import pytest
from unittest.mock import Mock
from core.models import Job, Product, AspectSummary as AspectSummaryModel
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig, ScraperConfig
from pipeline.types import ProductInfo, AspectSummary, PipelineResult
from pipeline.arctic_shift_scraper import ArcticShiftScraper
import tasks.pipeline_task as task_module
from tasks.pipeline_task import run_pipeline_task, NO_OPINIONS_MESSAGE, _build_scraper


def make_config(backend):
    return AppConfig(
        llms=LLMConfig(product_understanding="ollama/x", aspect_extraction="gemini/x", summarization="ollama/x"),
        embeddings=EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2"),
        scraper=ScraperConfig(backend=backend),
    )


# ── scraper backend selection ───────────────────────────────────────────────────

def test_build_scraper_defaults_to_arctic_shift():
    assert isinstance(_build_scraper(make_config("arctic_shift")), ArcticShiftScraper)


def test_build_scraper_uses_praw_when_configured(mocker):
    sentinel = object()
    praw_cls = mocker.patch("tasks.pipeline_task.RedditScraper", return_value=sentinel)
    assert _build_scraper(make_config("praw")) is sentinel
    praw_cls.assert_called_once()


def test_build_scraper_rejects_unknown_backend():
    with pytest.raises(ValueError):
        _build_scraper(make_config("nope"))


def make_job(query="Sony WH-1000XM5"):
    return Job.objects.create(id="job-1", product_query=query, status="pending")


def make_product():
    return ProductInfo("sony-xm5", "Sony WH-1000XM5", "headphones", ["XM5"], ["headphones"])


def patch_runner(mocker, result):
    """Replace the component factory so the task drives a fake runner — no PRAW,
    no embedding model, no LLM. Returns the fake runner for assertions."""
    runner = Mock()
    runner.run.return_value = result
    mocker.patch.object(task_module, "build_runner", return_value=runner)
    mocker.patch.object(task_module, "load_config", return_value=Mock())
    return runner


# ── happy path ──────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_rerun_replaces_stale_summaries(mocker):
    # An earlier analysis left a summary for this product; re-running must replace
    # it, not pile new rows on top of stale ones.
    product = Product.objects.create(
        id="sony-xm5", canonical_name="Sony WH-1000XM5", category="headphones",
    )
    AspectSummaryModel.objects.create(
        product=product, aspect="STALE", mention_pct=99.0, positive_pct=0.0,
        negative_pct=0.0, recent_trend="stable", headline="old", detail="old",
    )
    make_job()
    patch_runner(mocker, PipelineResult(make_product(), [
        AspectSummary("battery", 80.0, 75.0, 25.0, "improving", "h", "d", ""),
    ]))

    run_pipeline_task("job-1")

    labels = set(
        AspectSummaryModel.objects.filter(product_id="sony-xm5").values_list("aspect", flat=True)
    )
    assert labels == {"battery"}  # the stale "STALE" row is gone


@pytest.mark.django_db
def test_persists_summaries_and_marks_done(mocker):
    job = make_job()
    summaries = [
        AspectSummary("battery", 80.0, 75.0, 25.0, "improving", "Great battery", "Lasts days", "Up recently"),
        AspectSummary("ANC", 60.0, 50.0, 50.0, "stable", "Decent ANC", "Mixed views", ""),
    ]
    patch_runner(mocker, PipelineResult(make_product(), summaries))

    run_pipeline_task("job-1")

    job.refresh_from_db()
    assert job.status == "done"
    assert job.progress == 100
    assert job.canonical_id == "sony-xm5"
    assert job.completed_at is not None

    assert Product.objects.filter(id="sony-xm5").exists()
    rows = AspectSummaryModel.objects.filter(product_id="sony-xm5")
    assert rows.count() == 2

    battery = rows.get(aspect="battery")
    assert battery.headline == "Great battery"
    assert battery.detail == "Lasts days"
    assert battery.trend_note == "Up recently"
    assert battery.mention_pct == 80.0
    assert battery.recent_trend == "improving"


# ── empty result: a successful job with a meaningful message ─────────────────────

@pytest.mark.django_db
def test_empty_result_marks_done_with_message(mocker):
    make_job("Obscure Product 9000")
    patch_runner(mocker, PipelineResult(make_product(), []))

    run_pipeline_task("job-1")

    job = Job.objects.get(id="job-1")
    assert job.status == "done"
    assert job.status_message == NO_OPINIONS_MESSAGE
    assert AspectSummaryModel.objects.count() == 0
    # product is still recorded so the UI can name what was searched
    assert Product.objects.filter(id="sony-xm5").exists()


# ── failure handling ────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_failure_marks_job_failed_and_reraises(mocker):
    make_job()
    runner = Mock()
    runner.run.side_effect = RuntimeError("LLM timed out")
    mocker.patch.object(task_module, "build_runner", return_value=runner)
    mocker.patch.object(task_module, "load_config", return_value=Mock())

    with pytest.raises(RuntimeError, match="LLM timed out"):
        run_pipeline_task("job-1")

    job = Job.objects.get(id="job-1")
    assert job.status == "failed"
    assert "LLM timed out" in job.status_message
    assert job.completed_at is not None


# ── progress reporting writes live status to the DB ─────────────────────────────

@pytest.mark.django_db
def test_progress_callback_writes_running_status(mocker):
    make_job()

    def fake_run(query, progress_callback=None):
        progress_callback("scraping")
        fresh = Job.objects.get(id="job-1")
        assert fresh.status == "running"
        assert fresh.status_message == "scraping"
        return PipelineResult(make_product(), [])

    runner = Mock()
    runner.run.side_effect = fake_run
    mocker.patch.object(task_module, "build_runner", return_value=runner)
    mocker.patch.object(task_module, "load_config", return_value=Mock())

    run_pipeline_task("job-1")
