from celery import shared_task
from django.utils import timezone

from core.models import Job, Product, AspectSummary as AspectSummaryModel
from pipeline.config import load_config
from pipeline.runner import PipelineRunner
from pipeline.product_agent import ProductUnderstandingAgent
from pipeline.scraper import RedditScraper
from pipeline.noise_filter import NoiseFilter
from pipeline.aspect_extractor import AspectExtractor
from pipeline.embedder_clusterer import EmbedderClusterer
from pipeline.quantifier import Quantifier
from pipeline.summarizer import Summarizer
from pipeline.types import PipelineResult
from reddit_truth.env_config import env

NO_OPINIONS_MESSAGE = "No opinions found — try a more popular product or broader search terms."


def build_runner(config) -> PipelineRunner:
    """Factory: construct the real pipeline components from config + secrets.

    This is the one place that knows about PRAW credentials, model names, and the
    embedding model — keeping that knowledge out of the pure pipeline. Exercised
    end-to-end by the Task 15 smoke test rather than unit tests.
    """
    return PipelineRunner(
        product_agent=ProductUnderstandingAgent(config),
        scraper=RedditScraper(
            env.reddit_client_id, env.reddit_client_secret, env.reddit_user_agent
        ),
        noise_filter=NoiseFilter(),
        aspect_extractor=AspectExtractor(config),
        embedder_clusterer=EmbedderClusterer(config),
        quantifier=Quantifier(),
        summarizer=Summarizer(config),
    )


@shared_task
def run_pipeline_task(job_id: str) -> None:
    """Adapter between the web request and the pure pipeline.

    Owns the job lifecycle (running → done/failed), translates pipeline
    dataclasses into ORM rows, and reports each stage to the DB so the UI can
    show live progress.
    """
    job = Job.objects.get(id=job_id)
    try:
        config = load_config()
        runner = build_runner(config)

        def progress(stage: str) -> None:
            job.status = "running"
            job.status_message = stage
            job.save(update_fields=["status", "status_message"])

        result = runner.run(job.product_query, progress_callback=progress)
        _persist_result(job, result)
    except Exception as exc:
        _mark_failed(job, exc)
        raise


def _persist_result(job: Job, result: PipelineResult) -> None:
    product = _save_product(result)

    for summary in result.summaries:
        AspectSummaryModel.objects.create(
            product=product,
            aspect=summary.label,
            mention_pct=summary.mention_pct,
            positive_pct=summary.positive_pct,
            negative_pct=summary.negative_pct,
            recent_trend=summary.recent_trend,
            headline=summary.headline,
            detail=summary.detail,
            trend_note=summary.trend_note,
        )

    job.canonical_id = result.product.canonical_id
    job.status = "done"
    job.progress = 100
    # Empty summaries is a valid outcome — tell the user, don't fail the job.
    job.status_message = "" if result.summaries else NO_OPINIONS_MESSAGE
    job.completed_at = timezone.now()
    job.save()


def _save_product(result: PipelineResult) -> Product:
    info = result.product
    product, _ = Product.objects.update_or_create(
        id=info.canonical_id,
        defaults={
            "canonical_name": info.canonical_name,
            "category": info.category,
            "search_terms": info.search_terms,
            "subreddits": info.subreddits,
        },
    )
    return product


def _mark_failed(job: Job, exc: Exception) -> None:
    job.status = "failed"
    job.status_message = str(exc)[:500]
    job.completed_at = timezone.now()
    job.save()
