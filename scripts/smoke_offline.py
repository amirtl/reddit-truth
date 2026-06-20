#!/usr/bin/env python
"""Live-minus-Reddit smoke test.

Exercises the REAL pipeline end to end — real product-understanding + extraction
+ summarization LLM calls, real embeddings, real clustering, real persistence to
Postgres via the actual Celery task — with only the Reddit scraper replaced by a
fixture of real-style comments. Use this to validate everything except PRAW when
Reddit API credentials aren't available yet.

Run (with Postgres + Ollama up and GEMINI_API_KEY in .env):
    uv run python scripts/smoke_offline.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

# Allow `python scripts/smoke_offline.py` by putting the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reddit_truth.settings")

import django  # noqa: E402

django.setup()

from core.models import Job, AspectSummary  # noqa: E402
from pipeline.runner import PipelineRunner  # noqa: E402
from pipeline.product_agent import ProductUnderstandingAgent  # noqa: E402
from pipeline.noise_filter import NoiseFilter  # noqa: E402
from pipeline.aspect_extractor import AspectExtractor  # noqa: E402
from pipeline.embedder_clusterer import EmbedderClusterer  # noqa: E402
from pipeline.quantifier import Quantifier  # noqa: E402
from pipeline.summarizer import Summarizer  # noqa: E402
from pipeline.types import RawComment  # noqa: E402
import tasks.pipeline_task as task_module  # noqa: E402


def _comment(cid, text, days_ago):
    return RawComment(
        id=cid,
        text=text,
        score=20,
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        subreddit="headphones",
        post_url="https://reddit.com/r/headphones/comments/example",
    )


# Real-style comments about the Sony WH-1000XM5, each >10 words (noise filter),
# spanning several aspects and a deliberate trend (recent battery complaints).
FIXTURE_COMMENTS = [
    _comment("c1", "The noise cancelling on these is genuinely incredible, it blocks out the entire airplane engine drone completely.", 220),
    _comment("c2", "ANC is the best I have ever experienced, far better than the Bose QC45 in my testing.", 210),
    _comment("c3", "Active noise cancellation is outstanding for commuting, the subway noise basically disappears when I put them on.", 30),
    _comment("c4", "Battery used to last me a full week but after eight months it barely makes it two days now.", 12),
    _comment("c5", "Battery life has degraded noticeably for me recently, it drains much faster than when I first bought them.", 9),
    _comment("c6", "The battery on mine is dying after only a year, really disappointing for headphones at this price point.", 15),
    _comment("c7", "Early on the battery easily lasted thirty hours per charge which was more than enough for long trips.", 230),
    _comment("c8", "Comfort is excellent for long listening sessions, the earcups are soft and they never clamp too hard.", 40),
    _comment("c9", "They are so comfortable I sometimes forget I am wearing them during an entire eight hour work day.", 25),
    _comment("c10", "Sound quality is rich and detailed with deep bass, though slightly warm for purists who want neutral tuning.", 60),
    _comment("c11", "The sound signature is fantastic for most genres, vocals are clear and the soundstage feels surprisingly wide.", 50),
    _comment("c12", "Call quality is mediocre, several people told me I sounded muffled and distant during important work calls.", 35),
    _comment("c13", "Microphone performance on calls is honestly the weakest part, friends say my voice sounds thin and processed.", 20),
    _comment("c14", "For the high price I expected a more premium build, the plastic hinges feel a little fragile to me.", 45),
    _comment("c15", "Worth every penny in my opinion, the combination of comfort, sound, and noise cancelling justifies the cost.", 70),
    _comment("c16", "The price is steep but the multipoint bluetooth and app features make them feel like a complete package.", 55),
]


class FixtureScraper:
    def run(self, product, limit=100):
        print(f"  [fixture] returning {len(FIXTURE_COMMENTS)} canned comments "
              f"(real scraper skipped — no Reddit creds)")
        return FIXTURE_COMMENTS


def fake_build_runner(config):
    return PipelineRunner(
        product_agent=ProductUnderstandingAgent(config),
        scraper=FixtureScraper(),
        noise_filter=NoiseFilter(),
        aspect_extractor=AspectExtractor(config),
        embedder_clusterer=EmbedderClusterer(config),
        quantifier=Quantifier(),
        summarizer=Summarizer(config),
    )


def main():
    task_module.build_runner = fake_build_runner  # swap only the factory

    job_id = uuid4().hex
    Job.objects.create(id=job_id, product_query="Sony WH-1000XM5", status="pending")
    print(f"→ created job {job_id}")

    def progress(stage):
        print(f"  [stage] {stage}")

    # Drive the REAL task body, but report stages live for visibility.
    import pipeline.runner as runner_module
    original_run = runner_module.PipelineRunner.run

    def traced_run(self, raw_query, progress_callback=None):
        return original_run(self, raw_query, progress_callback=progress)

    runner_module.PipelineRunner.run = traced_run
    try:
        task_module.run_pipeline_task(job_id)
    finally:
        runner_module.PipelineRunner.run = original_run

    job = Job.objects.get(id=job_id)
    print(f"\n→ job status: {job.status}")
    print(f"  canonical_id: {job.canonical_id}")
    print(f"  message: {job.status_message or '(none)'}")

    if job.status == "failed":
        print("\n✗ FAILED")
        return 1

    summaries = AspectSummary.objects.filter(product_id=job.canonical_id)
    print(f"\n→ {summaries.count()} aspect summaries persisted:\n")
    for s in summaries.order_by("-mention_pct"):
        print(f"  ● {s.aspect}  ({s.mention_pct:.0f}% mention, "
              f"{s.positive_pct:.0f}% pos / {s.negative_pct:.0f}% neg, {s.recent_trend})")
        print(f"    headline: {s.headline}")
        print(f"    detail:   {s.detail}")
        if s.trend_note:
            print(f"    trend:    {s.trend_note}")
        print()

    if summaries.count() == 0:
        print("⚠ pipeline completed but produced no summaries")
        return 1
    print("✓ live-minus-Reddit smoke passed — LLMs + embeddings + DB all working")
    return 0


if __name__ == "__main__":
    sys.exit(main())
