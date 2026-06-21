from typing import Callable
from .types import PipelineResult

ProgressCallback = Callable[[str], None]


class PipelineRunner:
    """Orchestrates the 7 pipeline stages from a raw query to aspect summaries.

    Pure domain logic: it knows the ORDER of the stages, nothing about how the
    components are built (config, secrets, PRAW) or where the results are stored.
    Components are injected so the Celery task wires real ones and tests wire mocks.
    """

    def __init__(
        self,
        *,
        product_agent,
        scraper,
        noise_filter,
        aspect_extractor,
        embedder_clusterer,
        quantifier,
        summarizer,
    ):
        self.product_agent = product_agent
        self.scraper = scraper
        self.noise_filter = noise_filter
        self.aspect_extractor = aspect_extractor
        self.embedder_clusterer = embedder_clusterer
        self.quantifier = quantifier
        self.summarizer = summarizer

    def run(self, raw_query: str, progress_callback: ProgressCallback | None = None) -> PipelineResult:
        report = progress_callback or (lambda stage: None)

        report("understanding")
        product = self.product_agent.run(raw_query)

        report("scraping")
        comments = self.scraper.run(product)

        report("filtering")
        comments = self.noise_filter.run(comments)

        report("extracting")
        claims = self.aspect_extractor.run(comments)

        # No aspects discussed → a valid outcome, not an error. Stop here and let
        # the caller tell the user "no opinions found" instead of doing empty work.
        # We still return the product so the caller can name it in that message.
        if not claims:
            return PipelineResult(product=product, summaries=[], comment_count=len(comments))

        report("clustering")
        clusters = self.embedder_clusterer.run(claims)

        report("quantifying")
        aspects = self.quantifier.run(clusters, comments)

        report("summarizing")
        summaries = self.summarizer.run(aspects, clusters)
        return PipelineResult(product=product, summaries=summaries, comment_count=len(comments))
