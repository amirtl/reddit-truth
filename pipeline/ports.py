from typing import Protocol

from .types import ProductInfo, RawComment


class Scraper(Protocol):
    """The scraper port: turn a product into raw comments.

    Backends (PRAW, Arctic-Shift) conform structurally — the rest of the
    pipeline depends on this contract, not on any concrete implementation.
    """

    def run(self, product: ProductInfo, limit: int = 100) -> list[RawComment]:
        ...


class ProductAgent(Protocol):
    """The product-understanding port: resolve a raw query into a ProductInfo.

    The single-call agent and the agentic (self-correcting) agent both conform,
    so the runner depends on this contract, not on a concrete implementation.
    """

    def run(self, raw_query: str) -> ProductInfo:
        ...
