from typing import Protocol

from .types import ProductInfo, RawComment


class Scraper(Protocol):
    """The scraper port: turn a product into raw comments.

    Backends (PRAW, Arctic-Shift) conform structurally — the rest of the
    pipeline depends on this contract, not on any concrete implementation.
    """

    def run(self, product: ProductInfo, limit: int = 100) -> list[RawComment]:
        ...
