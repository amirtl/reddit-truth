import pytest
from datetime import datetime
from pipeline.types import ProductInfo, RawComment


@pytest.fixture
def sample_product_info():
    return ProductInfo(
        canonical_id="sony-wh-1000xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=["WH-1000XM5", "Sony XM5 headphones"],
        subreddits=["headphones", "audiophile", "sony"],
    )


@pytest.fixture
def sample_comments():
    return [
        RawComment("c1", "Battery life is incredible, lasts 3 days easily on a single charge", 42, datetime(2024, 6, 1), "headphones", "https://reddit.com/1"),
        RawComment("c2", "ANC is absolutely the best I have ever tried, destroys the Bose QC45", 38, datetime(2024, 6, 2), "audiophile", "https://reddit.com/2"),
        RawComment("c3", "Mine died after 8 months, ear cushions completely degraded", 29, datetime(2024, 5, 1), "headphones", "https://reddit.com/3"),
        RawComment("c4", "The companion app crashes constantly on Android", 15, datetime(2024, 4, 1), "sony", "https://reddit.com/4"),
        RawComment("c5", "lol", 1, datetime(2024, 3, 1), "headphones", "https://reddit.com/5"),
    ]
