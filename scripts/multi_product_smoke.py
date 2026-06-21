#!/usr/bin/env python
"""Live multi-product validation of the scraping path.

For several diverse products, runs the real product agent + Arctic-Shift scraper
and reports: generated search terms (specific, not generic?), subreddits, and how
many comments were harvested (recall > 0 after the title-focus filter?). Catches
generalization bugs that single-product testing hides.

Run with Ollama up + GEMINI_API_KEY set:  uv run python scripts/multi_product_smoke.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reddit_truth.settings")

import django  # noqa: E402
django.setup()

from pipeline.config import load_config  # noqa: E402
from pipeline.product_agent import ProductUnderstandingAgent  # noqa: E402
from pipeline.arctic_shift_scraper import ArcticShiftScraper  # noqa: E402

QUERIES = [
    "Sony WH-1000XM5",
    "iPhone 15 Pro",
    "Steam Deck",
    "MacBook Air M3",
    "Baldur's Gate 3",
    "Dyson V15",
]

config = load_config()
agent = ProductUnderstandingAgent(config)
scraper = ArcticShiftScraper(request_delay=0.3)

print(f"{'product':<22} {'#terms':>6} {'#comments':>10}  result")
print("-" * 70)
for q in QUERIES:
    try:
        product = ProductUnderstandingAgent(config).run(q)
        comments = scraper.run(product, limit=60)
        verdict = "OK" if comments else "NO COMMENTS (recall starved?)"
        print(f"{q:<22} {len(product.search_terms):>6} {len(comments):>10}  {verdict}")
        print(f"    terms: {product.search_terms}")
        print(f"    subs:  {product.subreddits}")
        if comments:
            print(f"    sample: {comments[0].text[:80].strip()!r}")
    except Exception as e:
        print(f"{q:<22}  ERROR: {type(e).__name__}: {str(e)[:80]}")
    print()
