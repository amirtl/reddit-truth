import pytest

from pipeline.agent_tools import validate_subreddit
from pipeline.arctic_shift_client import ArcticShiftClient


@pytest.mark.integration
def test_real_subreddit_has_threads_fake_one_does_not():
    """Contract test against the live Arctic-Shift API: a real subreddit yields
    product threads, a fabricated one yields none. Guards the assumptions our
    mocked unit tests bake in. Run with: pytest -m integration"""
    client = ArcticShiftClient()
    real = validate_subreddit(client, "headphones", ["Sony WH-1000XM5", "WH-1000XM5"])
    fake = validate_subreddit(client, "HeadphoneGears", ["Sony WH-1000XM5"])
    assert real and real > 0          # real subreddit yields threads
    assert fake == 0                   # fabricated subreddit yields none
