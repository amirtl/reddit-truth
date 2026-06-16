from datetime import datetime
from pipeline.types import RawComment
from pipeline.noise_filter import NoiseFilter


def make_comment(id, text, score=10):
    return RawComment(id, text, score, datetime(2024, 1, 1), "headphones", "https://reddit.com")


def test_filters_short_comments():
    f = NoiseFilter()
    short = make_comment("1", "Great product")
    assert f.run([short]) == []


def test_keeps_long_substantive_comments():
    f = NoiseFilter()
    long = make_comment("2", "Battery life is incredible and lasts me three full days on a single charge without ANC enabled")
    assert len(f.run([long])) == 1


def test_filters_bot_comments():
    f = NoiseFilter()
    bot = make_comment("3", "I am a bot and this action was performed automatically please contact the moderators")
    assert f.run([bot]) == []


def test_filters_heavily_downvoted():
    f = NoiseFilter()
    downvoted = make_comment("4", "Battery life is incredible and lasts me three full days on a single charge without ANC enabled", score=-10)
    assert f.run([downvoted]) == []


def test_mixed_batch_filters_correctly(sample_comments):
    f = NoiseFilter()
    result = f.run(sample_comments)
    ids = [c.id for c in result]
    assert "c5" not in ids   # "lol" is too short
    assert "c1" in ids
    assert "c2" in ids


def test_word_count_boundary():
    f = NoiseFilter()
    nine_words  = make_comment("a", "one two three four five six seven eight nine")
    ten_words   = make_comment("b", "one two three four five six seven eight nine ten")
    assert f.run([nine_words]) == []
    assert len(f.run([ten_words])) == 1


def test_score_boundary():
    f = NoiseFilter()
    text = "one two three four five six seven eight nine ten"
    at_limit    = make_comment("a", text, score=-5)
    below_limit = make_comment("b", text, score=-6)
    assert len(f.run([at_limit])) == 1
    assert f.run([below_limit]) == []
