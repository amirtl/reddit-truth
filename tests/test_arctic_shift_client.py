import requests

from pipeline.arctic_shift_client import ArcticShiftClient


def test_post_titles_returns_titles(mocker):
    resp = mocker.Mock()
    resp.json.return_value = {"data": [{"title": "Sony WH-1000XM5 review"}, {"title": "x"}]}
    mocker.patch("pipeline.arctic_shift_client.requests.get", return_value=resp)
    client = ArcticShiftClient(request_delay=0)
    assert client.post_titles("headphones", "XM5") == ["Sony WH-1000XM5 review", "x"]


def test_post_titles_returns_none_on_persistent_error(mocker):
    mocker.patch("pipeline.arctic_shift_client.requests.get",
                 side_effect=requests.exceptions.ConnectionError("boom"))
    client = ArcticShiftClient(request_delay=0, max_retries=2)
    assert client.post_titles("headphones", "XM5") is None


def test_alias_matcher_is_word_bounded():
    m = ArcticShiftClient.alias_matcher(["XM5"])
    assert m.search("my XM5 is great")
    assert not m.search("the XM500 model")
