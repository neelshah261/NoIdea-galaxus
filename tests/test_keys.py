import pytest

from galaxus.keys import KeyRing, load_keys


class HTTPError(Exception):
    def __init__(self, status):
        self.status = status


def test_load_order_skips_empty_and_duplicates():
    env = {"LLM_API_KEY_10": "j", "LLM_API_KEY_2": "b", "LLM_API_KEY": "a", "LLM_API_KEY_3": " ", "LLM_API_KEY_4": "a"}
    assert load_keys(env) == ["a", "b", "j"]


def test_switches_when_credits_run_out():
    used = []

    def fn(key):
        used.append(key)
        if key == "a":
            raise HTTPError(402)
        return "ok"

    ring = KeyRing(["a", "b", "c"])
    assert ring.call(fn) == "ok" and ring.key == "b"
    assert ring.call(fn) == "ok" and used == ["a", "b", "b"]  # stays on b


def test_other_errors_are_not_swallowed():
    def fn(key):
        raise HTTPError(500)

    with pytest.raises(HTTPError):
        KeyRing(["a", "b"]).call(fn)


def test_all_keys_spent():
    def fn(key):
        raise HTTPError(402)

    with pytest.raises(RuntimeError, match="all 2 API keys"):
        KeyRing(["a", "b"]).call(fn)


def test_no_keys():
    with pytest.raises(RuntimeError, match="no API key"):
        KeyRing([])
