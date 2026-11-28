"""#43: the zero-shot predictor on our own vLLM endpoint (bf16 vs FP8), chosen by environment. No network."""

from __future__ import annotations

import pytest

from galaxus.model import zero_shot


def test_without_the_variable_the_gateway_is_used(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr("galaxus.model.apertus.backend", lambda name, **options: calls.append((name, options))
                        or sentinel)
    assert zero_shot.default_backend({}) is sentinel
    assert calls[0][0] == "api" and calls[0][1]["extra_body"] == {"logprobs": True}


def test_vllm_reads_url_name_and_key_from_the_environment():
    backend = zero_shot.default_backend({
        "ZERO_SHOT_BACKEND": "vllm", "ZERO_SHOT_BASE_URL": "https://job.example/v1",
        "ZERO_SHOT_SERVED_MODEL": "apertus-8b-fp8", "ZERO_SHOT_API_KEY": "secret-value",
    })
    assert backend.name == "vllm" and backend.model == "apertus-8b-fp8"
    assert backend.url.startswith("https://job.example/v1") and backend.extra_body == {"logprobs": True}
    assert backend.retries == zero_shot.GATEWAY_RETRIES
    assert "secret-value" not in repr(vars(backend).get("url"))


def test_vllm_without_a_served_name_asks_for_the_hub_model():
    from galaxus.model.apertus.backend import APERTUS_8B

    backend = zero_shot.default_backend({"ZERO_SHOT_BACKEND": "vllm", "ZERO_SHOT_BASE_URL": "http://gpu:8000/v1"})
    assert backend.model == APERTUS_8B


def test_vllm_without_a_url_and_an_unknown_backend_are_refused():
    with pytest.raises(RuntimeError, match="ZERO_SHOT_BASE_URL"):
        zero_shot.default_backend({"ZERO_SHOT_BACKEND": "vllm"})
    with pytest.raises(ValueError, match="api .* or vllm"):
        zero_shot.default_backend({"ZERO_SHOT_BACKEND": "local"})


def test_a_predictor_built_without_a_backend_takes_the_environments(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_BACKEND", "vllm")
    monkeypatch.setenv("ZERO_SHOT_BASE_URL", "http://gpu:8000/v1")
    monkeypatch.setenv("ZERO_SHOT_SERVED_MODEL", "apertus-8b")
    predictor = zero_shot.ZeroShotPredictor(
        defaults={"subject": "a"}, dimensions=("subject",), allowed={"subject": frozenset({"a", "b"})},
        guideline="Rule one.",
    )
    assert predictor.model == "apertus-8b" and predictor.config()["backend"] == "vllm"
