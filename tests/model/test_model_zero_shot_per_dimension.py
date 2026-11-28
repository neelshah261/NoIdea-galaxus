"""#156: zero-shot with one question per dimension. A fake backend and synthetic images: no network, no data."""

from __future__ import annotations

import json
import math
import threading

import pytest
from model_synthetic import DIMENSIONS
from test_model_zero_shot import GOOD, FakeBackend, item, predictor, with_logprobs

from galaxus.model.apertus.backend import Generation
from galaxus.model.zero_shot import OBSERVATION_FIELD, PRESETS


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A synthetic ``$GALAXUS_DATA`` for the image cache, as in test_model_zero_shot."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("GALAXUS_DATA", str(root))
    return root


class SchemaBackend:
    """Answers each request from the one dimension its schema asks for; ``bad`` dimensions get garbage."""

    name = "fake"
    model = "apertus-test"

    def __init__(self, bad=()):
        self.bad = set(bad)
        self.asked: list[str] = []
        self._lock = threading.Lock()

    def generate(self, messages, images=(), json_schema=None, *, max_tokens=256, temperature=0.0):
        [dimension] = [name for name in json_schema["properties"] if name != OBSERVATION_FIELD]
        with self._lock:
            self.asked.append(dimension)
        text = "garbage" if dimension in self.bad else json.dumps({dimension: GOOD[dimension]})
        return Generation(text=text, prompt_tokens=100, completion_tokens=5)


def test_six_requests_per_image_each_with_only_its_dimension_and_its_classes(data_dir):
    backend = FakeBackend(with_logprobs(GOOD))
    zero_shot = predictor(backend, per_dimension=True, workers=4)
    predictions = zero_shot.predict([item(data_dir, "img_00000"), item(data_dir, "img_00001")])
    assert len(backend.calls) == 2 * len(DIMENSIONS)
    for call in backend.calls:
        [dimension] = call["schema"]["properties"]
        assert call["schema"]["required"] == [dimension]
        assert f"these dimensions: {dimension}.\n" in call["messages"][0]["content"]
    assert sorted(next(iter(c["schema"]["properties"])) for c in backend.calls) == sorted(DIMENSIONS * 2)
    for prediction in predictions:
        assert prediction.labels == GOOD
        assert set(prediction.confidence) == set(DIMENSIONS)
        assert all(math.isclose(value, math.exp(-0.1)) for value in prediction.confidence.values())
    assert [p.image_id for p in predictions] == ["img_00000", "img_00001"]
    assert zero_shot.config()["per_dimension"] is True and zero_shot.config()["images"] == 2


def test_observe_first_puts_the_observation_before_the_one_dimension(data_dir):
    backend = FakeBackend(json.dumps(GOOD))
    predictor(backend, per_dimension=True, observe_first=True).predict([item(data_dir)])
    for call in backend.calls:
        assert call["schema"]["required"][0] == OBSERVATION_FIELD and len(call["schema"]["required"]) == 2


def test_only_the_bad_dimension_is_asked_again_and_then_defaults(data_dir):
    backend = SchemaBackend(bad={DIMENSIONS[0]})
    zero_shot = predictor(backend, per_dimension=True)
    [prediction] = zero_shot.predict([item(data_dir)])
    assert backend.asked.count(DIMENSIONS[0]) == 2
    assert all(backend.asked.count(d) == 1 for d in DIMENSIONS[1:])
    assert prediction.labels[DIMENSIONS[0]] == zero_shot.defaults[DIMENSIONS[0]]
    assert {d: prediction.labels[d] for d in DIMENSIONS[1:]} == {d: GOOD[d] for d in DIMENSIONS[1:]}
    assert zero_shot.config()["retries"] == 1 and zero_shot.config()["defaulted_cells"] == 1


def test_the_prompt_hash_tells_the_two_modes_apart_and_the_plain_one_is_unchanged():
    plain, joint = predictor(FakeBackend("{}")), predictor(FakeBackend("{}"), per_dimension=True)
    assert plain.prompt_sha256() != joint.prompt_sha256()
    assert plain.questions == {}


def test_the_three_variants_ask_one_question_per_dimension():
    names = [name for name, options in PRESETS.items() if options.get("per_dimension")]
    assert names == ["zero-shot-per-dimension", "zero-shot-per-dimension-observe-first",
                     "zero-shot-per-dimension-observe-first-no-context"]
