"""#49 LoRA, the data side and the predictor: synthetic rows only, no torch, no model, no network."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import pytest

from galaxus.model import lora

DIMS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
CLASSES = {d: ["alpha", "beta", "none"] for d in DIMS}


def char_offsets(text):
    ids = [b + 1 for b in text.encode("utf-8")]
    return ids, [(i, i + 1) for i in range(len(ids))]


def test_serialized_answer_spans_point_at_the_values():
    text, spans = lora.serialize_answer({"subject": "alpha", "background": "none"})
    assert json.loads(text) == {"subject": "alpha", "background": "none"}
    assert text[slice(*spans["subject"])] == "alpha" and text[slice(*spans["background"])] == "none"


def test_labels_mask_the_prompt_and_prelabel_values_only():
    target = {d: "alpha" for d in DIMS}
    answer = lora.serialize_answer(target)[0]
    ids = char_offsets("<img> prompt text\n" + answer)[0]
    labels = lora.training_labels(ids, answer, ("subject", "background"), char_offsets)
    start, positions = lora.value_positions(ids, answer, char_offsets)
    assert all(x == -100 for x in labels[:start])
    for dim in DIMS:
        kept = all(labels[p] == ids[p] for p in positions[dim])
        masked = all(labels[p] == -100 for p in positions[dim])
        assert kept if dim in ("subject", "background") else masked
    # the JSON structure outside the values still trains the format
    assert labels[start] == ids[start]


def test_training_set_repeats_rare_human_classes_and_adds_mirrors():
    rows = []
    for i in range(450):  # 225 human text_language labels: above the 200 target, so only 'beta' is rare
        rare = i < 10
        row = {"image_id": f"i{i}", "image_file": f"/x/{i}.jpg", "product_title": "t"}
        for d in DIMS:
            row[d] = "beta" if (rare and d == "subject") else "alpha"
            row[f"{d}_source"] = "human" if d != "text_language" or i % 2 else "model"
        rows.append(row)
    frame = pd.DataFrame(rows)
    mirrored = frame.head(5).assign(mirrored=True)
    data = lora.build_training_set(frame, DIMS, mirrored)
    assert data.stats["images"] == 450 and data.stats["mirrored_added"] == 5
    assert data.stats["repeats_added"] == 10 * (lora.MAX_REPEATS - 1)  # 10 rare rows, capped at 3x
    assert data.stats["prelabels_masked"]["text_language"] == 225
    rare = [e for e in data.examples if e.target["subject"] == "beta" and not e.mirrored]
    assert len(rare) == 30
    assert sum(e.mirrored for e in data.examples) == 5


def test_prompts_list_classes_and_survive_missing_context():
    system = lora.system_prompt(CLASSES)
    assert all(d in system for d in DIMS) and "alpha, beta, none" in system
    assert "not available" in lora.user_prompt(None)
    assert "product title: Mouse" in lora.user_prompt({"product_title": "Mouse", "brand": ""})


def test_parse_answer_accepts_only_complete_valid_json():
    good = json.dumps({d: "alpha" for d in DIMS})
    assert lora.parse_answer(f"sure: {good} done", CLASSES) == {d: "alpha" for d in DIMS}
    assert lora.parse_answer(good.replace('"alpha"', '"gamma"', 1), CLASSES) is None
    assert lora.parse_answer("no json here", CLASSES) is None


def test_plan_arithmetic():
    plan = lora.Plan(examples_per_epoch=1600, epochs=2, batch=1, accumulation=16, tokens_per_example=1500,
                     tokens_per_second=1500, dollars_per_hour=1.8)
    assert plan.steps == 200 and plan.hours == pytest.approx(3200 / 3600) and plan.dollars == pytest.approx(1.6)
    assert lora.tokens_per_example(512) == 1024 + 450 + 70


@dataclass
class Generation:
    text: str


def test_predictor_parses_answers_and_falls_back_on_garbage():
    replies = iter([json.dumps({d: "beta" for d in DIMS}), "I cannot tell"])

    class Backend:
        def generate(self, messages, images, json_schema, max_tokens, temperature):
            assert messages[0]["role"] == "system" and images == [b"jpeg"] and temperature == 0.0
            return Generation(next(replies))

    @dataclass(frozen=True)
    class Item:
        image_id: str
        context: dict | None = None

    predictor = lora.LoraPredictor(Backend(), CLASSES, {d: "none" for d in DIMS}, lambda item: b"jpeg")
    first, second = predictor.predict([Item("a"), Item("b")])
    assert first.labels["subject"] == "beta" and second.labels["subject"] == "none"
    assert predictor.fallbacks == 1 and predictor.config()["fallbacks"] == 1


def test_the_eval_script_knows_the_lora_predictor_and_it_scores_through_a_fake_endpoint():
    from galaxus.model import eval as model_eval

    assert "lora" in model_eval.PREDICTORS

    class Data:
        def dimensions(self):
            return DIMS

        def allowed_classes(self):
            return {d: frozenset(CLASSES[d]) for d in DIMS}

        def scored_mask(self, rows, dim):
            return rows[f"{dim}_source"] == "human"

    rows = pd.DataFrame({"image_id": ["a", "b"], **{d: ["beta", "beta"] for d in DIMS},
                         **{f"{d}_source": ["human", "human"] for d in DIMS}})

    class Backend:
        model = "apertus-lora"

        def generate(self, messages, images, json_schema, max_tokens, temperature):
            return Generation(json.dumps({d: "alpha" for d in DIMS}))

    predictor = lora.eval_factory(backend=Backend())(rows, Data(), 0)
    predictor.image_bytes = lambda item: b"jpeg"
    assert predictor.model == "apertus-lora" and predictor.fallback == {d: "beta" for d in DIMS}

    @dataclass(frozen=True)
    class Item:
        image_id: str
        path: str = ""
        context: dict | None = None

    assert predictor.predict([Item("x")])[0].labels == {d: "alpha" for d in DIMS}


def test_without_an_endpoint_the_lora_predictor_says_what_to_set():
    with pytest.raises(RuntimeError, match="LORA_BASE_URL"):
        lora.eval_factory(environ={})(pd.DataFrame(), None, 0)


# ---------------------------------------------------------------- #100: from_env and the schema at inference


class _FakeVllm:
    """Stands in for VllmBackend: records how it was built and answers one valid class per dimension."""

    built: ClassVar[list] = []

    def __init__(self, base_url, model, **options):
        self.base_url, self.model, self.options = base_url, model, options
        self.schemas = []
        _FakeVllm.built.append(self)

    def generate(self, messages, images, json_schema=None, *, max_tokens, temperature):
        self.schemas.append(json_schema)
        return Generation(json.dumps({d: min(json_schema["properties"][d]["enum"]) for d in json_schema["required"]}))


@pytest.fixture
def fake_endpoint(monkeypatch):
    """No network and no $GALAXUS_DATA: a fake vLLM backend, and a majority answer from the contract's classes."""
    from galaxus.data.classes import allowed_classes

    _FakeVllm.built = []
    monkeypatch.setattr("galaxus.model.apertus.vllm.VllmBackend", _FakeVllm)
    monkeypatch.setattr("galaxus.model.baselines.fit_from_data_lane",
                        lambda: {d: min(names) for d, names in allowed_classes().items()})
    return _FakeVllm


@dataclass(frozen=True)
class _Item:
    image_id: str
    path: str = ""
    context: dict | None = None


@pytest.mark.real_contract
def test_from_env_builds_with_no_arguments_from_the_contract(fake_endpoint):
    from galaxus.data.classes import allowed_classes, dimensions

    predictor = lora.from_env({"LORA_BASE_URL": "http://gpu:8000/v1"})
    backend = fake_endpoint.built[-1]
    assert backend.base_url == "http://gpu:8000/v1" and backend.model == "apertus-lora"
    assert backend.options["retries"] == lora.RETRIES and predictor.model == "apertus-lora"
    assert list(predictor.classes) == list(dimensions())
    assert predictor.classes == {d: sorted(allowed_classes()[d]) for d in dimensions()}
    assert predictor.image_bytes is lora._cached_image  # the same 512 px pixels as eval_factory
    assert set(predictor.fallback) == set(dimensions())


@pytest.mark.real_contract
def test_a_served_name_can_be_chosen(fake_endpoint):
    lora.from_env({"LORA_BASE_URL": "http://gpu:8000/v1", "LORA_SERVED_MODEL": "run-7"})
    assert fake_endpoint.built[-1].model == "run-7"


def test_from_env_without_an_endpoint_stops_before_reading_anything(monkeypatch):
    monkeypatch.setattr("galaxus.model.baselines.fit_from_data_lane", lambda: pytest.fail("read the data"))
    with pytest.raises(RuntimeError, match="LORA_BASE_URL"):
        lora.from_env({})


@pytest.mark.real_contract
def test_the_class_schema_reaches_generate_and_the_prompt_is_the_training_one(fake_endpoint):
    predictor = lora.from_env({"LORA_BASE_URL": "http://gpu:8000/v1"})
    predictor.image_bytes = lambda item: b"jpeg"
    answer = predictor.predict([_Item("x")])[0]
    schema = fake_endpoint.built[-1].schemas[-1]
    assert schema["required"] == list(predictor.classes) and schema["additionalProperties"] is False
    assert all(schema["properties"][d]["enum"] == sorted(predictor.classes[d]) for d in predictor.classes)
    assert answer.labels == {d: min(predictor.classes[d]) for d in predictor.classes}
    assert predictor.fallbacks == 0 and predictor.system == lora.system_prompt(predictor.classes)


def test_an_answer_with_an_unknown_class_is_counted_as_a_fallback():
    class Backend:
        def generate(self, messages, images, json_schema=None, *, max_tokens, temperature):
            return Generation(json.dumps({**{d: "alpha" for d in DIMS}, "subject": "not_a_class"}))

    predictor = lora.LoraPredictor(Backend(), CLASSES, {d: "none" for d in DIMS}, lambda item: b"jpeg")
    assert predictor.predict([_Item("a")])[0].labels == {d: "none" for d in DIMS}
    assert predictor.fallbacks == 1


@pytest.mark.real_contract
def test_submit_and_final_run_load_it_by_name(fake_endpoint, monkeypatch):
    """``submit --run`` and ``final_run --predictor`` both build through ``load_predictor``."""
    from galaxus.serve.items import load_predictor

    monkeypatch.setenv("LORA_BASE_URL", "http://gpu:8000/v1")
    predictor = load_predictor("galaxus.model.lora:from_env")
    assert isinstance(predictor, lora.LoraPredictor) and fake_endpoint.built


def test_predictions_are_the_contract_type_so_eval_and_the_runner_can_read_them():
    """#76 step 4: ``eval lora`` reads ``prediction.image_id``; plain dicts crashed it."""
    from galaxus.types import Prediction

    class Backend:
        def generate(self, messages, images, json_schema=None, *, max_tokens, temperature):
            return Generation(json.dumps({d: "alpha" for d in DIMS}))

    predictor = lora.LoraPredictor(Backend(), CLASSES, {d: "none" for d in DIMS}, lambda item: b"jpeg")
    answer = predictor.predict([_Item("a")])[0]
    assert isinstance(answer, Prediction) and answer.image_id == "a"


def test_eval_lora_scores_end_to_end_through_a_fake_endpoint(tmp_path, monkeypatch):
    """``python -m galaxus.model.eval lora`` (#76 step 4) on synthetic data: one EXP row, no crash."""
    from model_synthetic import fake_data_lane, train_table

    from galaxus.model import eval as model_eval

    class Backend:
        model = "apertus-lora"

        def generate(self, messages, images, json_schema=None, *, max_tokens, temperature):
            return Generation(json.dumps({d: json_schema["properties"][d]["enum"][0] for d in json_schema["required"]}))

    monkeypatch.setitem(model_eval.PREDICTORS, "lora", lora.eval_factory(backend=Backend()))
    monkeypatch.setattr(lora, "_cached_image", lambda item: b"jpeg")
    row = model_eval.run_eval("lora", fake_data_lane(train_table(), tmp_path / "out"),
                              results_path=tmp_path / "metrics.json", exp_id="EXP-0004", allow_dirty=True)
    assert row["predictor"] == "lora" and row["model"] == "apertus-lora" and row["config"]["fallbacks"] == 0
    assert (tmp_path / "out" / "predictions" / "EXP-0004" / "holdout.jsonl").is_file()


def test_an_api_key_from_the_environment_reaches_the_backend(fake_endpoint):
    """#76: an HF Job's exposed port wants an HF token; ``$LORA_API_KEY`` carries it, unset means no header."""
    lora._vllm_backend({"LORA_BASE_URL": "http://gpu:8000/v1", "LORA_API_KEY": "secret-value"})
    assert fake_endpoint.built[-1].options.get("api_key") == "secret-value"
    lora._vllm_backend({"LORA_BASE_URL": "http://gpu:8000/v1"})
    assert fake_endpoint.built[-1].options.get("api_key") is None


# ---------------------------------------------------------------- #75: confidences from the LoRA's token logprobs


@dataclass
class _GenerationWithLogprobs:
    text: str
    logprobs: list | None = None


def test_confidences_come_from_the_class_name_tokens_and_a_fallback_gets_none():
    import math

    answer = json.dumps({d: "alpha" for d in DIMS})
    start = answer.index('"alpha"') + 1  # the first dimension's class name
    tokens = [{"token": ch, "logprob": (-0.1 if start <= i < start + len("alpha") else 0.0)}
              for i, ch in enumerate(answer)]
    replies = iter([_GenerationWithLogprobs(answer, tokens), _GenerationWithLogprobs("not json", None)])

    class Backend:
        def generate(self, messages, images, json_schema=None, *, max_tokens, temperature):
            return next(replies)

    predictor = lora.LoraPredictor(Backend(), CLASSES, {d: "none" for d in DIMS}, lambda item: b"jpeg")
    good, fallback = predictor.predict([_Item("a"), _Item("b")])
    assert good.confidence[DIMS[0]] == pytest.approx(math.exp(-0.5))
    assert all(good.confidence[d] == pytest.approx(1.0) for d in DIMS[1:])
    assert fallback.confidence is None and predictor.fallbacks == 1
    assert predictor.config()["no_confidence_cells"] == len(DIMS)


def test_without_logprobs_there_is_no_confidence_and_it_is_counted():
    class Backend:
        def generate(self, messages, images, json_schema=None, *, max_tokens, temperature):
            return Generation(json.dumps({d: "beta" for d in DIMS}))

    predictor = lora.LoraPredictor(Backend(), CLASSES, {d: "none" for d in DIMS}, lambda item: b"jpeg")
    assert predictor.predict([_Item("a")])[0].confidence is None
    assert predictor.config()["no_confidence_cells"] == len(DIMS)


def test_the_vllm_backend_asks_for_logprobs(fake_endpoint):
    lora._vllm_backend({"LORA_BASE_URL": "http://gpu:8000/v1"})
    assert fake_endpoint.built[-1].options.get("extra_body") == {"logprobs": True}
