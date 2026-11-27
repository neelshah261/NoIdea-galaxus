"""#49 LoRA, the data side and the predictor: synthetic rows only, no torch, no model, no network."""

from __future__ import annotations

import json
from dataclasses import dataclass

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
        def generate(self, messages, images, max_tokens, temperature):
            assert messages[0]["role"] == "system" and images == [b"jpeg"] and temperature == 0.0
            return Generation(next(replies))

    @dataclass(frozen=True)
    class Item:
        image_id: str
        context: dict | None = None

    predictor = lora.LoraPredictor(Backend(), CLASSES, {d: "none" for d in DIMS}, lambda item: b"jpeg")
    first, second = predictor.predict([Item("a"), Item("b")])
    assert first["labels"]["subject"] == "beta" and second["labels"]["subject"] == "none"
    assert predictor.fallbacks == 1 and predictor.config()["fallbacks"] == 1
