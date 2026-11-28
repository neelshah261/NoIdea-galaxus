"""#133: conditioning on product type (the hosts' idea 3, #17). Synthetic rows and answers only, no model, no network.

``evaluate`` is a test stand-in in metric.evaluate's shape (the modules never compute an F1 themselves).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from model_synthetic import FakeMetric, fake_data_lane, train_table

from galaxus.model import type_prior, zero_shot
from galaxus.model.eval import PREDICTORS, run_eval
from galaxus.types import ImageItem, Prediction

DIMS = ("subject", "perspective")


def scored(frame, dimension):
    return frame[f"{dimension}_source"] == "human"


def solution_frame(part, dims):
    return part[["image_id", *dims]].astype(object)


def evaluate(solution, submission):
    joined = solution.merge(submission, on="image_id", suffixes=("_t", "_p"))
    out = {}
    for d in [c for c in solution.columns if c != "image_id"]:
        t, p = joined[f"{d}_t"].to_numpy(), joined[f"{d}_p"].to_numpy()
        f1s = []
        for c in np.union1d(t.astype(str), p.astype(str)):
            tp = np.sum((p == c) & (t == c))
            fp, fn = np.sum((p == c) & (t != c)), np.sum((p != c) & (t == c))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        out[d] = {"f1_macro": float(np.mean(f1s))}
    return out


def catalog(n_products=60, per=5, seed=0):
    """Two product types: 'gadget' images are almost all subject 'b', 'thing' images are 'a'. Perspective is random.

    The model answers 'a' for everything, unsure on gadgets and sure on things: the type knows better.
    """
    rng = np.random.default_rng(seed)
    n = n_products * per
    frame = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "product_id": [f"p{k // per}" for k in range(n)]})
    frame["product_type"] = np.where((np.arange(n) // per) % 3 == 0, "gadget", "thing")
    frame["subject"] = np.where(frame["product_type"] == "gadget", "b", "a")
    frame["perspective"] = rng.choice(["l", "r"], n)
    for d in DIMS:
        frame[f"{d}_source"] = "human"
    records = {}
    for row in frame.itertuples():
        gadget = row.product_type == "gadget"
        records[row.image_id] = {"image_id": row.image_id, "labels": {"subject": "a", "perspective": row.perspective},
                                 "confidence": {"subject": 0.45 if gadget else 0.95, "perspective": 0.9}}
    return frame, records


# ---------------------------------------------------------------- the type table


def test_the_table_holds_each_frequent_types_top_class_and_share():
    frame, _ = catalog()
    frame.loc[frame.index[:3], "subject"] = "c"  # three noisy gadget labels
    table = type_prior.type_table(frame, DIMS, scored, min_cells=20)
    assert table["subject"]["gadget"]["class"] == "b" and table["subject"]["gadget"]["share"] < 1
    assert table["subject"]["thing"] == {"class": "a", "share": 1.0, "cells": 200}
    assert table["perspective"]["thing"]["share"] < 0.7  # random perspective: an impure type


def test_rare_missing_and_unscored_cells_stay_out_of_the_table():
    frame, _ = catalog()
    frame.loc[frame.index[:5], "product_type"] = "rare"
    frame.loc[frame.index[5:10], "product_type"] = None
    frame.loc[frame["product_type"] == "gadget", "subject_source"] = "model"  # prelabels never count
    table = type_prior.type_table(frame, DIMS, scored, min_cells=20)
    assert "rare" not in table["subject"] and "None" not in table["subject"] and "gadget" not in table["subject"]
    assert type_prior.type_table(frame.drop(columns="product_type"), DIMS, scored) == {"subject": {}, "perspective": {}}


# ---------------------------------------------------------------- the rule


def test_a_cell_changes_only_when_unsure_and_the_type_is_pure_enough():
    table = {"subject": {"gadget": {"class": "b", "share": 0.9, "cells": 50}}}
    rules = {"subject": {"below": 0.5, "share": 0.8}}
    assert type_prior.apply_rules({"subject": "a"}, {"subject": 0.3}, "gadget", table, rules) == {"subject": "b"}
    assert type_prior.apply_rules({"subject": "a"}, {"subject": 0.7}, "gadget", table, rules) == {"subject": "a"}
    # no confidence = a default answer, the least sure: it changes too (+0.015 on EXP-0041, #176)
    assert type_prior.apply_rules({"subject": "a"}, None, "gadget", table, rules) == {"subject": "b"}
    assert type_prior.apply_rules({"subject": "a"}, {"perspective": 0.9}, "gadget", table, rules) == {"subject": "b"}
    loose = {"subject": {"gadget": {"class": "b", "share": 0.35, "cells": 50}}}
    assert type_prior.apply_rules({"subject": "a"}, {"subject": 0.3}, "gadget", loose,
                                  {"subject": {"below": 0.5, "share": min(type_prior.SHARES)}}) == {"subject": "b"}
    assert type_prior.apply_rules({"subject": "a"}, {"subject": 0.3}, None, table, rules) == {"subject": "a"}
    assert type_prior.apply_rules({"subject": "a"}, {"subject": 0.3}, "thing", table, rules) == {"subject": "a"}
    strict = {"subject": {"below": 0.5, "share": 0.95}}
    assert type_prior.apply_rules({"subject": "a"}, {"subject": 0.3}, "gadget", table, strict) == {"subject": "a"}


def test_learning_finds_the_gate_that_hands_unsure_gadgets_to_their_type():
    frame, records = catalog()
    table = type_prior.type_table(frame, DIMS, scored)
    result = type_prior.learn(records, frame, table, DIMS, solution_frame, evaluate)
    rule = result["rules"]["subject"]
    assert 0.45 < rule["below"] <= 0.95 and rule["share"] <= 1.0
    assert result["gains"]["subject"]["after"] > result["gains"]["subject"]["before"]
    assert "perspective" not in result["rules"]  # already perfect, and its types are impure


def test_the_crossfitted_answers_use_the_other_halfs_rules_and_drop_stale_confidences():
    frame, records = catalog()
    table = type_prior.type_table(frame, DIMS, scored)
    learned, by_image, full = type_prior.crossfit(records, frame, table, DIMS, solution_frame, evaluate)
    assert len(by_image) == len(frame) and learned[0]["rules"] and learned[1]["rules"]
    predictor = type_prior.TypePriorPredictor(table=table, rules=full["rules"], saved=records, rules_by_image=by_image)
    items = [ImageItem(r.image_id, "/x", {"product_type": r.product_type}) for r in frame.itertuples()]
    answers = predictor.predict(items)
    truth = frame.set_index("image_id")["subject"]
    assert all(p.labels["subject"] == truth[p.image_id] for p in answers)
    changed = next(p for p in answers if p.labels["subject"] == "b")
    assert "subject" not in changed.confidence and changed.confidence["perspective"] == 0.9


def test_without_product_context_the_base_answers_come_back_unchanged():
    frame, records = catalog()
    table = type_prior.type_table(frame, DIMS, scored)
    predictor = type_prior.TypePriorPredictor(table=table, rules={"subject": {"below": 1.01, "share": 0.6}},
                                              saved=records)
    answers = predictor.predict([ImageItem(i, "/x") for i in frame["image_id"]])
    assert all(p.labels == records[p.image_id]["labels"] for p in answers)


# ---------------------------------------------------------------- dev / final and the eval script


def test_a_live_inner_predictor_is_wrapped(monkeypatch, tmp_path):
    class Inner:
        def predict(self, batch):
            return [Prediction(image_id=i.image_id, labels={"subject": "a"}, confidence={"subject": 0.2})
                    for i in batch]

    saved = tmp_path / "rules.json"
    saved.write_text(json.dumps({"base": "EXP-0002", "rules": {"subject": {"below": 0.5, "share": 0.8}},
                                 "table": {"subject": {"gadget": {"class": "b", "share": 0.9, "cells": 40}}}}))
    monkeypatch.setenv("TYPE_PRIOR_RULES", str(saved))
    monkeypatch.setenv("TYPE_PRIOR_INNER", "x:y")
    monkeypatch.setattr("galaxus.serve.items.load_predictor", lambda spec: Inner())
    predictor = type_prior.from_env()
    gadget, other = predictor.predict([ImageItem("i1", "/x", {"product_type": "gadget"}), ImageItem("i2", "/x")])
    assert gadget.labels == {"subject": "b"} and other.labels == {"subject": "a"}


def test_from_env_says_what_to_set(monkeypatch):
    monkeypatch.delenv("TYPE_PRIOR_RULES", raising=False)
    with pytest.raises(RuntimeError, match="TYPE_PRIOR_RULES"):
        type_prior.from_env()


def test_the_eval_script_writes_an_exp_row(tmp_path, monkeypatch):
    train = train_table(products=12)
    out = tmp_path / "out"
    data = fake_data_lane(train, out, FakeMetric())
    path = out / "predictions" / "EXP-0002" / "holdout.jsonl"
    path.parent.mkdir(parents=True)
    lines = [json.dumps({"image_id": i, "labels": {d: f"{d}_a" for d in data.dimensions()},
                         "confidence": {d: 0.9 for d in data.dimensions()}})
             for i in data.split_holdout(train)[1]["image_id"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results = tmp_path / "metrics.json"
    results.write_text(json.dumps([{"id": "EXP-0002", "images_per_second": 4.95}]), encoding="utf-8")
    monkeypatch.setattr("galaxus.model.eval.RESULTS_FILE", results)
    monkeypatch.setenv("TYPE_PRIOR_BASE", "EXP-0002")
    row = run_eval("type-prior", data, results_path=results, exp_id="EXP-0097", allow_dirty=True)
    assert row["predictor"] == "type-prior" and row["images_per_second"] == pytest.approx(4.95)
    assert row["config"]["base"] == "EXP-0002"


def test_the_eval_script_says_which_base_to_set(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPE_PRIOR_BASE", raising=False)
    data = fake_data_lane(train_table(products=12), tmp_path)
    with pytest.raises(RuntimeError, match="TYPE_PRIOR_BASE"):
        PREDICTORS["type-prior"](train_table(products=12), data, 0)


# ---------------------------------------------------------------- the prompt route: zero-shot-type-first


class Backend:
    name, model = "fake", "apertus-test"


def predictor(**options):
    allowed = {"subject": frozenset({"a", "b"}), "perspective": frozenset({"l", "r"})}
    return zero_shot.ZeroShotPredictor(Backend(), defaults={"subject": "a", "perspective": "l"}, dimensions=DIMS,
                                       allowed=allowed, guideline="Rule one.", **options)


def test_type_first_adds_its_line_last_and_leaves_every_other_prompt_as_it_was():
    plain = zero_shot.system_prompt("Rule one.", DIMS)
    assert zero_shot.TYPE_FIRST not in plain
    assert zero_shot.system_prompt("Rule one.", DIMS, type_first=False) == plain
    both = zero_shot.system_prompt("Rule one.", DIMS, image_first=True, type_first=True)
    assert both.index(zero_shot.IMAGE_FIRST) < both.index(zero_shot.TYPE_FIRST) < both.index("Rule one.")
    assert predictor().prompt_sha256() != predictor(type_first=True).prompt_sha256()
    assert predictor(type_first=True).config()["type_first"] is True


def test_type_first_puts_the_product_type_first_and_keeps_the_other_fields_in_order():
    context = {"product_title": "Mouse M1", "brand": "B", "product_type": "Computer mouse", "category": "C"}
    item = ImageItem("i1", "/x", context)
    assert list(predictor()._context(item)) == ["product_title", "brand", "product_type", "category"]
    assert list(predictor(type_first=True)._context(item)) == ["product_type", "product_title", "brand", "category"]
    without = ImageItem("i2", "/x", {"product_title": "Mouse M1"})
    assert predictor(type_first=True)._context(without) == {"product_title": "Mouse M1"}
    assert predictor(type_first=True, use_context=False)._context(item) is None


def test_the_variant_and_the_prior_are_registered_for_the_eval_script():
    assert zero_shot.VARIANTS["zero-shot-type-first"] == {"type_first": True}
    assert {"zero-shot-type-first", "mirror-tta:zero-shot-type-first", "type-prior"} <= set(PREDICTORS)
