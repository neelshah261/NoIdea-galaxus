"""Confidence-gated recalibration (#42) on synthetic answers. No model, no competition data, no network.

``evaluate`` is a test stand-in in metric.evaluate's shape (the module itself never computes an F1).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from model_synthetic import FakeMetric, fake_data_lane, train_table

from galaxus.model import recalibrate
from galaxus.model.contract import ImageItem
from galaxus.model.eval import run_eval

DIMS = ("subject", "perspective")
ALLOWED = {"subject": ["a", "b", "c"], "perspective": ["l", "r"]}


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


def swallowing_model(n_products=60, per=5, seed=0):
    """Truth: 'b' is rare. The model says 'a' for every 'b', but with low confidence there; 'a' itself is confident."""
    rng = np.random.default_rng(seed)
    n = n_products * per
    frame = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "product_id": [f"p{k // per}" for k in range(n)]})
    frame["subject"] = np.where(rng.random(n) < 0.2, "b", "a")
    frame["perspective"] = rng.choice(["l", "r"], n)
    records = {}
    for row in frame.itertuples():
        low = row.subject == "b"
        records[row.image_id] = {"image_id": row.image_id,
                                 "labels": {"subject": "a", "perspective": row.perspective},
                                 "confidence": {"subject": 0.45 if low else 0.95, "perspective": 0.9}}
    return frame, records


def test_a_rare_class_swallowed_at_low_confidence_is_given_back():
    frame, records = swallowing_model()
    result = recalibrate.learn(records, frame, DIMS, ALLOWED, solution_frame, evaluate)
    assert result["rules"]["subject"]["a"] == {"below": 0.5, "to": "b"}
    assert result["gains"]["subject"]["after"] > result["gains"]["subject"]["before"]
    assert "perspective" not in result["rules"]  # already perfect: no rule


def test_the_crossfitted_answers_use_the_other_halfs_rules_and_still_gain():
    frame, records = swallowing_model()
    learned, by_image, full = recalibrate.crossfit(records, frame, DIMS, ALLOWED, solution_frame, evaluate)
    assert len(by_image) == len(frame) and all(r for r in (learned[0]["rules"], learned[1]["rules"]))
    predictor = recalibrate.RecalibratedPredictor(rules=full["rules"], saved=records, rules_by_image=by_image)
    answers = predictor.predict([ImageItem(i, "/x") for i in frame["image_id"]])
    truth = frame.set_index("image_id")["subject"]
    assert sum(p.labels["subject"] == truth[p.image_id] for p in answers) == len(frame)


def test_rules_never_touch_a_cell_without_confidence_or_above_the_threshold():
    rules = {"subject": {"a": {"below": 0.5, "to": "b"}}}
    assert recalibrate.apply_rules({"subject": "a"}, None, rules) == {"subject": "a"}
    assert recalibrate.apply_rules({"subject": "a"}, {"subject": 0.7}, rules) == {"subject": "a"}
    assert recalibrate.apply_rules({"subject": "a"}, {"subject": 0.3}, rules) == {"subject": "b"}


def test_a_live_inner_predictor_is_wrapped(monkeypatch, tmp_path):
    from galaxus.types import Prediction

    class Inner:
        def predict(self, batch):
            return [Prediction(image_id=i.image_id, labels={"subject": "a"}, confidence={"subject": 0.2})
                    for i in batch]

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"base": "EXP-0004", "rules": {"subject": {"a": {"below": 0.5, "to": "b"}}}}))
    monkeypatch.setenv("RECAL_RULES", str(rules))
    monkeypatch.setenv("RECAL_INNER", "x:y")
    monkeypatch.setattr("galaxus.serve.items.load_predictor", lambda spec: Inner())
    assert recalibrate.from_env().predict([ImageItem("i1", "/x")])[0].labels == {"subject": "b"}


def test_from_env_says_what_to_set(monkeypatch):
    monkeypatch.delenv("RECAL_RULES", raising=False)
    with pytest.raises(RuntimeError, match="RECAL_RULES"):
        recalibrate.from_env()


def test_the_eval_script_writes_an_exp_row(tmp_path, monkeypatch):
    train = train_table(products=12)
    out = tmp_path / "out"
    data = fake_data_lane(train, out, FakeMetric())
    path = out / "predictions" / "EXP-0004" / "holdout.jsonl"
    path.parent.mkdir(parents=True)
    lines = [json.dumps({"image_id": i, "labels": {d: f"{d}_a" for d in data.dimensions()},
                         "confidence": {d: 0.9 for d in data.dimensions()}})
             for i in data.split_holdout(train)[1]["image_id"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results = tmp_path / "metrics.json"
    results.write_text(json.dumps([{"id": "EXP-0004", "images_per_second": 7.0}]), encoding="utf-8")
    monkeypatch.setattr("galaxus.model.eval.RESULTS_FILE", results)
    monkeypatch.setenv("RECAL_BASE", "EXP-0004")
    row = run_eval("recalibrated", data, results_path=results, exp_id="EXP-0098", allow_dirty=True)
    assert row["predictor"] == "recalibrated" and row["images_per_second"] == pytest.approx(7.0)
    assert row["config"]["base"] == "EXP-0004"
