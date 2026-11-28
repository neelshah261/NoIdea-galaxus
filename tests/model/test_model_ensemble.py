"""Ensemble routing per dimension (#87), on synthetic answers. No model, no competition data, no network.

``evaluate`` here is a test stand-in in metric.evaluate's shape; the module never computes an F1 itself, which
``test_every_number_comes_from_the_evaluate_it_is_given`` checks.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from model_synthetic import FakeMetric, fake_data_lane, train_table

from galaxus.model import ensemble
from galaxus.model.eval import run_eval
from galaxus.types import ImageItem

DIMS = ("subject", "perspective")


def holdout(products=40, per_product=5, seed=0):
    rng = np.random.default_rng(seed)
    n = products * per_product
    frame = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)],
                          "product_id": [f"p{k // per_product}" for k in range(n)]})
    frame["subject"] = rng.choice(["a", "b", "c"], n)
    frame["perspective"] = rng.choice(["l", "r"], n)
    return frame


def solution_frame(part, dimensions):
    return part[["image_id", *dimensions]].astype(object)


def evaluate(solution, submission):
    """metric.evaluate's shape: macro F1 per dimension over every class that is true or predicted."""
    joined = solution.merge(submission, on="image_id", suffixes=("_true", "_pred"))
    out = {}
    for d in [c for c in solution.columns if c != "image_id"]:
        true, pred = joined[f"{d}_true"].to_numpy(), joined[f"{d}_pred"].to_numpy()
        f1s = []
        for c in np.union1d(true.astype(str), pred.astype(str)):
            tp = np.sum((pred == c) & (true == c))
            fp, fn = np.sum((pred == c) & (true != c)), np.sum((pred != c) & (true == c))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        out[d] = {"f1_macro": float(np.mean(f1s))}
    return out


def member(frame, accuracy, seed, confidence=0.8):
    """Answers right with the given accuracy per dimension; an accuracy may be a function of the row index."""
    rng = np.random.default_rng(seed)
    records = {}
    for index, row in enumerate(frame.itertuples()):
        labels = {}
        for d in DIMS:
            truth = getattr(row, d)
            p = accuracy[d](index) if callable(accuracy[d]) else accuracy[d]
            options = sorted(o for o in frame[d].unique() if o != truth)
            labels[d] = truth if rng.random() < p else str(rng.choice(options))
        records[row.image_id] = {"image_id": row.image_id, "labels": labels, "confidence": {d: confidence for d in DIMS}}
    return records


def test_each_dimension_goes_to_its_best_member():
    frame = holdout()
    members = {"ctx": member(frame, {"subject": 0.95, "perspective": 0.55}, 1),
               "noctx": member(frame, {"subject": 0.55, "perspective": 0.95}, 2)}
    choice = ensemble.choose(members, frame, DIMS, solution_frame, evaluate)
    assert choice["routing"]["subject"] in ("ctx", ensemble.VOTE)
    assert choice["routing"]["perspective"] in ("noctx", ensemble.VOTE)


def test_vote_wins_when_three_members_err_independently():
    frame = holdout(80)
    members = {f"m{k}": member(frame, {"subject": 0.7, "perspective": 0.7}, k) for k in range(3)}
    choice = ensemble.choose(members, frame, DIMS, solution_frame, evaluate)
    assert all(choice["routing"][d] == ensemble.VOTE for d in DIMS)


def test_every_number_comes_from_the_evaluate_it_is_given():
    frame = holdout(10)
    members = {"a": member(frame, {"subject": 1, "perspective": 1}, 1),
               "b": member(frame, {"subject": 0, "perspective": 0}, 2)}
    calls = []

    def constant(solution, submission):
        calls.append(len(submission))
        return {d: {"f1_macro": 0.123} for d in DIMS}

    choice = ensemble.choose(members, frame, DIMS, solution_frame, constant)
    assert calls and all(v == 0.123 for per in choice["scores"].values() for v in per.values())
    assert choice["routing"] == {"subject": "a", "perspective": "a"}  # a tie: a member before the vote, by name


def test_the_estimate_is_not_optimistic_each_half_is_answered_by_the_other_halfs_routing():
    frame = holdout()
    halves = ensemble.holdout_halves(frame)
    rows = list(frame["image_id"])
    first = set(frame.loc[halves.to_numpy() == 0, "image_id"])
    # "a" is perfect on half 0 and useless on half 1 for subject; "b" is the reverse. Chosen on one half and scored
    # on the other, each routing picks the member that is useless where it is applied: an honest, bad number.
    good_on_0 = {"subject": lambda i: 1.0 if rows[i] in first else 0.0, "perspective": 0.8}
    good_on_1 = {"subject": lambda i: 0.0 if rows[i] in first else 1.0, "perspective": 0.8}
    members = {"a": member(frame, good_on_0, 1), "b": member(frame, good_on_1, 2)}
    fitted = ensemble.crossfit(members, frame, DIMS, solution_frame, evaluate, halves)
    assert fitted["chosen_on_half"][0]["routing"]["subject"] == "a"
    assert fitted["chosen_on_half"][1]["routing"]["subject"] == "b"
    predictor = ensemble.EnsemblePredictor(members, {"a": "EXP-1", "b": "EXP-2"}, fitted)
    answers = predictor.predict([ImageItem(i, f"/{i}.jpg") for i in rows])
    truth = frame.set_index("image_id")["subject"]
    assert sum(p.labels["subject"] == truth[p.image_id] for p in answers) < 0.6 * len(rows)


def test_halves_are_grouped_by_product():
    frame = holdout()
    halves = ensemble.holdout_halves(frame)
    assert set(halves) == {0, 1}
    assert frame.assign(h=halves).groupby("product_id")["h"].nunique().max() == 1


def test_an_image_outside_the_holdout_gets_the_whole_holdout_routing():
    frame = holdout(10)
    members = {"a": member(frame, {"subject": 1, "perspective": 1}, 1)}
    fitted = ensemble.crossfit(members, frame, DIMS, solution_frame, evaluate, ensemble.holdout_halves(frame))
    fitted["answered_by"].pop("i0")  # as if i0 were a dev image
    fitted["full"] = {**fitted["full"], "routing": {"subject": "a", "perspective": "a"}}
    assert ensemble.EnsemblePredictor(members, {"a": "EXP-1"}, fitted).predict([ImageItem("i0", "/x")])[0].labels


def test_apply_follows_the_routing_and_keeps_every_cell_filled():
    frame = holdout(10)
    members = {"a": member(frame, {"subject": 1.0, "perspective": 0.0}, 1),
               "b": member(frame, {"subject": 0.0, "perspective": 1.0}, 2)}
    choice = {"dimensions": list(DIMS), "routing": {"subject": "a", "perspective": "b"},
              "weights": {d: {"a": 1.0, "b": 1.0} for d in DIMS}}
    answers = ensemble.apply(choice, members)
    truth = frame.set_index("image_id")
    assert len(answers) == len(frame)
    assert all(p.labels[d] == truth.loc[p.image_id, d] for p in answers for d in DIMS)


def test_members_must_answer_the_same_images():
    frame = holdout(4)
    a = member(frame, {"subject": 1, "perspective": 1}, 1)
    b = dict(list(a.items())[:10])
    with pytest.raises(KeyError, match="missing answers"):
        ensemble.choose({"a": a, "b": b}, frame, DIMS, solution_frame, evaluate)
    with pytest.raises(KeyError, match="only one member"):
        ensemble.apply({"dimensions": list(DIMS), "routing": {}, "weights": {}}, {"a": a, "b": b})


@pytest.mark.parametrize("spec", ["", "a", "vote=EXP-1", "a=EXP-1 a=EXP-2"])
def test_member_specs_are_checked(spec):
    with pytest.raises(ValueError):
        ensemble.parse_members(spec)


def test_members_default_to_the_two_zero_shot_runs():
    assert ensemble.members_from_env({}) == {"zeroshot": "EXP-0002", "zeroshot_noctx": "EXP-0009"}
    assert ensemble.members_from_env({"ENSEMBLE_MEMBERS": "x=EXP-0004"}) == {"x": "EXP-0004"}


def test_combined_speed_is_every_image_through_every_member():
    assert ensemble.combined_speed([6.0, 6.0]) == pytest.approx(3.0)
    assert ensemble.combined_speed([6.0, None]) is None


def test_the_eval_script_writes_an_exp_row_with_the_members_speed(tmp_path, monkeypatch):
    train = train_table(products=12)
    out = tmp_path / "out"
    data = fake_data_lane(train, out, FakeMetric())
    for exp in ("EXP-0002", "EXP-0009"):
        path = out / "predictions" / exp / "holdout.jsonl"
        path.parent.mkdir(parents=True)
        lines = [json.dumps({"image_id": i, "labels": {d: f"{d}_a" for d in data.dimensions()}, "confidence": None})
                 for i in data.split_holdout(train)[1]["image_id"]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results = tmp_path / "metrics.json"
    results.write_text(json.dumps([{"id": "EXP-0002", "images_per_second": 6.0, "model": "m"},
                                   {"id": "EXP-0009", "images_per_second": 6.0, "model": "m"}]), encoding="utf-8")
    monkeypatch.setattr("galaxus.model.eval.RESULTS_FILE", results)
    monkeypatch.delenv("ENSEMBLE_MEMBERS", raising=False)
    row = run_eval("ensemble", data, results_path=results, exp_id="EXP-0099", allow_dirty=True)
    assert row["predictor"] == "ensemble" and row["images_per_second"] == pytest.approx(3.0)
    assert row["model"] == "m"
    assert row["config"]["members"] == {"zeroshot": "EXP-0002", "zeroshot_noctx": "EXP-0009"}
    assert (out / "predictions" / "EXP-0099" / "holdout.jsonl").is_file()


def test_a_missing_member_file_says_where_it_must_come_from(tmp_path):
    with pytest.raises(FileNotFoundError, match="EXP-0002's holdout answers"):
        ensemble.load_members({"zeroshot": "EXP-0002"}, tmp_path)
