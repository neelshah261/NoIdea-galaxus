"""#182: four or more members under ensemble-agree, with the product-type prior on top, for the holdout row and for
the dev or final file. Synthetic rows and invented classes only, no model, no network.

The members stand in for the 70B (the base), the frozen Apertus heads, the SigLIP helper and a Kaggle run. Only the
heads know the perspective; nobody knows that 'gadget' products are subject 'b', but the product type does.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from galaxus.model import ensemble, type_prior
from galaxus.model.eval import write_predictions
from galaxus.types import ImageItem, Prediction

DIMS = ("subject", "perspective")
MEMBERS = {"big": "EXP-0901", "heads": "EXP-0902", "siglip": "EXP-0903", "kaggle": "EXP-0904"}


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


def catalog(prefix, n_products=80, per=5, seed=0):
    """Every third product is a 'gadget' (subject 'b'), the rest 'thing' (subject 'a'); perspective is random."""
    rng = np.random.default_rng(seed)
    n = n_products * per
    frame = pd.DataFrame({"image_id": [f"{prefix}{k}" for k in range(n)],
                          "product_id": [f"{prefix}p{k // per}" for k in range(n)]})
    frame["product_type"] = np.where((np.arange(n) // per) % 3 == 0, "gadget", "thing")
    frame["subject"] = np.where(frame["product_type"] == "gadget", "b", "a")
    frame["perspective"] = rng.choice(["l", "r", "f"], n)
    for d in DIMS:
        frame[f"{d}_source"] = "human"
    return frame


def member_answers(frame, seed=1):
    """Each member's saved answers. All say subject 'a' (unsure on gadgets); only 'heads' gets perspective right."""
    rng = np.random.default_rng(seed)
    truth = frame["perspective"].to_numpy()
    noise = {name: rng.choice(["l", "r", "f"], len(frame)) for name in MEMBERS}
    keep = {"big": 0.55, "heads": 1.0, "siglip": 0.3, "kaggle": 0.6}
    out = {}
    for name in MEMBERS:
        right = rng.random(len(frame)) < keep[name]
        records = {}
        for k, row in enumerate(frame.itertuples()):
            gadget = row.product_type == "gadget"
            records[row.image_id] = {
                "image_id": row.image_id,
                "labels": {"subject": "a", "perspective": truth[k] if right[k] else noise[name][k]},
                "confidence": {"subject": 0.45 if gadget else 0.95, "perspective": 0.9 if name == "heads" else 0.6},
            }
        out[name] = records
    return out


def save(root, exp, part, records):
    path = root / "predictions" / exp / f"{part}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records.values()), encoding="utf-8")
    return path


def items(frame):
    return [ImageItem(image_id=r.image_id, path="unused", context={"product_type": r.product_type})
            for r in frame.itertuples()]


def macro(frame, predictions):
    answers = pd.DataFrame([{"image_id": p.image_id, **p.labels} for p in predictions])
    scores = evaluate(solution_frame(frame, DIMS), answers)
    return float(np.mean([scores[d]["f1_macro"] for d in DIMS]))


@pytest.fixture
def setup(tmp_path):
    rows, holdout = catalog("t", seed=3), catalog("h", seed=0)
    for name, records in member_answers(holdout).items():
        save(tmp_path, MEMBERS[name], "holdout", records)
    combined = ensemble.from_saved(MEMBERS, holdout, DIMS, solution_frame, evaluate, tmp_path, rule=ensemble.AGREE)
    return rows, holdout, combined


def test_ensemble_agree_takes_four_members_and_routes_perspective_to_the_one_that_knows(setup):
    _, _, combined = setup
    full = combined.fitted["full"]
    assert full["base"] == "big" and set(combined.members) == set(MEMBERS)
    assert full["routing"]["perspective"] == "heads"
    assert full["routing"]["subject"] == "big"  # nobody beats the base on subject: it stays


def test_the_type_prior_on_top_of_the_ensemble_beats_every_member_on_the_holdout(setup, tmp_path):
    rows, holdout, combined = setup
    write_predictions(tmp_path / "predictions" / "EXP-0905", combined.predict(items(holdout)))
    on_top = type_prior.from_saved("EXP-0905", rows, holdout, DIMS, scored, solution_frame, evaluate, tmp_path)
    final = on_top.predict(items(holdout))

    gadgets = holdout["product_type"] == "gadget"
    subject = pd.Series([p.labels["subject"] for p in final])
    assert (subject[gadgets.to_numpy()] == "b").mean() > 0.9
    best_member = max(
        macro(holdout, [Prediction(image_id=i, labels=r["labels"]) for i, r in records.items()])
        for records in ensemble.load_members(MEMBERS, tmp_path).values()
    )
    assert macro(holdout, final) > best_member + 0.1


def test_dev_answers_get_the_same_combination_and_then_the_prior(setup, tmp_path, monkeypatch):
    rows, holdout, combined = setup
    dev = catalog("d", n_products=30, seed=7)
    for name, records in member_answers(dev, seed=5).items():
        save(tmp_path, MEMBERS[name], "dev", records)
    monkeypatch.setattr("galaxus.data.paths.out_dir", lambda: tmp_path)
    monkeypatch.setattr("galaxus.data.load.read_dev", lambda: dev)

    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({**combined.fitted["full"], "member_exps": MEMBERS}), encoding="utf-8")
    assert ensemble.main(["apply", "--routing", str(routing), "--part", "dev", "--exp", "EXP-0905"]) == 0

    write_predictions(tmp_path / "predictions" / "EXP-0905", combined.predict(items(holdout)))
    table = type_prior.type_table(rows, DIMS, scored)
    _, _, full = type_prior.crossfit(ensemble.load_records(tmp_path / "predictions" / "EXP-0905" / "holdout.jsonl"),
                                     holdout, table, DIMS, solution_frame, evaluate)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"base": "EXP-0905", "table": table, **full}), encoding="utf-8")

    assert type_prior.main(["apply", "--rules", str(rules), "--base", "EXP-0905", "--part", "dev",
                            "--exp", "EXP-0906"]) == 0
    out = ensemble.load_records(tmp_path / "predictions" / "EXP-0906" / "dev.jsonl")
    assert list(out) == dev["image_id"].tolist()  # every dev image, in the order of dev.csv
    gadget = dev.loc[dev["product_type"] == "gadget", "image_id"].iloc[0]
    thing = dev.loc[dev["product_type"] == "thing", "image_id"].iloc[0]
    assert out[gadget]["labels"]["subject"] == "b" and "subject" not in (out[gadget]["confidence"] or {})
    assert out[thing]["labels"]["subject"] == "a" and out[thing]["confidence"]["subject"] == pytest.approx(0.95)
    truth = dict(zip(dev["image_id"], dev["perspective"], strict=True))
    assert all(r["labels"]["perspective"] == truth[i] for i, r in out.items())  # routed to the heads


def test_apply_refuses_rules_learned_on_another_runs_answers(tmp_path, monkeypatch):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"base": "EXP-0049", "table": {}, "rules": {}}), encoding="utf-8")
    monkeypatch.setattr("galaxus.data.paths.out_dir", lambda: tmp_path)
    with pytest.raises(SystemExit, match="learn --base EXP-0905"):
        type_prior.main(["apply", "--rules", str(rules), "--base", "EXP-0905", "--part", "dev", "--exp", "EXP-0906"])


def test_apply_saved_needs_an_answer_for_every_image_and_leaves_unknown_types_alone():
    part = pd.DataFrame({"image_id": ["x", "y"], "product_type": ["gadget", ""]})
    saved = {"table": {"subject": {"gadget": {"class": "b", "share": 1.0, "cells": 50}}},
             "rules": {"subject": {"below": 0.5, "share": 0.9}}}
    records = {i: {"image_id": i, "labels": {"subject": "a"}, "confidence": {"subject": 0.2}} for i in ("x", "y")}
    answers = type_prior.apply_saved(saved, records, part)
    assert [a.labels["subject"] for a in answers] == ["b", "a"]
    assert answers[0].confidence is None and answers[1].confidence == {"subject": 0.2}
    with pytest.raises(KeyError, match="no saved answer"):
        type_prior.apply_saved(saved, {"x": records["x"]}, part)
