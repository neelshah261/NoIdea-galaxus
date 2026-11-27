"""Ensemble routing per dimension, on synthetic answers. No model, no data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxus.model import ensemble

DIMS = ("subject", "perspective")


def holdout(n=200, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)]})
    frame["subject"] = rng.choice(["a", "b", "c"], n)
    frame["perspective"] = rng.choice(["left", "right"], n)
    for dim in DIMS:
        frame[f"{dim}_source"] = "human"
    return frame


def scored(rows, dim):
    return rows[f"{dim}_source"] == "human"


def member(frame, accuracy, seed, confidence=0.8):
    """Answers that are right with the given accuracy per dimension."""
    rng = np.random.default_rng(seed)
    records = {}
    for row in frame.itertuples():
        labels = {}
        for dim in DIMS:
            truth = getattr(row, dim)
            options = sorted(frame[dim].unique())
            labels[dim] = truth if rng.random() < accuracy[dim] else rng.choice([o for o in options if o != truth])
        records[row.image_id] = {"image_id": row.image_id, "labels": labels,
                                 "confidence": {d: confidence for d in DIMS}}
    return records


def test_each_dimension_goes_to_its_best_member():
    frame = holdout()
    members = {
        "heads": member(frame, {"subject": 0.95, "perspective": 0.55}, 1),
        "lora": member(frame, {"subject": 0.60, "perspective": 0.95}, 2),
    }
    choice = ensemble.choose(members, frame, DIMS, scored)
    assert choice["routing"]["subject"] in ("heads", ensemble.VOTE)
    assert choice["routing"]["perspective"] in ("lora", ensemble.VOTE)
    best_single = max(np.mean([choice["scores"][d][m] for d in DIMS]) for m in members)
    assert choice["holdout_estimate"] >= best_single and "optimistic" in choice["note"]


def test_vote_wins_when_three_members_err_independently():
    frame = holdout(400)
    members = {f"m{k}": member(frame, {"subject": 0.7, "perspective": 0.7}, k) for k in range(3)}
    choice = ensemble.choose(members, frame, DIMS, scored)
    assert all(choice["routing"][d] == ensemble.VOTE for d in DIMS)
    assert all(choice["scores"][d][ensemble.VOTE] > max(choice["scores"][d][m] for m in members) for d in DIMS)


def test_apply_follows_the_routing_and_keeps_every_cell_filled():
    frame = holdout(50)
    members = {"heads": member(frame, {"subject": 1.0, "perspective": 0.0}, 1),
               "lora": member(frame, {"subject": 0.0, "perspective": 1.0}, 2)}
    choice = {"routing": {"subject": "heads", "perspective": "lora"}, "members": ["heads", "lora"],
              "weights": {d: {"heads": 1.0, "lora": 1.0} for d in DIMS}}
    records = ensemble.apply(choice, members)
    truth = frame.set_index("image_id")
    assert len(records) == 50
    assert all(r["labels"]["subject"] == truth.loc[r["image_id"], "subject"] for r in records)
    assert all(r["labels"]["perspective"] == truth.loc[r["image_id"], "perspective"] for r in records)


def test_members_must_cover_the_same_images_and_the_routing_its_members():
    frame = holdout(20)
    a = member(frame, {"subject": 1, "perspective": 1}, 1)
    b = dict(list(a.items())[:10])
    with pytest.raises(KeyError, match="same images"):
        ensemble.choose({"a": a, "b": b}, frame, DIMS, scored)
    with pytest.raises(KeyError, match="needs member"):
        ensemble.apply({"routing": {"subject": "zzz"}, "members": ["a"], "weights": {}}, {"a": a})


def test_macro_f1_counts_predicted_classes_that_never_occur():
    assert ensemble.macro_f1(np.array(["a", "a"]), np.array(["a", "a"])) == 1.0
    assert ensemble.macro_f1(np.array(["a", "a"]), np.array(["a", "b"])) < 1.0
