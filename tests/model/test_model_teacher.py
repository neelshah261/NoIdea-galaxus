"""D (#211): teacher dev rows and self-training for the heads. Synthetic dimensions, classes, rules and rows only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from galaxus.data import split
from galaxus.model import helper, teacher
from galaxus.types import ImageItem

DIMS = ("shape", "words", "tongue")
ALLOWED = {"shape": frozenset({"round", "square", "badge"}), "words": frozenset({"none", "some"}),
           "tongue": frozenset({"none", "de", "fr"})}
#: Same structure as bench/scoring_rules.json, on made-up dimensions.
RULES = {
    "not_scored_when": [{"when": {"dimension": "shape", "in": ["badge"]}, "not_scored": ["words"]}],
    "scored_only_if": [{"dimension": "tongue", "any_of": [{"dimension": "words", "not_in": ["none"]}]}],
}


def dev(n=4):
    return pd.DataFrame({"image_id": [f"d{k}" for k in range(n)], "image_file": [f"f{k}" for k in range(n)],
                         "product_id": [f"q{k}" for k in range(n)], "product_title": "t"})


def record(shape, words, tongue, conf=None):
    out = {"labels": {"shape": shape, "words": words, "tongue": tongue}}
    if conf is not None:
        out["confidence"] = dict.fromkeys(DIMS, conf)
    return out


def test_relevance_follows_the_rules_on_the_teachers_own_labels():
    records = {"d0": record("round", "some", "de"), "d1": record("badge", "some", "de"),
               "d2": record("square", "none", "none")}
    rows = teacher.teacher_rows(records, dev(), dimensions=DIMS, allowed=ALLOWED, rules=RULES).set_index("image_id")
    assert rows.loc["d0", ["shape_relevant", "words_relevant", "tongue_relevant"]].tolist() == [True, True, True]
    assert not rows.loc["d1", "words_relevant"]  # badge: words not scored
    assert not rows.loc["d2", "tongue_relevant"]  # no words: tongue not scored
    assert rows.loc["d2", "tongue_source"] == teacher.DROPPED_SOURCE
    assert rows.loc["d0", "shape_source"] == teacher.TEACHER_SOURCE
    assert "d3" not in rows.index  # an image the teacher did not answer is not a row
    assert rows.loc["d0", "product_title"] == "t"  # catalog fields kept, for A's text features


def test_low_confidence_and_unknown_classes_are_dropped():
    records = {"d0": record("round", "some", "de", conf=0.4), "d1": {"labels": {"shape": "hexagon", "words": "some",
                                                                              "tongue": "de"}}}
    rows = teacher.teacher_rows(records, dev(), min_confidence=0.5, dimensions=DIMS, allowed=ALLOWED,
                                rules=RULES).set_index("image_id")
    assert all(rows.loc["d0", f"{d}_source"] == teacher.DROPPED_SOURCE for d in DIMS)
    assert rows.loc["d1", "shape_source"] == teacher.DROPPED_SOURCE  # not a class of shape
    assert rows.loc["d1", "words_source"] == teacher.TEACHER_SOURCE  # no confidence reported: the label counts
    assert not teacher.training_mask(rows.reset_index(), "shape").any()


def test_training_mask_takes_human_teacher_and_agree_cells_but_scoring_stays_human_only():
    frame = pd.DataFrame({"shape": ["round"] * 5,
                          "shape_source": ["human", "model", "teacher", "agree", "dropped"],
                          "shape_relevant": ["true", "true", True, True, False]})
    assert teacher.training_mask(frame, "shape").tolist() == [True, False, True, True, False]
    holdout = frame.iloc[:2]  # what an EXP row scores: train's own rows, human or model
    assert split.scored_mask(holdout, "shape").tolist() == [True, False]


def test_agreeing_rows_need_every_member_to_agree_above_the_floor():
    members = {
        "heads": {"d0": record("round", "some", "de", 0.95), "d1": record("round", "some", "de", 0.95),
                  "d2": record("square", "some", "fr", 0.5)},
        "big": {"d0": record("round", "some", "fr", 0.99), "d1": record("square", "some", "de", 0.99),
                "d2": record("square", "some", "fr", 0.99)},
        "teacher": {"d0": record("round", "some", "de"), "d1": record("round", "some", "de")},
    }
    rows = teacher.agreeing_rows(members, dev(), 0.9, DIMS, ALLOWED, RULES).set_index("image_id")
    assert set(rows.index) == {"d0", "d1"}  # d2: the teacher did not answer it
    assert rows.loc["d0", "shape_source"] == teacher.AGREE_SOURCE and rows.loc["d0", "shape"] == "round"
    assert rows.loc["d0", "tongue_source"] == teacher.DROPPED_SOURCE  # big said fr
    assert rows.loc["d1", "shape_source"] == teacher.DROPPED_SOURCE  # big said square
    assert rows.loc["d1", "tongue_source"] == teacher.AGREE_SOURCE


def test_from_environ_reads_the_switches(tmp_path):
    assert teacher.from_environ({}) is None
    with pytest.raises(FileNotFoundError):
        teacher.from_environ({"HELPER_TEACHER": str(tmp_path / "missing.jsonl")})
    path = tmp_path / "dev.jsonl"
    path.write_text("")
    setup = teacher.from_environ({"HELPER_TEACHER": str(path), "HELPER_SELF_TRAIN": "1",
                                  "HELPER_SELF_TRAIN_MIN_CONFIDENCE": "0.8"})
    assert setup.teacher == str(path) and setup.self_train and setup.agree_floor == 0.8
    assert teacher.from_environ({"HELPER_SELF_TRAIN": "1"}).teacher is None


# ---------------------------------------------------------------- the factory: rows appended, deterministic


def world(seed=0):
    """Train rows with few human shape labels, dev images a teacher labels correctly, and a separable feature."""
    rng = np.random.default_rng(seed)
    centre = {"round": 3.0, "square": -3.0}
    n_train, n_dev = 60, 200
    truth = {}
    train = pd.DataFrame({"image_id": [f"t{k}" for k in range(n_train)], "image_file": "u",
                          "product_id": [f"p{k // 2}" for k in range(n_train)]})
    train["shape"] = rng.choice(["round", "square"], n_train)
    train["shape_source"] = np.where(np.arange(n_train) < 12, "human", "model")  # only 12 human cells
    train["shape_relevant"] = "true"
    truth.update(dict(zip(train["image_id"], train["shape"], strict=True)))
    dev_rows = pd.DataFrame({"image_id": [f"d{k}" for k in range(n_dev)], "image_file": "u",
                             "product_id": [f"q{k // 2}" for k in range(n_dev)]})
    dev_shape = rng.choice(["round", "square"], n_dev)
    truth.update(dict(zip(dev_rows["image_id"], dev_shape, strict=True)))
    noise = {i: rng.normal(0, 2.5, 6) for i in truth}

    def embed(pairs, _name):
        return {i: (centre[truth[i]] * np.r_[1, np.zeros(5)] + noise[i]).astype(np.float32) for i, _ in pairs}

    records = {i: {"labels": {"shape": s}} for i, s in zip(dev_rows["image_id"], dev_shape, strict=True)}
    data = SimpleNamespace(dimensions=lambda: ("shape",), scored_mask=split.scored_mask,
                           allowed_classes=lambda: {"shape": frozenset({"round", "square"})})
    return train, dev_rows, records, embed, data, truth


def answers(predictor, ids):
    return [p.labels["shape"] for p in predictor.predict([ImageItem(image_id=i, path="u") for i in ids])]


def test_the_factory_appends_teacher_rows_trains_on_them_and_is_deterministic(monkeypatch):
    monkeypatch.setattr(teacher, "load_rules", lambda path=None: {})
    train, dev_rows, records, embed, data, truth = world()
    setup = teacher.TeacherSetup(teacher="EXP-0080/dev.jsonl", records=records, read_dev=lambda: dev_rows)
    with_teacher = helper.factory(encoder_name="fake", embed=embed, teacher=setup)(train, data, 0)
    again = helper.factory(encoder_name="fake", embed=embed, teacher=setup)(train, data, 0)
    meta = with_teacher.heads_model.meta["teacher"]
    assert meta["added_cells"] == {"shape": 200} and meta["teacher"] == "EXP-0080/dev.jsonl"
    probe = [f"probe{k}" for k in range(300)]
    rng = np.random.default_rng(7)
    probe_truth = rng.choice(["round", "square"], 300)
    truth.update(dict(zip(probe, probe_truth, strict=True)))
    for i, s in zip(probe, probe_truth, strict=True):
        with_teacher.features[i] = again.features[i] = (
            {"round": 3.0, "square": -3.0}[s] * np.r_[1, np.zeros(5)] + rng.normal(0, 2.5, 6)).astype(np.float32)
    assert answers(with_teacher, probe) == answers(again, probe)  # two runs, identical answers
    assert np.array_equal(with_teacher.heads_model.heads["shape"].weights, again.heads_model.heads["shape"].weights)


def test_self_training_without_a_teacher_uses_the_heads_confident_dev_answers(monkeypatch):
    monkeypatch.setattr(teacher, "load_rules", lambda path=None: {})
    train, dev_rows, _records, embed, data, _truth = world()
    setup = teacher.TeacherSetup(self_train=True, agree_floor=0.8, read_dev=lambda: dev_rows)
    predictor = helper.factory(encoder_name="fake", embed=embed, teacher=setup)(train, data, 0)
    added = predictor.heads_model.meta["teacher"]["added_cells"]["shape"]
    assert 0 < added < 200  # only the confident ones


def test_a_dev_image_among_the_training_rows_is_refused(monkeypatch):
    monkeypatch.setattr(teacher, "load_rules", lambda path=None: {})
    train, dev_rows, records, embed, data, _truth = world()
    leaky = pd.concat([train, dev_rows.iloc[:1].assign(shape="round", shape_source="human", shape_relevant="true")])
    setup = teacher.TeacherSetup(records=records, read_dev=lambda: dev_rows)
    with pytest.raises(ValueError, match="dev images only"):
        helper.factory(encoder_name="fake", embed=embed, teacher=setup)(leaky, data, 0)


def test_without_a_teacher_the_factory_is_unchanged():
    train, _dev, _records, embed, data, _truth = world()
    plain = helper.factory(encoder_name="fake", embed=embed)(train, data, 0)
    assert "teacher" not in plain.heads_model.meta


def test_the_eval_names_are_registered():
    from galaxus.model.eval import PREDICTORS

    assert {"helper-siglip-teacher", "helper-fused-teacher"} <= set(PREDICTORS)


def test_the_real_rules_file_loads():
    rules = teacher.load_rules()
    assert rules["not_scored_when"] and rules["scored_only_if"]
