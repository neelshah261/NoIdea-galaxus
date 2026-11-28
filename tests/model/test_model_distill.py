"""Teacher passes and the per-dimension merge (#132 D1, #172), with a fake teacher. No model, no data, no network."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest
from model_synthetic import FakeMetric, fake_data_lane, train_table

from galaxus.model import distill
from galaxus.model.ensemble import load_records
from galaxus.types import ImageItem, Prediction


class FakeTeacher:
    """Answers every image with the same labels and a confidence; records which images it was asked about."""

    def __init__(self, labels=None, confidence=0.9):
        self.labels = labels or {"subject": "subject_a", "perspective": "perspective_b"}
        self.confidence = confidence
        self.seen = []
        self.lock = threading.Lock()

    def predict(self, batch):
        with self.lock:
            self.seen += [item.image_id for item in batch]
        return [Prediction(image_id=item.image_id, labels=dict(self.labels),
                           confidence=dict.fromkeys(self.labels, self.confidence)) for item in batch]


def items(n, start=0):
    return [ImageItem(f"img_{k:03d}", f"/synthetic/img_{k:03d}.jpg") for k in range(start, start + n)]


def test_a_pass_writes_every_answer_with_its_confidence_and_a_rerun_skips_them(tmp_path):
    target = tmp_path / "distill" / "b70" / "train.jsonl"
    teacher = FakeTeacher()
    stats = distill.run_pass(teacher, items(10), target, batch_size=3, log=lambda line: None)
    assert stats["answered"] == 10 and stats["skipped"] == 0
    records = load_records(target)
    assert len(records) == 10
    assert records["img_000"]["labels"] == teacher.labels
    assert records["img_000"]["confidence"] == {"subject": 0.9, "perspective": 0.9}

    again = FakeTeacher()
    stats = distill.run_pass(again, items(15), target, batch_size=3, log=lambda line: None)
    assert stats["answered"] == 5 and stats["skipped"] == 10
    assert again.seen == [f"img_{k:03d}" for k in range(10, 15)]  # only the images not answered yet
    assert len(load_records(target)) == 15


def test_workers_answer_every_image_once(tmp_path):
    target = tmp_path / "train.jsonl"
    distill.run_pass(FakeTeacher(), items(50), target, batch_size=4, workers=4, log=lambda line: None)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50 and len({json.loads(line)["image_id"] for line in lines}) == 50


def test_the_log_holds_counts_only_no_ids_and_no_labels(tmp_path):
    lines = []
    teacher = FakeTeacher()
    distill.run_pass(teacher, items(40), tmp_path / "train.jsonl", batch_size=2, log=lines.append)
    assert lines and all("img_" not in line for line in lines)
    assert all(label not in line for line in lines for label in teacher.labels.values())


def test_the_teacher_answers_train_minus_the_official_holdout(tmp_path):
    train = train_table(products=12)
    data = fake_data_lane(train, tmp_path / "out", FakeMetric())
    rest, holdout = data.split_holdout(train)
    answered = {str(item.image_id) for item in distill.teacher_items(data)}
    assert answered == set(rest["image_id"])
    assert not answered & set(holdout["image_id"])

    leaky = dataclasses.replace(data, split_holdout=lambda table: (table, holdout))
    with pytest.raises(ValueError, match="holdout"):
        distill.teacher_items(leaky)


def test_merge_takes_each_dimension_from_its_teacher_else_the_default():
    dims = ("subject", "perspective", "background")
    a = {"x": {"image_id": "x", "labels": {"subject": "a1", "perspective": "a2", "background": "a3"},
               "confidence": {"subject": 0.5, "perspective": 0.6, "background": 0.7}}}
    b = {"x": {"image_id": "x", "labels": {"subject": "b1"}, "confidence": {"subject": 0.95}},
         "y": {"image_id": "y", "labels": {"subject": "b1"}, "confidence": {"subject": 0.9}}}
    merged = distill.merge_teachers({"a": a, "b": b}, {"subject": "b"}, dims, default="a")
    assert merged["x"]["labels"] == {"subject": "b1", "perspective": "a2", "background": "a3"}
    assert merged["x"]["confidence"] == {"subject": 0.95, "perspective": 0.6, "background": 0.7}
    # y has no answer from the default teacher: only the dimension its own teacher answered, so the rest stays masked.
    assert merged["y"]["labels"] == {"subject": "b1"}

    only_b = distill.merge_teachers({"b": b}, {"subject": "b"}, dims)
    assert set(only_b["x"]["labels"]) == {"subject"}
    with pytest.raises(ValueError, match="class contract"):
        distill.merge_teachers({"a": a}, {"colour": "a"}, dims)
    with pytest.raises(KeyError, match="gpt"):
        distill.merge_teachers({"a": a}, {"subject": "gpt"}, dims)


def test_a_source_is_a_teacher_name_or_an_exp_ids_saved_answers(tmp_path):
    assert distill.source_file(tmp_path, "b70") == tmp_path / "distill" / "b70" / "train.jsonl"
    assert distill.source_file(tmp_path, "EXP-0015", "dev") == tmp_path / "predictions" / "EXP-0015" / "dev.jsonl"


def test_a_merged_file_is_what_the_loras_teacher_option_reads(tmp_path):
    merged = distill.merge_teachers(
        {"a": {"x": {"image_id": "x", "labels": {"subject": "a1"}, "confidence": {"subject": 0.9}}}},
        {}, ("subject",), default="a")
    path = distill.write_records(merged, distill.teacher_file(tmp_path, "mixed"))
    assert load_records(path) == merged
