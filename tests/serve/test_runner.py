"""B4 runner: one line per image, resume without duplicates, contract checks, timing records."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from serve_helpers import DIMENSIONS, Pred, StubPredictor

from galaxus.serve import runner
from galaxus.serve.items import (
    PendingImageItem,
    context_columns,
    items_from_frame,
    load_predictor,
)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_line_per_image_in_item_order(items, tmp_path):
    report = runner.run(StubPredictor(), items, "EXP-0001", "dev", batch_size=2, root=tmp_path, log=None)
    path = runner.predictions_path("EXP-0001", "dev", tmp_path)
    assert [record["image_id"] for record in lines(path)] == [item.image_id for item in items]
    assert set(lines(path)[0]["labels"]) == set(DIMENSIONS)
    assert report.complete and report.predicted == 3 and report.already_done == 0
    assert report.images_per_second and report.images_per_second > 0
    timing = lines(runner.runs_path(path))
    assert timing[-1]["predicted"] == 3 and timing[-1]["batch_size"] == 2


def test_killed_halfway_then_restarted_finishes_without_duplicates(items, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        runner.run(StubPredictor(crash_after_batches=1), items, "E", "dev", batch_size=1, root=tmp_path, log=None)
    path = runner.predictions_path("E", "dev", tmp_path)
    assert len(lines(path)) == 1

    second = StubPredictor()
    report = runner.run(second, items, "E", "dev", batch_size=1, root=tmp_path, log=None)
    ids = [record["image_id"] for record in lines(path)]
    assert sorted(ids) == sorted(item.image_id for item in items) and len(ids) == len(set(ids))
    assert second.calls == [[items[1].image_id], [items[2].image_id]]
    assert report.already_done == 1 and report.predicted == 2 and report.complete

    third = StubPredictor()
    report = runner.run(third, items, "E", "dev", root=tmp_path, log=None)
    assert third.calls == [] and report.predicted == 0 and report.complete


def test_a_line_cut_off_by_a_crash_is_repaired_and_predicted_again(items, tmp_path):
    runner.run(StubPredictor(), items[:2], "E", "dev", root=tmp_path, log=None)
    path = runner.predictions_path("E", "dev", tmp_path)
    with path.open("ab") as handle:
        handle.write(b'{"image_id": "img_d3", "lab')
    predictor = StubPredictor()
    runner.run(predictor, items, "E", "dev", root=tmp_path, log=None)
    assert predictor.calls == [[items[2].image_id]]
    assert [record["image_id"] for record in lines(path)] == [item.image_id for item in items]


def test_a_broken_line_in_the_middle_is_an_error(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"image_id": "a", "labels": {"x": "y"}}\nnot json\n{"image_id": "b", "labels": {}}\n')
    with pytest.raises(ValueError, match="p.jsonl:2"):
        runner.read_predictions(path)


class Missing(StubPredictor):
    def predict(self, items):
        return super().predict(items)[1:]


class Extra(StubPredictor):
    def predict(self, items):
        return [*super().predict(items), Pred("intruder", {"subject": "x"})]


class Empty(StubPredictor):
    def predict(self, items):
        return [Pred(item.image_id, {"subject": ""}) for item in items]


@pytest.mark.parametrize("bad", [Missing, Extra, Empty])
def test_a_contract_violation_writes_nothing_of_that_batch(items, tmp_path, bad):
    with pytest.raises(runner.PredictorContractError):
        runner.run(bad(), items, "E", "dev", batch_size=3, root=tmp_path, log=None)
    path = runner.predictions_path("E", "dev", tmp_path)
    assert not path.exists() or path.read_text() == ""


def test_predictions_as_plain_mappings_are_accepted(items, tmp_path):
    class Mappings:
        def predict(self, batch):
            return [{"image_id": item.image_id, "labels": {"subject": "s"}, "confidence": None} for item in batch]

    report = runner.run(Mappings(), items, "E", "dev", root=tmp_path, log=None)
    assert report.complete


def test_duplicate_items_and_unsafe_names_are_rejected(items, tmp_path):
    with pytest.raises(ValueError, match="more than once"):
        runner.run(StubPredictor(), [items[0], items[0]], "E", "dev", root=tmp_path, log=None)
    for exp, part in (("../escape", "dev"), ("E", "dev/../../x"), ("", "dev")):
        with pytest.raises(ValueError):
            runner.predictions_path(exp, part, tmp_path)


def test_items_from_frame_keeps_context_and_drops_labels():
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "image_path": ["i/a.jpg", "i/b.jpg"],
            "image_file": ["/d/i/a.jpg", "/d/i/b.jpg"],
            "product_id": ["p", "p"],
            "product_title": ["Mouse", ""],
            "subject": ["x", "y"],
            "subject_source": ["human", "model"],
            "subject_relevant": ["true", "true"],
        }
    )
    assert context_columns(frame) == ["product_title"]
    first, second = items_from_frame(frame)
    assert first.path == "/d/i/a.jpg" and first.context == {"product_title": "Mouse"}
    assert second.context is None
    assert len(items_from_frame(frame, limit=1)) == 1
    with pytest.raises(KeyError, match="image_file"):
        items_from_frame(frame.drop(columns=["image_file"]))


def test_pending_item_matches_the_contract_fields():
    item = PendingImageItem("a", "/x.jpg")
    assert (item.image_id, item.path, item.context) == ("a", "/x.jpg", None)


def test_load_predictor_from_class_factory_or_object():
    assert isinstance(load_predictor("serve_helpers:StubPredictor"), StubPredictor)
    assert isinstance(load_predictor("serve_helpers:stub_factory"), StubPredictor)
    with pytest.raises(ValueError):
        load_predictor("no_colon")
    with pytest.raises(TypeError):
        load_predictor("serve_helpers:DIMENSIONS")
