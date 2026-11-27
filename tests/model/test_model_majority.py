"""Issue #12: the majority baseline (EXP-0001). Synthetic labels only."""

from __future__ import annotations

import pandas as pd
import pytest
from model_synthetic import DIMENSIONS, fake_scored_mask, train_table

from galaxus.model.baselines import MajorityPredictor
from galaxus.model.contract import ImageItem, Predictor


def table(cells: list[tuple[str, str, str]]) -> pd.DataFrame:
    """One dimension, ``subject``: (label, source, relevant) per image, as strings like the CSV reader returns."""
    return pd.DataFrame(
        {
            "image_id": [f"img_{index}" for index in range(len(cells))],
            "subject": [cell[0] for cell in cells],
            "subject_source": [cell[1] for cell in cells],
            "subject_relevant": [cell[2] for cell in cells],
        }
    )


def answer(cells: list[tuple[str, str, str]]) -> dict[str, str]:
    return MajorityPredictor.fit(table(cells), ["subject"], fake_scored_mask).answer


def test_the_most_frequent_human_label_wins():
    assert answer([("a", "human", "true"), ("b", "human", "true"), ("b", "human", "true")]) == {"subject": "b"}


def test_model_prelabels_never_count():
    assert answer([("a", "human", "true")] + [("b", "model", "true")] * 5) == {"subject": "a"}


def test_cells_that_are_not_scored_never_count():
    assert answer([("a", "human", "true")] + [("b", "human", "false")] * 5) == {"subject": "a"}


def test_empty_labels_never_count():
    assert answer([("", "human", "true")] * 3 + [("a", "human", "true")]) == {"subject": "a"}


def test_a_tie_goes_to_the_name_that_sorts_first_whatever_the_row_order():
    cells = [("b", "human", "true"), ("a", "human", "true")]
    assert answer(cells) == answer(list(reversed(cells))) == {"subject": "a"}


def test_a_dimension_without_a_scored_human_label_is_an_error():
    with pytest.raises(ValueError, match="no scored human label for subject"):
        answer([("a", "model", "true"), ("b", "human", "false")])


def test_every_image_gets_the_same_answer_in_order():
    predictor = MajorityPredictor({"subject": "a", "background": "x"})
    items = [ImageItem(image_id=f"img_{index}", path=f"/nowhere/{index}.jpg") for index in range(3)]
    predictions = predictor.predict(items)
    assert [prediction.image_id for prediction in predictions] == ["img_0", "img_1", "img_2"]
    assert all(dict(prediction.labels) == {"subject": "a", "background": "x"} for prediction in predictions)
    assert all(prediction.confidence is None for prediction in predictions)
    assert predictor.predict([]) == []


def test_it_is_a_predictor_and_records_its_answer():
    predictor = MajorityPredictor({"subject": "a"})
    assert isinstance(predictor, Predictor)
    assert predictor.config() == {"answer": {"subject": "a"}}


def test_an_empty_answer_is_refused():
    with pytest.raises(ValueError):
        MajorityPredictor({})


def test_the_data_lanes_scored_mask_gives_the_same_answer():
    split = pytest.importorskip("galaxus.data.split", reason="the data lane (#5) is not on this branch yet")
    cells = [("a", "human", "true")] + [("b", "model", "true")] * 3 + [("b", "human", "false")] * 3
    cells += [("c", "human", "true")] * 2
    assert MajorityPredictor.fit(table(cells), ["subject"], split.scored_mask).answer == {"subject": "c"}


@pytest.fixture
def synthetic_data_dir(tmp_path, monkeypatch):
    """A ``$GALAXUS_DATA`` with a synthetic train.csv, and the expected answer: train minus the official holdout."""
    pytest.importorskip("galaxus.data.split", reason="the data lane (#4, #5) is not on this branch yet")
    from galaxus.data import classes
    from galaxus.data.load import read_train
    from galaxus.data.split import scored_mask, split_holdout

    data_dir = tmp_path / "galaxus-data"
    data_dir.mkdir()
    train_table(products=20).drop(columns=["image_file"]).to_csv(data_dir / "train.csv", index=False)
    monkeypatch.setenv("GALAXUS_DATA", str(data_dir))
    monkeypatch.setattr(classes, "dimensions", lambda: DIMENSIONS)
    return MajorityPredictor.fit(split_holdout(read_train())[0], DIMENSIONS, scored_mask).answer


def test_without_an_answer_it_fits_itself_on_train_minus_the_official_holdout(synthetic_data_dir):
    assert MajorityPredictor().answer == synthetic_data_dir


def test_the_serve_lanes_runner_can_build_it_by_name(synthetic_data_dir):
    """``--predictor galaxus.model.baselines:MajorityPredictor``, the B5 command (#15)."""
    items = pytest.importorskip("galaxus.serve.items", reason="the serve lane (#14) is not on this branch yet")
    assert items.load_predictor("galaxus.model.baselines:MajorityPredictor").answer == synthetic_data_dir
