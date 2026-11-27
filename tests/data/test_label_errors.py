"""Issue #7: suspected label errors. Synthetic labels and predictions only; the csv never enters git."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
import pytest

from galaxus.data.label_errors import (
    PREDICTION_COLUMNS,
    PREDICTIONS_FILE,
    REPORT_COLUMNS,
    find_label_errors,
    out_of_fold_frame,
    write_label_errors,
)

DIMENSION = ("subject",)


def predictions(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(PREDICTION_COLUMNS))


def test_the_report_columns_are_exactly_what_the_hosts_asked_for():
    assert REPORT_COLUMNS == ("image_id", "dimension", "label", "suggested_label")


def test_a_high_confidence_disagreement_with_a_human_label_is_reported(train_table):
    report = find_label_errors(predictions(("img_1", "subject", "beta", 0.99)), train_table, dimension_names=DIMENSION)
    assert report.to_dict(orient="records") == [
        {"image_id": "img_1", "dimension": "subject", "label": "alpha", "suggested_label": "beta"}
    ]


def test_an_agreement_is_not_reported(train_table):
    report = find_label_errors(predictions(("img_1", "subject", "alpha", 0.99)), train_table, dimension_names=DIMENSION)
    assert report.empty
    assert list(report.columns) == list(REPORT_COLUMNS)


def test_a_disagreement_below_the_threshold_is_not_reported(train_table):
    report = find_label_errors(
        predictions(("img_1", "subject", "beta", 0.5)), train_table, threshold=0.9, dimension_names=DIMENSION
    )
    assert report.empty


def test_the_threshold_is_a_parameter(train_table):
    rows = predictions(("img_1", "subject", "beta", 0.5))
    assert find_label_errors(rows, train_table, threshold=0.4, dimension_names=DIMENSION).shape[0] == 1
    assert find_label_errors(rows, train_table, threshold=0.6, dimension_names=DIMENSION).empty


def test_a_model_prelabel_is_never_a_label_error(train_table):
    """About a fifth of train cells are unchecked prelabels (FAQ.md): disagreeing with one proves nothing."""
    report = find_label_errors(predictions(("img_3", "subject", "beta", 0.99)), train_table, dimension_names=DIMENSION)
    assert report.empty


def test_an_irrelevant_cell_with_a_human_label_is_reported_by_default(train_table):
    report = find_label_errors(predictions(("img_4", "subject", "alpha", 0.99)), train_table, dimension_names=DIMENSION)
    assert report["image_id"].tolist() == ["img_4"]


def test_scored_only_restricts_to_the_cells_the_metric_scores(train_table):
    report = find_label_errors(
        predictions(("img_4", "subject", "alpha", 0.99)), train_table, dimension_names=DIMENSION, scored_only=True
    )
    assert report.empty


def test_the_report_is_ranked_by_confidence(train_table):
    report = find_label_errors(
        predictions(
            ("img_1", "subject", "beta", 0.91),
            ("img_5", "subject", "beta", 0.99),
            ("img_7", "subject", "beta", 0.95),
        ),
        train_table,
        dimension_names=DIMENSION,
    )
    assert report["image_id"].tolist() == ["img_5", "img_7", "img_1"]


def test_an_image_that_is_not_in_train_is_ignored(train_table):
    report = find_label_errors(
        predictions(("img_unknown", "subject", "beta", 0.99)), train_table, dimension_names=DIMENSION
    )
    assert report.empty


def test_a_nested_mapping_of_predictions_is_accepted():
    frame = out_of_fold_frame(
        {"img_1": {"subject": ("beta", 0.9)}, "img_2": {"subject": {"prediction": "alpha", "confidence": 0.8}}}
    )
    assert frame["prediction"].tolist() == ["beta", "alpha"]
    assert frame["confidence"].tolist() == [0.9, 0.8]


def test_a_table_missing_a_column_is_refused():
    with pytest.raises(KeyError, match="confidence"):
        out_of_fold_frame(pd.DataFrame({"image_id": ["img_1"], "dimension": ["subject"], "prediction": ["beta"]}))


def test_a_confidence_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        out_of_fold_frame(predictions(("img_1", "subject", "beta", "very sure")))


def test_a_missing_dimension_column_in_train_is_an_error(train_table):
    with pytest.raises(KeyError, match="subject"):
        find_label_errors(
            predictions(("img_1", "subject", "beta", 0.99)),
            train_table.drop(columns=["subject_source"]),
            dimension_names=DIMENSION,
        )


def test_the_csv_is_written_to_the_out_directory_outside_the_repo(galaxus_data, train_table):
    report = find_label_errors(predictions(("img_1", "subject", "beta", 0.99)), train_table, dimension_names=DIMENSION)
    path = write_label_errors(report)
    assert path == galaxus_data / "out" / "label_errors.csv"
    assert pd.read_csv(path).columns.tolist() == list(REPORT_COLUMNS)


def test_a_report_with_the_wrong_columns_is_refused(galaxus_data):
    with pytest.raises(ValueError, match="exactly the columns"):
        write_label_errors(pd.DataFrame({"image_id": ["img_1"], "oops": ["x"]}))


def test_the_report_is_never_written_into_the_repository(tmp_path, train_table):
    """AGENTS.md "Galaxus product rules" 3: the label-error csv never enters git. A human sends it on Discord."""
    report = find_label_errors(predictions(("img_1", "subject", "beta", 0.99)), train_table, dimension_names=DIMENSION)
    target = tmp_path / "label_errors.csv"
    assert write_label_errors(report, path=target) == target


# The format design settled on: PLAN.md "Lanes and contracts", written by galaxus.model.eval.write_predictions.


def record(image_id: str, label: str, confidence: float) -> dict:
    return {"image_id": image_id, "labels": {"subject": label}, "confidence": {"subject": confidence}}


def write_jsonl(path, records, trailing_blank_line: bool = False) -> None:
    """One JSON object per line, the way ``galaxus.model.eval.write_predictions`` writes it."""
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            print(json.dumps(item), file=handle)
        if trailing_blank_line:
            print(file=handle)


def test_prediction_records_from_the_model_lane_are_accepted(train_table):
    report = find_label_errors([record("img_1", "beta", 0.99)], train_table, dimension_names=DIMENSION)
    assert report.to_dict(orient="records") == [
        {"image_id": "img_1", "dimension": "subject", "label": "alpha", "suggested_label": "beta"}
    ]


def test_prediction_objects_are_accepted_not_only_mappings(train_table):
    """``Prediction`` is a frozen dataclass in galaxus.types, so attributes must work as well as keys."""

    @dataclass(frozen=True)
    class Prediction:
        image_id: str
        labels: Mapping[str, str]
        confidence: Mapping[str, float] | None = None

    predicted = [Prediction(image_id="img_1", labels={"subject": "beta"}, confidence={"subject": 0.99})]
    report = find_label_errors(predicted, train_table, dimension_names=DIMENSION)
    assert report["suggested_label"].tolist() == ["beta"]


def test_the_jsonl_file_the_model_lane_writes_is_read(tmp_path, train_table):
    target = tmp_path / PREDICTIONS_FILE
    write_jsonl(target, [record("img_1", "beta", 0.99), record("img_5", "beta", 0.95)])
    report = find_label_errors(target, train_table, dimension_names=DIMENSION)
    assert report["image_id"].tolist() == ["img_1", "img_5"]


def test_the_directory_holding_the_jsonl_file_is_accepted(tmp_path, train_table):
    """The model lane writes it under ``out/predictions/<EXP id>/``, so the run directory is the natural handle."""
    write_jsonl(tmp_path / PREDICTIONS_FILE, [record("img_1", "beta", 0.99)])
    assert find_label_errors(tmp_path, train_table, dimension_names=DIMENSION).shape[0] == 1


def test_blank_lines_in_the_jsonl_file_are_skipped(tmp_path):
    target = tmp_path / PREDICTIONS_FILE
    write_jsonl(target, [record("img_1", "beta", 0.99)], trailing_blank_line=True)
    assert out_of_fold_frame(target).shape[0] == 1


def test_a_missing_jsonl_file_says_what_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match=PREDICTIONS_FILE):
        out_of_fold_frame(tmp_path)


def test_a_prediction_without_confidences_is_refused():
    """``Prediction.confidence`` is optional in the contract, but this report ranks by it (issue #7)."""
    with pytest.raises(ValueError, match="no confidence"):
        out_of_fold_frame([{"image_id": "img_1", "labels": {"subject": "beta"}, "confidence": None}])


def test_a_dimension_without_a_confidence_is_refused():
    with pytest.raises(KeyError, match="background"):
        out_of_fold_frame([{"image_id": "img_1", "labels": {"background": "white"}, "confidence": {"subject": 0.9}}])


def test_records_and_the_long_table_agree():
    """The old hand-written form and the contract form must produce the same table."""
    from_records = out_of_fold_frame([record("img_1", "beta", 0.99)])
    from_table = out_of_fold_frame(predictions(("img_1", "subject", "beta", 0.99)))
    pd.testing.assert_frame_equal(from_records, from_table)
