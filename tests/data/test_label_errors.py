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
    main,
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


# --- #7: the command line ------------------------------------------------------------------------------------
#
# The finder had a library API and no way to run it, unlike prep and validate. The bonus csv is due by 14:30 on
# 28 November, so the moment a predictor emits confidences (#75) this has to be one command, not a script someone
# writes under time pressure.


@pytest.fixture
def predictions_file(galaxus_data, train_table):
    """Out-of-fold predictions on disk, in the model lane's jsonl shape, disagreeing with img_1's human label."""
    train_table.to_csv(galaxy := galaxus_data / "train.csv", index=False)
    assert galaxy.is_file()
    folder = galaxus_data / "out" / "predictions" / "EXP-0099"
    folder.mkdir(parents=True)
    path = folder / PREDICTIONS_FILE
    path.write_text(
        "\n".join(
            json.dumps({"image_id": image_id, "labels": {"subject": cls}, "confidence": {"subject": confidence}})
            # img_1 is labelled alpha and img_2 beta, so both of these disagree with a human label; only the
            # first is confident enough for the default threshold.
            for image_id, cls, confidence in [("img_1", "beta", 0.99), ("img_2", "alpha", 0.4)]
        ),
        encoding="utf-8",
    )
    return path


def test_the_cli_writes_the_csv_and_reports_counts(predictions_file, galaxus_data, capsys):
    assert main(["--predictions", str(predictions_file), "--dimensions", "subject"]) == 0
    printed = capsys.readouterr().out
    assert "1 suspected error(s)" in printed
    assert "subject 1" in printed
    written = galaxus_data / "out" / "label_errors.csv"
    assert written.is_file()
    assert list(pd.read_csv(written).columns) == list(REPORT_COLUMNS)


def test_the_cli_prints_no_image_id_and_no_label(predictions_file, capsys):
    """Product rule 3: identifiers are masked in anything shown. Counts are safe, rows are not."""
    main(["--predictions", str(predictions_file), "--dimensions", "subject"])
    printed = capsys.readouterr().out
    assert "img_1" not in printed
    assert "alpha" not in printed and "beta" not in printed


def test_the_cli_refuses_to_write_inside_a_git_work_tree(predictions_file, tmp_path, capsys):
    """The one way this file could be committed by accident is a mistyped --out."""
    repository = tmp_path / "checkout"
    (repository / ".git").mkdir(parents=True)
    assert main(["--predictions", str(predictions_file), "--out", str(repository / "errors.csv")]) == 1
    assert "never enter git" in capsys.readouterr().err
    assert not (repository / "errors.csv").exists()


def test_the_cli_dry_run_writes_nothing(predictions_file, galaxus_data, capsys):
    assert main(["--predictions", str(predictions_file), "--dimensions", "subject", "--dry-run"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert not (galaxus_data / "out" / "label_errors.csv").exists()


def test_the_cli_threshold_is_a_confidence(predictions_file):
    with pytest.raises(SystemExit):
        main(["--predictions", str(predictions_file), "--threshold", "90"])


def test_the_cli_threshold_changes_what_is_reported(predictions_file, capsys):
    main(["--predictions", str(predictions_file), "--dimensions", "subject", "--threshold", "0.3", "--dry-run"])
    assert "2 suspected error(s)" in capsys.readouterr().out


def test_the_cli_top_keeps_only_the_most_confident(predictions_file, capsys):
    main(
        [
            "--predictions",
            str(predictions_file),
            "--dimensions",
            "subject",
            "--threshold",
            "0.3",
            "--top",
            "1",
            "--dry-run",
        ]
    )
    assert "1 suspected error(s)" in capsys.readouterr().out


def test_the_cli_reports_a_missing_predictions_file_without_a_traceback(galaxus_data, train_table, capsys):
    train_table.to_csv(galaxus_data / "train.csv", index=False)
    assert main(["--predictions", str(galaxus_data / "nothing-here.jsonl")]) == 1
    assert capsys.readouterr().err.startswith("cannot find label errors:")


# --- #92: reading the classes when no confidence exists ------------------------------------------------------


def test_predictions_without_confidence_can_be_read_when_not_ranking(galaxus_data):
    """A confusion table needs the class only, and the predictors it explains have no confidences yet (#75)."""
    path = galaxus_data / "holdout.jsonl"
    path.write_text(json.dumps({"image_id": "img_1", "labels": {"subject": "beta"}}), encoding="utf-8")
    frame = out_of_fold_frame(path, require_confidence=False)
    assert frame["prediction"].tolist() == ["beta"]
    assert frame["confidence"].isna().all()


def test_reading_without_confidence_is_opt_in_so_the_finder_still_refuses(galaxus_data):
    path = galaxus_data / "holdout.jsonl"
    path.write_text(json.dumps({"image_id": "img_1", "labels": {"subject": "beta"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="has no confidence"):
        out_of_fold_frame(path)


def test_a_dimension_without_a_confidence_is_kept_when_not_ranking(galaxus_data):
    """Half-filled confidences: the ranked report refuses them, the class-only read keeps the row."""
    record = {"image_id": "img_1", "labels": {"subject": "beta", "background": "white"}, "confidence": {"subject": 0.9}}
    path = galaxus_data / "holdout.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    frame = out_of_fold_frame(path, require_confidence=False)
    assert sorted(frame["dimension"]) == ["background", "subject"]
    with pytest.raises(KeyError, match="carries no confidence"):
        out_of_fold_frame(path)


def test_write_label_errors_itself_refuses_a_path_inside_a_work_tree(tmp_path):
    """The guard belongs to the writer, not just the command: #51 calls write_label_errors directly."""
    repository = tmp_path / "checkout"
    (repository / ".git").mkdir(parents=True)
    report = pd.DataFrame(columns=list(REPORT_COLUMNS))
    with pytest.raises(ValueError, match="never enter git"):
        write_label_errors(report, repository / "label_errors.csv")
    assert not (repository / "label_errors.csv").exists()


def test_write_label_errors_still_writes_the_default_outside_the_repository(galaxus_data):
    report = pd.DataFrame(columns=list(REPORT_COLUMNS))
    written = write_label_errors(report)
    assert written == galaxus_data / "out" / "label_errors.csv"
    assert written.is_file()


# --- the format this lane consumes, pinned against the lane that produces it --------------------------------


def test_the_model_lanes_writer_produces_what_this_lane_reads(galaxus_data, train_table):
    """#7 end to end across the lane boundary: `eval.write_predictions` into `find_label_errors`.

    Every other test here builds the jsonl by hand, so all of them would still pass if the model lane changed
    the record it writes. This one imports the real writer, so a divergence fails in CI instead of at 14:30 on
    the day the csv is due. Importing another lane's code is allowed (AGENTS.md "Lanes"); nothing of it is
    changed here, and the shape it pins is `contracts/prediction.md`.
    """
    from galaxus.model.eval import write_predictions
    from galaxus.types import Prediction

    predictions = [
        Prediction(image_id="img_1", labels={"subject": "beta"}, confidence={"subject": 0.97}),
        Prediction(image_id="img_2", labels={"subject": "alpha"}, confidence={"subject": 0.10}),
    ]
    path = write_predictions(galaxus_data / "out" / "predictions" / "EXP-0011", predictions)

    report = find_label_errors(path, train_table, dimension_names=("subject",))
    # img_1's human label is alpha and the model says beta at 0.97: a suspected error.
    # img_2 agrees with its label anyway, and at 0.10 would be under the threshold regardless.
    assert report.to_dict(orient="records") == [
        {"image_id": "img_1", "dimension": "subject", "label": "alpha", "suggested_label": "beta"}
    ]


def test_a_writer_that_dropped_confidence_would_be_caught_here(galaxus_data, train_table):
    """The other half of the pin: if the producer ever wrote `confidence: null`, the ranked finder must refuse."""
    from galaxus.model.eval import write_predictions
    from galaxus.types import Prediction

    path = write_predictions(
        galaxus_data / "out" / "predictions" / "EXP-0012",
        [Prediction(image_id="img_1", labels={"subject": "beta"}, confidence=None)],
    )
    assert '"confidence": null' in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="has no confidence"):
        find_label_errors(path, train_table, dimension_names=("subject",))
