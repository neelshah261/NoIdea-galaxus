"""Issue #6: the submission writer. Header, row order and the constant answers of sample_submission.csv."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import DIMENSIONS, FAKE_CLASSES

from galaxus.data.submission import write_submission
from galaxus.data.validate import SubmissionInvalid, read_submission


@pytest.fixture
def data_with_sample(galaxus_data, sample_submission):
    """$GALAXUS_DATA with a sample submission and a train table that between them hold every synthetic class."""
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    train = pd.DataFrame(
        {
            "image_id": [f"t_{index}" for index in range(2)],
            **{dimension: names[:2] for dimension, names in FAKE_CLASSES.items()},
        }
    )
    train.to_csv(galaxus_data / "train.csv", index=False)
    return galaxus_data


FULL = {
    "subject": "beta",
    "background": "color",
    "perspective": "back",
    "individuals": "present",
    "text_classification": "seal",
    "text_language": "German",
}


def test_header_and_row_order_follow_the_sample(data_with_sample, sample_submission):
    path = write_submission({"img_00002": FULL})
    written = read_submission(path)
    assert list(written.columns) == list(sample_submission.columns)
    assert written["image_id"].tolist() == sample_submission["image_id"].tolist()


def test_the_file_goes_to_the_out_directory_outside_the_repo(data_with_sample):
    path = write_submission({})
    assert path == data_with_sample / "out" / "submission.csv"


def test_unpredicted_rows_keep_the_constant_answers(data_with_sample, sample_submission):
    """This is how the final rows stay constant before Sat 13:00 CET."""
    written = read_submission(write_submission({"img_00002": FULL}))
    unchanged = written[written["image_id"] != "img_00002"]
    for dimension in DIMENSIONS:
        assert set(unchanged[dimension]) == {sample_submission[dimension].iloc[0]}


def test_predicted_rows_take_the_prediction(data_with_sample):
    written = read_submission(write_submission({"img_00002": FULL})).set_index("image_id")
    assert written.loc["img_00002"].to_dict() == FULL


def test_merging_final_rows_into_a_best_dev_file(data_with_sample, sample_submission):
    """The Saturday step: the best dev file, written again with the final rows replaced."""
    best_dev = read_submission(write_submission({"img_00001": FULL}, name="dev_best.csv"))
    merged = read_submission(
        write_submission(
            pd.concat([best_dev, pd.DataFrame([{"image_id": "img_00003", **FULL}])], ignore_index=True),
            name="final.csv",
        )
    ).set_index("image_id")
    assert merged.loc["img_00001"].to_dict() == FULL
    assert merged.loc["img_00003"].to_dict() == FULL
    assert merged.loc["img_00002", "subject"] == sample_submission["subject"].iloc[0]


def test_a_pandas_row_is_accepted_as_an_answer(data_with_sample):
    row = pd.Series(FULL)
    written = read_submission(write_submission({"img_00001": row})).set_index("image_id")
    assert written.loc["img_00001"].to_dict() == FULL


def test_a_partial_prediction_keeps_the_sample_value_in_the_other_cells(data_with_sample, sample_submission):
    written = read_submission(write_submission({"img_00002": {"subject": "beta"}})).set_index("image_id")
    assert written.loc["img_00002", "subject"] == "beta"
    assert written.loc["img_00002", "background"] == sample_submission["background"].iloc[0]


def test_a_table_of_predictions_works_too(data_with_sample):
    table = pd.DataFrame([{"image_id": "img_00003", **FULL}])
    written = read_submission(write_submission(table)).set_index("image_id")
    assert written.loc["img_00003"].to_dict() == FULL


def test_an_unknown_image_id_is_refused(data_with_sample):
    with pytest.raises(KeyError, match="img_99999") as caught:
        write_submission({"img_99999": FULL})
    # The id alone is not enough to pin this. Removing the explicit check and re-running, pandas' own `.loc`
    # assignment still raises KeyError and still names the id ("None of [Index(['img_99999'], ...)] are in the
    # [index]"), so a match on the id passes either way. What that message does not do is say what went wrong or
    # what to do about it, and at 13:00 on Saturday the message is what a human has to act on. So the sentence is
    # the contract, not just the exception type.
    assert "not in sample_submission.csv" in str(caught.value)


def test_an_invalid_class_is_refused_before_the_file_can_reach_kaggle(data_with_sample):
    with pytest.raises(SubmissionInvalid, match="not allowed in this dimension"):
        write_submission({"img_00001": {**FULL, "subject": "white"}})


def test_validation_can_be_switched_off_only_explicitly(data_with_sample):
    path = write_submission({"img_00001": {**FULL, "subject": "white"}}, validate=False)
    assert read_submission(path).set_index("image_id").loc["img_00001", "subject"] == "white"


def test_an_explicit_path_is_honoured(tmp_path, data_with_sample):
    target = tmp_path / "elsewhere" / "s.csv"
    assert write_submission({}, path=target) == target
    assert target.is_file()


def test_no_prediction_at_all_writes_the_sample_unchanged(data_with_sample, sample_submission):
    written = read_submission(write_submission({}))
    pd.testing.assert_frame_equal(written, sample_submission)
