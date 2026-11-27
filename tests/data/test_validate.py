"""Issue #6: the validator catches everything Kaggle would reject. Synthetic sample and classes."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import DIMENSIONS

from galaxus.data.validate import CAPITAL_NONE, SubmissionInvalid, main, validate_submission


@pytest.fixture
def valid(sample_submission) -> pd.DataFrame:
    return sample_submission.copy()


def check(frame, sample, allowed):
    return validate_submission(frame, sample=sample, allowed=allowed)


def problems_of(frame, sample, allowed) -> list[str]:
    with pytest.raises(SubmissionInvalid) as raised:
        check(frame, sample, allowed)
    return raised.value.problems


def test_a_valid_file_passes(valid, sample_submission, allowed):
    assert check(valid, sample_submission, allowed) is valid


def test_a_missing_image_id_is_caught(valid, sample_submission, allowed):
    problems = problems_of(valid.iloc[:2], sample_submission, allowed)
    assert any("missing" in line and "img_00003" in line for line in problems)


def test_an_unknown_image_id_is_caught(valid, sample_submission, allowed):
    extra = pd.concat([valid, valid.iloc[[0]].assign(image_id="img_99999")], ignore_index=True)
    problems = problems_of(extra, sample_submission, allowed)
    assert any("not in sample_submission.csv" in line and "img_99999" in line for line in problems)


def test_a_duplicated_image_id_is_caught(valid, sample_submission, allowed):
    duplicated = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    problems = problems_of(duplicated, sample_submission, allowed)
    assert any("duplicated" in line and "img_00001" in line for line in problems)


def test_a_missing_column_is_caught(valid, sample_submission, allowed):
    problems = problems_of(valid.drop(columns=["background"]), sample_submission, allowed)
    assert any("missing column" in line and "background" in line for line in problems)


def test_an_extra_column_is_caught(valid, sample_submission, allowed):
    problems = problems_of(valid.assign(confidence="0.9"), sample_submission, allowed)
    assert any("unexpected column" in line and "confidence" in line for line in problems)


def test_columns_in_the_wrong_order_are_caught(valid, sample_submission, allowed):
    reordered = valid[["image_id", "background", "subject", *DIMENSIONS[2:]]]
    problems = problems_of(reordered, sample_submission, allowed)
    assert any("wrong order" in line for line in problems)


def test_rows_in_the_wrong_order_are_caught(valid, sample_submission, allowed):
    problems = problems_of(valid.iloc[::-1], sample_submission, allowed)
    assert any("different order" in line for line in problems)


def test_an_empty_cell_is_caught(valid, sample_submission, allowed):
    valid.loc[1, "perspective"] = ""
    problems = problems_of(valid, sample_submission, allowed)
    assert any("empty cell" in line and "perspective" in line and "img_00002" in line for line in problems)


def test_a_class_that_is_not_allowed_in_its_dimension_is_caught(valid, sample_submission, allowed):
    valid.loc[0, "subject"] = allowed["background"] and "white"
    problems = problems_of(valid, sample_submission, allowed)
    assert any("not allowed in this dimension" in line and "subject" in line for line in problems)


def test_capital_none_is_caught_and_explained(valid, sample_submission, allowed):
    valid.loc[2, "text_language"] = CAPITAL_NONE
    problems = problems_of(valid, sample_submission, allowed)
    assert any("capital N" in line and "text_language" in line and "img_00003" in line for line in problems)


def test_lowercase_none_is_a_real_class_and_passes(valid, sample_submission, allowed):
    valid["text_language"] = "none"
    valid["individuals"] = "none"
    assert check(valid, sample_submission, allowed) is valid


def test_every_problem_is_reported_not_just_the_first(valid, sample_submission, allowed):
    valid.loc[0, "subject"] = ""
    valid.loc[1, "background"] = CAPITAL_NONE
    broken = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    assert len(problems_of(broken, sample_submission, allowed)) >= 3


def test_a_file_on_disk_is_read_verbatim(tmp_path, valid, sample_submission, allowed):
    valid.loc[0, "text_language"] = CAPITAL_NONE
    path = tmp_path / "submission.csv"
    valid.to_csv(path, index=False)
    problems = problems_of(path, sample_submission, allowed)
    assert any("capital N" in line for line in problems)


def test_the_error_message_names_the_file(tmp_path, valid, sample_submission, allowed):
    valid.loc[0, "subject"] = ""
    path = tmp_path / "submission.csv"
    valid.to_csv(path, index=False)
    with pytest.raises(SubmissionInvalid) as raised:
        validate_submission(path, sample=sample_submission, allowed=allowed)
    assert str(path) in str(raised.value)
    assert raised.value.path == path


def test_the_cli_returns_zero_for_a_valid_file(tmp_path, capsys, galaxus_data, sample_submission, valid):
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    path = tmp_path / "submission.csv"
    valid.to_csv(path, index=False)
    assert main([str(path)]) == 0
    assert "valid, 3 rows" in capsys.readouterr().out


def test_the_cli_returns_one_and_prints_the_problems(tmp_path, capsys, galaxus_data, sample_submission, valid):
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    valid.loc[0, "subject"] = ""
    path = tmp_path / "submission.csv"
    valid.to_csv(path, index=False)
    assert main([str(path)]) == 1
    assert "empty cell" in capsys.readouterr().err


def test_the_cli_fails_loudly_when_it_cannot_even_read_the_file(tmp_path, capsys, galaxus_data, sample_submission):
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    assert main([str(tmp_path / "nope.csv")]) == 1
    assert "cannot validate" in capsys.readouterr().err
