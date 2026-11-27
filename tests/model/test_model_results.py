"""Issue #12: results/metrics.json only grows, and an EXP id is never reused (AGENTS.md "State files")."""

from __future__ import annotations

import pytest

from galaxus.model.results import (
    REPOSITORY,
    RESULTS_FILE,
    ResultsFileError,
    append_row,
    next_exp_id,
    read_rows,
)


def test_a_missing_file_has_no_rows_and_the_first_id_is_one(tmp_path):
    assert read_rows(tmp_path / "metrics.json") == []
    assert next_exp_id([]) == "EXP-0001"


def test_the_next_id_follows_the_highest_one():
    assert next_exp_id([{"id": "EXP-0001"}, {"id": "EXP-0003"}, {"id": "draft"}]) == "EXP-0004"


def test_rows_are_appended_and_earlier_rows_kept(tmp_path):
    path = tmp_path / "results" / "metrics.json"
    append_row({"id": "EXP-0001", "score": 0.1}, path)
    append_row({"id": "EXP-0002", "score": 0.2}, path)
    assert [row["id"] for row in read_rows(path)] == ["EXP-0001", "EXP-0002"]
    assert path.read_text(encoding="utf-8").endswith("]\n")
    assert [entry.name for entry in path.parent.iterdir()] == ["metrics.json"], "no temporary file is left behind"


def test_an_id_is_never_reused(tmp_path):
    path = tmp_path / "metrics.json"
    append_row({"id": "EXP-0001"}, path)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ResultsFileError, match="never reused"):
        append_row({"id": "EXP-0001", "score": 1.0}, path)
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("exp_id", ["", "EXP-1", "exp-0001", "EXP-0001-b"])
def test_a_malformed_id_is_refused(tmp_path, exp_id):
    with pytest.raises(ResultsFileError, match="is not an EXP-#### id"):
        append_row({"id": exp_id}, tmp_path / "metrics.json")


def test_a_file_that_is_not_a_list_of_rows_is_an_error(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text('{"id": "EXP-0001"}', encoding="utf-8")
    with pytest.raises(ResultsFileError, match="JSON list"):
        read_rows(path)


def test_the_default_file_is_results_metrics_json_in_this_repository():
    assert RESULTS_FILE.parts[-2:] == ("results", "metrics.json")
    assert (REPOSITORY / "AGENTS.md").is_file()
