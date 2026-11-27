"""The ``$GALAXUS_DATA`` contract: data lives outside the repo and a missing variable says so clearly."""

from __future__ import annotations

import pytest

from galaxus.data.paths import GALAXUS_DATA_ENV, MissingDataDir, data_dir, data_file, out_dir


def test_unset_variable_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv(GALAXUS_DATA_ENV, raising=False)
    with pytest.raises(MissingDataDir, match=GALAXUS_DATA_ENV):
        data_dir()


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_variable_raises(monkeypatch, value):
    monkeypatch.setenv(GALAXUS_DATA_ENV, value)
    with pytest.raises(MissingDataDir, match=GALAXUS_DATA_ENV):
        data_dir()


def test_variable_pointing_at_a_non_directory_raises(monkeypatch, tmp_path):
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(file_path))
    with pytest.raises(MissingDataDir, match="not a directory"):
        data_dir()


def test_data_dir_points_at_the_variable(galaxus_data):
    assert galaxus_data.is_dir()
    assert data_dir() == galaxus_data


def test_missing_file_names_the_download(galaxus_data):
    with pytest.raises(FileNotFoundError, match="train.csv"):
        data_file("train.csv")


def test_existing_file_is_returned(galaxus_data):
    (galaxus_data / "train.csv").write_text("image_id\n", encoding="utf-8")
    assert data_file("train.csv") == galaxus_data / "train.csv"


def test_out_dir_is_created_inside_the_data_dir(galaxus_data):
    assert not (galaxus_data / "out").exists()
    created = out_dir()
    assert created == galaxus_data / "out"
    assert created.is_dir()
