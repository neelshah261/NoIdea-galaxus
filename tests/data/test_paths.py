"""The ``$GALAXUS_DATA`` contract: data lives outside the repo and a missing variable says so clearly."""

from __future__ import annotations

import pytest

from galaxus.data.paths import (
    FINAL_CSV,
    GALAXUS_DATA_ENV,
    MissingDataDir,
    data_dir,
    data_file,
    final_dir,
    kaggle_dir,
    locate_final_csv,
    out_dir,
)


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


# Issue #26: one layout for every lane. BASE_ARCHITECTURE.md section 3 puts the download in
# kaggle/ and the decrypted final images in final/, but teammates downloaded flat first, so both
# must work. serve's runner --part final depends on this agreeing with final_run.


def test_the_layout_folders_are_functions_so_other_lanes_stop_building_paths(galaxus_data):
    assert kaggle_dir() == galaxus_data / "kaggle"
    assert final_dir() == galaxus_data / "final"


def test_an_input_folder_is_not_created_just_by_asking_for_its_path(galaxus_data):
    """``kaggle/`` and ``final/`` are inputs: creating them would hide a missing download."""
    kaggle_dir()
    final_dir()
    assert not (galaxus_data / "kaggle").exists()
    assert not (galaxus_data / "final").exists()


def test_a_file_in_the_kaggle_folder_is_found(galaxus_data):
    (galaxus_data / "kaggle").mkdir()
    target = galaxus_data / "kaggle" / "train.csv"
    target.write_text("image_id", encoding="utf-8")
    assert data_file("train.csv") == target


def test_the_flat_layout_still_works(galaxus_data):
    """Whoever downloaded straight into $GALAXUS_DATA keeps working (PR #19, path mismatch)."""
    target = galaxus_data / "train.csv"
    target.write_text("image_id", encoding="utf-8")
    assert data_file("train.csv") == target


def test_the_kaggle_folder_wins_when_a_file_is_in_both(galaxus_data):
    (galaxus_data / "kaggle").mkdir()
    (galaxus_data / "train.csv").write_text("flat", encoding="utf-8")
    (galaxus_data / "kaggle" / "train.csv").write_text("kaggle", encoding="utf-8")
    assert data_file("train.csv").read_text(encoding="utf-8") == "kaggle"


def test_a_missing_file_names_both_places_it_looked(galaxus_data):
    with pytest.raises(FileNotFoundError) as refused:
        data_file("train.csv")
    message = str(refused.value)
    assert str(galaxus_data / "kaggle" / "train.csv") in message
    assert str(galaxus_data / "train.csv") in message


def test_there_is_no_final_csv_before_the_passphrase(galaxus_data):
    assert locate_final_csv() is None


def test_an_empty_final_folder_is_still_none(galaxus_data):
    (galaxus_data / "final").mkdir()
    assert locate_final_csv() is None


def test_final_csv_is_found_however_deep_the_archive_put_it(galaxus_data):
    """The same rule as galaxus.serve.final_run.find_final_csv, which searches the whole tree."""
    nested = galaxus_data / "final" / "final_images" / "release"
    nested.mkdir(parents=True)
    target = nested / FINAL_CSV
    target.write_text("image_id,image_path", encoding="utf-8")
    assert locate_final_csv() == target


def test_two_final_csv_files_are_refused_rather_than_guessed(galaxus_data):
    """Picking one could silently score the wrong images, so this stops instead."""
    for part in ("a", "b"):
        folder = galaxus_data / "final" / part
        folder.mkdir(parents=True)
        (folder / FINAL_CSV).write_text("image_id", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="expected one"):
        locate_final_csv()
