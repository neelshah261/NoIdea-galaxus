"""Issue #4: the CSV loaders. Columns stay exactly as the file has them."""

from __future__ import annotations

import pytest

from galaxus.data.load import IMAGE_FILE_COLUMN, read_dev, read_final, read_part, read_sample_submission, read_train

CSV = "image_id,product_id,image_path,subject,subject_source,subject_relevant\n"


def write_csv(path, body=""):
    path.write_text(CSV + body, encoding="utf-8")
    return path


def test_image_file_is_absolute_and_relative_to_the_csv(tmp_path):
    csv_path = write_csv(tmp_path / "train.csv", "img_1,p1,train_images/img_1.jpg,alpha,human,true\n")
    frame = read_part(csv_path)
    assert frame[IMAGE_FILE_COLUMN].tolist() == [str(tmp_path / "train_images/img_1.jpg")]


def test_every_column_is_kept_as_text(tmp_path):
    csv_path = write_csv(tmp_path / "train.csv", "img_1,00123,a.jpg,alpha,human,true\n")
    frame = read_part(csv_path)
    assert frame["product_id"].tolist() == ["00123"], "a numeric-looking product id must not lose its zeros"
    assert frame["subject_relevant"].tolist() == ["true"]
    assert frame["subject_source"].tolist() == ["human"]


def test_empty_cells_stay_empty_strings_not_nan(tmp_path):
    csv_path = write_csv(tmp_path / "train.csv", "img_1,p1,a.jpg,,human,true\n")
    frame = read_part(csv_path)
    assert frame["subject"].tolist() == [""]
    assert not frame["subject"].isna().any()


def test_the_text_none_is_not_turned_into_a_missing_value(tmp_path):
    """``none`` is a real class and ``None`` must stay the text ``None`` so the validator can catch it."""
    csv_path = write_csv(tmp_path / "train.csv", "img_1,p1,a.jpg,none,human,true\nimg_2,p1,b.jpg,None,human,true\n")
    frame = read_part(csv_path)
    assert frame["subject"].tolist() == ["none", "None"]


def test_a_csv_without_image_path_gets_no_image_file_column(tmp_path):
    csv_path = tmp_path / "sample_submission.csv"
    csv_path.write_text("image_id,subject\nimg_1,alpha\n", encoding="utf-8")
    assert IMAGE_FILE_COLUMN not in read_part(csv_path).columns


@pytest.mark.parametrize(
    ("reader", "name"),
    [(read_train, "train.csv"), (read_dev, "dev.csv"), (read_sample_submission, "sample_submission.csv")],
)
def test_named_readers_read_their_file_from_the_data_dir(galaxus_data, reader, name):
    write_csv(galaxus_data / name, "img_1,p1,a.jpg,alpha,human,true\n")
    assert reader()["image_id"].tolist() == ["img_1"]


def test_missing_final_csv_explains_the_passphrase_step(galaxus_data):
    with pytest.raises(FileNotFoundError, match="passphrase"):
        read_final()


# Issue #26: on Saturday a human decrypts the final images into $GALAXUS_DATA/final/, and the
# archive decides how deep final.csv lands. read_final must find it there, or runner --part final
# fails at the one moment it cannot be retried.

FINAL_ROW = "img_9,p9,images/img_9.jpg,alpha,human,true"


def test_read_final_finds_the_csv_in_the_decrypted_folder(galaxus_data):
    release = galaxus_data / "final" / "final_images"
    release.mkdir(parents=True)
    write_csv(release / "final.csv", FINAL_ROW)
    assert read_final()["image_id"].tolist() == ["img_9"]


def test_image_paths_of_the_final_rows_resolve_against_the_decrypted_csv(galaxus_data):
    """Not against $GALAXUS_DATA: the images sit beside the csv the archive produced."""
    release = galaxus_data / "final" / "final_images"
    release.mkdir(parents=True)
    write_csv(release / "final.csv", FINAL_ROW)
    assert read_final()[IMAGE_FILE_COLUMN].tolist() == [str(release / "images/img_9.jpg")]


def test_read_final_still_accepts_the_flat_layout(galaxus_data):
    write_csv(galaxus_data / "final.csv", FINAL_ROW)
    assert read_final()["image_id"].tolist() == ["img_9"]


def test_read_final_prefers_the_decrypted_folder_over_a_stale_flat_file(galaxus_data):
    """A leftover final.csv beside the Kaggle files must not win over Saturday's decryption."""
    write_csv(galaxus_data / "final.csv", "img_stale,p0,images/x.jpg,alpha,human,true")
    release = galaxus_data / "final"
    release.mkdir(parents=True, exist_ok=True)
    write_csv(release / "final.csv", FINAL_ROW)
    assert read_final()["image_id"].tolist() == ["img_9"]


def test_the_missing_final_message_names_the_decrypted_folder(galaxus_data):
    with pytest.raises(FileNotFoundError, match="final"):
        read_final()
