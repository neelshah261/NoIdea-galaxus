"""Reading the competition CSVs from ``$GALAXUS_DATA``.

Every column stays exactly as it is in the file: read as strings with empty cells kept empty, so a label, a
``{dimension}_source`` or a ``{dimension}_relevant`` flag is never silently retyped or turned into ``NaN``. The
one added column is ``image_file``, the absolute path of the image, resolved against the CSV's own directory as in
[reference/finetune_qwen35_reference.ipynb](../../../reference/finetune_qwen35_reference.ipynb) (``read_part``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from galaxus.data.paths import FINAL_CSV, data_file, final_dir, locate_final_csv

IMAGE_PATH_COLUMN = "image_path"
IMAGE_FILE_COLUMN = "image_file"


def read_part(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    """One competition CSV as strings, with the full path of each image in ``image_file``."""
    path = Path(csv_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if IMAGE_PATH_COLUMN in frame.columns:
        frame[IMAGE_FILE_COLUMN] = [str(path.parent / image_path) for image_path in frame[IMAGE_PATH_COLUMN]]
    return frame


def read_train() -> pd.DataFrame:
    """``train.csv``: the labeled images, with ``product_id`` and the label, ``_source`` and ``_relevant`` columns."""
    return read_part(data_file("train.csv"))


def read_dev() -> pd.DataFrame:
    """``dev.csv``: the images of the public leaderboard, unlabeled."""
    return read_part(data_file("dev.csv"))


def read_final() -> pd.DataFrame:
    """``final.csv``: the images of the private leaderboard. Only after the passphrase is published.

    Raises:
        FileNotFoundError: before the final images are decrypted, naming the next step.
    """
    decrypted = locate_final_csv()
    if decrypted is not None:
        return read_part(decrypted)
    try:
        return read_part(data_file(FINAL_CSV))
    except FileNotFoundError as missing:
        raise FileNotFoundError(
            f"{FINAL_CSV} is not in $GALAXUS_DATA yet: it is neither under {final_dir()} nor beside the other "
            "Kaggle files. The final images are decrypted with the passphrase published Sat 28 Nov 13:00 CET "
            "(challenge_description.md, Timeline); a human runs the decryption."
        ) from missing


def read_sample_submission() -> pd.DataFrame:
    """``sample_submission.csv``: the exact header and row order every submission must keep."""
    return read_part(data_file("sample_submission.csv"))
