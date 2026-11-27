"""Writing the Kaggle submission file.

The rules are in [challenge_description.md, "Submission
file"](../../../challenge_description.md#submission-file) and [FAQ.md,
"Submissions"](../../../FAQ.md#submissions). Two of them shape this module:

- the header and the row order are the ones of ``sample_submission.csv``;
- a row we did not predict keeps the value ``sample_submission.csv`` has. That is how the final rows stay constant
  before Sat 28 Nov 13:00 CET, and how the best dev file gets its final rows merged in afterwards.

The file is written to ``$GALAXUS_DATA/out/``, never into the repository, and validated before this function
returns (AGENTS.md "Galaxus product rules" 2). A human uploads it; agents never do.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from galaxus.data.classes import ID_COLUMN
from galaxus.data.load import IMAGE_FILE_COLUMN, read_sample_submission
from galaxus.data.paths import out_dir
from galaxus.data.validate import validate_submission

DEFAULT_NAME = "submission.csv"

Predictions = Mapping[str, Mapping[str, str]] | pd.DataFrame


def submission_columns(sample: pd.DataFrame) -> list[str]:
    """The columns of a submission: ``image_id`` plus the six dimensions, in the sample's own order."""
    return [column for column in sample.columns if column != IMAGE_FILE_COLUMN]


def _as_mapping(predictions: Predictions, dimension_names: list[str]) -> dict[str, dict[str, str]]:
    """Predictions as ``{image_id: {dimension: class}}``, dropping cells that hold nothing."""
    if isinstance(predictions, pd.DataFrame):
        if ID_COLUMN not in predictions.columns:
            raise KeyError(f"the predictions table needs an {ID_COLUMN} column")
        records = predictions.to_dict(orient="records")
        rows = {str(record[ID_COLUMN]): record for record in records}
    else:
        rows = {str(image_id): ({} if answer is None else dict(answer)) for image_id, answer in predictions.items()}
    mapping: dict[str, dict[str, str]] = {}
    for image_id, answer in rows.items():
        cells = {
            dimension: str(answer[dimension])
            for dimension in dimension_names
            if dimension in answer and answer[dimension] is not None and str(answer[dimension]) != ""
        }
        if cells:
            mapping[image_id] = cells
    return mapping


def write_submission(
    predictions: Predictions,
    path: str | os.PathLike[str] | None = None,
    sample: pd.DataFrame | None = None,
    name: str = DEFAULT_NAME,
    validate: bool = True,
) -> Path:
    """Write ``predictions`` over a copy of ``sample_submission.csv`` and return the path written.

    Args:
        predictions: ``{image_id: {dimension: class}}`` or a table with an ``image_id`` column and dimension
            columns. Unknown image ids are rejected; missing ones keep the sample's value.
        path: where to write. Defaults to ``$GALAXUS_DATA/out/<name>``.
        sample: the sample submission. Defaults to the one in ``$GALAXUS_DATA``.
        name: file name used under ``$GALAXUS_DATA/out/`` when ``path`` is not given.
        validate: run :func:`~galaxus.data.validate.validate_submission` on the result. Keep this on for any file
            that may reach Kaggle.

    Raises:
        KeyError: if ``predictions`` holds an image id that is not in the sample submission.
        SubmissionInvalid: if the written file would not be accepted.
    """
    sample_frame = read_sample_submission() if sample is None else sample
    columns = submission_columns(sample_frame)
    dimension_names = [column for column in columns if column != ID_COLUMN]
    answers = _as_mapping(predictions, dimension_names)

    submission = sample_frame[columns].astype(str).set_index(ID_COLUMN)
    unknown = sorted(set(answers) - set(submission.index))
    if unknown:
        raise KeyError(
            f"{len(unknown)} predicted image id(s) are not in sample_submission.csv, first: {unknown[:3]}. "
            "A submission holds exactly the sample's rows."
        )
    for dimension in dimension_names:
        predicted = {image_id: cells[dimension] for image_id, cells in answers.items() if dimension in cells}
        if predicted:
            ids = list(predicted)
            submission.loc[ids, dimension] = [predicted[image_id] for image_id in ids]
    submission = submission.reset_index()[columns]

    target = Path(path) if path is not None else out_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(target, index=False)
    if validate:
        validate_submission(target, sample=sample_frame)
    return target
