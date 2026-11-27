"""The gate in front of Kaggle: nothing invalid is ever uploaded (AGENTS.md "Galaxus product rules" 2).

What a valid file looks like is defined in [challenge_description.md, "Submission
file"](../../../challenge_description.md#submission-file) and [FAQ.md,
"Submissions"](../../../FAQ.md#submissions); the class names themselves come from
:mod:`galaxus.data.classes`, never from this file.

Checked here, each reported with the rows involved:

* the columns are exactly the ones of ``sample_submission.csv``, in that order;
* every ``image_id`` of the sample is present, none is duplicated, none is unknown, and the row order matches;
* no cell is empty;
* every cell holds a class allowed in its dimension;
* no cell holds ``None`` with a capital N, which Kaggle reads as an empty cell.

As a command line:

    python -m galaxus.data.validate $GALAXUS_DATA/out/submission.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from galaxus.data.classes import ID_COLUMN, allowed_classes
from galaxus.data.load import IMAGE_FILE_COLUMN, read_sample_submission

#: ``None`` with a capital N: a valid class name is lowercase ``none``, and Kaggle reads this as an empty cell.
CAPITAL_NONE = "None"
#: How many offending rows an error message names.
EXAMPLES = 3


class SubmissionInvalid(Exception):
    """The file would not be accepted. ``problems`` lists every failure found, not just the first."""

    def __init__(self, problems: list[str], path: Path | None = None) -> None:
        self.problems = problems
        self.path = path
        where = f"{path}: " if path is not None else ""
        super().__init__(f"{where}{len(problems)} problem(s)\n" + "\n".join(f"  - {line}" for line in problems))


def read_submission(path: str | os.PathLike[str]) -> pd.DataFrame:
    """A submission file read verbatim: every cell a string, empty cells empty, ``None`` still the text ``None``."""
    return pd.read_csv(Path(path), dtype=str, keep_default_na=False)


def _column_problems(frame: pd.DataFrame, expected: list[str]) -> list[str]:
    actual = list(frame.columns)
    if actual == expected:
        return []
    problems = []
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing:
        problems.append(f"missing column(s): {missing}")
    if extra:
        problems.append(f"unexpected column(s): {extra}")
    if not missing and not extra:
        problems.append(f"columns are in the wrong order: {actual}, expected {expected}")
    return problems


def _id_problems(frame: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    problems = []
    ids = frame[ID_COLUMN]
    duplicated = sorted(ids[ids.duplicated()].unique())
    if duplicated:
        problems.append(f"{len(duplicated)} duplicated {ID_COLUMN}(s), first: {duplicated[:EXAMPLES]}")
    expected = sample[ID_COLUMN].astype(str).tolist()
    # Both sets are built once. Writing ``image_id not in set(ids)`` inside the comprehension rebuilt the whole
    # set for every expected id, which made this check O(n^2): 124 s of the 124.5 s that validating a real
    # 10,553-row dev file took (#72). The comparisons themselves are unchanged, so the same files get the same
    # problems: ``present`` holds the submitted ids exactly as they are typed, as ``set(ids)`` did.
    present = set(ids)
    expected_set = set(expected)
    missing = [image_id for image_id in expected if image_id not in present]
    if missing:
        problems.append(
            f"{len(missing)} {ID_COLUMN}(s) of sample_submission.csv are missing, first: {missing[:EXAMPLES]}"
        )
    unknown = sorted(present - expected_set)
    if unknown:
        problems.append(f"{len(unknown)} {ID_COLUMN}(s) are not in sample_submission.csv, first: {unknown[:EXAMPLES]}")
    if not duplicated and not missing and not unknown and ids.tolist() != expected:
        problems.append("the rows are in a different order than sample_submission.csv")
    return problems


def _cell_problems(frame: pd.DataFrame, dimension_names: list[str], allowed: dict[str, frozenset[str]]) -> list[str]:
    problems = []
    for dimension in dimension_names:
        if dimension not in frame.columns:
            continue
        column = frame[dimension]
        empty = frame.loc[column == "", ID_COLUMN].tolist()
        if empty:
            problems.append(f"{dimension}: {len(empty)} empty cell(s), first: {empty[:EXAMPLES]}")
        capital = frame.loc[column == CAPITAL_NONE, ID_COLUMN].tolist()
        if capital:
            problems.append(
                f"{dimension}: {len(capital)} cell(s) hold {CAPITAL_NONE!r} with a capital N, which Kaggle reads as "
                f"an empty cell; the class name is lowercase. First: {capital[:EXAMPLES]}"
            )
        if dimension in allowed:
            bad = column[~column.isin(allowed[dimension]) & (column != "") & (column != CAPITAL_NONE)]
            if len(bad):
                names = sorted(bad.unique())
                problems.append(
                    f"{dimension}: {len(bad)} cell(s) hold a class that is not allowed in this dimension: "
                    f"{names[:EXAMPLES]}"
                )
    return problems


def validate_submission(
    submission: str | os.PathLike[str] | pd.DataFrame,
    sample: pd.DataFrame | None = None,
    allowed: dict[str, frozenset[str]] | None = None,
) -> pd.DataFrame:
    """Check a submission and return it, or raise :class:`SubmissionInvalid` listing every problem.

    Args:
        submission: the file to check, or an already-read table of strings.
        sample: the sample submission that fixes the header, the row order and the id set. Defaults to the one in
            ``$GALAXUS_DATA``.
        allowed: class names allowed per dimension. Defaults to :func:`galaxus.data.classes.allowed_classes`.
    """
    path = None if isinstance(submission, pd.DataFrame) else Path(submission)
    frame = submission if isinstance(submission, pd.DataFrame) else read_submission(path)
    sample_frame = read_sample_submission() if sample is None else sample
    expected = [column for column in sample_frame.columns if column != IMAGE_FILE_COLUMN]
    dimension_names = [column for column in expected if column != ID_COLUMN]
    allowed_per_dimension = allowed_classes() if allowed is None else allowed

    problems = _column_problems(frame, expected)
    if ID_COLUMN in frame.columns:
        problems += _id_problems(frame, sample_frame)
        problems += _cell_problems(frame, dimension_names, allowed_per_dimension)
    if problems:
        raise SubmissionInvalid(problems, path)
    return frame


def main(argv: list[str] | None = None) -> int:
    """``python -m galaxus.data.validate <file>``: 0 when the file would be accepted, 1 when it would not."""
    parser = argparse.ArgumentParser(
        prog="python -m galaxus.data.validate",
        description="Check a Kaggle submission file before a human uploads it.",
    )
    parser.add_argument("file", help="the submission csv to check")
    parser.add_argument(
        "--sample",
        default=None,
        help="sample_submission.csv to check against (default: the one in $GALAXUS_DATA)",
    )
    arguments = parser.parse_args(argv)
    sample = read_submission(arguments.sample) if arguments.sample else None
    try:
        frame = validate_submission(arguments.file, sample=sample)
    except SubmissionInvalid as invalid:
        print(invalid, file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as failure:
        print(f"cannot validate {arguments.file}: {failure}", file=sys.stderr)
        return 1
    print(f"{arguments.file}: valid, {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
