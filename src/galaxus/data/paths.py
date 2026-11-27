"""Where the competition data is: ``$GALAXUS_DATA``, always outside the repository.

AGENTS.md "Galaxus product rules" 3: competition data, labels, ``metric.py`` and ``labeling_guideline.*`` are
loaded from ``$GALAXUS_DATA`` at runtime and never committed. Outputs go to ``$GALAXUS_DATA/out/``.
"""

from __future__ import annotations

import os
from pathlib import Path

GALAXUS_DATA_ENV = "GALAXUS_DATA"


class MissingDataDir(RuntimeError):
    """``$GALAXUS_DATA`` is unset, empty or does not point at a directory."""


def data_dir() -> Path:
    """The directory that holds the Kaggle files, from ``$GALAXUS_DATA``.

    Raises:
        MissingDataDir: if the variable is unset or empty, or is not a directory.
    """
    raw = os.environ.get(GALAXUS_DATA_ENV, "").strip()
    if not raw:
        raise MissingDataDir(
            f"${GALAXUS_DATA_ENV} is not set. Point it at the directory holding the Kaggle files, outside this "
            f"repository, for example: export {GALAXUS_DATA_ENV}=~/galaxus-data"
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise MissingDataDir(f"${GALAXUS_DATA_ENV} is {path}, which is not a directory")
    return path


def data_file(name: str) -> Path:
    """The path of one Kaggle file inside ``$GALAXUS_DATA``.

    Raises:
        MissingDataDir: if ``$GALAXUS_DATA`` is unusable.
        FileNotFoundError: if the file is not there, naming what to download.
    """
    path = data_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download the competition files into ${GALAXUS_DATA_ENV} "
            f"(https://www.kaggle.com/competitions/hackapertus-galaxus/data)."
        )
    return path


def out_dir() -> Path:
    """``$GALAXUS_DATA/out/``, created if needed. Everything we generate goes here, never into the repository."""
    path = data_dir() / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path
