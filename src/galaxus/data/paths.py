"""Where the competition data is: ``$GALAXUS_DATA``, always outside the repository.

AGENTS.md "Galaxus product rules" 3: competition data, labels, ``metric.py`` and ``labeling_guideline.*`` are
loaded from ``$GALAXUS_DATA`` at runtime and never committed. Outputs go to ``$GALAXUS_DATA/out/``.
"""

from __future__ import annotations

import os
from pathlib import Path

GALAXUS_DATA_ENV = "GALAXUS_DATA"


#: The folders of BASE_ARCHITECTURE.md section 3, "Data layout (outside git)".
KAGGLE_SUBDIR = "kaggle"
FINAL_SUBDIR = "final"
FINAL_CSV = "final.csv"


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
    """The path of one Kaggle file: ``$GALAXUS_DATA/kaggle/<name>`` if it is there, else ``$GALAXUS_DATA/<name>``.

    BASE_ARCHITECTURE.md section 3 puts the download in ``kaggle/``. The flat layout is still accepted because
    teammates downloaded straight into ``$GALAXUS_DATA`` before that was written (issue #26, and the "Path mismatch
    to settle" note on PR #19).

    Raises:
        MissingDataDir: if ``$GALAXUS_DATA`` is unusable.
        FileNotFoundError: if the file is in neither place, naming both and what to download.
    """
    root = data_dir()
    candidates = (root / KAGGLE_SUBDIR / name, root / name)
    for path in candidates:
        if path.exists():
            return path
    looked = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"{name} is missing: looked for {looked}. Download the competition files into ${GALAXUS_DATA_ENV} "
        f"(https://www.kaggle.com/competitions/hackapertus-galaxus/data)."
    )


def kaggle_dir() -> Path:
    """``$GALAXUS_DATA/kaggle/``: the competition download, untouched.

    An input, so it is not created here. Read a file from it with :func:`data_file`, which also accepts the flat
    layout.
    """
    return data_dir() / KAGGLE_SUBDIR


def final_dir() -> Path:
    """``$GALAXUS_DATA/final/``: where a human decrypts the final images on Saturday. An input, not created."""
    return data_dir() / FINAL_SUBDIR


def locate_final_csv() -> Path | None:
    """The one ``final.csv`` anywhere below ``$GALAXUS_DATA/final/``, or ``None`` while it is not decrypted yet.

    The archive decides how deep it lands, so this searches the tree, which is what makes ``runner --part final``
    and :func:`galaxus.data.load.read_final` agree on Saturday (issue #26).

    **Serve imports this one** (``galaxus.serve.final_run``, #91) rather than keeping its own copy of the rule, so
    this function is on the 13:00 path of the final run and not only on this lane's readers: changing it changes
    that run. Measured, rather than assumed - making the search shallow (``glob`` for ``rglob``) fails 6 tests in
    ``tests/data`` and 5 in ``tests/serve``.

    Raises:
        FileNotFoundError: if the tree holds more than one, since guessing which is the real one could silently
            score the wrong images.
    """
    root = final_dir()
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob(FINAL_CSV))
    if len(candidates) > 1:
        found = [str(path.relative_to(root)) for path in candidates]
        raise FileNotFoundError(f"expected one {FINAL_CSV} under {root}, found {len(found)}: {found[:10]}")
    return candidates[0] if candidates else None


def out_dir() -> Path:
    """``$GALAXUS_DATA/out/``, created if needed. Everything we generate goes here, never into the repository."""
    path = data_dir() / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(*parts: str) -> Path:
    """``$GALAXUS_DATA/cache/<parts>/``, created if needed.

    The layout is BASE_ARCHITECTURE.md "Data layout (outside git)": ``cache/images/<size>/`` for prepared JPEGs,
    ``cache/features/<exp>/`` for hidden states. Derived files only, so deleting the cache costs time, never data.
    """
    path = data_dir().joinpath("cache", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
