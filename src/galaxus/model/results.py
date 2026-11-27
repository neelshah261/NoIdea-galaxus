"""``results/metrics.json``: one row per experiment, written only by the eval script (AGENTS.md product rule 1).

The file is a JSON list of EXP rows in the order they were run. Rows are only ever appended, and an ``EXP-####``
id is never reused or renumbered (AGENTS.md "State files").
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: The repository root, from this file's place in ``src/galaxus/model/``.
REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS_FILE = REPOSITORY / "results" / "metrics.json"
EXP_ID = re.compile(r"EXP-(\d{4,})")


class ResultsFileError(ValueError):
    """``results/metrics.json`` is not a list of EXP rows, or a row would reuse an id."""


def read_rows(path: Path = RESULTS_FILE) -> list[dict[str, Any]]:
    """The rows written so far; none when the file does not exist yet."""
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) and "id" in row for row in rows):
        raise ResultsFileError(f"{path} must be a JSON list of EXP rows, each with an id")
    return rows


def check_exp_id(exp_id: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Refuse an id that is not ``EXP-####`` or that a row already has."""
    if not EXP_ID.fullmatch(exp_id):
        raise ResultsFileError(f"{exp_id!r} is not an EXP-#### id")
    if any(str(row["id"]) == exp_id for row in rows):
        raise ResultsFileError(f"{exp_id} is already in results/metrics.json: EXP ids are never reused")


def next_exp_id(rows: Iterable[Mapping[str, Any]]) -> str:
    """The id after the highest one in ``rows``: ``EXP-0001`` for an empty file."""
    numbers = [int(match.group(1)) for row in rows if (match := EXP_ID.fullmatch(str(row["id"])))]
    return f"EXP-{max(numbers, default=0) + 1:04d}"


def append_row(row: Mapping[str, Any], path: Path = RESULTS_FILE) -> None:
    """Append ``row``. The file is replaced in one step, so a crash never leaves half a file."""
    rows = read_rows(path)
    check_exp_id(str(row.get("id", "")), rows)
    rows.append(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(text)
    os.replace(handle.name, path)
