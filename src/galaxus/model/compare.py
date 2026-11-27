"""``python -m galaxus.model.compare``: which variant works with which model, dimension by dimension (#106).

It reads ``results/metrics.json`` and nothing else (no competition data, no network), so it runs on any checkout.
Every number it prints is an EXP row's own, written by the eval script (AGENTS.md product rule 1).

    python -m galaxus.model.compare                          # every row that scored dimensions
    python -m galaxus.model.compare EXP-0002 EXP-0014 EXP-0030

The best row of each dimension is picked on the same holdout that scores it, which flatters it. The number to quote
is the ensemble's cross-fitted row (``python -m galaxus.model.eval ensemble``); this table says what to try next and
which runs to put into that ensemble.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from galaxus.model.results import RESULTS_FILE, read_rows
from galaxus.model.zero_shot import IMAGE_SIZE
from galaxus.serve.final_run import THROUGHPUT_FLOOR

#: Predictors that run no model: their images per second is not a product figure.
NO_MODEL = frozenset({"majority"})
#: Flags a zero-shot row's config may carry (#106), and how the table names them.
FLAGS = (("image_first", "image first"), ("macro_aware", "macro-aware"), ("observe_first", "observe first"))


def variant(config: Mapping[str, Any]) -> str:
    """A short name for what a run changed, read from its config. Rows written before #106 read as the default."""
    parts = []
    if config.get("context") is False:
        parts.append("no context")
    dropped = list(config.get("drop_context") or [])
    if dropped:
        parts.append("without " + ", ".join(dropped))
    parts += [label for flag, label in FLAGS if config.get(flag)]
    if config.get("image", "plain") != "plain":
        parts.append(str(config["image"]))
    if config.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
        parts.append(f"{config['image_size']} px")
    return ", ".join(parts) or "default"


def select(rows: Iterable[Mapping[str, Any]], exp_ids: Sequence[str] = ()) -> list[Mapping[str, Any]]:
    """The rows named by ``exp_ids``, in that order, or every row that scored dimensions.

    Raises:
        KeyError: naming every id that is not in the results file.
    """
    rows = list(rows)
    if not exp_ids:
        return [row for row in rows if row.get("macro_f1")]
    by_id = {row["id"]: row for row in rows}
    missing = [exp_id for exp_id in exp_ids if exp_id not in by_id]
    if missing:
        raise KeyError(f"not in the results file: {', '.join(missing)}")
    return [by_id[exp_id] for exp_id in exp_ids]


def dimensions(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every dimension the rows scored, in the order they first appear."""
    seen: list[str] = []
    for row in rows:
        seen += [dimension for dimension in row.get("macro_f1") or {} if dimension not in seen]
    return seen


def speed(row: Mapping[str, Any]) -> float | None:
    """Images per second, or None for a run without a model or without a measurement."""
    if row.get("predictor") in NO_MODEL or row.get("images_per_second") is None:
        return None
    return float(row["images_per_second"])


def _who(row: Mapping[str, Any]) -> str:
    return f"{row['id']} ({row.get('model') or row.get('predictor')}, {variant(row.get('config') or {})})"


def _cell(value: float | None, best: float | None) -> str:
    if value is None:
        return "–"
    text = f"{value:.3f}"
    return f"**{text}**" if best is not None and abs(value - best) < 1e-12 else text


def table(rows: Sequence[Mapping[str, Any]]) -> str:
    """A markdown table: one line per run, one column per dimension, the best of each column in bold."""
    columns = dimensions(rows)
    best = {
        dimension: max((row["macro_f1"][dimension] for row in rows if dimension in (row.get("macro_f1") or {})),
                       default=None)
        for dimension in columns
    }
    best_mean = max((row["score"] for row in rows if row.get("score") is not None), default=None)
    header = ["EXP", "predictor", "model", "variant", "mean", *columns, "img/s"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in rows:
        scores = row.get("macro_f1") or {}
        rate = speed(row)
        rate_text = "n/a (no model)" if row.get("predictor") in NO_MODEL else "–" if rate is None else f"{rate:.1f}"
        cells = [
            str(row["id"]),
            str(row.get("predictor", "")),
            str(row.get("model") or "–"),
            variant(row.get("config") or {}),
            _cell(row.get("score"), best_mean),
            *(_cell(scores.get(dimension), best[dimension]) for dimension in columns),
            rate_text,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def findings(rows: Sequence[Mapping[str, Any]], floor: float = THROUGHPUT_FLOOR) -> str:
    """The best run per dimension, overall, at the speed floor, and per model."""
    lines = ["Best per dimension (picked on the holdout that scores it, so flattering; quote the ensemble's row):"]
    for dimension in dimensions(rows):
        scored = [row for row in rows if dimension in (row.get("macro_f1") or {})]
        top = max(scored, key=lambda row: row["macro_f1"][dimension])
        lines.append(f"- {dimension}: {_who(top)} {top['macro_f1'][dimension]:.3f}")
    scored = [row for row in rows if row.get("score") is not None]
    if scored:
        top = max(scored, key=lambda row: row["score"])
        lines.append(f"Best mean: {_who(top)} {top['score']:.3f}")
    fast = [row for row in scored if (speed(row) or 0.0) >= floor]
    if fast:
        top = max(fast, key=lambda row: row["score"])
        lines.append(f"Best mean at {floor:g} img/s or more: {_who(top)} {top['score']:.3f}, {speed(top):.1f} img/s")
    models = [model for model in dict.fromkeys(row.get("model") for row in scored) if model]
    for model in models:
        top = max((row for row in scored if row.get("model") == model), key=lambda row: row["score"])
        lines.append(f"Best on {model}: {_who(top)} {top['score']:.3f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.compare", description=__doc__.splitlines()[0])
    parser.add_argument("exps", nargs="*", help="EXP ids, in the order to show them (default: every scored row)")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE, help="the results file to read")
    arguments = parser.parse_args(argv)
    try:
        rows = select(read_rows(arguments.results), arguments.exps)
    except KeyError as error:
        print(f"compare: {error.args[0]}", file=sys.stderr)
        return 2
    if not rows:
        print("compare: no row with per-dimension scores yet", file=sys.stderr)
        return 1
    print(table(rows))
    print()
    print(findings(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
