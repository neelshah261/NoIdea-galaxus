"""The eval script: any predictor × the official holdout → metric.py → one EXP row in ``results/metrics.json``.

    uv run python -m galaxus.model.eval majority --exp-id EXP-0001

Every number shown to judges comes from a row this script wrote (AGENTS.md product rule 1):

- the holdout is the fixed split of challenge_description.md "Scoring locally", from the data lane (#5);
- the scores come from metric.py through the data lane's wrapper, never from code of ours;
- the row records the commit, the config and the seed, and a checkout with uncommitted changes is refused, so the
  commit reproduces the row.

The data-lane functions come in as one :class:`DataLane`, so tests can pass synthetic ones. The real ones are
imported only when a run needs them, and read everything from ``$GALAXUS_DATA``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.model.baselines import MajorityPredictor
from galaxus.model.contract import ImageItem, Prediction, Predictor
from galaxus.model.results import (
    REPOSITORY,
    RESULTS_FILE,
    ResultsFileError,
    append_row,
    check_exp_id,
    next_exp_id,
    read_rows,
)

ID_COLUMN = "image_id"
IMAGE_FILE_COLUMN = "image_file"
PRODUCT_COLUMN = "product_id"
#: Never product context: identifiers and paths. Label columns are found by their ``_source`` sibling.
NOT_CONTEXT = frozenset({"image_id", "image_path", "image_file", "product_id"})
LABEL_METADATA_SUFFIXES = ("_source", "_relevant")
DEFAULT_BATCH_SIZE = 256


class InvalidPredictions(ValueError):
    """A predictor broke the contract: an image without a prediction, a duplicate, or an empty or unknown class."""


class DirtyCheckout(RuntimeError):
    """The checkout has uncommitted changes, so the row's commit would not reproduce its numbers."""


@dataclass(frozen=True)
class DataLane:
    """The data-lane functions a run needs (#4, #5). :meth:`real` gives the actual ones."""

    read_train: Callable[[], pd.DataFrame]
    split_holdout: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    dimensions: Callable[[], tuple[str, ...]]
    allowed_classes: Callable[[], Mapping[str, frozenset[str]]]
    scored_mask: Callable[[pd.DataFrame, str], pd.Series]
    solution_frame: Callable[[pd.DataFrame, tuple[str, ...]], pd.DataFrame]
    score: Callable[[pd.DataFrame, pd.DataFrame], float]
    evaluate: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]]
    out_dir: Callable[[], Path]

    @classmethod
    def real(cls) -> DataLane:
        """The data lane's own functions, reading from ``$GALAXUS_DATA``."""
        from galaxus.data import classes, load, metric, paths, split

        return cls(
            read_train=load.read_train,
            split_holdout=split.split_holdout,
            dimensions=classes.dimensions,
            allowed_classes=classes.allowed_classes,
            scored_mask=split.scored_mask,
            solution_frame=split.solution_frame,
            score=metric.score,
            evaluate=metric.evaluate,
            out_dir=paths.out_dir,
        )


#: ``factory(training_rows, data, seed) -> Predictor``. The training rows are train minus the official holdout.
PredictorFactory = Callable[[pd.DataFrame, DataLane, int], Predictor]


def _majority(rows: pd.DataFrame, data: DataLane, seed: int) -> Predictor:
    return MajorityPredictor.fit(rows, data.dimensions(), data.scored_mask)


#: The predictors this script runs, by name. A new method registers its factory here and changes nothing else.
PREDICTORS: dict[str, PredictorFactory] = {"majority": _majority}


def items_from_rows(rows: pd.DataFrame) -> list[ImageItem]:
    """Rows of a competition CSV as ``ImageItem``s, exactly as dev and final images become items.

    With the serve lane installed, its ``items_from_frame`` (#14) builds them, so a predictor sees the holdout the
    way the runner shows it dev and final. Until then the same rule applies here: the product context is every
    column that is not an id, a path, a label or label metadata, and an empty cell is left out.
    TODO(model): drop the copy of the rule once #14 is on the team branch.
    """
    if IMAGE_FILE_COLUMN in rows.columns:
        try:
            from galaxus.serve.items import items_from_frame
        except ImportError:
            pass
        else:
            return list(items_from_frame(rows))
    columns = [str(column) for column in rows.columns]
    labels = {column for column in columns if f"{column}_source" in columns}
    context_columns = [
        column
        for column in columns
        if column not in NOT_CONTEXT and column not in labels and not column.endswith(LABEL_METADATA_SUFFIXES)
    ]
    items = []
    for record in rows.to_dict(orient="records"):
        context = {column: str(record[column]) for column in context_columns if str(record[column]).strip()}
        path = record.get(IMAGE_FILE_COLUMN)
        items.append(
            ImageItem(
                image_id=str(record[ID_COLUMN]),
                path=str(path) if path is not None else "",
                context=context or None,
            )
        )
    return items


def predict_all(
    predictor: Predictor, items: Sequence[ImageItem], batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[list[Prediction], float]:
    """Every item's prediction, in batches, and the seconds the predictor took.

    TODO(model): go through the serve lane's runner once it exists (B4, #14), as BASE_ARCHITECTURE B3 says.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    predictions: list[Prediction] = []
    started = time.perf_counter()
    for start in range(0, len(items), batch_size):
        predictions.extend(predictor.predict(items[start : start + batch_size]))
    return predictions, time.perf_counter() - started


def predictions_frame(
    predictions: Sequence[Prediction],
    items: Sequence[ImageItem],
    dimensions: Sequence[str],
    allowed: Mapping[str, frozenset[str]],
) -> pd.DataFrame:
    """The predictions as a submission-shaped table, after checking them against the contract.

    Raises:
        InvalidPredictions: for an image without a prediction, a duplicate or unasked-for one, or an empty or unknown
            class. A predictor must map a failed answer to a valid class itself (PLAN.md, prediction.md).
    """
    expected = [item.image_id for item in items]
    by_id: dict[str, Prediction] = {}
    problems = []
    for prediction in predictions:
        if prediction.image_id in by_id:
            problems.append(f"{prediction.image_id} is predicted twice")
        by_id[prediction.image_id] = prediction
    missing = [image_id for image_id in expected if image_id not in by_id]
    if missing:
        problems.append(f"{len(missing)} image(s) have no prediction, first {missing[:3]}")
    unasked = sorted(set(by_id) - set(expected))
    if unasked:
        problems.append(f"{len(unasked)} prediction(s) are for images that were not asked for, first {unasked[:3]}")
    records = []
    for image_id in expected:
        prediction = by_id.get(image_id)
        if prediction is None:
            continue
        record = {ID_COLUMN: image_id}
        for dimension in dimensions:
            value = prediction.labels.get(dimension)
            if value is None or str(value).strip() == "":
                problems.append(f"{image_id} has no class for {dimension}")
            elif str(value) not in allowed.get(dimension, frozenset()):
                problems.append(f"{image_id}: {value!r} is not a class of {dimension}")
            record[dimension] = value
        records.append(record)
    if problems:
        shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more" if len(problems) > 5 else "")
        raise InvalidPredictions(f"the predictions break the contract (PLAN.md, prediction.md): {shown}")
    return pd.DataFrame(records, columns=[ID_COLUMN, *dimensions])


def git_state(root: Path = REPOSITORY) -> tuple[str | None, bool]:
    """The commit checked out, and whether anything outside ``results/`` differs from it."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        changes = git("status", "--porcelain", "--", ".", ":(exclude)results")
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit, bool(changes.strip())


def hardware() -> dict[str, Any]:
    """Where the run happened. The GPU fields are filled only when the predictor itself loaded torch with CUDA."""
    info: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "gpu": None,
        "peak_gpu_memory_gb": None,
    }
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    if cuda is not None and cuda.is_available():
        info["gpu"] = cuda.get_device_name(0)
        info["peak_gpu_memory_gb"] = round(cuda.max_memory_allocated() / 1e9, 2)
    return info


def fingerprint(image_ids: Sequence[str]) -> str:
    """A short hash of a set of image ids: two rows scored the same images only if theirs are equal."""
    return hashlib.sha256("\n".join(sorted(map(str, image_ids))).encode("utf-8")).hexdigest()[:16]


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _per_dimension(evaluation: Mapping[str, Any], dimensions: Sequence[str], key: str) -> dict[str, float | None]:
    """One number per dimension from ``metric.evaluate``, None where it has none."""
    return {dimension: _number((evaluation.get(dimension) or {}).get(key)) for dimension in dimensions}


def write_predictions(directory: Path, predictions: Sequence[Prediction]) -> Path:
    """The holdout answers as jsonl under ``$GALAXUS_DATA/out/predictions/<EXP id>/``, never in the repository."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "holdout.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            record = {
                "image_id": prediction.image_id,
                "labels": dict(prediction.labels),
                "confidence": None if prediction.confidence is None else dict(prediction.confidence),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def run_eval(
    predictor_name: str,
    data: DataLane | None = None,
    *,
    results_path: Path = RESULTS_FILE,
    exp_id: str | None = None,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    usd_per_hour: float | None = None,
    note: str = "",
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Score one predictor on the official holdout and append its EXP row. Returns the row.

    Raises:
        KeyError: for a predictor that is not in :data:`PREDICTORS`.
        DirtyCheckout: when the checkout has uncommitted changes and ``allow_dirty`` is false.
        ResultsFileError: for an id that is malformed or already used.
        InvalidPredictions: when the predictor breaks the contract. Nothing is written then.
    """
    if predictor_name not in PREDICTORS:
        raise KeyError(f"unknown predictor {predictor_name!r}, known: {', '.join(sorted(PREDICTORS))}")
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise DirtyCheckout(
            "the checkout has uncommitted changes outside results/, so its commit would not reproduce this row: "
            "commit them first, or pass --allow-dirty and the row says dirty"
        )
    rows = read_rows(results_path)
    exp_id = exp_id or next_exp_id(rows)
    check_exp_id(exp_id, rows)

    data = data or DataLane.real()
    dimensions = tuple(data.dimensions())
    training_rows, holdout = data.split_holdout(data.read_train())
    predictor = PREDICTORS[predictor_name](training_rows, data, seed)
    items = items_from_rows(holdout)
    predictions, seconds = predict_all(predictor, items, batch_size)
    submission = predictions_frame(predictions, items, dimensions, data.allowed_classes())
    solution = data.solution_frame(holdout, dimensions)
    score = data.score(solution, submission)
    evaluation = data.evaluate(solution, submission)

    images_per_second = len(items) / seconds if seconds > 0 else None
    cost = None
    if usd_per_hour is not None and images_per_second:
        cost = usd_per_hour / (images_per_second * 3600) * 100_000  # PLAN.md "Speed"
    config = getattr(predictor, "config", None)
    row = {
        "id": exp_id,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "predictor": predictor_name,
        "commit": commit,
        "dirty": dirty,
        "seed": seed,
        "config": {"batch_size": batch_size, **(config() if callable(config) else {})},
        "model": getattr(predictor, "model", None),
        "model_revision": getattr(predictor, "model_revision", None),
        "docker_digest": getattr(predictor, "docker_digest", None),
        "helpers": list(getattr(predictor, "helpers", ())),
        "holdout": {
            "images": len(holdout),
            "products": int(holdout[PRODUCT_COLUMN].nunique()) if PRODUCT_COLUMN in holdout.columns else None,
            "sha256": fingerprint(holdout[ID_COLUMN].tolist()),
            "training_images": len(training_rows),
        },
        "score": _number(score),
        "macro_f1": _per_dimension(evaluation, dimensions, "f1_macro"),
        "weighted_f1": _per_dimension(evaluation, dimensions, "f1_weighted"),
        "overall_weighted_f1": _number((evaluation.get("overall") or {}).get("f1_weighted")),
        "images_per_second": images_per_second,
        "hardware": hardware(),
        "usd_per_hour": usd_per_hour,
        "cost_per_100k_usd": cost,
        "note": note,
    }
    write_predictions(data.out_dir() / "predictions" / exp_id, predictions)
    append_row(row, results_path)
    return row


def _shown(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def format_row(row: Mapping[str, Any]) -> str:
    """A row's scores, laid out as the reference notebook prints its holdout scores."""
    commit = (row.get("commit") or "unknown")[:7] + (" (dirty)" if row.get("dirty") else "")
    speed = row.get("images_per_second")
    lines = [
        f"{row['id']}  {row['predictor']}  score {_shown(row['score'])}  commit {commit}",
        f"holdout {row['holdout']['images']} images  {f'{speed:,.0f} images/s' if speed else ''}",
        "",
        f"{'':22}{'F1 macro':>9}{'F1 weighted':>13}",
    ]
    for dimension, macro in row["macro_f1"].items():
        lines.append(f"{dimension:22}{_shown(macro):>9}{_shown(row['weighted_f1'].get(dimension)):>13}")
    lines.append(f"{'overall':22}{_shown(row['score']):>9}{_shown(row.get('overall_weighted_f1')):>13}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m galaxus.model.eval",
        description="Score a predictor on the official holdout and append its EXP row to results/metrics.json.",
    )
    parser.add_argument("predictor", choices=sorted(PREDICTORS))
    parser.add_argument("--exp-id", help="the id to use, for example EXP-0001 (default: the next free one)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--usd-per-hour", type=float, help="price of the machine, for the cost per 100,000 images")
    parser.add_argument("--note", default="", help="one line on what this run tests")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE, help="default: results/metrics.json")
    parser.add_argument("--allow-dirty", action="store_true", help="run on uncommitted changes; the row says dirty")
    args = parser.parse_args(argv)
    try:
        row = run_eval(
            args.predictor,
            results_path=args.results,
            exp_id=args.exp_id,
            seed=args.seed,
            batch_size=args.batch_size,
            usd_per_hour=args.usd_per_hour,
            note=args.note,
            allow_dirty=args.allow_dirty,
        )
    except (RuntimeError, ResultsFileError, InvalidPredictions, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(format_row(row))
    print(f"\nappended {row['id']} to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
