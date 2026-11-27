"""B4: one runner for the holdout, the dev images and the final images (BASE_ARCHITECTURE.md, section 4).

It batches ``ImageItem``s through any ``Predictor`` and appends one JSON line per image to
``$GALAXUS_DATA/out/predictions/<exp>/<part>.jsonl``:

    {"image_id": "...", "labels": {"<dimension>": "<class>", ...}, "confidence": {...} | null}

- **Resume.** Ids already in the file are skipped, so a run killed halfway finishes on the next start without
  duplicates. A line cut off by the crash is removed before the run continues.
- **Safe writes.** A batch is checked against the contract before anything is written, then written and flushed
  (``fsync``) as a whole, so the file only ever holds complete batches plus at most one cut-off line.
- **Timing.** Every run appends one record (images, seconds, images per second, batch size) to
  ``<part>.runs.jsonl`` next to the predictions. The benchmark (:mod:`galaxus.serve.bench`) builds on the same numbers.

As a command line (``$GALAXUS_DATA`` set, data lane and the predictor's lane installed):

    python -m galaxus.serve.runner --predictor galaxus.model.baselines:MajorityPredictor --exp EXP-0001 --part dev
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREDICTIONS_DIR = "predictions"
DEFAULT_BATCH_SIZE = 16
PARTS = ("holdout", "dev", "final")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PredictorContractError(RuntimeError):
    """A predictor returned something the contract does not allow. Nothing of that batch was written."""


@dataclass(frozen=True)
class RunReport:
    """What one call of :func:`run` did."""

    exp: str
    part: str
    path: str
    requested: int
    already_done: int
    predicted: int
    seconds: float
    images_per_second: float | None
    batch_size: int
    finished_at: str

    @property
    def complete(self) -> bool:
        """True when every requested image has a prediction in the file."""
        return self.already_done + self.predicted == self.requested


def _check_name(kind: str, value: str) -> str:
    if not _SAFE_NAME.match(value):
        raise ValueError(f"{kind} {value!r} must be letters, digits, '.', '_' or '-', starting with a letter or digit")
    return value


def predictions_path(exp: str, part: str, root: str | os.PathLike[str] | None = None) -> Path:
    """``<root>/predictions/<exp>/<part>.jsonl``; ``root`` defaults to ``$GALAXUS_DATA/out``."""
    if root is None:
        from galaxus.data.paths import out_dir

        root = out_dir()
    return Path(root) / PREDICTIONS_DIR / _check_name("exp", exp) / f"{_check_name('part', part)}.jsonl"


def runs_path(predictions: Path) -> Path:
    """Where the timing records of a predictions file go: ``<part>.runs.jsonl`` next to it."""
    return predictions.with_name(predictions.stem + ".runs.jsonl")


def read_predictions(path: str | os.PathLike[str], repair: bool = True) -> dict[str, dict[str, Any]]:
    """The records of a predictions file by ``image_id``; empty when the file does not exist.

    A last line without its newline is what a crash mid-write leaves behind. With ``repair`` it is cut off the
    file (its image is predicted again); without, it is ignored. A broken line anywhere else is an error.
    """
    file = Path(path)
    if not file.exists():
        return {}
    raw = file.read_bytes()
    complete_end = raw.rfind(b"\n") + 1
    if complete_end < len(raw) and repair:
        with file.open("r+b") as handle:
            handle.truncate(complete_end)
    records: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(raw[:complete_end].decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            image_id = str(record["image_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as broken:
            raise ValueError(f"{file}:{number} is not a prediction record: {broken}") from broken
        if image_id in records:
            raise ValueError(f"{file}:{number} repeats image_id {image_id!r}")
        records[image_id] = record
    return records


def read_labels(path: str | os.PathLike[str]) -> dict[str, dict[str, str]]:
    """``{image_id: {dimension: class}}`` from a predictions file, the shape the submission writer takes."""
    return {image_id: dict(record["labels"]) for image_id, record in read_predictions(path, repair=False).items()}


def _field(prediction: Any, name: str) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(name)
    return getattr(prediction, name, None)


def _as_record(prediction: Any) -> dict[str, Any]:
    """A ``Prediction`` (dataclass, object or mapping) as the JSON record we store."""
    image_id = _field(prediction, "image_id")
    labels = _field(prediction, "labels")
    confidence = _field(prediction, "confidence")
    if image_id is None or not isinstance(labels, Mapping) or not labels:
        raise PredictorContractError(f"a prediction needs an image_id and a non-empty labels mapping: {prediction!r}")
    empty = [dimension for dimension, value in labels.items() if value is None or not str(value).strip()]
    if empty:
        raise PredictorContractError(f"{image_id}: empty class in {empty}; a predictor never leaves a cell empty")
    record: dict[str, Any] = {
        "image_id": str(image_id),
        "labels": {str(dimension): str(value) for dimension, value in labels.items()},
        "confidence": None,
    }
    if confidence is not None:
        if not isinstance(confidence, Mapping):
            raise PredictorContractError(f"{image_id}: confidence must be a mapping or None")
        record["confidence"] = {str(dimension): float(value) for dimension, value in confidence.items()}
    return record


def _check_batch(batch: Sequence[Any], predictions: Sequence[Any]) -> list[dict[str, Any]]:
    expected = [str(item.image_id) for item in batch]
    records = [_as_record(prediction) for prediction in predictions]
    got = [record["image_id"] for record in records]
    if sorted(got) != sorted(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        repeated = sorted({image_id for image_id in got if got.count(image_id) > 1})
        raise PredictorContractError(
            f"the predictor must return exactly one prediction per item of the batch; "
            f"missing {missing[:3]}, unexpected {extra[:3]}, repeated {repeated[:3]}"
        )
    order = {image_id: index for index, image_id in enumerate(expected)}
    return sorted(records, key=lambda record: order[record["image_id"]])


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run(
    predictor: Any,
    items: Sequence[Any],
    exp: str,
    part: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    root: str | os.PathLike[str] | None = None,
    log: Callable[[str], None] | None = print,
) -> RunReport:
    """Predict every item that has no line in the predictions file yet, and return what happened.

    Raises:
        ValueError: duplicate image ids among ``items``, a bad ``exp``/``part`` name or a bad ``batch_size``.
        PredictorContractError: the predictor broke the contract; the lines of earlier batches stay valid.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    ids = [str(item.image_id) for item in items]
    if len(set(ids)) != len(ids):
        repeated = sorted({image_id for image_id in ids if ids.count(image_id) > 1})
        raise ValueError(f"{len(repeated)} image id(s) appear more than once among the items, first: {repeated[:3]}")

    path = predictions_path(exp, part, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_predictions(path)
    todo = [item for item in items if str(item.image_id) not in done]
    already = len(items) - len(todo)
    if log and already:
        log(f"[runner] {exp}/{part}: {already} of {len(items)} already predicted, resuming with {len(todo)}")

    predicted = 0
    started = time.perf_counter()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for batch in _batches(todo, batch_size):
            records = _check_batch(batch, list(predictor.predict(list(batch))))
            handle.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
            handle.flush()
            os.fsync(handle.fileno())
            predicted += len(records)
            if log:
                elapsed = time.perf_counter() - started
                log(f"[runner] {exp}/{part}: {already + predicted}/{len(items)} ({predicted / elapsed:.2f} img/s)")
    seconds = time.perf_counter() - started

    report = RunReport(
        exp=exp,
        part=part,
        path=str(path),
        requested=len(items),
        already_done=already,
        predicted=predicted,
        seconds=round(seconds, 3),
        images_per_second=round(predicted / seconds, 3) if predicted and seconds > 0 else None,
        batch_size=batch_size,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    with runs_path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(report)) + "\n")
    return report


def items_for_part(part: str, limit: int | None = None) -> list[Any]:
    """The items of ``holdout`` (the official split), ``dev`` or ``final``, read through the data lane."""
    from galaxus.data import load, split
    from galaxus.serve.items import items_from_frame

    if part == "holdout":
        frame = split.split_holdout(load.read_train())[1]
    elif part == "dev":
        frame = load.read_dev()
    elif part == "final":
        frame = load.read_final()
    else:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    return items_from_frame(frame, limit)


def main(argv: list[str] | None = None) -> int:
    """``python -m galaxus.serve.runner``: run a predictor over one part, resuming if it ran before."""
    from galaxus.serve.items import load_predictor

    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.runner", description=__doc__.split("\n")[0])
    parser.add_argument("--predictor", required=True, help="package.module:name, e.g. galaxus.model.baselines:...")
    parser.add_argument("--exp", required=True, help="experiment id, e.g. EXP-0001")
    parser.add_argument("--part", required=True, choices=PARTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (smoke runs)")
    arguments = parser.parse_args(argv)

    items = items_for_part(arguments.part, arguments.limit)
    report = run(load_predictor(arguments.predictor), items, arguments.exp, arguments.part, arguments.batch_size)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    sys.exit(main())
