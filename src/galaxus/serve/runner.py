"""B4: one runner for the holdout, the dev images and the final images (BASE_ARCHITECTURE.md, section 4).

It batches ``ImageItem``s through any ``Predictor`` and appends one JSON line per image to
``$GALAXUS_DATA/out/predictions/<exp>/<part>.jsonl``:

    {"image_id": "...", "labels": {"<dimension>": "<class>", ...}, "confidence": {...} | null}

- **Resume.** Ids already in the file are skipped, so a run killed halfway finishes on the next start without
  duplicates. A line cut off by the crash is removed before the run continues.
- **Safe writes.** A batch is checked against the contract before anything is written, then written and flushed
  (``fsync``) as a whole, so the file only ever holds complete batches plus at most one cut-off line.
- **Parallel batches.** ``workers=N`` keeps up to N batches in flight on a thread pool (#36). Over an HTTP backend
  most of the time is network wait, so this is what reaches the team's 5-10 images per second. Use it only with a
  thread-safe predictor; the default (1) is sequential, as a local GPU model needs.
- **An unreadable image keeps the constant answer (#216).** With a ``fallback`` (the entry points pass
  :func:`constant_fallback`), a batch that fails with an image error (``OSError``, which covers PIL's unidentified
  and truncated files, or ``ValueError``) is retried one image at a time; an image that still fails gets its row of
  ``sample_submission.csv``, marked ``"fallback": "constant"``, and is counted, never named (product rule 3). Any other
  error still stops the run, and so do more than ``max_fallbacks`` such images: that many is a broken method, not a
  broken file. **A restart retries every constant row** (they are taken out of the file first), so a run stopped by
  a systemic error predicts those images once the method is fixed, and a truly broken file just falls back again. This is what the hosts' reference notebook does ("cannot be read, it keeps the constant answer").
- **New batches (#216).** ``--csv <table>`` runs any table shaped like dev.csv (``image_id``, ``image_path``
  relative to the table, optional catalog columns) as part ``csv-<file stem>``.
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
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREDICTIONS_DIR = "predictions"
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 1
PARTS = ("holdout", "dev", "final")
#: The errors of one image file (PIL raises ``UnidentifiedImageError``, an ``OSError``, or ``OSError`` for a
#: truncated file; a bad size or mode surfaces as ``ValueError``). Anything else is the method's own failure.
IMAGE_ERRORS = (OSError, ValueError)
#: ...but never these: a network or timeout failure (``urllib``'s errors and ``requests``' are ``OSError`` too) or a
#: reply that is not JSON is the method or the gateway failing, so it stops the run like any other error.
NOT_IMAGE_ERRORS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, json.JSONDecodeError)
FALLBACK_MARK = "constant"
#: More images than this falling back in one run stops it: a systemic error, not a few broken files.
DEFAULT_MAX_FALLBACKS = 50
CSV_PREFIX = "csv-"
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
    workers: int = DEFAULT_WORKERS
    #: Images that could not be read and keep the constant answer (``"fallback": "constant"`` in the file).
    fallbacks: int = 0

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


def is_image_error(error: BaseException) -> bool:
    """True for the error of one unreadable image file (see :data:`IMAGE_ERRORS`), never for a network failure."""
    import urllib.error

    excluded = (*NOT_IMAGE_ERRORS, urllib.error.URLError)
    try:
        import requests

        excluded = (*excluded, requests.RequestException)
    except ImportError:  # pragma: no cover - requests is optional
        pass
    return isinstance(error, IMAGE_ERRORS) and not isinstance(error, excluded)


class TooManyFallbacks(RuntimeError):
    """More images failed with an image error than ``max_fallbacks``: the run stops, as for any other error."""


def constant_fallback(sample: Any = None) -> Callable[[str], dict[str, str]]:
    """``image_id -> labels``: the image's row of ``sample_submission.csv``, or for an image not in it (a ``--csv``
    batch) the sample's most common row, which is the constant answer of every row."""
    if sample is None:
        from galaxus.data.load import read_sample_submission

        sample = read_sample_submission()
    columns = [c for c in sample.columns if c not in ("image_id", "image_file", "image_path")]
    rows = {str(r["image_id"]): {c: str(r[c]) for c in columns} for r in sample.to_dict(orient="records")}
    if not rows:
        raise ValueError("the sample submission has no rows: no constant answer to fall back to")
    most_common = sample[columns].astype(str).value_counts().idxmax()
    most_common = most_common if isinstance(most_common, tuple) else (most_common,)
    constant = dict(zip(columns, most_common, strict=True))

    def labels(image_id: str) -> dict[str, str]:
        return dict(rows.get(str(image_id), constant))

    return labels


def default_fallback(log: Callable[[str], None] | None = print) -> Callable[[str], dict[str, str]] | None:
    """:func:`constant_fallback` from ``$GALAXUS_DATA``'s sample_submission.csv, or ``None`` (every error stops the
    run, as before #216) when that file cannot be read; the entry points use this so a start never fails on it."""
    try:
        return constant_fallback()
    except (OSError, KeyError, ValueError, RuntimeError) as problem:
        if log:
            log(f"[runner] no constant answer to fall back to ({type(problem).__name__}): an unreadable image stops "
                "the run")
        return None


def fallback_record(image_id: str, labels: Mapping[str, str]) -> dict[str, Any]:
    """The predictions-file line of an image that keeps the constant answer."""
    return {"image_id": str(image_id), "labels": {str(k): str(v) for k, v in labels.items()}, "confidence": None,
            "fallback": FALLBACK_MARK}


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
    workers: int = DEFAULT_WORKERS,
    fallback: Callable[[str], Mapping[str, str]] | None = None,
    max_fallbacks: int = DEFAULT_MAX_FALLBACKS,
) -> RunReport:
    """Predict every item that has no line in the predictions file yet, and return what happened.

    With ``fallback``, an image that fails with an image error keeps ``fallback(image_id)`` (see the module
    docstring); without it, every error stops the run, as before #216.

    With ``workers`` above 1, up to that many batches run at the same time; the lines of the file then follow the
    order in which batches finish, which nothing depends on (records are read back by ``image_id``).

    Raises:
        ValueError: duplicate image ids among ``items``, or a bad ``exp``/``part``, ``batch_size`` or ``workers``.
        PredictorContractError: the predictor broke the contract; the lines of batches already finished stay valid.
        TooManyFallbacks: more than ``max_fallbacks`` images failed with an image error.
        Exception: whatever the predictor raised. With several workers, the batches still in flight are finished and
            written first, so a restart only redoes the batches that never ran.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    ids = [str(item.image_id) for item in items]
    if len(set(ids)) != len(ids):
        repeated = sorted({image_id for image_id in ids if ids.count(image_id) > 1})
        raise ValueError(f"{len(repeated)} image id(s) appear more than once among the items, first: {repeated[:3]}")

    path = predictions_path(exp, part, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_predictions(path)
    retry = sum(1 for record in done.values() if record.get("fallback"))
    if retry:
        # A constant row is not a prediction: take those lines out and predict their images again (#216 review).
        done = {image_id: record for image_id, record in done.items() if not record.get("fallback")}
        _rewrite(path, done.values())
        if log:
            log(f"[runner] {exp}/{part}: retrying {retry} image(s) that kept the constant answer last time")
    todo = [item for item in items if str(item.image_id) not in done]
    already = len(items) - len(todo)
    if log and already:
        log(f"[runner] {exp}/{part}: {already} of {len(items)} already predicted, resuming with {len(todo)}")

    predicted = 0
    started = time.perf_counter()
    lock = threading.Lock()
    with path.open("a", encoding="utf-8", newline="\n") as handle:

        def write(records: list[dict[str, Any]]) -> None:
            nonlocal predicted
            with lock:
                handle.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
                handle.flush()
                os.fsync(handle.fileno())
                predicted += len(records)
                if log:
                    elapsed = time.perf_counter() - started
                    log(f"[runner] {exp}/{part}: {already + predicted}/{len(items)} ({predicted / elapsed:.2f} img/s)")

        fell_back = 0

        def one_by_one(batch: Sequence[Any], first: BaseException) -> list[dict[str, Any]]:
            nonlocal fell_back
            records = []
            for item in batch:
                try:
                    records += _check_batch([item], list(predictor.predict([item])))
                except IMAGE_ERRORS as error:
                    if not is_image_error(error):
                        raise
                    with lock:
                        fell_back += 1
                        count = fell_back
                    if count > max_fallbacks:
                        raise TooManyFallbacks(
                            f"{count} images failed with an image error ({type(error).__name__}), more than "
                            f"max_fallbacks={max_fallbacks}: this looks like the method failing, not a few broken "
                            "files, so the run stops") from first
                    records.append(fallback_record(str(item.image_id), fallback(str(item.image_id))))
                    if log:
                        log(f"[runner] {exp}/{part}: {count} image(s) so far could not be read "
                            f"({type(error).__name__}) and keep the constant answer")
            return records

        def predict(batch: Sequence[Any]) -> list[dict[str, Any]]:
            try:
                return _check_batch(batch, list(predictor.predict(list(batch))))
            except IMAGE_ERRORS as error:
                if fallback is None or not is_image_error(error):
                    raise
                return one_by_one(batch, error)

        if workers == 1:
            for batch in _batches(todo, batch_size):
                write(predict(batch))
        else:
            _run_parallel(predict, write, list(_batches(todo, batch_size)), workers)
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
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        workers=workers,
        fallbacks=fell_back,
    )
    if log and fell_back:
        log(f"[runner] {exp}/{part}: {fell_back} image(s) could not be read and keep the constant answer")
    with runs_path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(report)) + "\n")
    return report


def _rewrite(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """``path`` holding exactly ``records``, replaced in one step (a crash leaves the old file or the new one)."""
    partial = path.with_name(path.name + ".part")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _run_parallel(
    predict: Callable[[Sequence[Any]], list[dict[str, Any]]],
    write: Callable[[list[dict[str, Any]]], None],
    batches: list[Sequence[Any]],
    workers: int,
) -> None:
    """Keep up to ``workers`` batches in flight and write each as it finishes.

    On a failure no new batch starts; the ones in flight are finished and written, then the first error is raised.
    """
    pending: set[Future[list[dict[str, Any]]]] = set()
    queue = iter(batches)
    failure: BaseException | None = None

    def fill(pool: ThreadPoolExecutor) -> None:
        while len(pending) < workers:
            batch = next(queue, None)
            if batch is None:
                return
            pending.add(pool.submit(predict, batch))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="runner") as pool:
        fill(pool)
        while pending:
            done, still_running = wait(pending, return_when=FIRST_COMPLETED)
            pending.clear()
            pending.update(still_running)
            for future in done:
                try:
                    write(future.result())
                except BaseException as error:  # noqa: BLE001 - re-raised below, after the other batches are saved
                    failure = failure or error
            if failure is None:
                fill(pool)
    if failure is not None:
        raise failure


def csv_part_name(table: str | os.PathLike[str]) -> str:
    """``csv-<file stem>``, the part a ``--csv`` table runs as (the model lane's ``heads extract --csv`` matches)."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", Path(table).stem).strip("-.") or "table"
    return _check_name("part", CSV_PREFIX + stem)


def items_for_csv(table: str | os.PathLike[str], limit: int | None = None) -> list[Any]:
    """The items of any table shaped like dev.csv, read with the data lane's ``read_part`` (image_path resolved
    relative to the table)."""
    from galaxus.data.load import read_part
    from galaxus.serve.items import items_from_frame

    path = Path(table)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such table")
    frame = read_part(path)
    if "image_file" not in frame.columns:
        raise KeyError(f"{path.name} needs image_id and image_path columns, as dev.csv has")
    return items_from_frame(frame, limit)


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--part", choices=PARTS)
    source.add_argument("--csv", help="any table shaped like dev.csv; runs as part csv-<file stem>")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (smoke runs)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="batches in flight (thread-safe only)")
    parser.add_argument("--max-fallbacks", type=int, default=DEFAULT_MAX_FALLBACKS,
                        help="unreadable images that may keep the constant answer before the run stops")
    parser.add_argument("--no-fallback", action="store_true", help="stop on an unreadable image instead")
    arguments = parser.parse_args(argv)

    if arguments.csv:
        part, items = csv_part_name(arguments.csv), items_for_csv(arguments.csv, arguments.limit)
    else:
        part, items = arguments.part, items_for_part(arguments.part, arguments.limit)
    predictor = load_predictor(arguments.predictor)
    report = run(predictor, items, arguments.exp, part, arguments.batch_size, workers=arguments.workers,
                 fallback=None if arguments.no_fallback else default_fallback(),
                 max_fallbacks=arguments.max_fallbacks)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    sys.exit(main())
