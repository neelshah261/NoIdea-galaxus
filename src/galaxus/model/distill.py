"""Teacher passes for distillation (#132 D1, #172): a strong model's answers become the student's training labels.

The FAQ allows any model in development "to create training targets or pseudo-labels on the dev images"
(FAQ.md:30-31). At runtime only the student, a fine-tuned Apertus, classifies.

* **The pass:** any predictor the eval script registers is the teacher. The model behind it comes from ``LLM_NAME``,
  ``LLM_BASE_URL`` and ``LLM_API_KEY``: the 70B on the gateway, or any OpenAI-compatible endpoint (#172 says how to
  run a non-gateway teacher safely). It answers **train minus the official holdout**; the holdout is checked out
  before the first call. Answers and confidences go to ``$GALAXUS_DATA/out/distill/<name>/train.jsonl`` in the format
  the LoRA's ``--teacher`` reads (``ensemble.load_records``). A rerun skips the images already answered, and the log
  holds counts and images per second only: no labels, no ids.
* **The merge:** one file that takes each dimension from the teacher named for it (``--take``), else from
  ``--default``. A source is a teacher name (``distill/<name>/<part>.jsonl``) or an EXP id whose answers the eval
  script or the runner saved (``predictions/<EXP>/<part>.jsonl``, e.g. a teacher's dev pass for ``--pseudo-dev``).
  A dimension with no source is left out of the record, so it stays out of the loss.

    LLM_NAME=apertus-v1-5-70b uv run --env-file .env python -m galaxus.model.distill run --name b70 \\
        --predictor zero-shot-observe-first --workers 4
    uv run python -m galaxus.model.distill merge --out mixed --take individuals=b70 --default gpt

Pick each dimension's teacher from the teachers' holdout rows (``python -m galaxus.model.compare``): a teacher's
labels are worth training on only where it beats the student there.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from galaxus.model.ensemble import ID_COLUMN, Records, load_records
from galaxus.types import ImageItem, Predictor

DISTILL_DIR = "distill"
TRAIN_PART = "train"


def teacher_file(out_dir: Path, name: str, part: str = TRAIN_PART) -> Path:
    """Where a teacher's answers for ``part`` live: ``<out>/distill/<name>/<part>.jsonl``."""
    return out_dir / DISTILL_DIR / name / f"{part}.jsonl"


def source_file(out_dir: Path, source: str, part: str = TRAIN_PART) -> Path:
    """A merge source: an EXP id's saved answers (``predictions/<EXP>/<part>.jsonl``), else a teacher's file."""
    if source.startswith("EXP-"):
        return out_dir / "predictions" / source / f"{part}.jsonl"
    return teacher_file(out_dir, source, part)


def teacher_items(data: Any) -> list[ImageItem]:
    """Train minus the official holdout, as the eval script builds items. Raises if a holdout image slipped in."""
    from galaxus.model.eval import items_from_rows

    rows, holdout = data.split_holdout(data.read_train())
    items = items_from_rows(rows)
    leaked = {str(item.image_id) for item in items} & set(holdout[ID_COLUMN].astype(str))
    if leaked:
        raise ValueError(f"{len(leaked)} holdout images among the teacher's items: the holdout is never a target")
    return items


def _record(prediction: Any) -> dict[str, Any]:
    return {
        ID_COLUMN: str(prediction.image_id),
        "labels": dict(prediction.labels),
        "confidence": dict(prediction.confidence or {}),
    }


def run_pass(
    predictor: Predictor,
    items: Sequence[ImageItem],
    target: Path,
    batch_size: int = 16,
    workers: int = 1,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Answer every item not yet in ``target`` and append the answers to it, batch by batch. Returns the counts."""
    done = set(load_records(target)) if target.is_file() else set()
    todo = [item for item in items if str(item.image_id) not in done]
    batches = [todo[start : start + batch_size] for start in range(0, len(todo), batch_size)]
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    answered = 0
    started = clock()

    def answer(batch: Sequence[ImageItem]) -> None:
        nonlocal answered
        records = [_record(p) for p in predictor.predict(batch)]
        with lock, target.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            answered += len(records)
            if answered % (batch_size * 10) < len(records) or answered == len(todo):
                seconds = clock() - started
                rate = answered / seconds if seconds > 0 else float("nan")
                log(f"{answered}/{len(todo)} answered this run, {len(done)} before | {rate:.2f} images/s")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(answer, batches))
    else:
        for batch in batches:
            answer(batch)
    seconds = clock() - started
    return {
        "answered": answered,
        "skipped": len(done & {str(item.image_id) for item in items}),
        "images": len(items),
        "images_per_second": answered / seconds if answered and seconds > 0 else None,
    }


def merge_teachers(
    teachers: Mapping[str, Records],
    take: Mapping[str, str],
    dimensions: Iterable[str],
    default: str | None = None,
) -> Records:
    """One record per image any chosen teacher answered: each dimension from its named teacher, else ``default``."""
    dimensions = tuple(dimensions)
    unknown = sorted(set(take) - set(dimensions))
    if unknown:
        raise ValueError(f"--take names dimensions that are not in the class contract: {unknown}")
    chosen = {d: take.get(d, default) for d in dimensions}
    missing = sorted({name for name in chosen.values() if name is not None} - set(teachers))
    if missing:
        raise KeyError(f"no answers loaded for teacher(s) {missing}")
    used = {name for name in chosen.values() if name is not None}
    merged: Records = {}
    for image_id in sorted(set().union(*(teachers[name] for name in used))) if used else []:
        labels: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for d, name in chosen.items():
            answer = teachers[name].get(image_id) if name else None
            if not answer or d not in (answer.get("labels") or {}):
                continue
            labels[d] = answer["labels"][d]
            value = (answer.get("confidence") or {}).get(d)
            if value is not None:
                confidence[d] = float(value)
        if labels:
            merged[image_id] = {ID_COLUMN: image_id, "labels": labels, "confidence": confidence}
    return merged


def write_records(records: Mapping[str, Mapping[str, Any]], target: Path) -> Path:
    """``records`` as jsonl, one image per line, sorted by id (replaces ``target``)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for image_id in sorted(records):
            handle.write(json.dumps(records[image_id], ensure_ascii=False) + "\n")
    return target


def pseudo_examples(
    pseudo: Mapping[str, Mapping[str, Any]],
    dev_rows: Any,
    dimensions: Sequence[str],
    min_confidence: float | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """#172 D6: dev images as training examples, labelled by a teacher (FAQ.md:30-31 allows dev pseudo-labels).

    ``pseudo`` is a teacher's dev answers (a ``galaxus.serve.submit --run`` pass, or ``merge --part dev``). The answer
    is the whole target JSON, so a dev image needs a label for every dimension; only its cells at ``min_confidence``
    or above enter the loss, the others are masked as prelabels are. An image with no confident cell is left out.
    Only rows of ``dev_rows`` (dev.csv) become examples, so no train, holdout or final image can join this way.
    Returns the examples and the number of cells per dimension that entered the loss.
    """
    from galaxus.model.lora import TEACHER_MIN_CONFIDENCE, Example, _context

    floor = TEACHER_MIN_CONFIDENCE if min_confidence is None else min_confidence
    examples, cells = [], dict.fromkeys(dimensions, 0)
    for record in dev_rows.to_dict(orient="records"):
        answer = pseudo.get(str(record[ID_COLUMN])) or {}
        labels, confidence = answer.get("labels") or {}, answer.get("confidence") or {}
        if any(d not in labels for d in dimensions):
            continue
        trained = tuple(d for d in dimensions if confidence.get(d) is not None and float(confidence[d]) >= floor)
        if not trained:
            continue
        examples.append(Example(str(record[ID_COLUMN]), str(record.get("image_file", "")), False, _context(record),
                                {d: str(labels[d]) for d in dimensions}, trained))
        for d in trained:
            cells[d] += 1
    return examples, cells


def with_pseudo(data: Any, examples: Sequence[Any], cells: Mapping[str, int]) -> Any:
    """The training set plus the pseudo-labelled dev examples, with their counts in the stats."""
    from galaxus.model.lora import TrainingSet

    stats = {**data.stats, "pseudo_added": len(examples), "pseudo_cells": dict(cells),
             "examples_per_epoch": len(data.examples) + len(examples)}
    return TrainingSet(list(data.examples) + list(examples), stats)


def _pairs(values: Sequence[str]) -> dict[str, str]:
    pairs = {}
    for value in values:
        dimension, _, source = value.partition("=")
        if not dimension or not source:
            raise ValueError(f"--take looks like dimension=teacher, got {value!r}")
        pairs[dimension] = source
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.distill", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="a teacher's answers on train minus the official holdout (resumable)")
    run_cmd.add_argument("--name", required=True, help="the teacher's name, e.g. b70 or gpt: its folder under distill/")
    run_cmd.add_argument("--predictor", required=True, help="a predictor the eval script registers, e.g. zero-shot-observe-first")
    run_cmd.add_argument("--batch-size", type=int, default=16)
    run_cmd.add_argument("--workers", type=int, default=1, help="batches in flight (thread-safe predictors only)")
    run_cmd.add_argument("--seed", type=int, default=0)
    merge_cmd = sub.add_parser("merge", help="one teacher file, each dimension from the teacher named for it")
    merge_cmd.add_argument("--out", required=True, help="name of the merged teacher (its folder under distill/)")
    merge_cmd.add_argument("--take", nargs="*", default=[], help="dimension=source, a teacher name or an EXP id")
    merge_cmd.add_argument("--default", default=None, help="the source for every dimension --take does not name")
    merge_cmd.add_argument("--part", default=TRAIN_PART, help="train (teacher passes) or dev (pseudo-labels)")
    arguments = parser.parse_args(argv)

    from galaxus.model.eval import PREDICTORS, DataLane

    data = DataLane.real()
    out = data.out_dir()
    if arguments.command == "run":
        if arguments.predictor not in PREDICTORS:
            parser.error(f"unknown predictor {arguments.predictor!r}; the eval script registers: {sorted(PREDICTORS)}")
        rows = data.split_holdout(data.read_train())[0]
        predictor = PREDICTORS[arguments.predictor](rows, data, arguments.seed)
        target = teacher_file(out, arguments.name)
        stats = run_pass(predictor, teacher_items(data), target, arguments.batch_size, arguments.workers)
        print(f"{stats['answered']} answered, {stats['skipped']} already there, of {stats['images']} -> {target}")
        return 0

    take = _pairs(arguments.take)
    sources = sorted(set(take.values()) | ({arguments.default} if arguments.default else set()))
    teachers = {name: load_records(source_file(out, name, arguments.part)) for name in sources}
    merged = merge_teachers(teachers, take, data.dimensions(), arguments.default)
    target = write_records(merged, teacher_file(out, arguments.out, arguments.part))
    print(f"{len(merged)} images -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
