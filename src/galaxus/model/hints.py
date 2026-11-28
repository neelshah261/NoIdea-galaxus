"""Retrieval hints in Apertus' zero-shot prompt: Part B of ``galaxus.data.neighbours`` (Q-0003, decisions.md, #237).

For each image, its ``k`` most similar **training** images (SigLIP embeddings, the data lane's index) and their human
labels, written as one line of text. Apertus decides: the line goes into the user message of the ``zero-shot-hints``
variant, framed as a hint and not the answer, and no rule turns a neighbour label into the label.

* **The index holds training images only.** Holdout queries search train minus the official holdout; dev and final
  queries search all of train. Never a dev or final image, never a pseudo-label. A query never gets itself, and a
  training query never gets its own product (``neighbours.nearest``).
* **Every image gets a hint:** neighbours are taken with no similarity floor, so each image has its ``k``.
* **The labels** are the neighbours' scored human labels (``split.scored_mask``). Names come from the data, never typed.

    python -m galaxus.model.hints measure                  # the hint alone on the holdout, through metric.py (dev only)
    python -m galaxus.model.hints write --part holdout     # then dev, then final once the final images are out
    ZERO_SHOT_PRESET=zero-shot-hints ZERO_SHOT_HINTS=$GALAXUS_DATA/out/hints/holdout.jsonl \\
        python -m galaxus.model.eval zero-shot-hints --exp-id EXP-0099
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_K = 5
#: No similarity floor: every image gets its k neighbours, so every image has a hint.
NO_FLOOR = -1.0
PREFIX = "Similar catalog images were labelled:"
PARTS = ("holdout", "dev", "final")
ScoredMask = Callable[[pd.DataFrame, str], pd.Series]


def hint_line(neighbour_labels: Sequence[Mapping[str, str]], dimensions: Sequence[str]) -> str:
    """``Similar catalog images were labelled: <dimension>: <label> (n of k); ...``: per dimension, the neighbours'
    most common label, ``n`` of the ``k`` neighbours that carry a label there. A dimension none of them carries is
    left out."""
    parts = []
    for d in dimensions:
        labels = [labels_of[d] for labels_of in neighbour_labels if d in labels_of]
        if labels:
            label, n = Counter(labels).most_common(1)[0]
            parts.append(f"{d}: {label} ({n} of {len(labels)})")
    return f"{PREFIX} " + "; ".join(parts) if parts else f"{PREFIX} no labelled neighbours"


def scored_labels(rows: pd.DataFrame, dimensions: Sequence[str], scored_mask: ScoredMask,
                  id_column: str = "image_id") -> dict[str, dict[str, str]]:
    """``{image_id: {dimension: label}}`` for the scored human cells of ``rows``."""
    ids = rows[id_column].astype(str).tolist()
    labels: dict[str, dict[str, str]] = {i: {} for i in ids}
    for d in dimensions:
        for i, value, ok in zip(ids, rows[d].astype(str).tolist(), scored_mask(rows, d).to_numpy(), strict=True):
            if ok:
                labels[i][d] = value
    return labels


def query_index(rows: pd.DataFrame, directory: Path, id_column: str = "image_id",
                product_column: str = "product_id") -> Any:
    """The queries as a ``neighbours.Index`` from the embedding cache. A query without a product column (dev, final)
    gets an empty product, which no training image has."""
    from galaxus.data import neighbours

    ids = rows[id_column].astype(str).tolist()
    products = rows[product_column].astype(str).tolist() if product_column in rows.columns else [""] * len(ids)
    missing = [i for i in ids if not (directory / f"{i}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} query image(s) have no embedding in {directory}, e.g. {missing[:3]}: "
                                "embed them first (`python -m galaxus.data.neighbours embed`, or `write` embeds them)")
    vectors = np.vstack([np.load(directory / f"{i}.npy").astype(np.float32).reshape(1, -1) for i in ids])
    return neighbours.Index(tuple(ids), tuple(products), neighbours.unit_rows(vectors))


def neighbours_of(queries: Any, index: Any, k: int = DEFAULT_K) -> dict[str, list[str]]:
    """``{query id: [neighbour ids, closest first]}``, ``k`` each (no similarity floor)."""
    from galaxus.data import neighbours

    found = neighbours.nearest(queries, index, k=k, min_similarity=NO_FLOOR)
    return {q: [n for n, _ in found.get(q, [])] for q in queries.image_ids}


def hints_for(queries: Any, index: Any, labels: Mapping[str, Mapping[str, str]], dimensions: Sequence[str],
              k: int = DEFAULT_K) -> dict[str, str]:
    """``{image_id: hint line}`` for every query."""
    found = neighbours_of(queries, index, k)
    return {q: hint_line([labels.get(n, {}) for n in found[q]], dimensions) for q in queries.image_ids}


def majority_answers(queries: Any, index: Any, labels: Mapping[str, Mapping[str, str]], dimensions: Sequence[str],
                     fallback: Mapping[str, str], k: int = DEFAULT_K) -> dict[str, dict[str, str]]:
    """The hint alone as an answer: per dimension the neighbours' most common label, ``fallback`` where none."""
    found = neighbours_of(queries, index, k)
    out = {}
    for q in queries.image_ids:
        answer = {}
        for d in dimensions:
            got = [labels.get(n, {}).get(d) for n in found[q]]
            got = [g for g in got if g is not None]
            answer[d] = Counter(got).most_common(1)[0][0] if got else fallback[d]
        out[q] = answer
    return out


def write_jsonl(hints: Mapping[str, str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for image_id, hint in hints.items():
            handle.write(json.dumps({"image_id": image_id, "hint": hint}, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> dict[str, str]:
    """``{image_id: hint}`` from a hints file."""
    hints = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            hints[str(record["image_id"])] = str(record["hint"])
    return hints


def _part_rows(part: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(queries, index rows)``: holdout searches train minus the holdout; dev and final search all of train."""
    from galaxus.data import load, split

    train = load.read_train()
    if part == "holdout":
        rest, holdout = split.split_holdout(train)
        return holdout, rest
    if part == "dev":
        return load.read_dev(), train
    if part == "final":
        return load.read_final(), train
    raise ValueError(f"part must be one of {PARTS}, got {part!r}")


def _embed_missing(rows: pd.DataFrame, encoder_name: str) -> dict[str, object]:
    from galaxus.data import neighbours

    folder = neighbours.embedding_dir(encoder_name)
    ids = rows["image_id"].astype(str)
    todo = rows[[not (folder / f"{i}.npy").is_file() for i in ids]]
    if not len(todo):
        return {"written": 0}
    return neighbours.embed_rows(todo, neighbours.load_encoder(encoder_name), encoder_name)


def main(argv: Iterable[str] | None = None) -> int:
    from galaxus.data import classes, metric, neighbours, split

    parser = argparse.ArgumentParser(prog="python -m galaxus.model.hints", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="the hint alone on the holdout, scored through metric.py (development)")
    write = sub.add_parser("write", help="write $GALAXUS_DATA/out/hints/<part>.jsonl")
    write.add_argument("--part", required=True, choices=PARTS)
    for command in (measure, write):
        command.add_argument("--k", type=int, default=DEFAULT_K)
        command.add_argument("--encoder", default=neighbours.DEFAULT_ENCODER)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    dims = tuple(classes.dimensions())
    part = "holdout" if arguments.command == "measure" else arguments.part
    queries_rows, index_rows = _part_rows(part)
    print(f"[hints] {part}: {len(queries_rows)} queries, {len(index_rows)} training images in the index", flush=True)
    print(f"[hints] embeddings: {_embed_missing(pd.concat([queries_rows, index_rows]), arguments.encoder)}", flush=True)
    folder = neighbours.embedding_dir(arguments.encoder)
    index = neighbours.build_index(index_rows, arguments.encoder)
    queries = query_index(queries_rows, folder)
    labels = scored_labels(index_rows, dims, split.scored_mask)

    if arguments.command == "measure":
        from galaxus.model.baselines import MajorityPredictor

        fallback = MajorityPredictor.fit(index_rows, dims, split.scored_mask).answer
        answers = majority_answers(queries, index, labels, dims, fallback, arguments.k)
        submission = pd.DataFrame([{"image_id": i, **answers[i]} for i in queries.image_ids])
        solution = split.solution_frame(queries_rows, dims)
        evaluation = metric.evaluate(solution, submission)
        print(f"[hints] the hint alone (k={arguments.k}, majority of the neighbours), holdout macro F1: "
              f"{metric.score(solution, submission):.4f}")
        for d in dims:
            print(f"  {d:22} {float((evaluation.get(d) or {}).get('f1_macro') or 0):.3f}")
        print("[hints] a development measurement, not a method: Apertus decides (Q-0003).")
        return 0

    hints = hints_for(queries, index, labels, dims, arguments.k)
    from galaxus.data.paths import out_dir

    path = write_jsonl(hints, out_dir() / "hints" / f"{part}.jsonl")
    print(f"[hints] {len(hints)} hints -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
