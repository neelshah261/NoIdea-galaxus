"""Combine several methods' saved answers, per dimension, on CPU: each model runs once, the combining is free.

Every predictor (majority, zero-shot #47, heads #41, LoRA #49) already writes the same records: the eval script to
``$GALAXUS_DATA/out/predictions/<EXP>/holdout.jsonl``, the serve runner to ``…/<EXP>/<part>.jsonl``. This module:

1. ``choose``: scores every member on the official holdout, per dimension, and also a **confidence-weighted vote**
   (weight = the member's holdout macro F1 on that dimension x its confidence). For each dimension it keeps the
   best of {each member, vote}. The routing is saved as json.
2. ``apply``: applies a routing to the members' dev or final answers and writes a new predictions file, which
   ``galaxus.serve.submit`` turns into a validated file. No model runs again.

The holdout score of a routing is **optimistic**: the routing was chosen on that same holdout. With a handful of
options per dimension and 2,444 holdout images the bias is small, but it is reported as "chosen on the holdout".

    python -m galaxus.model.ensemble choose --members heads=EXP-0010 lora=EXP-0011 zeroshot=EXP-0002
    python -m galaxus.model.ensemble apply --routing <routing.json> --part dev --exp EXP-0020
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VOTE = "vote"
Records = dict[str, dict[str, Any]]


def macro_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    """Macro F1 over every class that is true or predicted, as metric.py counts it (no metric.py import, so this
    module stays independent of $GALAXUS_DATA for the per-dimension choice)."""
    classes = np.union1d(true, predicted)
    scores = []
    for value in classes:
        tp = np.sum((predicted == value) & (true == value))
        fp = np.sum((predicted == value) & (true != value))
        fn = np.sum((predicted != value) & (true == value))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def load_records(path: str | Path) -> Records:
    """``{image_id: record}`` from a predictions jsonl."""
    records: Records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["image_id"])] = record
    return records


def _confidence(record: Mapping[str, Any], dimension: str) -> float:
    confidence = record.get("confidence") or {}
    value = confidence.get(dimension)
    return 1.0 if value is None else float(value)


def vote(
    members: Mapping[str, Records], image_id: str, dimension: str, weights: Mapping[str, float]
) -> tuple[str, float]:
    """Confidence-weighted vote of the members on one cell; ties go to the class name that sorts first."""
    tally: dict[str, float] = defaultdict(float)
    for name, records in members.items():
        record = records[image_id]
        tally[record["labels"][dimension]] += weights.get(name, 1.0) * _confidence(record, dimension)
    best = min(tally, key=lambda label: (-tally[label], label))
    total = sum(tally.values()) or 1.0
    return best, tally[best] / total


def _common_ids(members: Mapping[str, Records]) -> list[str]:
    ids = None
    for records in members.values():
        ids = set(records) if ids is None else ids & set(records)
    missing = {name: len(set(records) - (ids or set())) for name, records in members.items()}
    if any(missing.values()):
        raise KeyError(f"members do not cover the same images: extra ids per member {missing}")
    return sorted(ids or [])


def choose(
    members: Mapping[str, Records],
    holdout: pd.DataFrame,
    dimensions: Sequence[str],
    scored_mask: Callable[[pd.DataFrame, str], pd.Series],
) -> dict[str, Any]:
    """Per dimension: macro F1 of each member and of the vote on the scored holdout cells, and the best of them."""
    if not members:
        raise ValueError("no members to combine")
    _common_ids(members)
    scores: dict[str, dict[str, float]] = {}
    routing: dict[str, str] = {}
    weights: dict[str, dict[str, float]] = {}
    for dimension in dimensions:
        scored = holdout[scored_mask(holdout, dimension)]
        ids = scored["image_id"].astype(str).tolist()
        truth = scored[dimension].astype(str).to_numpy()
        per = {
            name: macro_f1(truth, np.array([records[i]["labels"][dimension] for i in ids]))
            for name, records in members.items()
        }
        weights[dimension] = dict(per)
        if len(members) > 1:
            per[VOTE] = macro_f1(truth, np.array([vote(members, i, dimension, per)[0] for i in ids]))
        scores[dimension] = {k: round(v, 4) for k, v in per.items()}
        routing[dimension] = max(per, key=lambda k: (per[k], k != VOTE))  # a tie prefers a single member
    estimate = float(np.mean([scores[d][routing[d]] for d in dimensions]))
    return {"routing": routing, "weights": weights, "scores": scores, "holdout_estimate": round(estimate, 4),
            "members": sorted(members), "note": "chosen on the holdout, so the estimate is optimistic"}


def apply(choice: Mapping[str, Any], members: Mapping[str, Records]) -> list[dict[str, Any]]:
    """Combined records for every image all members answered, following ``choice['routing']``."""
    routing: Mapping[str, str] = choice["routing"]
    missing = sorted({m for m in routing.values() if m != VOTE} - set(members))
    if missing:
        raise KeyError(f"the routing needs member(s) {missing}, which were not given")
    voters = {name: members[name] for name in choice["members"] if name in members}
    combined = []
    for image_id in _common_ids(members):
        labels, confidence = {}, {}
        for dimension, source in routing.items():
            if source == VOTE:
                labels[dimension], confidence[dimension] = vote(voters, image_id, dimension,
                                                                choice["weights"][dimension])
            else:
                record = members[source][image_id]
                labels[dimension] = record["labels"][dimension]
                confidence[dimension] = _confidence(record, dimension)
        combined.append({"image_id": image_id, "labels": labels, "confidence": confidence})
    return combined


def _member_files(specs: Sequence[str], part: str) -> dict[str, Path]:
    from galaxus.data.paths import out_dir

    files = {}
    for spec in specs:
        name, _, exp = spec.partition("=")
        if not name or not exp or name == VOTE:
            raise ValueError(f"members look like name=EXP-0001 (not '{VOTE}'), got {spec!r}")
        files[name] = out_dir() / "predictions" / exp / f"{part}.jsonl"
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.ensemble", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    choose_cmd = sub.add_parser("choose", help="score members and the vote on the holdout, save the routing")
    choose_cmd.add_argument("--members", nargs="+", required=True, help="name=EXP-#### (holdout.jsonl of each)")
    choose_cmd.add_argument("--out", type=Path, default=None)
    apply_cmd = sub.add_parser("apply", help="apply a routing to the members' dev or final answers")
    apply_cmd.add_argument("--routing", type=Path, required=True)
    apply_cmd.add_argument("--part", required=True, choices=("dev", "final", "holdout"))
    apply_cmd.add_argument("--exp", required=True, help="EXP id for the combined answers")
    arguments = parser.parse_args(argv)

    from galaxus.data import classes, load, split
    from galaxus.data.paths import out_dir

    if arguments.command == "choose":
        files = _member_files(arguments.members, "holdout")
        members = {name: load_records(path) for name, path in files.items()}
        holdout = split.split_holdout(load.read_train())[1]
        choice = choose(members, holdout, classes.dimensions(), split.scored_mask)
        choice["member_exps"] = dict(spec.split("=", 1) for spec in arguments.members)
        target = arguments.out or out_dir() / "ensemble" / ("routing-" + "-".join(sorted(members)) + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(choice, indent=2), encoding="utf-8")
        width = max(len(n) for n in [*members, VOTE])
        print(f"{'dimension':22}" + "".join(f"{n:>{width + 2}}" for n in [*sorted(members), VOTE]) + "   chosen")
        for dimension, per in choice["scores"].items():
            row = "".join(f"{per.get(n, float('nan')):>{width + 2}.3f}" for n in [*sorted(members), VOTE])
            print(f"{dimension:22}{row}   {choice['routing'][dimension]}")
        print(f"\nholdout mean macro F1 of the routing: {choice['holdout_estimate']} ({choice['note']})")
        print(f"routing saved to {target}")
        return 0

    choice = json.loads(arguments.routing.read_text(encoding="utf-8"))
    files = _member_files([f"{n}={e}" for n, e in choice["member_exps"].items()], arguments.part)
    members = {name: load_records(path) for name, path in files.items()}
    records = apply(choice, members)
    from galaxus.serve.runner import predictions_path

    target = predictions_path(arguments.exp, arguments.part)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} combined answers -> {target}")
    print(f"next: python -m galaxus.serve.submit --exp {arguments.exp} --part {arguments.part}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
