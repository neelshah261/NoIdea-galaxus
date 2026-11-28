"""Pick the main method per dimension (BUILD_PLAN.md phase 2 step 3) from the members' saved answers, on CPU.

Every method (zero-shot with and without context #47, LoRA #49 / #76, heads #41) writes its holdout answers to
``$GALAXUS_DATA/out/predictions/<EXP>/holdout.jsonl`` through ``python -m galaxus.model.eval``. This module combines
them per dimension without running any model again (#87):

* **Options per dimension:** each member, and a vote of the members weighted by each member's macro F1 on that
  dimension times its confidence.
* **Scored through metric.py:** every macro F1 comes from the data lane's ``metric.evaluate`` on
  ``split.solution_frame``, never from a copy of the formula (AGENTS.md product rule 1).
* **No optimistic number:** the holdout is cut into two product-grouped halves (``split.product_folds``). The
  routing is chosen on one half and applied to the other, in both directions, so no holdout answer is scored by a
  routing that saw it. That cross-fitted set is what the EXP row scores.
* **An EXP row:** ``python -m galaxus.model.eval ensemble`` runs :class:`EnsemblePredictor`, so the number lands in
  ``results/metrics.json`` with commit, config and holdout hash.
* For dev and final, ``apply`` uses the routing chosen on the whole holdout.

    ENSEMBLE_MEMBERS="zeroshot=EXP-0002 zeroshot_noctx=EXP-0009" python -m galaxus.model.eval ensemble
    python -m galaxus.model.ensemble choose --members zeroshot=EXP-0002 zeroshot_noctx=EXP-0009
    python -m galaxus.model.ensemble apply --routing <routing.json> --part dev --exp EXP-0020

Dimensions and classes come from ``galaxus.data.classes``; nothing here names one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.types import ImageItem, Prediction

VOTE = "vote"
ID_COLUMN = "image_id"
MEMBERS_ENV = "ENSEMBLE_MEMBERS"
#: BUILD_PLAN.md phase 2's methods that exist today (#87 step 5); the LoRA (EXP-0004) joins once it is scored.
DEFAULT_MEMBERS = "zeroshot=EXP-0002 zeroshot_noctx=EXP-0009"
HALVES = 2
Records = dict[str, dict[str, Any]]
#: ``evaluate(solution, submission)`` -> metric.py's result: ``{dimension: {"f1_macro": ...}, "overall": ...}``.
Evaluate = Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]]
SolutionFrame = Callable[[pd.DataFrame, tuple[str, ...]], pd.DataFrame]


def load_records(path: str | Path) -> Records:
    """``{image_id: record}`` from a predictions jsonl."""
    records: Records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record[ID_COLUMN])] = record
    return records


def parse_members(spec: str | Sequence[str]) -> dict[str, str]:
    """``"a=EXP-0002 b=EXP-0009"`` (or its parts) -> ``{"a": "EXP-0002", "b": "EXP-0009"}``."""
    parts = spec.split() if isinstance(spec, str) else list(spec)
    members: dict[str, str] = {}
    for part in parts:
        name, _, exp = part.partition("=")
        if not name or not exp or name == VOTE:
            raise ValueError(f"members look like name=EXP-0002 (not '{VOTE}'), got {part!r}")
        if name in members:
            raise ValueError(f"member {name!r} is named twice")
        members[name] = exp
    if not members:
        raise ValueError("no members to combine")
    return members


def _confidence(record: Mapping[str, Any], dimension: str) -> float:
    value = (record.get("confidence") or {}).get(dimension)
    return 1.0 if value is None else float(value)


def vote(
    members: Mapping[str, Records], image_id: str, dimension: str, weights: Mapping[str, float]
) -> tuple[str, float]:
    """Weighted vote on one cell (weight = the member's macro F1 x its confidence); ties go to the name sorting first."""
    tally: dict[str, float] = defaultdict(float)
    for name, records in members.items():
        record = records[image_id]
        tally[record["labels"][dimension]] += weights.get(name, 1.0) * _confidence(record, dimension)
    best = min(tally, key=lambda label: (-tally[label], label))
    total = sum(tally.values()) or 1.0
    return best, tally[best] / total


def check_coverage(members: Mapping[str, Records], ids: Sequence[str]) -> None:
    """Every member must answer every image in ``ids``: a gap would silently change what is compared."""
    missing = {name: sum(i not in records for i in ids) for name, records in members.items()}
    if any(missing.values()):
        raise KeyError(f"members do not answer every image; missing answers per member: {missing}")


def _frame(ids: Sequence[str], labels: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    return pd.DataFrame({ID_COLUMN: list(ids), **{d: list(v) for d, v in labels.items()}})


def _macro(evaluation: Mapping[str, Any], dimensions: Sequence[str]) -> dict[str, float]:
    return {d: float((evaluation.get(d) or {}).get("f1_macro") or 0.0) for d in dimensions}


def choose(
    members: Mapping[str, Records],
    part: pd.DataFrame,
    dimensions: Sequence[str],
    solution_frame: SolutionFrame,
    evaluate: Evaluate,
) -> dict[str, Any]:
    """Per dimension on ``part``: metric.py's macro F1 of each member and of the vote, and the best of them."""
    if not members:
        raise ValueError("no members to combine")
    dimensions = tuple(dimensions)
    ids = part[ID_COLUMN].astype(str).tolist()
    check_coverage(members, ids)
    solution = solution_frame(part, dimensions)
    per_member = {
        name: _macro(evaluate(solution, _frame(ids, {d: [records[i]["labels"][d] for i in ids] for d in dimensions})),
                     dimensions)
        for name, records in members.items()
    }
    weights = {d: {name: per_member[name][d] for name in members} for d in dimensions}
    scores = {d: {name: per_member[name][d] for name in members} for d in dimensions}
    if len(members) > 1:
        voted = {d: [vote(members, i, d, weights[d])[0] for i in ids] for d in dimensions}
        for d, value in _macro(evaluate(solution, _frame(ids, voted)), dimensions).items():
            scores[d][VOTE] = value
    # Best option per dimension; on a tie a single member beats the vote (simpler), then the name sorting first.
    routing = {d: min(per, key=lambda k: (-per[k], k == VOTE, k)) for d, per in scores.items()}
    return {
        "dimensions": list(dimensions),
        "routing": routing,
        "weights": weights,
        "scores": {d: {k: round(v, 4) for k, v in per.items()} for d, per in scores.items()},
        "images": len(ids),
    }


def combine(choice: Mapping[str, Any], members: Mapping[str, Records], image_id: str) -> Prediction:
    """One image's answer under a routing: each dimension from its chosen member, or from the vote."""
    labels: dict[str, str] = {}
    confidence: dict[str, float] = {}
    for dimension in choice["dimensions"]:
        source = choice["routing"][dimension]
        if source == VOTE:
            labels[dimension], confidence[dimension] = vote(members, image_id, dimension, choice["weights"][dimension])
        else:
            record = members[source][image_id]
            labels[dimension] = record["labels"][dimension]
            confidence[dimension] = _confidence(record, dimension)
    return Prediction(image_id=image_id, labels=labels, confidence=confidence)


def apply(choice: Mapping[str, Any], members: Mapping[str, Records]) -> list[Prediction]:
    """A routing applied to every image the members all answer (dev or final answers: no model runs again)."""
    ids = sorted(set.intersection(*(set(records) for records in members.values())))
    extra = {name: len(set(records) - set(ids)) for name, records in members.items()}
    if any(extra.values()):
        raise KeyError(f"members do not answer the same images; answers only one member has: {extra}")
    return [combine(choice, members, i) for i in ids]


def holdout_halves(holdout: pd.DataFrame, seed: int = 0) -> pd.Series:
    """0 or 1 per holdout row, grouped by product, from the data lane's ``split.product_folds``."""
    from galaxus.data.split import product_folds

    return product_folds(holdout, n_folds=HALVES, random_state=seed)


def crossfit(
    members: Mapping[str, Records],
    holdout: pd.DataFrame,
    dimensions: Sequence[str],
    solution_frame: SolutionFrame,
    evaluate: Evaluate,
    halves: pd.Series,
) -> dict[str, Any]:
    """A routing chosen on each half (applied to the other one), and the routing chosen on the whole holdout."""
    half_of_row = halves.to_numpy()
    chosen_on_half = [
        choose(members, holdout[half_of_row == half], dimensions, solution_frame, evaluate) for half in range(HALVES)
    ]
    # Holdout image -> index of the half whose routing answers it: always the other half.
    answered_by = {str(i): HALVES - 1 - int(h) for i, h in zip(holdout[ID_COLUMN], half_of_row, strict=True)}
    return {
        "chosen_on_half": chosen_on_half,
        "answered_by": answered_by,
        "full": choose(members, holdout, dimensions, solution_frame, evaluate),
    }


@dataclass
class EnsemblePredictor:
    """Answers holdout images with the routing chosen on the *other* half, so its eval row is an honest estimate.

    Images outside the holdout get the routing chosen on the whole holdout, which is what dev and final use.
    """

    members: dict[str, Records]
    member_exps: dict[str, str]
    fitted: dict[str, Any]
    images_per_second: float | None = None
    model: str | None = None
    helpers: tuple[str, ...] = field(default_factory=tuple)

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        ids = [str(item.image_id) for item in batch]
        check_coverage(self.members, ids)
        predictions = []
        for image_id in ids:
            half = self.fitted["answered_by"].get(image_id)
            choice = self.fitted["full"] if half is None else self.fitted["chosen_on_half"][half]
            predictions.append(combine(choice, self.members, image_id))
        return predictions

    def config(self) -> dict[str, Any]:
        return {
            "method": "per-dimension routing, chosen on one product half of the holdout and scored on the other (#87)",
            "members": dict(self.member_exps),
            "routing_chosen_on_half": [c["routing"] for c in self.fitted["chosen_on_half"]],
            "routing_full_holdout": self.fitted["full"]["routing"],
            "speed": "images_per_second is the members' combined speed: every image goes through every member",
        }


def combined_speed(rates: Sequence[float | None]) -> float | None:
    """Images per second when every image goes through every member in turn: 1 / sum(1 / rate)."""
    if not rates or any(rate is None or rate <= 0 for rate in rates):
        return None
    return 1.0 / sum(1.0 / float(rate) for rate in rates)


def load_members(member_exps: Mapping[str, str], out_dir: Path, part: str = "holdout") -> dict[str, Records]:
    """Each member's saved answers from ``out_dir/predictions/<EXP>/<part>.jsonl``."""
    members = {}
    for name, exp in member_exps.items():
        path = out_dir / "predictions" / exp / f"{part}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing: member {name} needs {exp}'s {part} answers on this machine "
                f"(the eval script writes them where {exp} ran)"
            )
        members[name] = load_records(path)
    return members


def from_saved(
    member_exps: Mapping[str, str],
    holdout: pd.DataFrame,
    dimensions: Sequence[str],
    solution_frame: SolutionFrame,
    evaluate: Evaluate,
    out_dir: Path,
    result_rows: Sequence[Mapping[str, Any]] = (),
    seed: int = 0,
) -> EnsemblePredictor:
    """The predictor from the members' saved holdout answers; speed and model from their EXP rows."""
    members = load_members(member_exps, out_dir)
    fitted = crossfit(members, holdout, dimensions, solution_frame, evaluate, holdout_halves(holdout, seed))
    rows = {str(row.get("id")): row for row in result_rows}
    member_rows = [rows.get(exp, {}) for exp in member_exps.values()]
    models = sorted({str(row["model"]) for row in member_rows if row.get("model")})
    return EnsemblePredictor(
        members=members,
        member_exps=dict(member_exps),
        fitted=fitted,
        images_per_second=combined_speed([row.get("images_per_second") for row in member_rows]),
        model=", ".join(models) or None,
    )


def members_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``$ENSEMBLE_MEMBERS``, or BUILD_PLAN.md's two zero-shot runs when it is unset."""
    environ = os.environ if environ is None else environ
    return parse_members(environ.get(MEMBERS_ENV) or DEFAULT_MEMBERS)


def _write(records: Sequence[Prediction], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for p in records:
            record = {ID_COLUMN: p.image_id, "labels": dict(p.labels), "confidence": dict(p.confidence or {})}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.ensemble", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    choose_cmd = sub.add_parser("choose", help="score the options per dimension and save the whole-holdout routing")
    choose_cmd.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS.split(), help="name=EXP-#### ...")
    choose_cmd.add_argument("--seed", type=int, default=0, help="seed of the product halves")
    choose_cmd.add_argument("--out", type=Path, default=None)
    apply_cmd = sub.add_parser("apply", help="apply the whole-holdout routing to the members' dev or final answers")
    apply_cmd.add_argument("--routing", type=Path, required=True)
    apply_cmd.add_argument("--part", required=True, choices=("dev", "final"))
    apply_cmd.add_argument("--exp", required=True, help="EXP id for the combined answers")
    arguments = parser.parse_args(argv)

    from galaxus.data import classes, load, metric, split
    from galaxus.data.paths import out_dir

    if arguments.command == "choose":
        member_exps = parse_members(arguments.members)
        holdout = split.split_holdout(load.read_train())[1]
        predictor = from_saved(member_exps, holdout, classes.dimensions(), split.solution_frame, metric.evaluate,
                               out_dir(), seed=arguments.seed)
        full = predictor.fitted["full"]
        names = [*member_exps, VOTE]
        width = max(len(n) for n in names) + 2
        print(f"{'dimension':22}" + "".join(f"{n:>{width}}" for n in names) + "   chosen: whole / half 0 / half 1")
        for d in full["dimensions"]:
            row = "".join(f"{full['scores'][d].get(n, float('nan')):>{width}.3f}" for n in names)
            halves = " / ".join(c["routing"][d] for c in predictor.fitted["chosen_on_half"])
            print(f"{d:22}{row}   {full['routing'][d]} / {halves}")
        spec = " ".join(f"{k}={v}" for k, v in member_exps.items())
        print("\nThese columns are chosen and scored on the same images, so they are optimistic. The honest number is"
              f'\nthe EXP row: {MEMBERS_ENV}="{spec}" python -m galaxus.model.eval ensemble')
        target = arguments.out or out_dir() / "ensemble" / ("routing-" + "-".join(sorted(member_exps)) + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**full, "member_exps": member_exps}, indent=2), encoding="utf-8")
        print(f"routing saved to {target}")
        return 0

    choice = json.loads(arguments.routing.read_text(encoding="utf-8"))
    records = apply(choice, load_members(choice["member_exps"], out_dir(), arguments.part))
    from galaxus.serve.runner import predictions_path

    target = predictions_path(arguments.exp, arguments.part)
    _write(records, target)
    print(f"{len(records)} combined answers -> {target}")
    print(f"next: python -m galaxus.serve.submit --exp {arguments.exp} --part {arguments.part}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
