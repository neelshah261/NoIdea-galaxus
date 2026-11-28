"""Product type as a prior (#133, the hosts' idea 3 in #17), on any predictor's saved answers with confidences.

The product type predicts some labels strongly. #17 point 1: "each product type has a hero-shot rule". Point 9: the
context decides accessory vs product. Per dimension, this module learns from **train minus the holdout** how pure each
product type is: its most common human class and that class's share of the type's scored cells. Then it learns one rule
per dimension, ``(below, share)``: **when the predictor's confidence is below ``below`` and the image's product type has
a top class with at least ``share`` of its cells, answer that class.**

* **Type table:** from the scored cells of train minus the holdout (human labels in relevant cells,
  ``split.scored_mask``). A type with fewer than ``MIN_TYPE_CELLS`` scored cells for a dimension is not in the table.
* **No optimistic number:** each dimension's rule is chosen by ``metric.evaluate`` on one product half of the holdout
  and applied to the other, in both directions (``ensemble.holdout_halves``, as #42 and #87). A dimension where no rule
  raises macro F1 keeps the base answer.
* **Missing or rare type:** the base answer stays. Falling back to the global prior would mean the majority class,
  which costs macro F1. So a run without product context scores exactly as its base.
* **Why a gate and not a Bayes product:** the saved answers carry the confidence of the chosen class only (#75), not a
  distribution over all classes. This rule uses exactly what is saved, like the recalibration of #42.
* **An EXP row:** ``TYPE_PRIOR_BASE=EXP-0002 python -m galaxus.model.eval type-prior`` scores the cross-fitted answers.
* **For dev and final:** ``python -m galaxus.model.type_prior learn --base EXP-0002`` saves the table and the rules
  learned on the whole holdout. :func:`from_env` wraps the live predictor with them (``$TYPE_PRIOR_RULES``,
  ``$TYPE_PRIOR_INNER``).

Dimensions and classes come from ``galaxus.data.classes``; no class name is typed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.types import ImageItem, Prediction

ID_COLUMN = "image_id"
#: The catalog field (challenge_description.md "The task"), as the items carry it in ``ImageItem.context``.
TYPE_FIELD = "product_type"
BASE_ENV = "TYPE_PRIOR_BASE"
RULES_ENV = "TYPE_PRIOR_RULES"
INNER_ENV = "TYPE_PRIOR_INNER"
#: Confidence gates tried; 1.01 means every cell that has a confidence.
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)
#: Minimum share of the type's top class tried.
SHARES = (0.6, 0.7, 0.8, 0.9, 0.95)
MIN_TYPE_CELLS = 20      # scored cells a type needs in train before its top class is trusted
MIN_CHANGED = 5          # a rule must move at least this many cells on the half it is learned on
Records = dict[str, dict[str, Any]]
Evaluate = Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]]
SolutionFrame = Callable[[pd.DataFrame, tuple[str, ...]], pd.DataFrame]
ScoredMask = Callable[[pd.DataFrame, str], pd.Series]
#: ``{dimension: {product type: {"class": c, "share": s, "cells": n}}}``
Table = dict[str, dict[str, dict[str, Any]]]
#: ``{dimension: {"below": t, "share": s}}``
Rules = dict[str, dict[str, float]]


def product_type(value: Any) -> str | None:
    """The product type as text, or None when it is missing (empty, NaN)."""
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() != "nan" else None


def type_table(rows: pd.DataFrame, dims: Sequence[str], scored_mask: ScoredMask,
               min_cells: int = MIN_TYPE_CELLS) -> Table:
    """Per dimension and product type: the top human class, its share and the cell count, from ``rows``' scored cells.

    ``rows`` is train minus the official holdout, never the holdout itself.
    """
    table: Table = {dimension: {} for dimension in dims}
    if TYPE_FIELD not in rows.columns:
        return table
    for dimension in dims:
        part = rows[scored_mask(rows, dimension)]
        types = part[TYPE_FIELD].map(product_type)
        for name, labels in part[dimension].astype(str).groupby(types):
            if len(labels) < min_cells:
                continue
            top, count = min(labels.value_counts().items(), key=lambda pair: (-pair[1], pair[0]))
            table[dimension][str(name)] = {"class": str(top), "share": round(count / len(labels), 4),
                                           "cells": len(labels)}
    return table


def apply_rules(labels: Mapping[str, str], confidence: Mapping[str, float] | None, kind: Any, table: Table,
                rules: Rules) -> dict[str, str]:
    """One answer with the rules applied. A cell changes only when its confidence is known and below the rule's gate,
    and the product type ``kind`` is in the table with a top-class share of at least the rule's."""
    out = dict(labels)
    name = product_type(kind)
    if name is None:
        return out
    for dimension, rule in rules.items():
        entry = table.get(dimension, {}).get(name)
        value = (confidence or {}).get(dimension)
        if entry and value is not None and entry["share"] >= rule["share"] and float(value) < rule["below"]:
            out[dimension] = entry["class"]
    return out


def learn(records: Records, part: pd.DataFrame, table: Table, dims: Sequence[str], solution_frame: SolutionFrame,
          evaluate: Evaluate) -> dict[str, Any]:
    """The rules for ``part``: per dimension, the ``(below, share)`` that raises its macro F1 most, if any does."""
    from galaxus.model.recalibrate import _f1 as dimension_f1

    dims = tuple(dims)
    ids = part[ID_COLUMN].astype(str).tolist()
    missing = [i for i in ids if i not in records]
    if missing:
        raise KeyError(f"{len(missing)} holdout images have no saved answer, e.g. {missing[:3]}")
    kinds = dict(zip(ids, part[TYPE_FIELD].map(product_type), strict=True)) if TYPE_FIELD in part.columns else {}
    fill = {d: records[ids[0]]["labels"][d] for d in dims}
    rules: Rules = {}
    gains: dict[str, dict[str, float]] = {}
    for dimension in dims:
        answers = {i: records[i]["labels"][dimension] for i in ids}
        conf = {i: (records[i].get("confidence") or {}).get(dimension) for i in ids}
        entries = {i: table.get(dimension, {}).get(kinds.get(i)) for i in ids}
        base = best = dimension_f1(answers, part, dimension, fill, dims, solution_frame, evaluate)
        chosen = None
        for share in SHARES:
            for below in THRESHOLDS:
                moved = {i: entry["class"] for i, entry in entries.items()
                         if entry and conf[i] is not None and entry["share"] >= share and float(conf[i]) < below
                         and answers[i] != entry["class"]}
                if len(moved) < MIN_CHANGED:
                    continue
                score = dimension_f1({**answers, **moved}, part, dimension, fill, dims, solution_frame, evaluate)
                if score > best + 1e-9:
                    best, chosen = score, {"below": below, "share": share}
        if chosen:
            rules[dimension] = chosen
        gains[dimension] = {"before": round(base, 4), "after": round(best, 4)}
    return {"rules": rules, "gains": gains, "images": len(ids)}


@dataclass
class TypePriorPredictor:
    """Answers with an inner predictor (or a base EXP's saved answers), then applies the product-type rules."""

    table: Table
    rules: Rules
    inner: Any = None                                  # a live predictor (dev/final), or None to read saved answers
    saved: Records | None = None                        # the base EXP's saved holdout answers
    rules_by_image: dict[str, Rules] | None = None      # holdout image -> rules learned on the other half
    base_exp: str | None = None
    images_per_second: float | None = None

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        if self.inner is not None:
            raw = [(str(p.image_id), dict(p.labels), dict(p.confidence or {})) for p in self.inner.predict(batch)]
        else:
            missing = [str(item.image_id) for item in batch if str(item.image_id) not in (self.saved or {})]
            if missing:
                raise KeyError(f"no saved answer for {missing[:3]}")
            raw = [(str(item.image_id), dict(self.saved[str(item.image_id)]["labels"]),
                    dict(self.saved[str(item.image_id)].get("confidence") or {})) for item in batch]
        kinds = {str(item.image_id): (item.context or {}).get(TYPE_FIELD) for item in batch}
        out = []
        for image_id, labels, confidence in raw:
            rules = (self.rules_by_image or {}).get(image_id, self.rules)
            answer = apply_rules(labels, confidence, kinds.get(image_id), self.table, rules)
            # a changed cell's confidence was about the old class: it is dropped, never carried over
            kept = {d: v for d, v in confidence.items() if answer.get(d) == labels.get(d)}
            out.append(Prediction(image_id=image_id, labels=answer, confidence=kept or None))
        return out

    def config(self) -> dict[str, Any]:
        return {"method": "product-type prior, gated by confidence, learned on one product half and applied to the "
                          "other (#133)",
                "base": self.base_exp, "rules_full_holdout": self.rules, "min_type_cells": MIN_TYPE_CELLS,
                "types_in_table": {d: len(v) for d, v in self.table.items()}}


def crossfit(records: Records, holdout: pd.DataFrame, table: Table, dims: Sequence[str], solution_frame: SolutionFrame,
             evaluate: Evaluate, seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Rules], dict[str, Any]]:
    """Rules learned on each half (applied to the other), the per-image map, and the whole-holdout rules."""
    from galaxus.model.ensemble import holdout_halves

    halves = holdout_halves(holdout, seed).to_numpy()
    learned = [learn(records, holdout[halves == h], table, dims, solution_frame, evaluate) for h in (0, 1)]
    by_image = {str(i): learned[1 - int(h)]["rules"] for i, h in zip(holdout[ID_COLUMN], halves, strict=True)}
    full = learn(records, holdout, table, dims, solution_frame, evaluate)
    return learned, by_image, full


def from_saved(base_exp: str, rows: pd.DataFrame, holdout: pd.DataFrame, dims: Sequence[str], scored_mask: ScoredMask,
               solution_frame: SolutionFrame, evaluate: Evaluate, out_dir: Path,
               result_rows: Sequence[Mapping[str, Any]] = (), seed: int = 0) -> TypePriorPredictor:
    """The EXP-row predictor: the base EXP's saved holdout answers with the type prior, cross-fitted.

    ``rows`` is train minus the holdout (the type table); ``holdout`` the official holdout (the rules).
    """
    from galaxus.model.ensemble import load_records

    path = out_dir / "predictions" / base_exp / "holdout.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `python -m galaxus.model.eval` for {base_exp} first")
    records = load_records(path)
    table = type_table(rows, dims, scored_mask)
    _, by_image, full = crossfit(records, holdout, table, dims, solution_frame, evaluate, seed)
    speed = {str(r.get("id")): r.get("images_per_second") for r in result_rows}.get(base_exp)
    return TypePriorPredictor(table=table, rules=full["rules"], saved=records, rules_by_image=by_image,
                              base_exp=base_exp, images_per_second=speed)


def from_env() -> TypePriorPredictor:
    """For ``final_run`` / ``submit``: ``$TYPE_PRIOR_INNER`` (e.g. galaxus.model.zero_shot:from_preset) with the table
    and rules in ``$TYPE_PRIOR_RULES`` (written by ``learn``)."""
    from galaxus.serve.items import load_predictor

    rules_path = os.environ.get(RULES_ENV)
    inner_spec = os.environ.get(INNER_ENV)
    if not rules_path or not inner_spec:
        raise RuntimeError(f"set ${RULES_ENV} (a json from `python -m galaxus.model.type_prior learn`) and "
                           f"${INNER_ENV} (the live predictor, e.g. galaxus.model.zero_shot:from_preset)")
    saved = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    return TypePriorPredictor(table=saved["table"], rules=saved["rules"], inner=load_predictor(inner_spec),
                              base_exp=saved.get("base"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.type_prior", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    learn_cmd = sub.add_parser("learn", help="the type table (train) and rules (whole holdout) from a base EXP")
    learn_cmd.add_argument("--base", required=True, help="EXP id whose holdout.jsonl has confidences, e.g. EXP-0002")
    learn_cmd.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)

    from galaxus.data import classes, load, metric, split
    from galaxus.data.paths import out_dir
    from galaxus.model.ensemble import load_records

    rows, holdout = split.split_holdout(load.read_train())
    dims = tuple(classes.dimensions())
    table = type_table(rows, dims, split.scored_mask)
    records = load_records(out_dir() / "predictions" / arguments.base / "holdout.jsonl")
    _, _, full = crossfit(records, holdout, table, dims, split.solution_frame, metric.evaluate)
    print(f"{'dimension':22}{'types':>6}{'before':>8}{'after (whole, optimistic)':>28}   rule")
    for d in dims:
        g = full["gains"][d]
        print(f"{d:22}{len(table[d]):>6}{g['before']:>8.3f}{g['after']:>28.3f}   {full['rules'].get(d, '-')}")
    print(f"\nThe honest number is the EXP row: {BASE_ENV}={arguments.base} python -m galaxus.model.eval type-prior")
    target = arguments.out or out_dir() / "type_prior" / f"rules-{arguments.base}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"base": arguments.base, "table": table, **full}, indent=2), encoding="utf-8")
    print(f"table and rules saved to {target}  (for final_run: {RULES_ENV}={target} {INNER_ENV}=<predictor>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
