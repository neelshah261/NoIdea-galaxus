"""Confidence-gated recalibration for macro F1 (#42, #101 item 6), on any predictor's saved answers with confidences.

Macro F1 counts every class equally, and a fine-tuned model leans towards the frequent classes. Per dimension and per
predicted class ``p``, this learns one rule from the holdout: **when the model says ``p`` with confidence below ``t``,
answer ``r`` instead**, where ``r`` is chosen (often a rarer class that ``p`` swallows) only if it raises the
dimension's macro F1. Then:

* **Scored through metric.py:** every candidate rule is judged by ``metric.evaluate`` on ``split.solution_frame``.
* **No optimistic number:** rules are learned on one product half of the holdout and applied to the other, in both
  directions (``ensemble.holdout_halves``, as #87), so no holdout answer is scored by a rule that saw it.
* **An EXP row:** ``RECAL_BASE=EXP-0004 python -m galaxus.model.eval recalibrated`` scores the cross-fitted answers.
* **For dev and final:** :class:`RecalibratedPredictor` wraps the live predictor with the rules learned on the whole
  holdout (``python -m galaxus.model.recalibrate learn --base EXP-0004`` saves them).

Only the predictor's own answer and confidence are used: no model runs again, and a cell without a confidence is left
as it is. Dimensions and classes come from ``galaxus.data.classes``.
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

from galaxus.model.contract import ImageItem, Prediction

ID_COLUMN = "image_id"
BASE_ENV = "RECAL_BASE"
RULES_ENV = "RECAL_RULES"
INNER_ENV = "RECAL_INNER"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_CHANGED = 5          # a rule must move at least this many cells on the half it is learned on
Records = dict[str, dict[str, Any]]
Evaluate = Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]]
SolutionFrame = Callable[[pd.DataFrame, tuple[str, ...]], pd.DataFrame]
#: ``{dimension: {predicted class: {"below": t, "to": r}}}``
Rules = dict[str, dict[str, dict[str, Any]]]


def apply_rules(labels: Mapping[str, str], confidence: Mapping[str, float] | None, rules: Rules) -> dict[str, str]:
    """One answer with the rules applied: a cell changes only when its confidence is known and below the rule's t."""
    out = dict(labels)
    for dimension, by_class in rules.items():
        rule = by_class.get(out.get(dimension, ""))
        value = (confidence or {}).get(dimension)
        if rule and value is not None and float(value) < float(rule["below"]):
            out[dimension] = rule["to"]
    return out


def _f1(answers: Mapping[str, str], part: pd.DataFrame, dimension: str, fill: Mapping[str, str],
        dims: tuple[str, ...], solution_frame: SolutionFrame, evaluate: Evaluate) -> float:
    ids = part[ID_COLUMN].astype(str).tolist()
    submission = pd.DataFrame({ID_COLUMN: ids, **{d: [fill[d]] * len(ids) for d in dims}})
    submission[dimension] = [answers[i] for i in ids]
    return float((evaluate(solution_frame(part, dims), submission).get(dimension) or {}).get("f1_macro") or 0.0)


def learn(
    records: Records, part: pd.DataFrame, dims: Sequence[str], allowed: Mapping[str, Sequence[str]],
    solution_frame: SolutionFrame, evaluate: Evaluate,
) -> dict[str, Any]:
    """Rules for ``part``: per dimension, greedily over the predicted classes, the (t, r) that raises macro F1 most."""
    dims = tuple(dims)
    ids = part[ID_COLUMN].astype(str).tolist()
    missing = [i for i in ids if i not in records]
    if missing:
        raise KeyError(f"{len(missing)} holdout images have no saved answer, e.g. {missing[:3]}")
    fill = {d: records[ids[0]]["labels"][d] for d in dims}
    rules: Rules = {}
    gains: dict[str, dict[str, float]] = {}
    for dimension in dims:
        answers = {i: records[i]["labels"][dimension] for i in ids}
        conf = {i: (records[i].get("confidence") or {}).get(dimension) for i in ids}
        base = best = _f1(answers, part, dimension, fill, dims, solution_frame, evaluate)
        chosen: dict[str, dict[str, Any]] = {}
        predicted = pd.Series(answers).value_counts().index.tolist()   # frequent classes first
        for p in predicted:
            candidates = [i for i in ids if answers[i] == p and conf[i] is not None]
            best_rule, best_answers = None, None
            for t in THRESHOLDS:
                low = [i for i in candidates if float(conf[i]) < t]
                if len(low) < MIN_CHANGED:
                    continue
                for r in sorted(allowed[dimension]):
                    if r == p:
                        continue
                    trial = dict(answers)
                    trial.update({i: r for i in low})
                    score = _f1(trial, part, dimension, fill, dims, solution_frame, evaluate)
                    if score > best + 1e-9:
                        best, best_rule, best_answers = score, {"below": t, "to": r}, trial
            if best_rule is not None:
                chosen[p] = best_rule
                answers = best_answers
        if chosen:
            rules[dimension] = chosen
        gains[dimension] = {"before": round(base, 4), "after": round(best, 4)}
    return {"rules": rules, "gains": gains, "images": len(ids)}


@dataclass
class RecalibratedPredictor:
    """Answers with an inner predictor, then applies rules (one set, or per holdout half for an honest EXP row)."""

    rules: Rules
    inner: Any = None                                  # a live predictor (dev/final), or None to read saved answers
    saved: Records | None = None                        # the base EXP's saved holdout answers
    rules_by_image: dict[str, Rules] | None = None      # holdout image -> rules learned on the other half
    base_exp: str | None = None
    images_per_second: float | None = None
    #: The base run's model and revision (#175), so the EXP row, compare and the report name it.
    model: str | None = None
    model_revision: str | None = None

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        if self.inner is not None:
            raw = [(p.image_id, dict(p.labels), dict(p.confidence or {})) for p in self.inner.predict(batch)]
        else:
            missing = [str(item.image_id) for item in batch if str(item.image_id) not in (self.saved or {})]
            if missing:
                raise KeyError(f"no saved answer for {missing[:3]}")
            raw = [(str(item.image_id), dict(self.saved[str(item.image_id)]["labels"]),
                    dict(self.saved[str(item.image_id)].get("confidence") or {})) for item in batch]
        out = []
        for image_id, labels, confidence in raw:
            rules = (self.rules_by_image or {}).get(str(image_id), self.rules)
            out.append(Prediction(image_id=image_id, labels=apply_rules(labels, confidence, rules),
                                  confidence=confidence or None))
        return out

    def config(self) -> dict[str, Any]:
        return {"method": "confidence-gated recalibration, learned on one product half, applied to the other (#42)",
                "base": self.base_exp, "rules_full_holdout": self.rules}


def crossfit(
    records: Records, holdout: pd.DataFrame, dims: Sequence[str], allowed: Mapping[str, Sequence[str]],
    solution_frame: SolutionFrame, evaluate: Evaluate, seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Rules], dict[str, Any]]:
    """Rules learned on each half (applied to the other), the per-image map, and the whole-holdout rules."""
    from galaxus.model.ensemble import holdout_halves

    halves = holdout_halves(holdout, seed).to_numpy()
    learned = [learn(records, holdout[halves == h], dims, allowed, solution_frame, evaluate) for h in (0, 1)]
    by_image = {str(i): learned[1 - int(h)]["rules"] for i, h in zip(holdout[ID_COLUMN], halves, strict=True)}
    full = learn(records, holdout, dims, allowed, solution_frame, evaluate)
    return learned, by_image, full


def from_saved(base_exp: str, holdout: pd.DataFrame, dims: Sequence[str], allowed: Mapping[str, Sequence[str]],
               solution_frame: SolutionFrame, evaluate: Evaluate, out_dir: Path,
               result_rows: Sequence[Mapping[str, Any]] = (), seed: int = 0) -> RecalibratedPredictor:
    """The EXP-row predictor: the base EXP's saved holdout answers, recalibrated cross-fitted."""
    from galaxus.model.ensemble import load_records

    path = out_dir / "predictions" / base_exp / "holdout.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `python -m galaxus.model.eval` for {base_exp} first")
    records = load_records(path)
    _, by_image, full = crossfit(records, holdout, dims, allowed, solution_frame, evaluate, seed)
    base_row = {str(r.get("id")): r for r in result_rows}.get(base_exp, {})
    speed = base_row.get("images_per_second")
    return RecalibratedPredictor(rules=full["rules"], saved=records, rules_by_image=by_image, base_exp=base_exp,
                                 images_per_second=speed,
                                 model=base_row.get("model"), model_revision=base_row.get("model_revision"))


def from_env() -> RecalibratedPredictor:
    """For ``final_run`` / ``submit``: ``$RECAL_INNER`` (e.g. galaxus.model.lora:from_env) with ``$RECAL_RULES``."""
    from galaxus.serve.items import load_predictor

    rules_path = os.environ.get(RULES_ENV)
    inner_spec = os.environ.get(INNER_ENV)
    if not rules_path or not inner_spec:
        raise RuntimeError(f"set ${RULES_ENV} (a rules json from `recalibrate learn`) and ${INNER_ENV} "
                           "(the live predictor, e.g. galaxus.model.lora:from_env)")
    saved = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    return RecalibratedPredictor(rules=saved["rules"], inner=load_predictor(inner_spec), base_exp=saved.get("base"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.recalibrate", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    learn_cmd = sub.add_parser("learn", help="learn the rules on the whole holdout from a base EXP's saved answers")
    learn_cmd.add_argument("--base", required=True, help="EXP id whose holdout.jsonl has confidences, e.g. EXP-0004")
    learn_cmd.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)

    from galaxus.data import classes, load, metric, split
    from galaxus.data.paths import out_dir
    from galaxus.model.ensemble import load_records

    holdout = split.split_holdout(load.read_train())[1]
    dims = tuple(classes.dimensions())
    allowed = {d: sorted(v) for d, v in classes.allowed_classes().items()}
    records = load_records(out_dir() / "predictions" / arguments.base / "holdout.jsonl")
    _, _, full = crossfit(records, holdout, dims, allowed, split.solution_frame, metric.evaluate)
    print(f"{'dimension':22}{'before':>8}{'after (whole, optimistic)':>28}   rules")
    for d in dims:
        g = full["gains"][d]
        print(f"{d:22}{g['before']:>8.3f}{g['after']:>28.3f}   {full['rules'].get(d, {})}")
    print("\nThe honest number is the EXP row: RECAL_BASE=" + arguments.base + " python -m galaxus.model.eval recalibrated")
    target = arguments.out or out_dir() / "recalibrate" / f"rules-{arguments.base}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"base": arguments.base, **full}, indent=2), encoding="utf-8")
    print(f"rules saved to {target}  (for final_run: RECAL_RULES={target} RECAL_INNER=<predictor>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
