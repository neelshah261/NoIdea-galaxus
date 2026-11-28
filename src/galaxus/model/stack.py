"""S1 (#200, umbrella #214): stack the members per dimension from their saved answers, then per-class biases for
macro F1.

``ensemble`` hands a whole dimension to one member (or a vote). Here every class learns which members to trust:
the 70B on the text dimensions, the heads on Apertus' hidden states (``apertus-core``) where they read the image
better. It runs on saved answers in seconds, with no model and no GPU.

* **Apertus runs only** (the hosts' answer, ``decisions.md``, #232): combining Apertus runs is fine, a vote with a
  helper model's answers is not. :func:`check_apertus_only` refuses any member whose EXP row is not a single Apertus
  run: another model, a helper, or the product-type prior (an override).
* **Features, per dimension:** for every member and every allowed class, two numbers: did the member answer this
  class (one-hot), and with which confidence (one-hot x confidence). Any Apertus run that wrote
  ``predictions/<EXP>/<part>.jsonl`` works: the 8B and 70B zero-shot runs, the LoRA, ``apertus-core``.
* **The stacker:** #51's softmax head (``heads.fit_head``: tempered class weights, L2, deterministic) on the scored
  human cells, then a per-class bias (``calibration.fit_bias``) for macro F1 on out-of-fold answers.
* **Honest:** cross-fitted on the holdout's two product halves (``ensemble.holdout_halves``). A half's stacker
  answers the other half, and its bias comes from out-of-fold answers on its own two product quarters. That
  cross-fitted set is what the EXP row scores.
* **Dev and final:** the stacker fitted on the whole holdout, with its bias from the halves' cross-fitted answers.

    STACK_MEMBERS="core=EXP-0092 70b=EXP-0049" uv run --env-file .env python -m galaxus.model.stack eval \\
        --exp-id EXP-0098
    uv run --env-file .env python -m galaxus.model.stack fit --members core=EXP-0092 70b=EXP-0049 \\
        --out $GALAXUS_DATA/out/stack/stack-EXP-0098.json
    uv run --env-file .env python -m galaxus.model.stack apply --stack <that json> --part dev --exp EXP-0098

**Live, for the final run:** ``final_run --predictor galaxus.model.stack:from_env``, with ``$STACK_FILE`` (the json
``fit`` saved) and ``$ENSEMBLE_PREDICTORS`` / ``$ENSEMBLE_CACHE`` as for the live ensemble (one live predictor per
member, under the names ``fit`` used).

Dimensions and classes come from ``galaxus.data.classes``; nothing here names one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxus.model import ensemble, heads
from galaxus.model.calibration import apply_bias, fit_bias
from galaxus.types import ImageItem, Prediction

MEMBERS_ENV = "STACK_MEMBERS"
FILE_ENV = "STACK_FILE"
QUARTERS = 2
#: Never a member: the product-type prior overrules Apertus (``decisions.md``, #232).
EXCLUDED_PREDICTORS = ("type-prior",)
ID_COLUMN = ensemble.ID_COLUMN
PRODUCT_COLUMN = "product_id"
Records = ensemble.Records


def check_apertus_only(member_exps: Mapping[str, str], result_rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse every member that is not a single Apertus run, from its EXP row: each model it names is Apertus, it
    lists no helper, and it is not the product-type prior. A member without a row cannot be checked, so it is refused
    too. A combination of Apertus runs (an ensemble row) is refused as well: give the stack its members directly."""
    rows = {str(row.get("id")): row for row in result_rows}
    refused = []
    for name, exp in member_exps.items():
        row = rows.get(exp)
        if row is None:
            refused.append(f"{name}={exp} (no row in results/metrics.json to check)")
            continue
        models = [part.strip() for part in str(row.get("model") or "").split(",") if part.strip()]
        predictor = str(row.get("predictor") or "")
        apertus = bool(models) and all("apertus" in model.lower() for model in models)
        if not apertus or row.get("helpers") or predictor.startswith("helper") or predictor in EXCLUDED_PREDICTORS:
            refused.append(f"{name}={exp} (predictor {predictor or 'unknown'}, model {', '.join(models) or 'none'})")
    if refused:
        raise ValueError("the stack combines Apertus runs only (decisions.md, #232); refused: " + "; ".join(refused))


def features(members: Mapping[str, Records], names: Sequence[str], ids: Sequence[str], dimension: str,
             layout: Sequence[str]) -> np.ndarray:
    """Per image: for each member (in ``names``' order) and each class of ``layout``, one-hot, then one-hot x
    confidence. A class outside the layout adds no evidence."""
    column = {name: k for k, name in enumerate(layout)}
    width = len(layout)
    x = np.zeros((len(ids), 2 * width * len(names)), dtype=np.float32)
    for row, image_id in enumerate(ids):
        for m, name in enumerate(names):
            record = members[name][image_id]
            k = column.get(record["labels"][dimension])
            if k is not None:
                x[row, 2 * width * m + k] = 1.0
                x[row, 2 * width * m + width + k] = ensemble._confidence(record, dimension)
    return x


def _full_log_probs(head: heads.Head, x: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """``head``'s log-probabilities in the column order of ``classes`` (a class it never saw gets -1e9)."""
    out = np.full((len(x), len(classes)), -1e9)
    lp = head.log_probs(x)
    for k, name in enumerate(head.classes):
        out[:, list(classes).index(name)] = lp[:, k]
    return out


def fit_dimension(x: np.ndarray, truth: Sequence[str], groups: pd.DataFrame, seed: int = 0) -> heads.Head:
    """The stacker on ``x`` with its per-class bias from out-of-fold answers over ``QUARTERS`` product folds."""
    from galaxus.data.split import product_folds

    truth = [str(t) for t in truth]
    head = heads.fit_head(x, truth, seed=seed)
    if len(head.classes) < 2 or groups[PRODUCT_COLUMN].nunique() < QUARTERS:
        return head
    fold = product_folds(groups, n_folds=QUARTERS, random_state=seed).to_numpy()
    oof = np.full((len(x), len(head.classes)), -1e9)
    for q in range(QUARTERS):
        train, held = fold != q, fold == q
        if not train.any() or not held.any() or len(set(np.asarray(truth)[train])) < 2:
            return head  # too few cells to calibrate honestly: keep the stacker without a bias
        inner = heads.fit_head(x[train], [t for t, keep in zip(truth, train, strict=True) if keep], seed=seed)
        oof[held] = _full_log_probs(inner, x[held], head.classes)
    head.bias = fit_bias(oof, np.searchsorted(head.classes, truth))
    return head


@dataclass
class Stack:
    """One stacker per dimension, the member order its features use, and each dimension's class layout."""

    names: list[str]
    layouts: dict[str, list[str]]
    heads: dict[str, heads.Head]

    def predict(self, members: Mapping[str, Records], ids: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
        """``{dimension: [(label, probability), ...]}`` for ``ids``, in order."""
        out: dict[str, list[tuple[str, float]]] = {}
        for dimension, head in self.heads.items():
            x = features(members, self.names, ids, dimension, self.layouts[dimension])
            lp = head.log_probs(x)
            index = apply_bias(lp, head.bias if head.bias.size else np.zeros(len(head.classes)))
            out[dimension] = [(head.classes[i], float(np.exp(lp[row, i]))) for row, i in enumerate(index)]
        return out

    def to_json(self) -> dict[str, Any]:
        return {"names": self.names, "layouts": self.layouts, "heads": {
            d: {"classes": h.classes, "weights": h.weights.tolist(), "intercept": h.intercept.tolist(),
                "mean": h.mean.tolist(), "scale": h.scale.tolist(), "bias": h.bias.tolist()}
            for d, h in self.heads.items()}}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Stack:
        return cls(names=list(data["names"]), layouts={d: list(v) for d, v in data["layouts"].items()}, heads={
            d: heads.Head(classes=list(h["classes"]), weights=np.asarray(h["weights"], dtype=np.float32),
                          intercept=np.asarray(h["intercept"], dtype=np.float32),
                          mean=np.asarray(h["mean"], dtype=np.float32), scale=np.asarray(h["scale"], dtype=np.float32),
                          bias=np.asarray(h["bias"], dtype=np.float64))
            for d, h in data["heads"].items()})


def fit_stack(members: Mapping[str, Records], names: Sequence[str], rows: pd.DataFrame, solution: pd.DataFrame,
              dimensions: Sequence[str], allowed: Mapping[str, Any], seed: int = 0) -> Stack:
    """A stacker per dimension on the scored cells of ``rows`` (``solution``: the human label or empty, by image)."""
    ids = rows[ID_COLUMN].astype(str).tolist()
    labels = solution.set_index(solution[ID_COLUMN].astype(str)).reindex(ids)
    layouts = {d: sorted(str(c) for c in allowed[d]) for d in dimensions}
    fitted = {}
    for d in dimensions:
        scored = labels[d].notna().to_numpy()
        if not scored.any():
            continue
        x = features(members, names, [i for i, keep in zip(ids, scored, strict=True) if keep], d, layouts[d])
        fitted[d] = fit_dimension(x, labels[d][scored].astype(str).tolist(), rows[scored], seed)
    return Stack(names=list(names), layouts=layouts, heads=fitted)


def crossfit(members: Mapping[str, Records], names: Sequence[str], holdout: pd.DataFrame, solution: pd.DataFrame,
             dimensions: Sequence[str], allowed: Mapping[str, Any], halves: pd.Series, seed: int = 0) -> dict[str, Any]:
    """A stack fitted on each half (it answers the other), and the whole-holdout stack with the halves' bias."""
    half_of_row = halves.to_numpy()
    on_half = [fit_stack(members, names, holdout[half_of_row == h], solution, dimensions, allowed, seed)
               for h in range(ensemble.HALVES)]
    answered_by = {str(i): ensemble.HALVES - 1 - int(h) for i, h in zip(holdout[ID_COLUMN], half_of_row, strict=True)}
    full = fit_stack(members, names, holdout, solution, dimensions, allowed, seed)
    # The whole-holdout stack's bias from the cross-fitted answers: each holdout image answered by the other half.
    ids = holdout[ID_COLUMN].astype(str).tolist()
    labels = solution.set_index(solution[ID_COLUMN].astype(str)).reindex(ids)
    for d, head in full.heads.items():
        scored = labels[d].notna().to_numpy()
        lp = np.full((len(ids), len(head.classes)), -1e9)
        for h, stack in enumerate(on_half):
            rows = np.array([answered_by[i] == h for i in ids]) & scored
            if d in stack.heads and rows.any():
                x = features(members, names, [i for i, keep in zip(ids, rows, strict=True) if keep], d, full.layouts[d])
                lp[rows] = _full_log_probs(stack.heads[d], x, head.classes)
        known = scored & (lp.max(axis=1) > -1e8)
        if known.any() and len(head.classes) > 1:
            head.bias = fit_bias(lp[known], np.searchsorted(head.classes, labels[d][known].astype(str).to_numpy()))
    return {"on_half": on_half, "answered_by": answered_by, "full": full}


@dataclass
class StackPredictor:
    """Holdout images are answered by the stack of the *other* half (an honest EXP row); others by the full stack."""

    members: dict[str, Records]
    member_exps: dict[str, str]
    fitted: dict[str, Any]
    images_per_second: float | None = None
    model: str | None = None
    helpers: tuple[str, ...] = field(default_factory=tuple)

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        ids = [str(item.image_id) for item in batch]
        ensemble.check_coverage(self.members, ids)
        labels: dict[str, dict[str, str]] = {i: {} for i in ids}
        confidence: dict[str, dict[str, float]] = {i: {} for i in ids}
        groups: dict[int | None, list[str]] = {}
        for image_id in ids:
            groups.setdefault(self.fitted["answered_by"].get(image_id), []).append(image_id)
        for half, group in groups.items():
            stack = self.fitted["full"] if half is None else self.fitted["on_half"][half]
            for d, answers in stack.predict(self.members, group).items():
                for image_id, (label, p) in zip(group, answers, strict=True):
                    labels[image_id][d], confidence[image_id][d] = label, p
        return [Prediction(image_id=i, labels=labels[i], confidence=confidence[i]) for i in ids]

    def config(self) -> dict[str, Any]:
        return {"method": "per-dimension stack of the members' saved answers (one-hot and confidence per member and "
                          "class), #51's softmax head and a per-class bias for macro F1, cross-fitted on the holdout's "
                          "product halves (#200)",
                "members": dict(self.member_exps), "member_order": list(self.fitted["full"].names)}


def from_saved(member_exps: Mapping[str, str], holdout: pd.DataFrame, dimensions: Sequence[str], solution_frame: Any,
               allowed: Mapping[str, Any], out_dir: Path, result_rows: Sequence[Mapping[str, Any]] = (),
               seed: int = 0) -> StackPredictor:
    """The predictor from the members' saved holdout answers; speed and models from their EXP rows, which must show
    Apertus runs only (:func:`check_apertus_only`)."""
    check_apertus_only(member_exps, result_rows)
    members = ensemble.load_members(member_exps, out_dir)
    ensemble.check_coverage(members, holdout[ID_COLUMN].astype(str).tolist())
    solution = solution_frame(holdout, tuple(dimensions))
    fitted = crossfit(members, list(member_exps), holdout, solution, dimensions, allowed,
                      ensemble.holdout_halves(holdout, seed), seed)
    rows = {str(row.get("id")): row for row in result_rows}
    member_rows = [rows.get(exp, {}) for exp in member_exps.values()]
    models = sorted({str(row["model"]) for row in member_rows if row.get("model")})
    helpers = sorted({str(h) for row in member_rows for h in (row.get("helpers") or [])})
    return StackPredictor(members=members, member_exps=dict(member_exps), fitted=fitted,
                          images_per_second=ensemble.combined_speed([row.get("images_per_second") for row in member_rows]),
                          model=", ".join(models) or None, helpers=tuple(helpers))


def members_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    environ = os.environ if environ is None else environ
    spec = environ.get(MEMBERS_ENV, "")
    if not spec:
        raise RuntimeError(f"set ${MEMBERS_ENV} to Apertus runs, e.g. \"core=EXP-0092 70b=EXP-0049\"")
    return ensemble.parse_members(spec)


def eval_factory(rows: pd.DataFrame, data: Any, seed: int) -> StackPredictor:
    """For the eval script: ``$STACK_MEMBERS``' saved holdout answers, cross-fitted (the training rows are unused)."""
    from galaxus.model.results import RESULTS_FILE, read_rows

    holdout = data.split_holdout(data.read_train())[1]
    return from_saved(members_from_env(), holdout, data.dimensions(), data.solution_frame, data.allowed_classes(),
                      data.out_dir(), read_rows(RESULTS_FILE), seed)


# ---------------------------------------------------------------- live, for the final run


@dataclass
class LiveStackPredictor(ensemble.LiveEnsemblePredictor):
    """Every member answers the batch live, at the same time (the live ensemble's machinery and answer cache), then
    the frozen whole-holdout stack combines them. ``choice`` holds the stack json ``fit`` saved."""

    def __post_init__(self) -> None:
        self.stack = Stack.from_json(self.choice)
        missing = [name for name in self.stack.names if name not in self.predictors]
        if missing:
            raise ValueError(f"the stack needs members {missing}, which have no predictor")
        models = sorted({str(m) for m in (getattr(p, "model", None) for p in self.predictors.values()) if m})
        self.model = ", ".join(models) or None
        self.model_revision = None
        self.helpers = tuple(sorted({h for p in self.predictors.values() for h in getattr(p, "helpers", ())}))

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        names = list(self.stack.names)
        with ThreadPoolExecutor(max_workers=max(1, len(names)), thread_name_prefix="member") as pool:
            futures = {name: pool.submit(self._answers, name, batch) for name in names}
            answers = {name: futures[name].result() for name in names}
        ids = [str(item.image_id) for item in batch]
        combined = self.stack.predict(answers, ids)
        return [Prediction(image_id=i, labels={d: combined[d][row][0] for d in combined},
                           confidence={d: combined[d][row][1] for d in combined}) for row, i in enumerate(ids)]

    def config(self) -> dict[str, Any]:
        members = {}
        for name in sorted(self.predictors):
            inner = getattr(self.predictors[name], "config", None)
            members[name] = {"predictor": self.specs.get(name), "config": inner() if callable(inner) else None}
        return {"method": "live per-dimension stack (#200): every member answers each image, the frozen stack from "
                          "`stack fit` combines them", "stack_file": self.routing_file,
                "member_exps": self.choice.get("member_exps"), "members": members,
                "answer_cache": None if self.cache_dir is None else str(self.cache_dir)}


def from_env(environ: Mapping[str, str] | None = None) -> LiveStackPredictor:
    """For ``final_run --predictor galaxus.model.stack:from_env``: ``$STACK_FILE`` and ``$ENSEMBLE_PREDICTORS``."""
    from galaxus.serve.items import load_predictor

    environ = os.environ if environ is None else environ
    stack_file, specs = environ.get(FILE_ENV), environ.get(ensemble.PREDICTORS_ENV)
    if not stack_file or not specs:
        raise RuntimeError(f"set ${FILE_ENV} (the json `python -m galaxus.model.stack fit` saved) and "
                           f"${ensemble.PREDICTORS_ENV} (name=package.module:factory per member)")
    choice = json.loads(Path(stack_file).read_text(encoding="utf-8"))
    member_specs = ensemble.parse_members(specs)
    missing = [name for name in choice["names"] if name not in member_specs]
    if missing:
        raise RuntimeError(f"the stack in {stack_file} reads members {missing}: add them to ${ensemble.PREDICTORS_ENV}")
    cache = environ.get(ensemble.CACHE_ENV)
    return LiveStackPredictor(choice=choice, predictors={n: load_predictor(member_specs[n]) for n in choice["names"]},
                              specs={n: member_specs[n] for n in choice["names"]},
                              cache_dir=Path(cache) if cache else None, routing_file=stack_file)


# ---------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.stack", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eval", help="the EXP row through the eval script ($STACK_MEMBERS), e.g. eval --exp-id EXP-0097",
                   add_help=False)
    fit_cmd = sub.add_parser("fit", help="fit the whole-holdout stack and save it as json")
    fit_cmd.add_argument("--members", nargs="+", required=True, help="name=EXP-#### ...")
    fit_cmd.add_argument("--seed", type=int, default=0)
    fit_cmd.add_argument("--out", type=Path, default=None)
    apply_cmd = sub.add_parser("apply", help="apply a saved stack to the members' dev or final answers")
    apply_cmd.add_argument("--stack", type=Path, required=True)
    apply_cmd.add_argument("--part", required=True, choices=("dev", "final"))
    apply_cmd.add_argument("--exp", required=True, help="EXP id for the stacked answers")
    arguments, rest = parser.parse_known_args(argv)

    if arguments.command == "eval":
        from galaxus.model import eval as model_eval

        model_eval.PREDICTORS["stack"] = eval_factory  # registered here, so eval.py itself needs no change
        return model_eval.main(["stack", *rest])
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    from galaxus.data import classes, load, split
    from galaxus.data.paths import out_dir

    if arguments.command == "fit":
        from galaxus.model.results import RESULTS_FILE, read_rows

        member_exps = ensemble.parse_members(arguments.members)
        holdout = split.split_holdout(load.read_train())[1]
        predictor = from_saved(member_exps, holdout, classes.dimensions(), split.solution_frame,
                               classes.allowed_classes(), out_dir(), read_rows(RESULTS_FILE), seed=arguments.seed)
        target = arguments.out or out_dir() / "stack" / ("stack-" + "-".join(member_exps) + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**predictor.fitted["full"].to_json(), "member_exps": member_exps}),
                          encoding="utf-8")
        print(f"stack of {', '.join(member_exps)} saved to {target}")
        return 0

    data = json.loads(arguments.stack.read_text(encoding="utf-8"))
    stack = Stack.from_json(data)
    members = ensemble.load_members(data["member_exps"], out_dir(), arguments.part)
    ids = sorted(set().union(*(set(records) for records in members.values())))
    ensemble.check_coverage(members, ids)  # a member missing an image would silently drop its row from the file
    combined = stack.predict(members, ids)
    from galaxus.serve.runner import predictions_path

    target = predictions_path(arguments.exp, arguments.part)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row, image_id in enumerate(ids):
            record = {ID_COLUMN: image_id, "labels": {d: combined[d][row][0] for d in combined},
                      "confidence": {d: combined[d][row][1] for d in combined}}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(ids)} stacked answers written to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
