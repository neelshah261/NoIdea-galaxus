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
* **Two rules** pick the routing. ``best`` takes each dimension's best option. ``agree`` (#106) keeps the
  **first member listed** (the base) on every dimension, except where the best other option **beats the base on both
  product halves** of the choosing data: a gain that flips between halves is noise, so that dimension stays on the
  base. Two near-equal alternatives (a member and the vote that mostly copies it) do not split the decision, since
  only the comparison with the base counts. Chosen on one half of the holdout, the rule splits that half into two
  product quarters, so the row stays honest.

    ENSEMBLE_MEMBERS="zeroshot=EXP-0002 zeroshot_noctx=EXP-0009" python -m galaxus.model.eval ensemble
    ENSEMBLE_MEMBERS="zeroshot=EXP-0002 zeroshot_noctx=EXP-0009" python -m galaxus.model.eval ensemble-agree
    python -m galaxus.model.ensemble choose --members zeroshot=EXP-0002 zeroshot_noctx=EXP-0009 --rule agree
    python -m galaxus.model.ensemble apply --routing <routing.json> --part dev --exp EXP-0020

* **Live, for the final run (#182):** :func:`from_env` builds each member's live predictor, answers a batch with all
  of them and combines the answers with the frozen routing file, so ``final_run --predictor
  galaxus.model.ensemble:from_env`` runs the whole combination in one pass. ``$ENSEMBLE_CACHE`` keeps every member's
  answers, so a rerun gives the same file even when a member is not bit-reproducible.

Dimensions and classes come from ``galaxus.data.classes``; nothing here names one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.types import ImageItem, Prediction

VOTE = "vote"
ID_COLUMN = "image_id"
MEMBERS_ENV = "ENSEMBLE_MEMBERS"
#: The live predictor (#182): the routing json ``choose`` saved, each member's predictor, and an answer cache.
ROUTING_ENV, PREDICTORS_ENV, CACHE_ENV = "ENSEMBLE_ROUTING", "ENSEMBLE_PREDICTORS", "ENSEMBLE_CACHE"
#: BUILD_PLAN.md phase 2's methods that exist today (#87 step 5); the LoRA (EXP-0004) joins once it is scored.
DEFAULT_MEMBERS = "zeroshot=EXP-0002 zeroshot_noctx=EXP-0009"
HALVES = 2
#: ``best``: each dimension's best option. ``agree`` (#106): the base, unless the best other option beats it on both halves.
BEST, AGREE = "best", "agree"
RULES = (BEST, AGREE)
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


def agree(
    members: Mapping[str, Records],
    part: pd.DataFrame,
    dimensions: Sequence[str],
    solution_frame: SolutionFrame,
    evaluate: Evaluate,
    folds: pd.Series,
    base: str,
) -> dict[str, Any]:
    """#106's rule on ``part``: ``base`` on every dimension, except where the best option beats it on both folds.

    ``folds`` is 0 or 1 per row of ``part``, grouped by product. The candidate is the best option on the whole of
    ``part``; the dimension switches to it only when it scores above ``base`` on each fold's own rows. Only the
    comparison with ``base`` counts, so a member and a vote that score alike cannot split the decision (#106,
    Vernika's run: both beat the base on every fold, yet each fold named a different one best). The scores and the
    vote weights returned are those of the whole of ``part``.
    """
    if base not in members:
        raise ValueError(f"the base {base!r} is not one of the members {sorted(members)}")
    fold_of_row = folds.to_numpy()
    on_fold = [choose(members, part[fold_of_row == fold], dimensions, solution_frame, evaluate) for fold in range(HALVES)]
    whole = choose(members, part, dimensions, solution_frame, evaluate)
    routing, beats_base = {}, {}
    for dimension in whole["dimensions"]:
        candidate = whole["routing"][dimension]
        beats_base[dimension] = [
            candidate != base and fold["scores"][dimension][candidate] > fold["scores"][dimension][base] for fold in on_fold
        ]
        routing[dimension] = candidate if all(beats_base[dimension]) else base
    return {
        **whole,
        "routing": routing,
        "rule": AGREE,
        "base": base,
        "candidates": dict(whole["routing"]),
        "beats_base_on_folds": beats_base,
        "routing_on_folds": [fold["routing"] for fold in on_fold],
    }


def choose_by_rule(
    members: Mapping[str, Records],
    part: pd.DataFrame,
    dimensions: Sequence[str],
    solution_frame: SolutionFrame,
    evaluate: Evaluate,
    rule: str = BEST,
    seed: int = 0,
    folds: pd.Series | None = None,
) -> dict[str, Any]:
    """The routing ``rule`` picks on ``part``; ``agree`` splits ``part`` into product halves unless given ``folds``.

    Under ``agree`` the base is the first member, so the order of ``$ENSEMBLE_MEMBERS`` decides it.
    """
    if rule == BEST:
        return choose(members, part, dimensions, solution_frame, evaluate)
    if rule == AGREE:
        folds = holdout_halves(part, seed) if folds is None else folds
        return agree(members, part, dimensions, solution_frame, evaluate, folds, next(iter(members)))
    raise ValueError(f"the rule must be one of {RULES}, got {rule!r}")


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
    rule: str = BEST,
    seed: int = 0,
) -> dict[str, Any]:
    """A routing chosen on each half (applied to the other one), and the routing chosen on the whole holdout.

    Under ``agree`` a half's routing needs the two product quarters of that half to agree, so no holdout answer is
    scored by a routing that saw it; the whole-holdout routing (dev and final) needs these two halves to agree.
    """
    half_of_row = halves.to_numpy()
    chosen_on_half = [
        choose_by_rule(members, holdout[half_of_row == half], dimensions, solution_frame, evaluate, rule, seed)
        for half in range(HALVES)
    ]
    # Holdout image -> index of the half whose routing answers it: always the other half.
    answered_by = {str(i): HALVES - 1 - int(h) for i, h in zip(holdout[ID_COLUMN], half_of_row, strict=True)}
    return {
        "chosen_on_half": chosen_on_half,
        "answered_by": answered_by,
        "full": choose_by_rule(members, holdout, dimensions, solution_frame, evaluate, rule, seed, folds=halves),
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
    rule: str = BEST

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
        config = {
            "method": "per-dimension routing, chosen on one product half of the holdout and scored on the other (#87)",
            "members": dict(self.member_exps),
            "routing_chosen_on_half": [c["routing"] for c in self.fitted["chosen_on_half"]],
            "routing_full_holdout": self.fitted["full"]["routing"],
            "speed": "images_per_second is the members' combined speed: every image goes through every member",
        }
        if self.rule == AGREE:
            config.update(
                rule=AGREE,
                base=self.fitted["full"]["base"],
                method="per-dimension routing under the halves-agree rule (#106): the base member everywhere, except "
                "where the best other option beats it on both product halves of the choosing data; chosen on one "
                "half of the holdout (it must win on both of that half's product quarters) and scored on the other",
                routing_full_holdout_by_half=self.fitted["full"]["routing_on_folds"],
            )
        return config


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
    rule: str = BEST,
) -> EnsemblePredictor:
    """The predictor from the members' saved holdout answers; speed and model from their EXP rows."""
    if rule not in RULES:
        raise ValueError(f"the rule must be one of {RULES}, got {rule!r}")
    members = load_members(member_exps, out_dir)
    fitted = crossfit(members, holdout, dimensions, solution_frame, evaluate, holdout_halves(holdout, seed), rule, seed)
    rows = {str(row.get("id")): row for row in result_rows}
    member_rows = [rows.get(exp, {}) for exp in member_exps.values()]
    models = sorted({str(row["model"]) for row in member_rows if row.get("model")})
    return EnsemblePredictor(
        members=members,
        member_exps=dict(member_exps),
        fitted=fitted,
        images_per_second=combined_speed([row.get("images_per_second") for row in member_rows]),
        model=", ".join(models) or None,
        rule=rule,
    )


def members_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``$ENSEMBLE_MEMBERS``, or BUILD_PLAN.md's two zero-shot runs when it is unset."""
    environ = os.environ if environ is None else environ
    return parse_members(environ.get(MEMBERS_ENV) or DEFAULT_MEMBERS)


def needed_members(choice: Mapping[str, Any]) -> list[str]:
    """The members a routing reads: those it routes a dimension to, and every member weighted in a dimension that
    takes the vote (the vote counts them all, as it did when ``choose`` scored it)."""
    needed = {source for source in choice["routing"].values() if source != VOTE}
    for dimension, source in choice["routing"].items():
        if source == VOTE:
            needed.update(choice["weights"][dimension])
    return sorted(needed)


def _record(prediction: Prediction) -> dict[str, Any]:
    return {ID_COLUMN: str(prediction.image_id), "labels": dict(prediction.labels),
            "confidence": dict(prediction.confidence or {})}


@dataclass
class LiveEnsemblePredictor:
    """The final run's predictor (#182): every member answers the batch live, then a frozen routing combines them.

    * **The routing is frozen:** it is the json ``choose`` saved on the holdout, never re-chosen here, so the final
      answers follow exactly the routing the EXP row scored. Rechoosing could flip a near-tie dimension.
    * **Deterministic reruns:** with ``cache_dir``, each member's answers are appended to ``<cache_dir>/<member>.jsonl``
      as they arrive and read back first on every later call. A member that is not bit-reproducible (a gateway model
      under server batching) is then asked once per image, and a rerun or a resume after a crash rebuilds the same
      file. The combination itself is pure: the same member answers always give the same file.
    """

    choice: dict[str, Any]
    predictors: dict[str, Any]
    specs: dict[str, str] = field(default_factory=dict)
    cache_dir: Path | None = None
    routing_file: str | None = None
    images_per_second: float | None = None
    _memory: dict[str, Records] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        missing = [name for name in needed_members(self.choice) if name not in self.predictors]
        if missing:
            raise ValueError(f"the routing needs members {missing}, which have no predictor")
        models = sorted({str(m) for m in (getattr(p, "model", None) for p in self.predictors.values()) if m})
        self.model = ", ".join(models) or None
        self.model_revision = None
        self.helpers = tuple(sorted({h for p in self.predictors.values() for h in getattr(p, "helpers", ())}))

    def _cached(self, name: str) -> Records:
        if name not in self._memory:
            path = None if self.cache_dir is None else self.cache_dir / f"{name}.jsonl"
            self._memory[name] = load_records(path) if path is not None and path.is_file() else {}
        return self._memory[name]

    def _answers(self, name: str, batch: Sequence[ImageItem]) -> Records:
        """One member's answers for ``batch``: cached ones first, the rest asked once and cached."""
        with self._lock:
            known = self._cached(name)
            todo = [item for item in batch if str(item.image_id) not in known]
        if todo:
            fresh = {str(p.image_id): _record(p) for p in self.predictors[name].predict(todo)}
            lost = [str(item.image_id) for item in todo if str(item.image_id) not in fresh]
            if lost:
                raise KeyError(f"member {name} gave no answer for {len(lost)} image(s), e.g. {lost[:3]}")
            with self._lock:
                new = {i: r for i, r in fresh.items() if i not in known}
                known.update(new)
                if self.cache_dir is not None and new:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    with (self.cache_dir / f"{name}.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                        handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new.values())
        with self._lock:
            return {str(item.image_id): known[str(item.image_id)] for item in batch}

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        """Every member answers the batch **at the same time**, one thread each (#182: in turn, the final drill ran
        at 1.98 img/s; the 70B alone runs at 2.4). A member's exception propagates, and the combination runs only
        once every member has answered."""
        names = sorted(self.predictors)
        if len(names) == 1:
            answers = {names[0]: self._answers(names[0], batch)}
        else:
            with ThreadPoolExecutor(max_workers=len(names), thread_name_prefix="member") as pool:
                futures = {name: pool.submit(self._answers, name, batch) for name in names}
                answers = {name: futures[name].result() for name in names}
        return [combine(self.choice, answers, str(item.image_id)) for item in batch]

    def config(self) -> dict[str, Any]:
        members = {}
        for name in sorted(self.predictors):
            inner = getattr(self.predictors[name], "config", None)
            members[name] = {"predictor": self.specs.get(name), "config": inner() if callable(inner) else None}
        return {
            "method": "live per-dimension routing (#182): every member answers each image, the frozen routing "
                      "from `ensemble choose` combines them",
            "routing_file": self.routing_file,
            "routing_sha256": _sha256(self.choice),
            "routing": dict(self.choice["routing"]),
            "rule": self.choice.get("rule", BEST),
            "members": members,
            "member_exps": self.choice.get("member_exps"),
            "answer_cache": None if self.cache_dir is None else str(self.cache_dir),
        }


def _sha256(choice: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(choice, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def from_env(environ: Mapping[str, str] | None = None) -> LiveEnsemblePredictor:
    """For ``final_run --predictor galaxus.model.ensemble:from_env`` and ``submit --run`` (#182, #181).

    ``$ENSEMBLE_ROUTING`` is the routing json ``choose`` saved; ``$ENSEMBLE_PREDICTORS`` names one live predictor per
    member the routing reads, under the names ``choose`` used, e.g. ``siglip=galaxus.model.helper:from_env
    70b=galaxus.model.type_prior:from_env``; ``$ENSEMBLE_CACHE`` (optional, one folder per part) keeps every member's
    answers so a rerun is identical. A member the routing never reads is not built.
    """
    from galaxus.serve.items import load_predictor

    environ = os.environ if environ is None else environ
    routing_file, specs = environ.get(ROUTING_ENV), environ.get(PREDICTORS_ENV)
    if not routing_file or not specs:
        raise RuntimeError(f"set ${ROUTING_ENV} (the routing json `python -m galaxus.model.ensemble choose` saved) and "
                           f"${PREDICTORS_ENV} (name=package.module:factory per member, e.g. "
                           "siglip=galaxus.model.helper:from_env 70b=galaxus.model.type_prior:from_env)")
    choice = json.loads(Path(routing_file).read_text(encoding="utf-8"))
    member_specs = parse_members(specs)
    needed = needed_members(choice)
    missing = [name for name in needed if name not in member_specs]
    if missing:
        raise RuntimeError(f"the routing in {routing_file} reads members {missing}: add them to ${PREDICTORS_ENV}")
    cache = environ.get(CACHE_ENV)
    return LiveEnsemblePredictor(
        choice=choice,
        predictors={name: load_predictor(member_specs[name]) for name in needed},
        specs={name: member_specs[name] for name in needed},
        cache_dir=Path(cache) if cache else None,
        routing_file=routing_file,
    )


def summary(fitted: Mapping[str, Any], member_exps: Mapping[str, str], rule: str = BEST) -> list[str]:
    """``choose``'s printout: whole-holdout macro F1 per option, and what each half picked."""
    full = fitted["full"]
    names = [*member_exps, VOTE]
    width = max(len(n) for n in names) + 2
    if rule == AGREE:
        picked = f"   best overall, beats the base on half 0 / half 1 -> routing (base: {full['base']})"
    else:
        picked = "   chosen: whole / half 0 / half 1"
    lines = [f"{'dimension':22}" + "".join(f"{n:>{width}}" for n in names) + picked]
    for d in full["dimensions"]:
        row = "".join(f"{full['scores'][d].get(n, float('nan')):>{width}.3f}" for n in names)
        if rule == AGREE:
            wins = " / ".join("yes" if win else "no" for win in full["beats_base_on_folds"][d])
            lines.append(f"{d:22}{row}   {full['candidates'][d]}: {wins} -> {full['routing'][d]}")
        else:
            halves = " / ".join(c["routing"][d] for c in fitted["chosen_on_half"])
            lines.append(f"{d:22}{row}   {full['routing'][d]} / {halves}")
    spec = " ".join(f"{k}={v}" for k, v in member_exps.items())
    predictor = "ensemble-agree" if rule == AGREE else "ensemble"
    lines += ["", "These columns are chosen and scored on the same images, so they are optimistic. The honest number is",
              f'the EXP row: {MEMBERS_ENV}="{spec}" python -m galaxus.model.eval {predictor}']
    return lines


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
    choose_cmd.add_argument("--rule", choices=RULES, default=BEST,
                            help="best: each dimension's best option; agree (#106): the first member listed, unless "
                                 "both product halves pick the same other option")
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
                               out_dir(), seed=arguments.seed, rule=arguments.rule)
        full = predictor.fitted["full"]
        print("\n".join(summary(predictor.fitted, member_exps, arguments.rule)))
        prefix = "routing-agree-" if arguments.rule == AGREE else "routing-"
        target = arguments.out or out_dir() / "ensemble" / (prefix + "-".join(sorted(member_exps)) + ".json")
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
