"""D (#211): teacher labels on the dev images as extra training rows for the heads, plus self-training.

Built by system S (Swarup's desktop) on branch ``model/ad-d-teacher`` (#211, #213).

Why it should help: ``gpt-6.2-astra`` with EXP-0049's prompt scores 0.858 on 1,000 random holdout images, where the
70B scores 0.472 (docs/method.md). The labels are learnable from the image, so 5,261 dev images labelled by that
teacher are more, and mostly clean, training rows for every head. The FAQ allows any model in development "to create
training targets or pseudo-labels on the dev images" (FAQ.md:30-31); at runtime only open models classify.

The teacher pass already exists (#172, #173): a person runs it, since it is paid (AGENTS.md rule 7), from a worktree
**without** a ``.env`` and with the teacher's own key (#172: the client rotates every ``LLM_API_KEY*`` it finds):

    LLM_NAME=gpt-6.2-astra LLM_BASE_URL=<OpenAI-compatible URL> LLM_API_KEY=<Parth's teacher key> \\
    ZERO_SHOT_PRESET=zero-shot-observe-first-no-context \\
      uv run python -m galaxus.serve.submit --exp EXP-0080 --part dev --run galaxus.model.zero_shot:from_preset
    # -> $GALAXUS_DATA/out/predictions/EXP-0080/dev.jsonl  (never upload that submission file)

The contract (fixed by #211 so A and D fit together):

* :func:`teacher_rows`: the teacher's dev answers as rows in train.csv's shape (``image_id``, ``image_file``,
  ``product_id``, the catalog fields, one label per dimension), with ``<dimension>_source = "teacher"`` and
  ``<dimension>_relevant`` from the challenge's scoring rules applied to the teacher's own labels
  (challenge_description.md "Which cells are scored"). The rules come from data (``bench/scoring_rules.json``, #82),
  never typed here. A cell under ``min_confidence``, without a label or outside the class set is left out
  (``_source = "dropped"``).
* :func:`training_mask`: the cells a head trains on: human, teacher or agreeing, and relevant. **Scoring never
  changes:** EXP rows still score only the human cells of the holdout through ``split.scored_mask``, and the heads'
  calibration bias is still fitted out of fold on human cells only.
* :func:`agreeing_rows`: self-training. Dev cells where every member agrees above ``min_confidence``, in the same
  shape, ``<dimension>_source = "agree"``. With ``HELPER_SELF_TRAIN=1`` the members are the teacher (when given), the
  heads' own first round on dev, and any ``HELPER_SELF_TRAIN_MEMBERS`` (the 70B's dev answers on Neel's laptop). With
  a teacher this keeps the teacher cells the heads confirm (cleaner labels); without one it is classic self-training.
* **Wiring (the only change in helper.py):** ``helper.factory(..., teacher=...)`` appends these rows to the training
  rows; ``HELPER_TEACHER=<dev.jsonl>`` and ``HELPER_SELF_TRAIN=1`` turn them on in ``from_env``/``fused_from_env``;
  eval names ``helper-siglip-teacher`` and ``helper-fused-teacher``; whichever of A and D merges second adds
  ``helper-fused-text-teacher``.
* **With A (#211):** the teacher rows are dev images, which carry the catalog fields, so A's text features reach
  them through ``ContextFeatures.transform``; the text model never trains on teacher labels.
* **Never:** a final image, a holdout image, or a typed class name. The teacher is in MODELS.md as development only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

#: ``HELPER_TEACHER=<path to the teacher's dev.jsonl>`` adds the teacher's dev rows to the heads' training rows.
TEACHER_ENV = "HELPER_TEACHER"
#: ``HELPER_SELF_TRAIN=1`` adds the dev cells where the members agree (with a teacher: the teacher and the heads).
SELF_TRAIN_ENV = "HELPER_SELF_TRAIN"
#: ``HELPER_SELF_TRAIN_MEMBERS="name=<dev.jsonl> ..."``: more members whose dev answers must agree too (the 70B's).
MEMBERS_ENV = "HELPER_SELF_TRAIN_MEMBERS"
#: The confidence floors of a teacher cell and of an agreeing cell.
TEACHER_FLOOR_ENV = "HELPER_TEACHER_MIN_CONFIDENCE"
AGREE_FLOOR_ENV = "HELPER_SELF_TRAIN_MIN_CONFIDENCE"
TEACHER_SOURCE = "teacher"
AGREE_SOURCE = "agree"
HUMAN_SOURCE = "human"
#: A cell left out (no label, outside the class set, under the floor, or not relevant): never trained on.
DROPPED_SOURCE = "dropped"
DEFAULT_AGREE_CONFIDENCE = 0.9
#: The challenge's scored-cell rules as data (#82): the only place a dimension or class of a rule is named.
RULES_FILE = Path(__file__).resolve().parents[3] / "bench" / "scoring_rules.json"

Records = Mapping[str, Mapping[str, Any]]


def load_rules(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """``bench/scoring_rules.json``: the written scored-cell rules, as bench/metric_parity.py reads them."""
    return json.loads(Path(path or RULES_FILE).read_text(encoding="utf-8"))


def relevance(labels: pd.DataFrame, dimensions: Sequence[str], rules: Mapping[str, Any]) -> dict[str, pd.Series]:
    """``<dimension>_relevant`` from the written rules, applied to ``labels`` (the teacher's or the members' own).

    A cell without a label is not relevant. A missing label never triggers a ``not_scored_when`` rule, and a
    ``scored_only_if`` condition on a missing label is not met: the cell is left out rather than guessed in."""
    out = {}
    for d in dimensions:
        keep = labels[d].notna() if d in labels else pd.Series(False, index=labels.index)
        for rule in rules.get("not_scored_when", []):
            when = rule["when"]
            if d in rule["not_scored"] and when["dimension"] in labels:
                keep &= ~labels[when["dimension"]].isin(when["in"])
        for rule in rules.get("scored_only_if", []):
            if rule["dimension"] != d:
                continue
            met = pd.Series(False, index=labels.index)
            for condition in rule["any_of"]:
                if condition["dimension"] not in labels:
                    continue
                value = labels[condition["dimension"]]
                if "in" in condition:
                    met |= value.isin(condition["in"])
                if "not_in" in condition:
                    met |= value.notna() & ~value.isin(condition["not_in"])
            keep &= met
        out[d] = keep.fillna(False).astype(bool)
    return out


def _confidence(record: Mapping[str, Any], dimension: str) -> float:
    """A record's confidence for one cell; a teacher that reports none counts as sure (its label is all we have)."""
    value = (record.get("confidence") or {}).get(dimension)
    return 1.0 if value is None else float(value)


def _rows(labels: Mapping[str, Mapping[str, Any]], dev: pd.DataFrame, dimensions: Sequence[str], source: str,
          rules: Mapping[str, Any]) -> pd.DataFrame:
    """Dev rows (catalog fields kept) with one label per dimension, ``_source`` and ``_relevant`` per the rules."""
    keep = dev[dev["image_id"].astype(str).isin(set(labels))].copy()
    ids = keep["image_id"].astype(str).tolist()
    frame = pd.DataFrame({d: [labels[i].get(d) for i in ids] for d in dimensions}, index=keep.index, dtype=object)
    relevant = relevance(frame, dimensions, rules)
    for d in dimensions:
        keep[d] = frame[d]
        keep[f"{d}_relevant"] = relevant[d]
        keep[f"{d}_source"] = [source if ok else DROPPED_SOURCE for ok in relevant[d]]
    return keep.reset_index(drop=True)


def teacher_rows(records: Records, dev: pd.DataFrame, min_confidence: float = 0.0,
                 dimensions: Sequence[str] | None = None, allowed: Mapping[str, Any] | None = None,
                 rules: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """The teacher's dev answers as training rows in train.csv's shape (see the module docstring)."""
    from galaxus.data import classes

    dimensions = tuple(dimensions or classes.dimensions())
    allowed = allowed or classes.allowed_classes()
    labels: dict[str, dict[str, Any]] = {}
    for image_id, record in records.items():
        given = record.get("labels") or {}
        labels[str(image_id)] = {d: given[d] for d in dimensions
                                 if given.get(d) in allowed[d] and _confidence(record, d) >= min_confidence}
    return _rows(labels, dev, dimensions, TEACHER_SOURCE, rules if rules is not None else load_rules())


def training_mask(frame: pd.DataFrame, dimension: str) -> pd.Series:
    """The cells a head trains on: a human, teacher or agreeing label where the dimension is relevant. A model
    prelabel and a dropped cell never count. Scoring is not this: EXP rows score ``split.scored_mask`` cells only."""
    source = frame[f"{dimension}_source"].astype(str)
    relevant = frame[f"{dimension}_relevant"]
    if relevant.dtype != bool:
        relevant = relevant.astype(str).str.strip().str.lower().isin(("true", "1"))
    return source.isin((HUMAN_SOURCE, TEACHER_SOURCE, AGREE_SOURCE)) & relevant


def agreeing_rows(members: Mapping[str, Records], dev: pd.DataFrame, min_confidence: float = DEFAULT_AGREE_CONFIDENCE,
                  dimensions: Sequence[str] | None = None, allowed: Mapping[str, Any] | None = None,
                  rules: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Self-training rows: the dev cells where every member gives the same class above ``min_confidence``.

    An image counts only where every member answered it. A member without a confidence (the teacher) counts as
    sure, so with a teacher this keeps the teacher's cells that the heads confirm."""
    from galaxus.data import classes

    dimensions = tuple(dimensions or classes.dimensions())
    allowed = allowed or classes.allowed_classes()
    rules = rules if rules is not None else load_rules()
    names = sorted(members)
    if not names:
        return _rows({}, dev, dimensions, AGREE_SOURCE, rules)
    common = set.intersection(*({str(i) for i in members[n]} for n in names))
    labels: dict[str, dict[str, Any]] = {}
    for image_id in sorted(common):
        records = [members[n][image_id] for n in names]
        agreed = {}
        for d in dimensions:
            answers = {(r.get("labels") or {}).get(d) for r in records}
            label = next(iter(answers)) if len(answers) == 1 else None
            if label in allowed[d] and all(_confidence(r, d) >= min_confidence for r in records):
                agreed[d] = label
        labels[image_id] = agreed
    return _rows(labels, dev, dimensions, AGREE_SOURCE, rules)


def read_members(spec: str) -> dict[str, dict[str, dict[str, Any]]]:
    """``"name=<dev.jsonl> name=<dev.jsonl>"`` as ``{name: records}`` (``runner.read_predictions``)."""
    from galaxus.serve.runner import read_predictions

    members = {}
    for part in spec.split():
        name, _, path = part.partition("=")
        if not name or not path:
            raise ValueError(f"{MEMBERS_ENV}: expected name=<dev.jsonl>, got {part!r}")
        members[name] = read_predictions(path)
    return members


def counts(extra: pd.DataFrame, dimensions: Sequence[str]) -> dict[str, int]:
    """Training cells added per dimension (counts only: safe to print and to record in the EXP row)."""
    if extra.empty:
        return dict.fromkeys(dimensions, 0)
    return {d: int(training_mask(extra, d).sum()) for d in dimensions}


class TeacherSetup:
    """What ``helper.factory(teacher=...)`` calls: the extra dev rows for one fit. Deterministic: no sampling."""

    def __init__(self, teacher: str | None = None, self_train: bool = False, members: str = "",
                 teacher_floor: float = 0.0, agree_floor: float = DEFAULT_AGREE_CONFIDENCE,
                 records: Records | None = None, read_dev: Callable[[], pd.DataFrame] | None = None) -> None:
        self.teacher, self.self_train, self.members = teacher, self_train, members
        self.teacher_floor, self.agree_floor = teacher_floor, agree_floor
        self._records = records
        self._read_dev = read_dev

    def dev_rows(self) -> pd.DataFrame:
        """dev.csv's rows (the only part D ever labels: never final, never the holdout)."""
        if self._read_dev is not None:
            return self._read_dev()
        from galaxus.data import load

        return load.read_dev()

    def teacher_records(self) -> dict[str, Mapping[str, Any]]:
        if self._records is not None:
            return {str(k): v for k, v in self._records.items()}
        if not self.teacher:
            return {}
        from galaxus.serve.runner import read_predictions

        return read_predictions(self.teacher)

    def describe(self) -> dict[str, Any]:
        """For the EXP row's config: file names and switches only, never a label or an image id."""
        return {"teacher": Path(self.teacher).parent.name + "/" + Path(self.teacher).name if self.teacher else None,
                "self_train": self.self_train, "members": [p.partition("=")[0] for p in self.members.split()],
                "teacher_min_confidence": self.teacher_floor, "agree_min_confidence": self.agree_floor}

    def extra_rows(self, dev: pd.DataFrame, data: Any,
                   first_round: Callable[[pd.DataFrame], Records] | None = None) -> pd.DataFrame:
        """The teacher's rows; with self-training, the cells where the teacher (if any), the heads' first round on
        dev and any extra members all agree."""
        dimensions, allowed = tuple(data.dimensions()), data.allowed_classes()
        records = self.teacher_records()
        if not self.self_train:
            return teacher_rows(records, dev, self.teacher_floor, dimensions, allowed)
        members: dict[str, Records] = {}
        if records:
            members["teacher"] = records
        if first_round is not None:
            members["heads"] = first_round(dev)
        if self.members:
            members.update(read_members(self.members))
        return agreeing_rows(members, dev, self.agree_floor, dimensions, allowed)


def from_environ(environ: Mapping[str, str] | None = None) -> TeacherSetup | None:
    """The D switches as one setup, or ``None`` when neither ``HELPER_TEACHER`` nor ``HELPER_SELF_TRAIN`` is on."""
    environ = os.environ if environ is None else environ
    path = environ.get(TEACHER_ENV, "").strip()
    self_train = environ.get(SELF_TRAIN_ENV, "") not in ("", "0")
    if not path and not self_train:
        return None
    if path and not Path(path).is_file():
        raise FileNotFoundError(f"{TEACHER_ENV}={path}: no such file (the teacher's dev.jsonl, EXP-0080)")
    return TeacherSetup(teacher=path or None, self_train=self_train, members=environ.get(MEMBERS_ENV, "").strip(),
                        teacher_floor=float(environ.get(TEACHER_FLOOR_ENV, "") or 0.0),
                        agree_floor=float(environ.get(AGREE_FLOOR_ENV, "") or DEFAULT_AGREE_CONFIDENCE))
