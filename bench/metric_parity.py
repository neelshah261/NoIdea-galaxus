"""#68: metric parity on the real data. Does our scored-cell mask equal the official rules, and do scores agree?

``metric.py`` (Kaggle's) scores exactly the cells where the solution holds a label; the hosts build their solution
from the rules in [challenge_description.md, "Which cells are scored"](../challenge_description.md#which-cells-are-scored).
Our local solution comes from ``galaxus.data.split.solution_frame``, which uses train.csv's ``_source`` and
``_relevant`` flags. This script rebuilds the mask **independently from the written rules** (applied to human labels
only, as the description says) and compares, cell by cell and score by score, on the official holdout and on all of
train. Run where ``$GALAXUS_DATA`` holds the Kaggle files; it prints counts only, never labels or image ids.

    python bench/metric_parity.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# challenge_description.md "Which cells are scored", quoted rule by rule. Class names as written there.
NO_CONTEXT_SUBJECTS = {"logo", "placeholder", "energy_label"}         # bg, perspective, individuals, text_cls
INFOGRAPHIC = "infographic"                                           # bg, perspective, individuals
NO_PERSPECTIVE_SUBJECTS = {"delivery_scope", "detail_closeup", "product_grouping"}
SCENE = "scene_lifestyle"                                             # background -> no subject, no perspective
TEXT_LANGUAGE_SUBJECTS = {"infographic", "packaging", "package_and_product", "full_product"}
NO_OVERLAY = "none"


def rule_mask(frame: pd.DataFrame, dimensions: tuple[str, ...]) -> dict[str, pd.Series]:
    """Scored cells per the written rules, triggered by human labels only."""
    human = {d: frame[f"{d}_source"].eq("human") for d in dimensions}
    subject = frame["subject"].where(human["subject"])
    background = frame["background"].where(human["background"])
    text = frame["text_classification"].where(human["text_classification"])
    masks = {}
    for d in dimensions:
        scored = human[d].copy()
        if d in ("background", "perspective", "individuals", "text_classification"):
            scored &= ~subject.isin(NO_CONTEXT_SUBJECTS)
        if d in ("background", "perspective", "individuals"):
            scored &= subject.ne(INFOGRAPHIC)
        if d == "perspective":
            scored &= ~subject.isin(NO_PERSPECTIVE_SUBJECTS)
        if d in ("subject", "perspective"):
            scored &= background.ne(SCENE)
        if d == "text_language":
            overlay = text.notna() & text.ne(NO_OVERLAY)
            scored &= overlay | subject.isin(TEXT_LANGUAGE_SUBJECTS)
        masks[d] = scored.fillna(False).astype(bool)
    return masks


def solution_from(frame: pd.DataFrame, masks: dict[str, pd.Series], dimensions: tuple[str, ...]) -> pd.DataFrame:
    solution = frame[["image_id", *dimensions]].copy().astype({d: object for d in dimensions})
    for d in dimensions:
        solution.loc[~masks[d], d] = None
    return solution


def submissions(frame: pd.DataFrame, dimensions: tuple[str, ...], classes: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Three test submissions: the majority answer, the labels with 20% noise, and uniformly random classes."""
    rng = np.random.default_rng(0)
    majority = frame[["image_id"]].copy()
    noisy = frame[["image_id"]].copy()
    random = frame[["image_id"]].copy()
    for d in dimensions:
        options = sorted(classes[d])
        majority[d] = frame[d].mode().iloc[0]
        truth = frame[d].astype(str).to_numpy()
        flip = rng.random(len(frame)) < 0.2
        noisy[d] = np.where(flip, rng.choice(options, len(frame)), truth)
        random[d] = rng.choice(options, len(frame))
    return {"majority": majority, "labels + 20% noise": noisy, "random": random}


def main() -> int:
    from galaxus.data import classes, load, metric, split

    dimensions = tuple(classes.dimensions())
    allowed = {d: sorted(v) for d, v in classes.allowed_classes().items()}
    train = load.read_train()
    _, holdout = split.split_holdout(train)
    ok = True
    for name, frame in (("official holdout", holdout), ("all of train", train)):
        ours = {d: split.scored_mask(frame, d) for d in dimensions}
        rules = rule_mask(frame, dimensions)
        print(f"\n== {name}: {len(frame)} images")
        print(f"{'dimension':22}{'scored (ours)':>15}{'scored (rules)':>16}{'only ours':>11}{'only rules':>12}")
        for d in dimensions:
            only_ours = int((ours[d] & ~rules[d]).sum())
            only_rules = int((rules[d] & ~ours[d]).sum())
            ok &= only_ours == 0 and only_rules == 0
            print(f"{d:22}{int(ours[d].sum()):>15}{int(rules[d].sum()):>16}{only_ours:>11}{only_rules:>12}")
        ours_solution = solution_from(frame, ours, dimensions)
        rules_solution = solution_from(frame, rules, dimensions)
        local_solution = split.solution_frame(frame, dimensions)
        same_as_wrapper = ours_solution.equals(local_solution)
        ok &= same_as_wrapper
        print(f"our mask == galaxus.data.split.solution_frame: {same_as_wrapper}")
        for label, submission in submissions(frame, dimensions, allowed).items():
            a = metric.score(ours_solution, submission)
            b = metric.score(rules_solution, submission)
            ok &= abs(a - b) < 1e-12
            print(f"  metric.py score, {label:20} ours {a:.6f}  rules {b:.6f}  diff {a - b:+.2e}")
    print("\nPARITY OK" if ok else "\nPARITY MISMATCH: see the counts above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
