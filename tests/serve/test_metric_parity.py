"""bench/metric_parity.py's rule engine on bench/scoring_rules.json, with tiny synthetic frames (#68, #82).

Tests may name dimensions and classes: they check the rules file says what challenge_description.md says.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "bench" / "metric_parity.py"
spec = importlib.util.spec_from_file_location("galaxus_bench_metric_parity", MODULE_PATH)
parity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parity
spec.loader.exec_module(parity)

DIMS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")


def frame(rows):
    """Rows of (labels, sources) -> a train-like frame."""
    records = []
    for i, (labels, sources) in enumerate(rows):
        record = {"image_id": f"i{i}"}
        for d in DIMS:
            record[d] = labels.get(d, "x")
            record[f"{d}_source"] = sources.get(d, "human")
        records.append(record)
    return pd.DataFrame(records)


def test_a_logo_leaves_only_the_subject_scored():
    masks = parity.rule_mask(frame([({"subject": "logo"}, {})]), DIMS, parity.load_rules())
    assert masks["subject"][0] and not any(masks[d][0] for d in ("background", "perspective", "individuals",
                                                                  "text_classification"))


def test_a_prelabel_never_triggers_a_rule():
    masks = parity.rule_mask(frame([({"subject": "logo"}, {"subject": "model"})]), DIMS, parity.load_rules())
    assert not masks["subject"][0] and masks["background"][0] and masks["perspective"][0]


def test_text_language_needs_an_overlay_or_a_text_bearing_subject():
    rows = [
        ({"subject": "full_product", "text_classification": "none"}, {}),
        ({"subject": "logo", "text_classification": "promotional_text"}, {}),
        ({"subject": "accessories", "text_classification": "none"}, {}),
    ]
    masks = parity.rule_mask(frame(rows), DIMS, parity.load_rules())
    assert masks["text_language"].tolist() == [True, True, False]


def test_scene_lifestyle_unscores_subject_and_perspective():
    masks = parity.rule_mask(frame([({"background": "scene_lifestyle"}, {})]), DIMS, parity.load_rules())
    assert not masks["subject"][0] and not masks["perspective"][0] and masks["individuals"][0]


def test_the_rules_file_is_checked_against_the_classes():
    every_class = {d: ["x"] for d in DIMS}
    with pytest.raises(ValueError, match="not a class of subject"):
        parity.load_rules(allowed=every_class)
