"""bench/ocr_language.py's decision rule and roles file, without OCR or a language model (#101 item 9).

Tests may name classes: they check the roles file and the rule do what the docstring says.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "bench" / "ocr_language.py"
spec = importlib.util.spec_from_file_location("galaxus_bench_ocr_language", MODULE_PATH)
ocr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ocr
spec.loader.exec_module(ocr)

CLASSES = ["Dutch", "English", "French", "German", "Italian", "none", "other"]
ROLES = {"language_dimension": "text_language", "no_text_class": "none", "other_class": "other"}


def words(*pairs):
    return [{"text": text, "confidence": confidence} for text, confidence in pairs]


def test_too_little_confident_text_is_no_text():
    def never(text):
        raise AssertionError("no text should not reach the detector")

    assert ocr.decide(words(("Ab", 0.9), ("Hallo Welt", 0.05)), never, CLASSES, ROLES, 8, 0.3) == "none"


def test_a_detected_class_language_is_matched_by_name():
    assert ocr.decide(words(("Bitte nicht werfen", 0.9)), lambda t: "GERMAN", CLASSES, ROLES, 8, 0.3) == "German"


def test_a_language_that_is_not_a_class_is_other_and_so_is_no_detection():
    assert ocr.decide(words(("Prosze nie rzucac", 0.9)), lambda t: "POLISH", CLASSES, ROLES, 8, 0.3) == "other"
    assert ocr.decide(words(("xqzt vbnm plkj", 0.9)), lambda t: None, CLASSES, ROLES, 8, 0.3) == "other"


def test_only_words_above_the_confidence_floor_are_read():
    seen = []
    ocr.decide(words(("keep this text", 0.8), ("drop", 0.1)), lambda t: seen.append(t) or "ENGLISH",
               CLASSES, ROLES, 4, 0.5)
    assert seen == ["keep this text"]


@pytest.mark.real_contract
def test_the_roles_file_names_real_classes():
    from galaxus.data.classes import allowed_classes

    roles = ocr.load_roles(allowed=allowed_classes())
    assert roles["no_text_class"] in allowed_classes()[roles["language_dimension"]]


def test_a_roles_file_with_an_unknown_class_is_refused(tmp_path):
    bad = tmp_path / "roles.json"
    bad.write_text('{"language_dimension": "d", "no_text_class": "none", "other_class": "zzz"}', encoding="utf-8")
    with pytest.raises(ValueError, match="other_class"):
        ocr.load_roles(bad, allowed={"d": ["none", "other"]})
