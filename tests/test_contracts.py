"""contracts/classes.json is challenge_description.md "Classes", verbatim, and it is what the code reads.

Every expected value here is parsed from challenge_description.md at test time: nothing is typed from memory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "classes.json"
DESCRIPTION = ROOT / "challenge_description.md"

pytestmark = pytest.mark.real_contract


def _description() -> str:
    return DESCRIPTION.read_text(encoding="utf-8")


def _classes_from_description() -> dict[str, list[str]]:
    section = _description().split("### Classes", 1)[1].split("\n### ", 1)[0]
    found: dict[str, list[str]] = {}
    for line in section.splitlines():
        row = re.match(r"\|\s*([a-z_]+)\s*\|\s*(.+?)\s*\|\s*$", line)
        if row and row.group(1) != "dimension":
            found[row.group(1)] = [name.strip() for name in row.group(2).split(",")]
    assert found, "the Classes table of challenge_description.md was not found"
    return found


def _submission_columns() -> list[str]:
    section = _description().split("### Submission file", 1)[1]
    header = re.search(r"^\s*image_id,([a-z_,]+)\s*$", section, re.MULTILINE)
    assert header, "the header line of the Submission file section was not found"
    return header.group(1).split(",")


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_the_contract_is_the_classes_table_verbatim_in_order():
    assert _contract()["dimensions"] == _classes_from_description()


def test_the_dimensions_follow_the_submission_file_columns():
    assert list(_contract()["dimensions"]) == _submission_columns()


def test_no_dimension_repeats_a_class_and_none_is_lowercase():
    for dimension, names in _contract()["dimensions"].items():
        assert len(names) == len(set(names)), dimension
        assert "None" not in names, f"{dimension}: a capital None is read as an empty cell"


def test_the_class_count_matches_the_description():
    total = sum(len(names) for names in _contract()["dimensions"].values())
    stated = re.search(r"(\d+) classes in total", _description())
    assert stated, "challenge_description.md no longer states the class total"
    assert total == int(stated.group(1))


def test_the_roles_name_a_real_dimension_and_class_and_quote_their_source():
    contract = _contract()
    roles = contract["roles"]
    dimension, cls = roles["overlay_dimension"], roles["no_overlay_class"]
    assert cls in contract["dimensions"][dimension]
    quote = roles["source"].strip('"')
    assert quote in _description(), "the role's quote must appear word for word in challenge_description.md"


def test_the_code_reads_the_contract(monkeypatch):
    monkeypatch.delenv("GALAXUS_DATA", raising=False)
    from galaxus.data import classes

    assert classes.CONTRACTS_CLASSES == CONTRACT
    expected = _classes_from_description()
    assert classes.dimensions() == tuple(expected)
    assert classes.allowed_classes() == {d: frozenset(names) for d, names in expected.items()}


@pytest.mark.skipif(not os.environ.get("GALAXUS_DATA"), reason="needs the Kaggle files: metric.py in $GALAXUS_DATA")
def test_the_contract_matches_the_scoring_code():
    from galaxus.data.metric import load_metric

    metric = load_metric()
    contract = _contract()["dimensions"]
    assert list(metric.DIMENSIONS) == list(contract)
    assert {d: set(metric.VALID_CLASSES[d]) for d in contract} == {d: set(n) for d, n in contract.items()}
