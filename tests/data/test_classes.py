"""The dimensions and allowed classes are read from a source file, never hard-coded here."""

from __future__ import annotations

import json

import pytest
from conftest import DIMENSIONS, FAKE_CLASSES

from galaxus.data import classes as classes_module
from galaxus.data.classes import ClassesUnavailable, allowed_classes, dimensions


@pytest.fixture(autouse=True)
def no_contract(tmp_path, monkeypatch):
    """By default no contract file exists, as on main today."""
    monkeypatch.setattr(classes_module, "CONTRACTS_CLASSES", tmp_path / "contracts" / "classes.json")


def write_contract(path, mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping), encoding="utf-8")


def test_the_contract_wins_when_design_has_added_it(tmp_path, monkeypatch, galaxus_data):
    contract = tmp_path / "contracts" / "classes.json"
    write_contract(contract, FAKE_CLASSES)
    monkeypatch.setattr(classes_module, "CONTRACTS_CLASSES", contract)
    assert dimensions() == DIMENSIONS
    assert allowed_classes()["background"] == frozenset(FAKE_CLASSES["background"])


def test_a_contract_nested_under_dimensions_is_also_accepted(tmp_path, monkeypatch):
    contract = tmp_path / "contracts" / "classes.json"
    write_contract(contract, {"dimensions": FAKE_CLASSES})
    monkeypatch.setattr(classes_module, "CONTRACTS_CLASSES", contract)
    assert dimensions() == DIMENSIONS


def test_without_a_contract_the_sample_submission_and_train_labels_are_used(galaxus_data, sample_submission):
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    train = sample_submission.copy()
    train["subject"] = ["beta", "beta", "beta"]
    train.to_csv(galaxus_data / "train.csv", index=False)
    assert dimensions() == DIMENSIONS
    assert allowed_classes()["subject"] == frozenset({"alpha", "beta"})


def test_the_sample_submission_alone_is_enough(galaxus_data, sample_submission):
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    assert allowed_classes()["subject"] == frozenset({"alpha"})


def test_with_no_source_at_all_the_error_says_what_to_do(monkeypatch):
    from galaxus.data.paths import GALAXUS_DATA_ENV

    monkeypatch.delenv(GALAXUS_DATA_ENV, raising=False)
    with pytest.raises(ClassesUnavailable, match="contracts/classes.json"):
        dimensions()
