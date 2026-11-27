"""The dimensions and allowed classes are read from a source file, never hard-coded here."""

from __future__ import annotations

import json

import pytest
from conftest import DIMENSIONS, FAKE_CLASSES

from galaxus.data import classes as classes_module
from galaxus.data.classes import ClassesUnavailable, allowed_classes, dimensions


@pytest.fixture(autouse=True)
def no_contract(request, tmp_path, monkeypatch):
    """By default this file runs as if no contract existed, so the fallback chain stays covered.

    Since #80 ``contracts/classes.json`` is on ``main``, so "no contract" has to be arranged rather than assumed.
    A test that wants the real one says ``@pytest.mark.real_contract``, as ``tests/conftest.py`` defines it.
    """
    if request.node.get_closest_marker("real_contract"):
        return
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


# --- #81: the two roles --------------------------------------------------------------------------------------


@pytest.mark.real_contract
def test_the_roles_come_from_the_contract_and_name_real_things():
    """``contracts/classes.json`` "roles" is the only source for these two names (AGENTS.md "Hard rules")."""
    from galaxus.data.classes import allowed_classes, dimensions, no_overlay_class, overlay_dimension

    dimension, cls = overlay_dimension(), no_overlay_class()
    assert dimension in dimensions(), f"{dimension!r} is not one of {dimensions()}"
    assert cls in allowed_classes()[dimension], f"{cls!r} is not a class of {dimension}"


def test_without_the_contract_the_roles_raise_and_name_the_file():
    """The contract is hidden for every test that does not ask for it, so this is the default path."""
    from galaxus.data.classes import ClassesUnavailable, no_overlay_class, overlay_dimension

    for accessor in (overlay_dimension, no_overlay_class):
        with pytest.raises(ClassesUnavailable, match="classes.json"):
            accessor()
