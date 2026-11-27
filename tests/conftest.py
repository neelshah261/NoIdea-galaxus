"""Test-wide setup owned by design (top-level ``tests/``), applied to every lane's tests.

``contracts/classes.json`` holds the real dimensions and classes, and ``galaxus.data.classes`` reads it before
``metric.py``. Most lane tests deliberately run on invented classes (a fake ``metric.py`` in a synthetic
``$GALAXUS_DATA``) to prove that no class name is typed in code. So for every test the contract is hidden by default
and the classes come from whatever ``$GALAXUS_DATA`` the test builds, exactly as before the contract existed.

A test that needs the real contract says so with ``@pytest.mark.real_contract`` (see ``tests/test_contracts.py``).
Production code is untouched: outside the tests the contract is always read first.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_contract: read the real contracts/classes.json instead of hiding it")


@pytest.fixture(autouse=True)
def _hide_the_class_contract_unless_asked(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path_factory):
    if request.node.get_closest_marker("real_contract"):
        return
    from galaxus.data import classes

    monkeypatch.setattr(classes, "CONTRACTS_CLASSES", tmp_path_factory.getbasetemp() / "no-contract" / "classes.json")
