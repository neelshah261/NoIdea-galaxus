"""``galaxus.data`` re-exports the lane's public names lazily (PEP 562); these guard that contract.

The laziness is deliberate: see the module docstring of ``galaxus.data``. Importing the submodules eagerly would
preload ``galaxus.data.validate``, and ``python -m galaxus.data.validate`` would then run it a second time as
``__main__`` with a second copy of ``SubmissionInvalid``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import galaxus.data


def test_all_matches_the_export_table():
    """``__all__`` is spelled out for the linters, so nothing may drift from ``_EXPORTS``.

    Names, not order. These two agreed under ``sorted()`` until #92 exported the first CamelCase name
    (``DimensionConfusions``), and plain ``sorted()`` puts every capitalised name before every lowercase one,
    which is not the order ruff's RUF022 wants. The ordering is ruff's to enforce - it is a required check and it
    fixes the list itself - so this asserts the invariant that matters instead of repeating the sort.
    """
    assert sorted(galaxus.data.__all__) == sorted(galaxus.data._EXPORTS)


def test_all_is_a_list_of_unique_names():
    assert isinstance(galaxus.data.__all__, list)
    assert len(galaxus.data.__all__) == len(set(galaxus.data.__all__))


@pytest.mark.parametrize("name", sorted(galaxus.data._EXPORTS))
def test_each_name_resolves_to_the_object_its_submodule_defines(name):
    module = importlib.import_module(f"galaxus.data.{galaxus.data._EXPORTS[name]}")
    assert getattr(galaxus.data, name) is getattr(module, name)


def test_dir_lists_exactly_the_public_names():
    assert sorted(galaxus.data.__dir__()) == sorted(galaxus.data.__all__)


def test_unknown_attribute_raises_attribute_error():
    missing = "unmistakably_not_an_export"
    with pytest.raises(AttributeError, match=missing):
        getattr(galaxus.data, missing)


def test_importing_the_package_does_not_preload_the_submodules():
    """The reason the re-exports are lazy: ``import galaxus.data`` must not pull in ``validate``."""
    code = "import galaxus.data, sys; print([m for m in sys.modules if m.startswith('galaxus.data.')])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", result.stdout
