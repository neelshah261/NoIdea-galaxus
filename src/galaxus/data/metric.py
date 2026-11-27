"""The official metric, loaded from ``$GALAXUS_DATA`` at runtime.

``metric.py`` is Kaggle data: AGENTS.md "Galaxus product rules" 3 forbids copying it into the repository, so it is
imported from the file in ``$GALAXUS_DATA`` instead. It is the same code the leaderboard runs, so a local number
and a leaderboard number differ only in which images they score (see
[challenge_description.md, "Scoring locally"](../../../challenge_description.md#scoring-locally)).

It is loaded under a private module name, so it can never shadow or be shadowed by this module.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any

import pandas as pd

from galaxus.data.classes import ID_COLUMN
from galaxus.data.paths import data_file

#: The name ``metric.py`` is registered under in :data:`sys.modules`, chosen not to collide with anything.
METRIC_MODULE_NAME = "galaxus_official_metric"

_loaded: ModuleType | None = None


def load_metric() -> ModuleType:
    """``metric.py`` from ``$GALAXUS_DATA``, loaded once and cached.

    Raises:
        MissingDataDir: if ``$GALAXUS_DATA`` is unset or unusable.
        FileNotFoundError: if ``metric.py`` is not in ``$GALAXUS_DATA``.
    """
    global _loaded
    if _loaded is not None:
        return _loaded
    path = data_file("metric.py")
    spec = importlib.util.spec_from_file_location(METRIC_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} cannot be imported as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[METRIC_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[METRIC_MODULE_NAME]
        raise
    _loaded = module
    return module


def score(solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = ID_COLUMN) -> float:
    """The leaderboard number: the mean of the six macro F1 scores."""
    return load_metric().score(solution, submission, id_column)


def evaluate(solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = ID_COLUMN) -> dict[str, Any]:
    """F1, precision and recall per dimension and per class, as on the Kaggle Results page."""
    return load_metric().evaluate(solution, submission, id_column)
