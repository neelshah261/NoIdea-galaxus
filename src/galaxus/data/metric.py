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
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from galaxus.data.classes import ID_COLUMN
from galaxus.data.paths import data_file

#: The name ``metric.py`` is registered under in :data:`sys.modules`, chosen not to collide with anything.
METRIC_MODULE_NAME = "galaxus_official_metric"

#: Loaded metrics, keyed by the resolved path of the ``metric.py`` they came from. A single global cached the
#: first file forever, so pointing ``$GALAXUS_DATA`` at another folder kept returning the first folder's metric
#: and its classes (#73).
_loaded: dict[Path, ModuleType] = {}


def load_metric() -> ModuleType:
    """``metric.py`` from ``$GALAXUS_DATA``, loaded once per file and cached by its path.

    Called twice with the same ``$GALAXUS_DATA``, as on Saturday, it reads and executes the file once. Pointed at
    a different ``$GALAXUS_DATA``, it loads that folder's file rather than handing back the first one.

    Raises:
        MissingDataDir: if ``$GALAXUS_DATA`` is unset or unusable.
        FileNotFoundError: if ``metric.py`` is not in ``$GALAXUS_DATA``.
    """
    # Resolved before the cache is consulted, because the path is the key: where $GALAXUS_DATA points now is what
    # decides which metric this is, not which one happened to be loaded first.
    path = data_file("metric.py")
    key = Path(path).resolve()
    cached = _loaded.get(key)
    if cached is not None:
        return cached
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
    # The module name is deliberately kept: it shadows nothing. Two different metric.py files in one process
    # therefore both register under it, the later overwriting the earlier in sys.modules, while the module
    # objects themselves stay distinct and cached per path.
    _loaded[key] = module
    return module


def score(solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = ID_COLUMN) -> float:
    """The leaderboard number: the mean of the six macro F1 scores."""
    return load_metric().score(solution, submission, id_column)


def evaluate(solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = ID_COLUMN) -> dict[str, Any]:
    """F1, precision and recall per dimension and per class, as on the Kaggle Results page."""
    return load_metric().evaluate(solution, submission, id_column)
