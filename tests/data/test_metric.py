"""Issue #5: the metric wrapper loads metric.py from $GALAXUS_DATA and never from the repository.

The real metric.py is Kaggle data, so these tests use a stub with the same two entry points. The test that needs
the real file is skipped when ``$GALAXUS_DATA`` has no metric.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from galaxus.data import metric as metric_module
from galaxus.data.metric import METRIC_MODULE_NAME, evaluate, load_metric, score
from galaxus.data.paths import GALAXUS_DATA_ENV, MissingDataDir

STUB = """
DIMENSIONS = ("subject", "background")
VALID_CLASSES = {"subject": ["alpha", "beta"], "background": ["white", "color"]}


def score(solution, submission, id_column):
    return 0.5


def evaluate(solution, submission, id_column):
    return {"overall": {"f1_macro": 0.5}}
"""


@pytest.fixture(autouse=True)
def fresh_cache():
    """The module caches the loaded metric; every test starts without it."""
    metric_module._loaded = None
    sys.modules.pop(METRIC_MODULE_NAME, None)
    yield
    metric_module._loaded = None
    sys.modules.pop(METRIC_MODULE_NAME, None)


def test_metric_py_is_not_in_the_repository():
    """AGENTS.md "Galaxus product rules" 3: metric.py is loaded from $GALAXUS_DATA, never committed."""
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "metric.py").exists()


def test_unset_data_dir_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(GALAXUS_DATA_ENV, raising=False)
    with pytest.raises(MissingDataDir):
        load_metric()


def test_missing_metric_py_is_a_clear_error(galaxus_data):
    with pytest.raises(FileNotFoundError, match="metric.py"):
        load_metric()


def test_the_stub_is_loaded_from_the_data_dir(galaxus_data):
    (galaxus_data / "metric.py").write_text(STUB, encoding="utf-8")
    loaded = load_metric()
    assert loaded.DIMENSIONS == ("subject", "background")
    assert Path(loaded.__file__) == galaxus_data / "metric.py"


def test_it_is_loaded_under_a_name_that_cannot_shadow_our_module(galaxus_data):
    (galaxus_data / "metric.py").write_text(STUB, encoding="utf-8")
    load_metric()
    assert METRIC_MODULE_NAME in sys.modules
    assert sys.modules["galaxus.data.metric"] is metric_module


def test_score_and_evaluate_are_forwarded(galaxus_data):
    (galaxus_data / "metric.py").write_text(STUB, encoding="utf-8")
    frame = pd.DataFrame({"image_id": ["img_1"], "subject": ["alpha"]})
    assert score(frame, frame) == 0.5
    assert evaluate(frame, frame) == {"overall": {"f1_macro": 0.5}}


def test_a_broken_metric_py_is_not_left_half_imported(galaxus_data):
    (galaxus_data / "metric.py").write_text("raise ValueError('broken')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken"):
        load_metric()
    assert METRIC_MODULE_NAME not in sys.modules


@pytest.mark.skipif(
    not (os.environ.get(GALAXUS_DATA_ENV) and Path(os.environ[GALAXUS_DATA_ENV], "metric.py").is_file()),
    reason=f"the real metric.py is only available where ${GALAXUS_DATA_ENV} holds the Kaggle files",
)
def test_the_real_metric_exposes_the_entry_points_we_use():
    loaded = load_metric()
    assert len(loaded.DIMENSIONS) == 6
    assert callable(loaded.score) and callable(loaded.evaluate)
