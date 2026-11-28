"""train/one_look_pilot.ipynb copies the official holdout constants for Kaggle self-containment (#118).

Notebooks are not run in ci, so this test reads the copy and fails if it ever drifts from galaxus.data.split:
a silent drift would let the pilot train on holdout products.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from galaxus.data import split

NOTEBOOK = Path(__file__).resolve().parents[2] / "train" / "one_look_pilot.ipynb"


def _notebook_code() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_the_notebook_holdout_constants_equal_the_official_ones():
    copy = re.search(r"^HOLDOUT_FRAC, HOLDOUT_SEED = ([0-9.]+), ([0-9]+)$", _notebook_code(), re.MULTILINE)
    assert copy, "the notebook no longer names HOLDOUT_FRAC, HOLDOUT_SEED"
    assert (float(copy.group(1)), int(copy.group(2))) == (split.HOLDOUT_FRAC, split.HOLDOUT_SEED)


def test_the_notebook_samples_with_the_named_constants_not_literals():
    code = _notebook_code()
    assert "sample(frac=HOLDOUT_FRAC, random_state=HOLDOUT_SEED)" in code
    assert "sample(frac=0.2, random_state=0)" not in code
