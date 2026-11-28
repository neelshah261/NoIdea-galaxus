"""The Kaggle notebooks copy the official holdout split to stay self-contained on Kaggle (#118, #130).

Notebooks are not run in ci, so this test runs each notebook's own split code on a synthetic train table and checks
that it holds out exactly the products ``galaxus.data.split`` holds out. A drift would let a notebook train on the
holdout its rounds are scored on. No competition data, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from galaxus.data import split

TRAIN = Path(__file__).resolve().parents[2] / "train"


def _code(name: str) -> str:
    cells = json.loads((TRAIN / name).read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def _constants(code: str) -> tuple[float, int]:
    found = re.search(r"^HOLDOUT_FRAC, HOLDOUT_SEED = ([0-9.]+), ([0-9]+)$", code, re.MULTILINE)
    assert found, "the notebook no longer names HOLDOUT_FRAC, HOLDOUT_SEED"
    return float(found.group(1)), int(found.group(2))


def _train(products: int = 60, per_product: int = 3) -> pd.DataFrame:
    ids = range(products * per_product)
    return pd.DataFrame({"image_id": [f"i{k}" for k in ids], "product_id": [f"p{k // per_product:03d}" for k in ids]})


def _official_holdout(train: pd.DataFrame) -> set[str]:
    return set(split.split_holdout(train)[1]["product_id"])


def test_the_self_training_notebook_holds_out_the_official_products():
    code = _code("self_training_heads.ipynb")
    frac, seed = _constants(code)
    assert (frac, seed) == (split.HOLDOUT_FRAC, split.HOLDOUT_SEED)
    function = re.search(r"^def split_holdout\(.*?(?=^\S)", code, re.MULTILINE | re.DOTALL)
    assert function, "the notebook no longer defines split_holdout"
    namespace = {"HOLDOUT_FRAC": frac, "HOLDOUT_SEED": seed, "PRODUCT": "product_id"}
    exec(function.group(0), namespace)  # noqa: S102 (the notebook's own function, in the repo, on synthetic rows)
    train = _train()
    assert set(namespace["split_holdout"](train)[1]["product_id"]) == _official_holdout(train)


def test_the_pilot_notebook_holds_out_the_official_products():
    code = _code("one_look_pilot.ipynb")
    frac, seed = _constants(code)
    expression = re.search(r"^holdout_products = (set\(.*\))$", code, re.MULTILINE)
    assert expression, "the pilot no longer computes holdout_products in one line"
    train = _train()
    held = eval(expression.group(1), {"train": train, "HOLDOUT_FRAC": frac, "HOLDOUT_SEED": seed})
    assert held == _official_holdout(train)
