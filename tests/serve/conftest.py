"""Fixtures for the serve-lane tests. No competition data, no network (AGENTS.md "Work → PR" 5)."""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image as PillowImage
from serve_helpers import DEV_IDS, DIMENSIONS, FAKE_CLASSES, FINAL_IDS, Item


@pytest.fixture
def items(tmp_path):
    """Three synthetic images on disk, as items."""
    result = []
    for index, image_id in enumerate(DEV_IDS):
        path = tmp_path / f"{image_id}.png"
        PillowImage.new("RGBA", (40 + index, 30), (0, 0, 0, 0)).save(path)
        result.append(Item(image_id, str(path), {"title": f"product {index}"}))
    return result


@pytest.fixture
def galaxus_data(tmp_path, monkeypatch):
    """A synthetic ``$GALAXUS_DATA``: sample_submission.csv, dev.csv and three dev images."""
    directory = tmp_path / "galaxus-data"
    (directory / "images").mkdir(parents=True)
    monkeypatch.setenv("GALAXUS_DATA", str(directory))
    ids = DEV_IDS + FINAL_IDS
    pd.DataFrame(
        {"image_id": ids, **{d: [FAKE_CLASSES[d][0]] * len(ids) for d in DIMENSIONS}}
    ).to_csv(directory / "sample_submission.csv", index=False)
    rows = []
    for index, image_id in enumerate(DEV_IDS):
        PillowImage.new("RGB", (32, 32), (index, 0, 0)).save(directory / "images" / f"{image_id}.jpg")
        rows.append({"image_id": image_id, "image_path": f"images/{image_id}.jpg", "product_title": f"p{index}",
                     "brand": "" if index == 0 else "acme"})
    pd.DataFrame(rows).to_csv(directory / "dev.csv", index=False)
    # train.csv holds every fake class once, so the validator (which reads the allowed classes from the data files
    # until contracts/classes.json exists) knows all of them.
    train = {"image_id": [f"img_t{n}" for n in range(3)], "product_id": ["p1", "p2", "p3"]}
    for dimension in DIMENSIONS:
        train[dimension] = FAKE_CLASSES[dimension]
        train[f"{dimension}_source"] = ["human"] * 3
        train[f"{dimension}_relevant"] = ["true"] * 3
    pd.DataFrame(train).to_csv(directory / "train.csv", index=False)
    return directory


@pytest.fixture(autouse=True)
def _isolated_from_the_real_data(monkeypatch):
    """#68: ``galaxus.data.metric.load_metric`` caches metric.py once per process, whatever ``$GALAXUS_DATA`` says.

    On a machine where ``$GALAXUS_DATA`` holds the real Kaggle files, a test that resolves the classes without the
    synthetic ``galaxus_data`` fixture loads the *real* metric.py, and every later test then validates its invented
    class names against the real ones and fails. Each serve test starts with an empty cache; the cache itself is the
    data lane's (reported on #6).
    """
    # Serve tests are synthetic: run them exactly as ci does, without the machine's real $GALAXUS_DATA. Tests that
    # need a data folder build their own with the ``galaxus_data`` fixture, which sets the variable again.
    monkeypatch.delenv("GALAXUS_DATA", raising=False)
    try:
        from galaxus.data import metric
    except ImportError:
        yield
        return
    metric.clear_metric_cache()  # the public reset since #84 (#90)
    yield
