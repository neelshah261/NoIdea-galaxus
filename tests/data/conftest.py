"""Synthetic fixtures for the data lane. No competition data, no network (AGENTS.md "Work → PR" 5)."""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image as PillowImage

from galaxus.data.paths import GALAXUS_DATA_ENV

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")

# Invented class names: the real ones are read at runtime from a source file, never written down here
# (AGENTS.md "Hard rules"). These only have to be self-consistent.
FAKE_CLASSES = {
    "subject": ["alpha", "beta"],
    "background": ["white", "color"],
    "perspective": ["front", "back"],
    "individuals": ["present", "none"],
    "text_classification": ["seal", "none"],
    "text_language": ["German", "none"],
}


@pytest.fixture
def allowed() -> dict[str, frozenset[str]]:
    """Allowed classes per dimension, synthetic."""
    return {dimension: frozenset(names) for dimension, names in FAKE_CLASSES.items()}


@pytest.fixture
def galaxus_data(tmp_path, monkeypatch):
    """An empty ``$GALAXUS_DATA`` in ``tmp_path``."""
    directory = tmp_path / "galaxus-data"
    directory.mkdir()
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(directory))
    return directory


@pytest.fixture
def sample_submission() -> pd.DataFrame:
    """A three-row sample submission holding the constant answers."""
    return pd.DataFrame(
        {
            "image_id": ["img_00001", "img_00002", "img_00003"],
            **{dimension: [FAKE_CLASSES[dimension][0]] * 3 for dimension in DIMENSIONS},
        }
    )


@pytest.fixture
def train_table() -> pd.DataFrame:
    """A tiny train table: four products, seven images, strings as the CSV reader returns them."""
    rows = [
        ("img_1", "p1", "alpha", "human", "true"),
        ("img_2", "p1", "beta", "human", "true"),
        ("img_3", "p2", "alpha", "model", "true"),
        ("img_4", "p2", "beta", "human", "false"),
        ("img_5", "p3", "alpha", "human", "true"),
        ("img_6", "p4", "beta", "model", "false"),
        ("img_7", "p4", "alpha", "human", "true"),
    ]
    frame = pd.DataFrame(
        {
            "image_id": [row[0] for row in rows],
            "product_id": [row[1] for row in rows],
        }
    )
    for dimension in DIMENSIONS:
        frame[dimension] = [row[2] if dimension == "subject" else FAKE_CLASSES[dimension][0] for row in rows]
        frame[f"{dimension}_source"] = [row[3] for row in rows]
        frame[f"{dimension}_relevant"] = [row[4] for row in rows]
    return frame


def write_image(path, mode, size, color=None, save_format="PNG"):
    """A synthetic image file in ``mode``, written to ``path``."""
    fill = color if color is not None else {"RGBA": (0, 0, 0, 0), "LA": (0, 0), "RGB": (10, 20, 30)}[mode]
    image = PillowImage.new(mode, size, fill)
    image.save(path, format=save_format)
    return path


def write_palette_image_with_transparency(path, size):
    """A P-mode PNG whose palette index 0 is transparent, and opaque red everywhere else in the palette."""
    image = PillowImage.new("P", size, 0)
    image.putpalette([255, 0, 0] * 256)
    image.save(path, format="PNG", transparency=0)
    return path
