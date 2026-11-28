"""#218 (umbrella #214): a new batch of images. Any table can be extracted (``--csv``), and one unreadable image is skipped and
counted, never named, instead of stopping the whole run.

Synthetic images and tables under ``tmp_path``. No model, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from galaxus.model import heads
from galaxus.types import ImageItem


def test_extract_skips_an_unreadable_image_and_keeps_going(tmp_path):
    store = heads.FeatureStore(tmp_path / "store")
    items = [ImageItem(image_id=f"img_{k}", path="unused") for k in range(5)]

    def image_bytes(item):
        if item.image_id == "img_2":
            raise OSError("cannot identify image file")  # what PIL raises for a broken file
        return b"jpeg"

    logs = []
    computed = heads.extract(lambda images, prompt: np.ones((1, 1, 3)), items, store, "csv-new", image_bytes,
                             log=logs.append)
    assert computed == 4
    assert set(store.ids("csv-new")) == {"img_0", "img_1", "img_3", "img_4"}
    assert any("1 image(s) could not be read" in line for line in logs)
    assert not any("img_2" in line for line in logs)  # counted, never named (AGENTS.md rule 3)


def test_errors_of_the_model_itself_still_stop_the_run(tmp_path):
    store = heads.FeatureStore(tmp_path / "store")

    def broken_forward(images, prompt):
        raise RuntimeError("CUDA out of memory")

    try:
        heads.extract(broken_forward, [ImageItem(image_id="a", path="u")], store, "dev", lambda item: b"x", log=None)
    except RuntimeError:
        pass
    else:  # pragma: no cover - the assertion below explains the failure
        raise AssertionError("a failing model must not be skipped like an unreadable image")


def test_any_table_of_new_images_is_a_source(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "a.png")
    table = tmp_path / "batch_0612.csv"
    pd.DataFrame({"image_id": ["a"], "image_path": ["images/a.png"], "product_title": ["Kettle"]}).to_csv(
        table, index=False)
    frame, part = heads.extract_source(None, str(table))
    assert part == "csv-batch_0612"
    assert frame.loc[0, "image_file"] == str(tmp_path / "images" / "a.png")  # resolved next to the table
    assert frame.loc[0, "product_title"] == "Kettle"
