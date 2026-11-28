"""#216: an unreadable image keeps the constant row instead of stopping the run, and ``--csv`` for new batches.

Synthetic images and invented classes only (serve_helpers); the broken file is a few bytes that are not an image.
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest
from PIL import Image as PillowImage
from serve_helpers import DEV_IDS, DIMENSIONS, FAKE_CLASSES, Item, Pred

from galaxus.serve import runner, submit


class OpensEveryImage:
    """Opens each file with PIL, as the live members do, then answers the second fake class."""

    def __init__(self, error: type[BaseException] | None = None) -> None:
        self.error = error
        self.calls = 0

    def predict(self, items):
        self.calls += 1
        if self.error is not None:
            raise self.error("the method itself failed")
        for item in items:
            with PillowImage.open(item.path) as image:
                image.load()
        return [Pred(item.image_id, {d: FAKE_CLASSES[d][1] for d in DIMENSIONS}) for item in items]


def pictures(folder, count, broken=(), prefix="img_x"):
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(count):
        path = folder / f"{prefix}{k}.png"
        if k in broken:
            path.write_bytes(b"not an image at all")
        else:
            PillowImage.new("RGB", (8, 8), (k, 0, 0)).save(path)
        out.append(Item(f"{prefix}{k}", str(path)))
    return out


def sample_for(ids):
    ids = list(ids)
    return pd.DataFrame({"image_id": ids, **{d: [FAKE_CLASSES[d][2]] * len(ids) for d in DIMENSIONS}})


def test_one_broken_image_in_a_batch_of_16_gives_15_predictions_and_1_constant_row(tmp_path):
    items = pictures(tmp_path / "img", 16, broken={5})
    lines = []
    report = runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=16, root=tmp_path / "out",
                        log=lines.append, fallback=runner.constant_fallback(sample_for(i.image_id for i in items)))
    records = runner.read_predictions(runner.predictions_path("EXP-0001", "dev", tmp_path / "out"))
    assert report.complete and report.fallbacks == 1 and len(records) == 16
    constant = [r for r in records.values() if r.get("fallback") == runner.FALLBACK_MARK]
    assert len(constant) == 1 and constant[0]["labels"] == {d: FAKE_CLASSES[d][2] for d in DIMENSIONS}
    assert sum(r["labels"]["subject"] == FAKE_CLASSES["subject"][1] for r in records.values()) == 15
    text = "\n".join(lines)
    assert "1 image(s) could not be read and keep the constant answer" in text
    assert "img_x5" not in text  # counted, never named (product rule 3)


def test_a_restart_does_not_retry_the_constant_row(tmp_path):
    items = pictures(tmp_path / "img", 4, broken={0})
    fallback = runner.constant_fallback(sample_for(i.image_id for i in items))
    runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=4, root=tmp_path, log=None, fallback=fallback)
    again = runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=4, root=tmp_path, log=None,
                       fallback=fallback)
    assert again.already_done == 4 and again.predicted == 0


def test_any_other_error_still_stops_the_run(tmp_path):
    items = pictures(tmp_path / "img", 4)
    with pytest.raises(RuntimeError, match="the method itself failed"):
        runner.run(OpensEveryImage(RuntimeError), items, "EXP-0001", "dev", batch_size=4, root=tmp_path, log=None,
                   fallback=runner.constant_fallback(sample_for(i.image_id for i in items)))


def test_without_a_fallback_an_unreadable_image_stops_the_run_as_before(tmp_path):
    items = pictures(tmp_path / "img", 4, broken={1})
    with pytest.raises(OSError):
        runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=4, root=tmp_path, log=None)


def test_too_many_unreadable_images_stop_the_run(tmp_path):
    items = pictures(tmp_path / "img", 6, broken={0, 1, 2, 3})
    with pytest.raises(runner.TooManyFallbacks):
        runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=6, root=tmp_path, log=None,
                   fallback=runner.constant_fallback(sample_for(i.image_id for i in items)), max_fallbacks=3)


def test_parallel_workers_fall_back_too(tmp_path):
    items = pictures(tmp_path / "img", 12, broken={2, 9})
    report = runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=3, root=tmp_path, log=None,
                        workers=3, fallback=runner.constant_fallback(sample_for(i.image_id for i in items)))
    assert report.complete and report.fallbacks == 2


def test_the_constant_row_merges_into_a_file_that_validates(galaxus_data):
    from galaxus.data.validate import read_submission

    items = pictures(galaxus_data / "imgs", 3, prefix="img_d")
    (galaxus_data / "imgs" / "img_d0.png").write_bytes(b"broken")
    items = [Item(image_id, item.path) for image_id, item in zip(DEV_IDS, items, strict=True)]
    report = runner.run(OpensEveryImage(), items, "EXP-0001", "dev", batch_size=3, log=None,
                        fallback=runner.default_fallback(None))
    assert report.fallbacks == 1
    labels = runner.read_labels(runner.predictions_path("EXP-0001", "dev"))
    path = submit.write_file(labels, "with_fallback.csv")
    frame = read_submission(path).set_index("image_id")
    sample = pd.read_csv(galaxus_data / "sample_submission.csv").set_index("image_id")
    assert frame.loc[DEV_IDS[0]].tolist() == sample.loc[DEV_IDS[0]].tolist()  # the constant answer of the sample


# ---------------------------------------------------------------- gap 2: --csv for a new batch


def table(folder, broken=()):
    """A dev.csv-shaped table of four generated images, image_path relative to the table."""
    rows = []
    (folder / "pics").mkdir(parents=True)
    for k in range(4):
        path = folder / "pics" / f"new_{k}.png"
        if k in broken:
            path.write_bytes(b"\x00\x01 not a png")
        else:
            PillowImage.new("RGB", (8, 8), (0, k, 0)).save(path)
        rows.append({"image_id": f"new_{k}", "image_path": f"pics/new_{k}.png", "product_title": f"t{k}"})
    target = folder / "Batch 2026-11-28.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def test_csv_part_name_and_items(tmp_path):
    path = table(tmp_path)
    assert runner.csv_part_name(path) == "csv-Batch-2026-11-28"
    items = runner.items_for_csv(path)
    assert [i.image_id for i in items] == [f"new_{k}" for k in range(4)]
    assert all(str(tmp_path / "pics") in i.path for i in items)
    with pytest.raises(FileNotFoundError):
        runner.items_for_csv(tmp_path / "missing.csv")


def test_submit_csv_runs_end_to_end_into_a_six_column_file_with_the_broken_image_constant(galaxus_data, capsys,
                                                                                          monkeypatch):
    path = table(galaxus_data / "batch", broken={3})
    monkeypatch.setattr("galaxus.serve.items.load_predictor", lambda spec: OpensEveryImage())
    assert submit.main(["--exp", "EXP-0001", "--csv", str(path), "--run", "unused:unused"]) == 0
    out = capsys.readouterr().out
    assert "not a Kaggle file" in out and "kaggle competitions submit" not in out
    written = galaxus_data / "out" / "submissions" / "EXP-0001_csv-Batch-2026-11-28.csv"
    with written.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["image_id", *DIMENSIONS]
    assert [r[0] for r in rows[1:]] == [f"new_{k}" for k in range(4)]
    assert rows[4][1:] == [FAKE_CLASSES[d][0] for d in DIMENSIONS]  # the sample's constant row
    assert rows[1][1:] == [FAKE_CLASSES[d][1] for d in DIMENSIONS]


def test_write_table_refuses_a_missing_prediction_or_an_unknown_class(galaxus_data):
    good = {d: FAKE_CLASSES[d][1] for d in DIMENSIONS}
    with pytest.raises(KeyError):
        submit.write_table({"a": good}, ["a", "b"], "t.csv")
    with pytest.raises(ValueError, match="not a class"):
        submit.write_table({"a": {**good, "subject": "made_up"}}, ["a"], "t.csv")
    assert not (galaxus_data / "out" / "submissions" / "t.csv").exists()


@pytest.mark.parametrize("error", [ConnectionError, TimeoutError])
def test_a_network_failure_is_not_an_unreadable_image_and_stops_the_run(tmp_path, error):
    items = pictures(tmp_path / "img", 4)
    with pytest.raises(error):
        runner.run(OpensEveryImage(error), items, "EXP-0001", "dev", batch_size=4, root=tmp_path, log=None,
                   fallback=runner.constant_fallback(sample_for(i.image_id for i in items)))


def test_is_image_error_tells_a_broken_file_from_the_gateway():
    import json
    import urllib.error

    from PIL import UnidentifiedImageError

    assert runner.is_image_error(UnidentifiedImageError("x")) and runner.is_image_error(OSError("truncated"))
    assert runner.is_image_error(ValueError("bad mode")) and runner.is_image_error(FileNotFoundError("gone"))
    for error in (urllib.error.URLError("down"), json.JSONDecodeError("x", "y", 0), ConnectionResetError(),
                  TimeoutError(), RuntimeError("cuda")):
        assert not runner.is_image_error(error), error
