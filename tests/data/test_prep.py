"""Issue #27: the prepared-image cache and ``python -m galaxus.data.prep``. Synthetic images only, no network."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import write_image
from PIL import Image as PillowImage

from galaxus.data import prep
from galaxus.data.images import (
    MIRROR_SUFFIX,
    PATCH_SIZE,
    cached_image,
    cached_jpeg,
    fill_image_cache,
    image_cache_dir,
    prepare_image,
    prepared_jpeg,
)

SIZE = 64


@pytest.fixture
def dev_part(galaxus_data):
    """A synthetic ``$GALAXUS_DATA`` with a three-row ``dev.csv`` and its images, shaped like a real part."""
    images = galaxus_data / "images"
    images.mkdir()
    rows = []
    for index in range(3):
        name = f"img_{index}"
        # An asymmetric picture, so a mirrored copy is genuinely different bytes.
        path = write_image(images / f"{name}.png", "RGBA", (40, 30), color=(10 + index * 40, 60, 90, 255))
        with PillowImage.open(path) as picture:
            picture.paste((250, 250, 250, 255), (0, 0, 8, 30))
            picture.save(path)
        rows.append({"image_id": name, "image_path": f"images/{name}.png"})
    pd.DataFrame(rows).to_csv(galaxus_data / "dev.csv", index=False)
    return galaxus_data


def test_the_cache_folder_names_the_patch_size_not_only_the_image_size(galaxus_data):
    """The prepared pixels depend on the patch size, so a different patch size must not reuse this folder."""
    assert image_cache_dir(SIZE) == galaxus_data / "cache" / "images" / f"{SIZE}-p{PATCH_SIZE}"


def test_another_patch_size_gets_its_own_folder(galaxus_data):
    """Since #28 the patch size is real, so a second one keeps a second cache instead of being refused.

    What must never happen is two patch sizes sharing a folder; `test_patch_size.py` covers the pixels.
    """
    assert image_cache_dir(SIZE, patch_size=PATCH_SIZE * 2).name == f"{SIZE}-p{PATCH_SIZE * 2}"
    assert image_cache_dir(SIZE, PATCH_SIZE) != image_cache_dir(SIZE, PATCH_SIZE * 2)


def test_the_patch_size_we_do_use_is_accepted(galaxus_data):
    assert image_cache_dir(SIZE, patch_size=PATCH_SIZE) == image_cache_dir(SIZE)


def test_the_mirrored_copy_has_its_own_name(dev_part):
    """`<image_id>.mirror.jpg`, so test-time augmentation never evicts the original."""
    source = dev_part / "images" / "img_0.png"
    plain = cached_jpeg("img_0", source, image_size=SIZE)
    mirrored = cached_jpeg("img_0", source, mirrored=True, image_size=SIZE)
    folder = image_cache_dir(SIZE)
    assert (folder / "img_0.jpg").read_bytes() == plain
    assert (folder / f"img_0{MIRROR_SUFFIX}.jpg").read_bytes() == mirrored
    assert plain != mirrored


def test_the_cached_pixels_are_the_pixels_prepare_image_gives(dev_part):
    """The acceptance criterion: the cache must not be a different picture from the live path."""
    source = dev_part / "images" / "img_1.png"
    from_cache = cached_jpeg("img_1", source, image_size=SIZE)
    assert from_cache == prepared_jpeg(source, image_size=SIZE)
    # Decode the cached file as it is. Preparing it *again* would re-encode an already prepared JPEG, which is a
    # different picture and not what any lane does with the cache.
    with PillowImage.open(image_cache_dir(SIZE) / "img_1.jpg") as cached:
        assert cached.convert("RGB").tobytes() == prepare_image(source, image_size=SIZE).tobytes()


def test_a_second_run_writes_nothing(dev_part):
    """The acceptance criterion: safe to re-run after a crash."""
    frame = pd.read_csv(dev_part / "dev.csv")
    frame["image_file"] = [str(dev_part / path) for path in frame["image_path"]]
    first = fill_image_cache(frame, image_size=SIZE)
    assert first["written"] == 3
    assert first["skipped"] == 0
    second = fill_image_cache(frame, image_size=SIZE)
    assert second["written"] == 0
    assert second["skipped"] == 3


def test_a_second_run_does_not_rewrite_the_bytes_on_disk(dev_part):
    frame = pd.read_csv(dev_part / "dev.csv")
    frame["image_file"] = [str(dev_part / path) for path in frame["image_path"]]
    fill_image_cache(frame, image_size=SIZE)
    cached = image_cache_dir(SIZE) / "img_0.jpg"
    stamp = cached.stat().st_mtime_ns
    fill_image_cache(frame, image_size=SIZE)
    assert cached.stat().st_mtime_ns == stamp


def test_cached_image_returns_a_path_and_prepares_it_when_missing(dev_part):
    source = dev_part / "images" / "img_2.png"
    target = image_cache_dir(SIZE) / "img_2.jpg"
    assert not target.exists()
    assert cached_image("img_2", source, image_size=SIZE) == target
    assert target.is_file()


def test_cached_image_finds_the_source_by_id_alone(dev_part):
    """A caller holding only an image_id still works: the id is looked up in the parts on disk."""
    assert cached_image("img_1", image_size=SIZE).is_file()


def test_cached_image_refuses_an_id_that_is_in_no_part(dev_part):
    with pytest.raises(KeyError, match="no part on disk"):
        cached_image("not_an_image", image_size=SIZE)


def test_cached_image_does_not_reprepare_what_is_there(dev_part):
    source = dev_part / "images" / "img_0.png"
    first = cached_image("img_0", source, image_size=SIZE)
    stamp = first.stat().st_mtime_ns
    assert cached_image("img_0", source, image_size=SIZE).stat().st_mtime_ns == stamp


def test_an_image_id_that_is_not_a_plain_name_cannot_escape_the_cache(dev_part):
    """The ids come from a competition CSV, so they are untrusted input."""
    source = dev_part / "images" / "img_0.png"
    for attack in ("../escaped", "a/b", "..", ""):
        with pytest.raises(ValueError, match="plain file name"):
            cached_image(attack, source, image_size=SIZE)


def test_workers_produce_the_same_bytes_as_one_thread(dev_part):
    frame = pd.read_csv(dev_part / "dev.csv")
    frame["image_file"] = [str(dev_part / path) for path in frame["image_path"]]
    parallel = fill_image_cache(frame, image_size=SIZE, workers=4)
    assert parallel["written"] == 3
    assert parallel["workers"] == 4
    for _, row in frame.iterrows():
        expected = prepared_jpeg(row["image_file"], image_size=SIZE)
        assert (image_cache_dir(SIZE) / f"{row['image_id']}.jpg").read_bytes() == expected


def test_the_fill_reports_a_rate_and_a_duration(dev_part):
    frame = pd.read_csv(dev_part / "dev.csv")
    frame["image_file"] = [str(dev_part / path) for path in frame["image_path"]]
    counts = fill_image_cache(frame, image_size=SIZE)
    assert counts["seconds"] > 0
    assert counts["images_per_second"] > 0


def test_nothing_written_means_no_invented_rate(dev_part):
    """A run that wrote nothing has no images per second, rather than a zero that reads like a measurement."""
    frame = pd.read_csv(dev_part / "dev.csv")
    frame["image_file"] = [str(dev_part / path) for path in frame["image_path"]]
    fill_image_cache(frame, image_size=SIZE)
    assert fill_image_cache(frame, image_size=SIZE)["images_per_second"] is None


def test_a_missing_source_image_can_be_counted_instead_of_stopping(dev_part):
    frame = pd.DataFrame({"image_id": ["gone"], "image_file": [str(dev_part / "images" / "nothing-here.png")]})
    with pytest.raises(FileNotFoundError):
        fill_image_cache(frame, image_size=SIZE)
    assert fill_image_cache(frame, image_size=SIZE, skip_missing=True)["missing"] == 1


def test_no_part_files_are_left_behind(dev_part):
    frame = pd.read_csv(dev_part / "dev.csv")
    frame["image_file"] = [str(dev_part / path) for path in frame["image_path"]]
    fill_image_cache(frame, image_size=SIZE, workers=4)
    assert list(image_cache_dir(SIZE).glob("*.part")) == []


def test_the_cli_fills_the_cache_and_prints_the_counts(dev_part, capsys):
    assert prep.main(["--part", "dev", "--size", str(SIZE), "--workers", "2"]) == 0
    printed = capsys.readouterr().out
    assert "3 written" in printed
    assert "images/s" in printed
    assert len(list(image_cache_dir(SIZE).glob("*.jpg"))) == 3


def test_the_cli_second_run_skips_everything(dev_part, capsys):
    prep.main(["--part", "dev", "--size", str(SIZE)])
    capsys.readouterr()
    prep.main(["--part", "dev", "--size", str(SIZE)])
    assert "3 already cached" in capsys.readouterr().out


def test_the_cli_mirror_writes_both_copies(dev_part):
    prep.main(["--part", "dev", "--size", str(SIZE), "--mirror"])
    folder = image_cache_dir(SIZE)
    assert len(list(folder.glob("*.jpg"))) == 6
    assert (folder / f"img_0{MIRROR_SUFFIX}.jpg").is_file()
    assert (folder / "img_0.jpg").read_bytes() != (folder / f"img_0{MIRROR_SUFFIX}.jpg").read_bytes()


def test_the_cli_limit_stops_early(dev_part):
    prep.main(["--part", "dev", "--size", str(SIZE), "--limit", "1"])
    assert len(list(image_cache_dir(SIZE).glob("*.jpg"))) == 1


def test_the_cli_refuses_a_part_that_is_not_a_part(dev_part):
    with pytest.raises(SystemExit):
        prep.main(["--part", "holdout", "--size", str(SIZE)])


def test_the_cli_refuses_zero_workers(dev_part):
    with pytest.raises(SystemExit):
        prep.main(["--part", "dev", "--workers", "0"])


def test_the_cli_refuses_a_size_that_is_not_whole_patches(dev_part):
    with pytest.raises(ValueError, match="multiple of"):
        prep.main(["--part", "dev", "--size", str(PATCH_SIZE + 1)])


def test_the_default_worker_count_is_at_least_one(dev_part):
    assert prep.default_workers() >= 1
