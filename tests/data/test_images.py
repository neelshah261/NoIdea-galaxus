"""Issue #4: the image pipeline. Synthetic images only."""

from __future__ import annotations

import pytest
from conftest import write_image, write_palette_image_with_transparency
from PIL import Image as PillowImage

from galaxus.data.images import (
    IMAGE_SIZE,
    MAX_SIDE_FACTOR,
    MIRROR_SUFFIX,
    PATCH_SIZE,
    cached_jpeg,
    fill_image_cache,
    image_cache_dir,
    native_aspect_image,
    opaque_rgb_on_white,
    prepare_image,
    prepared_jpeg,
    vision_tokens,
)

TRANSPARENT_MODES = ("RGBA", "LA")


@pytest.mark.parametrize("mode", TRANSPARENT_MODES)
def test_transparent_pixels_become_white_not_black(tmp_path, mode):
    """FAQ.md "Transparent PNGs turn black": a plain convert("RGB") would make these black."""
    path = write_image(tmp_path / f"{mode}.png", mode, (64, 64))
    with PillowImage.open(path) as image:
        rgb = opaque_rgb_on_white(image)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((32, 32)) == (255, 255, 255)
    assert rgb.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_palette_transparency_becomes_white(tmp_path):
    """A P-mode image carries its transparency in ``info``, not in its mode."""
    path = write_palette_image_with_transparency(tmp_path / "p.png", (64, 64))
    with PillowImage.open(path) as image:
        assert image.mode == "P"
        assert "transparency" in image.info
        rgb = opaque_rgb_on_white(image)
    assert rgb.getpixel((32, 32)) == (255, 255, 255)


def test_opaque_image_is_unchanged_apart_from_mode(tmp_path):
    path = write_image(tmp_path / "rgb.png", "RGB", (64, 64), color=(10, 20, 30))
    with PillowImage.open(path) as image:
        rgb = opaque_rgb_on_white(image)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((32, 32)) == (10, 20, 30)


def test_half_transparent_image_keeps_the_opaque_half(tmp_path):
    image = PillowImage.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(32, 64):
        for y in range(64):
            image.putpixel((x, y), (10, 20, 200, 255))
    path = tmp_path / "half.png"
    image.save(path)
    with PillowImage.open(path) as opened:
        rgb = opaque_rgb_on_white(opened)
    assert rgb.getpixel((8, 32)) == (255, 255, 255)
    assert rgb.getpixel((56, 32)) == (10, 20, 200)


@pytest.mark.parametrize("size", [(100, 100), (640, 480), (33, 97), (2000, 10), (7, 5), (1024, 1024)])
def test_sides_are_multiples_of_the_patch_size(size):
    prepared = native_aspect_image(PillowImage.new("RGB", size, (255, 255, 255)), IMAGE_SIZE)
    assert prepared.width % PATCH_SIZE == 0
    assert prepared.height % PATCH_SIZE == 0
    assert prepared.width >= PATCH_SIZE
    assert prepared.height >= PATCH_SIZE


def test_aspect_ratio_is_kept_within_one_patch():
    source = PillowImage.new("RGB", (640, 480), (255, 255, 255))
    prepared = native_aspect_image(source, IMAGE_SIZE)
    assert prepared.width / prepared.height == pytest.approx(640 / 480, abs=0.05)
    assert prepared.width * prepared.height == pytest.approx(IMAGE_SIZE**2, rel=0.1)


def test_very_elongated_image_is_capped_on_its_longest_side():
    prepared = native_aspect_image(PillowImage.new("RGB", (2000, 10), (255, 255, 255)), IMAGE_SIZE)
    assert max(prepared.size) <= MAX_SIDE_FACTOR * IMAGE_SIZE
    assert min(prepared.size) == PATCH_SIZE


@pytest.mark.parametrize("image_size", [384, 512, 768])
def test_image_size_is_a_parameter(tmp_path, image_size):
    """We ablate 384 / 512 / 768, so the size must be settable, with the same function train and predict."""
    path = write_image(tmp_path / "rgb.png", "RGB", (600, 400))
    prepared = prepare_image(path, image_size=image_size)
    assert prepared.width % PATCH_SIZE == 0 and prepared.height % PATCH_SIZE == 0
    assert prepared.width * prepared.height == pytest.approx(image_size**2, rel=0.15)
    assert vision_tokens(prepared) == (prepared.width // PATCH_SIZE) * (prepared.height // PATCH_SIZE)


@pytest.mark.parametrize("image_size", [0, 7, 500, 512.0, -32])
def test_an_image_size_that_is_not_a_patch_multiple_is_refused(tmp_path, image_size):
    path = write_image(tmp_path / "rgb.png", "RGB", (64, 64))
    with pytest.raises(ValueError, match="image_size"):
        prepared_jpeg(path, image_size=image_size)


def test_prepared_output_is_rgb_jpeg_bytes(tmp_path):
    path = write_image(tmp_path / "rgba.png", "RGBA", (200, 100))
    data = prepared_jpeg(path)
    assert data.startswith(b"\xff\xd8\xff")
    assert prepare_image(path).mode == "RGB"


def test_mirroring_flips_horizontally_only(tmp_path):
    image = PillowImage.new("RGB", (128, 64), (255, 255, 255))
    for y in range(64):
        for x in range(16):
            image.putpixel((x, y), (0, 0, 0))
    path = tmp_path / "left_black.png"
    image.save(path)
    straight = prepare_image(path, image_size=512)
    mirrored = prepare_image(path, mirrored=True, image_size=512)
    assert straight.size == mirrored.size
    middle = straight.height // 2
    assert straight.getpixel((2, middle))[0] < 128
    assert mirrored.getpixel((2, middle))[0] > 128
    assert mirrored.getpixel((mirrored.width - 3, middle))[0] < 128


def test_transparency_survives_the_jpeg_round_trip_as_white(tmp_path):
    path = write_image(tmp_path / "rgba.png", "RGBA", (256, 256))
    prepared = prepare_image(path)
    assert min(prepared.getpixel((prepared.width // 2, prepared.height // 2))) >= 250


# The image cache: BASE_ARCHITECTURE.md "Data layout" puts prepared JPEGs in
# $GALAXUS_DATA/cache/images/<size>/, and B1 is done when `just prep` fills it.


def an_image(tmp_path, name: str = "one.png"):
    """A synthetic RGBA image on disk, and the (image_id, path) pair for it."""
    path = tmp_path / name
    write_image(path, "RGBA", (70, 40), (10, 200, 30, 0))
    return name.split(".")[0], path


def test_the_cache_directory_is_under_galaxus_data_and_named_by_size(galaxus_data):
    assert image_cache_dir(512) == galaxus_data / "cache" / "images" / f"512-p{PATCH_SIZE}"
    assert image_cache_dir(512).is_dir()


def test_each_size_gets_its_own_cache_directory(galaxus_data):
    """We ablate 384 / 512 / 768, so one size must never overwrite another's pixels."""
    assert image_cache_dir(384) != image_cache_dir(768)


def test_a_bad_image_size_is_refused_before_a_directory_appears(galaxus_data):
    with pytest.raises(ValueError, match="multiple of"):
        image_cache_dir(500)


def test_the_first_call_writes_the_cache_file_and_the_second_reads_it(tmp_path, galaxus_data):
    image_id, path = an_image(tmp_path)
    first = cached_jpeg(image_id, path, image_size=64)
    cached = image_cache_dir(64) / f"{image_id}.jpg"
    assert cached.is_file()
    assert cached.read_bytes() == first

    path.unlink()  # the source is gone, so a second call can only come from the cache
    assert cached_jpeg(image_id, path, image_size=64) == first


def test_the_cached_bytes_are_the_same_as_preparing_directly(tmp_path, galaxus_data):
    image_id, path = an_image(tmp_path)
    assert cached_jpeg(image_id, path, image_size=64) == prepared_jpeg(path, image_size=64)


def test_the_mirror_is_cached_under_its_own_name(tmp_path, galaxus_data):
    """Test-time augmentation must not evict the original."""
    path = tmp_path / "lopsided.png"
    image = PillowImage.new("RGB", (64, 32), (10, 200, 30))
    image.paste((240, 240, 240), (0, 0, 16, 32))  # a light stripe down one side only
    image.save(path)

    plain = cached_jpeg("lopsided", path, image_size=64)
    mirrored = cached_jpeg("lopsided", path, mirrored=True, image_size=64)
    assert plain != mirrored
    assert (image_cache_dir(64) / "lopsided.jpg").read_bytes() == plain
    assert (image_cache_dir(64) / f"lopsided{MIRROR_SUFFIX}.jpg").read_bytes() == mirrored


def test_refresh_rewrites_a_stale_entry(tmp_path, galaxus_data):
    image_id, path = an_image(tmp_path)
    cached = image_cache_dir(64) / f"{image_id}.jpg"
    cached.write_bytes(b"stale")
    assert cached_jpeg(image_id, path, image_size=64) == b"stale"
    assert cached_jpeg(image_id, path, image_size=64, refresh=True) != b"stale"


def test_no_half_written_file_is_left_behind(tmp_path, galaxus_data):
    """Each entry is written to a .part name and renamed, so an interrupted run is safe to resume."""
    image_id, path = an_image(tmp_path)
    cached_jpeg(image_id, path, image_size=64)
    assert list(image_cache_dir(64).glob("*.part")) == []


@pytest.mark.parametrize("image_id", ["../escape", "sub/dir", "..", ""])
def test_an_image_id_that_is_not_a_plain_name_is_refused(tmp_path, galaxus_data, image_id):
    """The ids come from a competition CSV: one must never write outside the cache."""
    _, path = an_image(tmp_path)
    with pytest.raises(ValueError, match="plain file name"):
        cached_jpeg(image_id, path, image_size=64)


def test_filling_the_cache_reports_what_it_did(tmp_path, galaxus_data):
    pairs = [an_image(tmp_path, "a.png"), an_image(tmp_path, "b.png")]
    report = fill_image_cache(pairs, image_size=64)
    assert (report["written"], report["skipped"], report["missing"]) == (2, 0, 0)
    assert report["directory"] == image_cache_dir(64)


def test_filling_the_cache_twice_skips_everything_the_second_time(tmp_path, galaxus_data):
    """`just prep` is re-run after an interrupted download, so this must be cheap and idempotent."""
    pairs = [an_image(tmp_path, "a.png")]
    fill_image_cache(pairs, image_size=64)
    assert fill_image_cache(pairs, image_size=64)["skipped"] == 1


def test_filling_the_cache_accepts_a_table_from_read_part(tmp_path, galaxus_data):
    import pandas as pd

    image_id, path = an_image(tmp_path)
    table = pd.DataFrame({"image_id": [image_id], "image_file": [str(path)]})
    assert fill_image_cache(table, image_size=64)["written"] == 1


def test_a_table_without_the_path_column_says_how_to_read_it(galaxus_data):
    import pandas as pd

    with pytest.raises(KeyError, match="read_part"):
        fill_image_cache(pd.DataFrame({"image_id": ["img_1"]}), image_size=64)


def test_a_missing_source_image_raises_by_default(tmp_path, galaxus_data):
    with pytest.raises(FileNotFoundError):
        fill_image_cache([("gone", tmp_path / "not-there.png")], image_size=64)


def test_skip_missing_counts_the_gaps_instead(tmp_path, galaxus_data):
    pairs = [an_image(tmp_path, "a.png"), ("gone", tmp_path / "not-there.png")]
    report = fill_image_cache(pairs, image_size=64, skip_missing=True)
    assert (report["written"], report["missing"]) == (1, 1)
