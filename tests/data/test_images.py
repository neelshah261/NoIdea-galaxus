"""Issue #4: the image pipeline. Synthetic images only."""

from __future__ import annotations

import pytest
from conftest import write_image, write_palette_image_with_transparency
from PIL import Image as PillowImage

from galaxus.data.images import (
    IMAGE_SIZE,
    MAX_SIDE_FACTOR,
    PATCH_SIZE,
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
