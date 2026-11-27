"""Issue #28: the patch size is a parameter, not a constant buried in the code.

32 is Qwen's, from the reference notebook. Apertus v1.5's is Q-0002, measured by the B6 smoke test (#46). These
tests pin the behaviour at several patch sizes so answering Q-0002 is a one-line change to ``PATCH_SIZE``.
"""

from __future__ import annotations

import pytest
from conftest import write_image
from PIL import Image as PillowImage

from galaxus.data.images import (
    IMAGE_SIZE,
    PATCH_SIZE,
    image_cache_dir,
    native_aspect_image,
    prepare_image,
    prepared_jpeg,
    vision_tokens,
)

# Patch sizes the issue names, plus the current default. 14 and 28 are the awkward ones: 512 is not a multiple of
# either, which is exactly why ``image_size`` has to be checked against the patch size rather than against 32.
PATCH_SIZES = (14, 16, 28, 32)


def areas_for(patch: int) -> tuple[int, ...]:
    """Target areas that are whole multiples of this patch size."""
    return (patch * 8, patch * 16)


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_both_sides_are_whole_patches(patch, tmp_path):
    source = write_image(tmp_path / "wide.png", "RGB", (300, 170))
    prepared = prepare_image(source, image_size=patch * 16, patch_size=patch)
    assert prepared.width % patch == 0
    assert prepared.height % patch == 0


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_the_token_count_matches_the_prepared_pixels(patch, tmp_path):
    source = write_image(tmp_path / "tall.png", "RGB", (120, 400))
    prepared = prepare_image(source, image_size=patch * 8, patch_size=patch)
    assert vision_tokens(prepared, patch) == (prepared.width // patch) * (prepared.height // patch)


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_a_smaller_patch_size_means_more_tokens_for_the_same_area(patch, tmp_path):
    """The point of the parameter: tokens scale with the patch size, and that is what costs money."""
    source = write_image(tmp_path / "square.png", "RGB", (256, 256))
    area = patch * 16
    prepared = prepare_image(source, image_size=area, patch_size=patch)
    # About (area / patch)^2 tokens, give or take the rounding to whole patches.
    assert vision_tokens(prepared, patch) == pytest.approx((area / patch) ** 2, rel=0.35)


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_no_side_is_ever_smaller_than_one_patch(patch, tmp_path):
    """A sliver of an image must still be a whole patch, or the model gets a fractional one."""
    source = write_image(tmp_path / "sliver.png", "RGB", (400, 3))
    prepared = prepare_image(source, image_size=patch * 8, patch_size=patch)
    assert prepared.height >= patch
    assert prepared.width >= patch


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_the_aspect_ratio_survives_at_every_patch_size(patch, tmp_path):
    source = write_image(tmp_path / "wide.png", "RGB", (400, 100))
    prepared = prepare_image(source, image_size=patch * 16, patch_size=patch)
    assert prepared.width > prepared.height


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_an_area_that_is_not_whole_patches_is_refused(patch, tmp_path):
    source = write_image(tmp_path / "x.png", "RGB", (64, 64))
    with pytest.raises(ValueError, match="multiple of the patch size"):
        prepared_jpeg(source, image_size=patch * 8 + 1, patch_size=patch)


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_each_patch_size_gets_its_own_cache_folder(patch, galaxus_data):
    """Two patch sizes must never serve each other's pixels."""
    folder = image_cache_dir(patch * 16, patch)
    assert folder.name == f"{patch * 16}-p{patch}"
    assert folder != image_cache_dir(patch * 16, patch * 2)


@pytest.mark.parametrize("patch", PATCH_SIZES)
def test_the_patch_size_really_changes_what_the_model_sees(patch, tmp_path):
    """If it did not, the folder name would be the only difference and the cache would be a lie.

    The area is a multiple of both patch sizes, so the only variable is the patch. Doubling it must roughly
    quarter the token count, which is what the cost of a run is made of.
    """
    source = write_image(tmp_path / "x.png", "RGB", (300, 200))
    area = patch * 16
    fine = prepare_image(source, image_size=area, patch_size=patch)
    coarse = prepare_image(source, image_size=area, patch_size=patch * 2)
    assert vision_tokens(fine, patch) > vision_tokens(coarse, patch * 2)
    assert vision_tokens(fine, patch) == pytest.approx(4 * vision_tokens(coarse, patch * 2), rel=0.35)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "32", None, True])
def test_a_patch_size_that_is_not_a_positive_whole_number_is_refused(bad, tmp_path):
    image = PillowImage.new("RGB", (64, 64), (1, 2, 3))
    with pytest.raises(ValueError, match="patch_size must be a positive whole number"):
        native_aspect_image(image, 64, bad)


def test_the_default_is_still_qwens_until_q_0002_is_answered():
    """Deliberate: the measurement is #46's. This test is the reminder, not a preference."""
    assert PATCH_SIZE == 32
    assert IMAGE_SIZE % PATCH_SIZE == 0


def test_the_default_patch_size_is_used_when_none_is_given(tmp_path):
    source = write_image(tmp_path / "x.png", "RGB", (200, 150))
    assert prepared_jpeg(source, image_size=256) == prepared_jpeg(source, image_size=256, patch_size=PATCH_SIZE)


def test_vision_tokens_defaults_to_the_same_patch_size(tmp_path):
    source = write_image(tmp_path / "x.png", "RGB", (200, 150))
    prepared = prepare_image(source, image_size=256)
    assert vision_tokens(prepared) == vision_tokens(prepared, PATCH_SIZE)
