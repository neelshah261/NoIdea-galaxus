"""#112 part A: unwrapping whatever an encoder returns into one vector per image.

The merged encoder called ``.detach()`` on the result of ``get_image_features`` and could not encode a single
image on ``transformers`` 5.24, which returns a ``BaseModelOutputWithPooling`` there instead of a tensor. The
stub encoder the other tests use cannot catch that, because a stub returns an array already. These tests cover
the unwrapping with plain stand-in objects: no model download, no network, no competition data.
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxus.data.neighbours import pooled_features


class Pooled:
    """Stands in for ``BaseModelOutputWithPooling``: the shape transformers 5.24 returns for SigLIP."""

    def __init__(self, last_hidden_state=None, pooler_output=None, image_embeds=None):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.image_embeds = image_embeds


class Hidden:
    """A patch-grid tensor whose ``mean(dim=1)`` is the only thing the fallback uses."""

    def __init__(self, array: np.ndarray):
        self.array = array

    def mean(self, dim: int) -> np.ndarray:
        return self.array.mean(axis=dim)


def test_a_plain_tensor_is_returned_unchanged():
    """Older transformers returned the pooled tensor directly, and that must keep working."""
    tensor = np.zeros((3, 768), dtype=np.float32)
    assert pooled_features(tensor) is tensor


def test_the_pooled_output_is_preferred_over_the_patch_grid():
    """This is the case that was broken: an output object, not a tensor."""
    pooler = np.ones((2, 768), dtype=np.float32)
    output = Pooled(last_hidden_state=Hidden(np.zeros((2, 196, 768), dtype=np.float32)), pooler_output=pooler)
    assert pooled_features(output) is pooler


def test_image_embeds_wins_when_a_model_offers_it():
    embeds = np.ones((2, 512), dtype=np.float32)
    output = Pooled(pooler_output=np.zeros((2, 768), dtype=np.float32), image_embeds=embeds)
    assert pooled_features(output) is embeds


def test_the_patch_grid_is_averaged_only_as_a_last_resort():
    """A mean over the patch grid is permutation-invariant, so it is the fallback and not the default.

    #127 measured the consequence: under a mean, an image and its mirror come out bit-identical, which erases
    exactly the left-from-right information the perspective dimension needs.
    """
    grid = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    output = Pooled(last_hidden_state=Hidden(grid))
    result = pooled_features(output)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, grid.mean(axis=1))


def test_a_mean_really_is_blind_to_the_patch_order():
    """Guards the docstring's claim, so nobody promotes the fallback to the default by accident."""
    grid = np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)
    mirrored = grid[:, ::-1, :]
    first = pooled_features(Pooled(last_hidden_state=Hidden(grid)))
    second = pooled_features(Pooled(last_hidden_state=Hidden(mirrored)))
    np.testing.assert_array_equal(first, second)


def test_an_output_with_nothing_usable_is_handed_back_for_the_caller_to_fail_on():
    """Returning the object unchanged gives the real AttributeError at the call site, naming the real type.

    Raising something of our own here would hide which encoder misbehaved.
    """
    empty = Pooled()
    assert pooled_features(empty) is empty


@pytest.mark.parametrize("width", [128, 768])
def test_the_vector_width_is_whatever_the_encoder_gives(width):
    pooler = np.zeros((5, width), dtype=np.float32)
    assert pooled_features(Pooled(pooler_output=pooler)).shape == (5, width)
