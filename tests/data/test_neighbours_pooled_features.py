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


@pytest.mark.filterwarnings("ignore:.*mean-pooled:RuntimeWarning")  # asserted in its own test below
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


@pytest.mark.filterwarnings("ignore:.*mean-pooled:RuntimeWarning")  # asserted in its own test below
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


def test_falling_back_to_the_mean_warns_instead_of_degrading_quietly():
    """The mirror-aware perspective head (#189) reads these features, and a mean would blind it to left/right.

    Without a warning that switch is invisible: the run finishes, the file validates, and perspective is quietly
    scored on features that cannot represent a side. The review of #189 found no guard against this, and the mean
    is still reachable for any encoder exposing only ``last_hidden_state``.
    """
    output = Pooled(last_hidden_state=Hidden(np.zeros((2, 196, 768), dtype=np.float32)))
    with pytest.warns(RuntimeWarning, match="mean-pooled"):
        pooled_features(output)


def test_the_warning_names_the_consequence_not_just_the_mechanism():
    """A reader of the log needs to know what breaks, not only which attribute was absent."""
    output = Pooled(last_hidden_state=Hidden(np.zeros((1, 4, 3), dtype=np.float32)))
    with pytest.warns(RuntimeWarning) as caught:
        pooled_features(output)
    message = str(caught[0].message)
    assert "mirror" in message
    assert "perspective" in message
    assert "Pooled" in message  # the type that misbehaved, so the log says which encoder


@pytest.mark.parametrize("attribute", ["image_embeds", "pooler_output"])
def test_the_preferred_paths_stay_silent(attribute, recwarn):
    """Only the lossy path warns; a warning on the normal path would train everyone to ignore it."""
    pooled_features(Pooled(**{attribute: np.zeros((2, 768), dtype=np.float32)}))
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_an_unusable_output_does_not_warn_about_pooling_it_never_did(recwarn):
    pooled_features(Pooled())
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []
