"""Issue #29: mirrored training rows and the label side of mirroring. Synthetic classes and rows only."""

from __future__ import annotations

import pandas as pd
import pytest

from galaxus.data.mirror import (
    MIRRORED_COLUMN,
    mirror_class,
    mirrorable_dimensions,
    mirrored_rows,
    swap_mirrored_confidence,
    with_mirrored_rows,
)

# Synthetic classes, not the competition's. Only the words "left" and "right" matter to the code under test.
ALLOWED = {
    "perspective": ["angle_left", "angle_right", "head_on", "top_down"],
    "subject": ["alpha", "beta"],
    "text_classification": ["seal", "none"],
}
NO_TEXT = "none"


def row(
    image_id: str,
    perspective: str,
    text: str = NO_TEXT,
    perspective_source: str = "human",
    text_source: str = "human",
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "product_id": "p1",
        "perspective": perspective,
        "perspective_source": perspective_source,
        "perspective_relevant": "true",
        "text_classification": text,
        "text_classification_source": text_source,
        "text_classification_relevant": "true",
    }


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            row("img_1", "angle_left"),
            row("img_2", "angle_right"),
            row("img_3", "head_on"),
            row("img_4", "angle_left", text="seal"),
            row("img_5", "angle_left", perspective_source="model"),
            row("img_6", "angle_right", text_source="model"),
        ]
    )


# --- mirror_class -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("angle_left", "angle_right"), ("angle_right", "angle_left"), ("head_on", "head_on"), ("top_down", "top_down")],
)
def test_left_and_right_swap_and_everything_else_stays(name, expected):
    assert mirror_class("perspective", name, ALLOWED) == expected


def test_a_dimension_with_no_left_or_right_classes_is_untouched():
    for name in ALLOWED["subject"]:
        assert mirror_class("subject", name, ALLOWED) == name


def test_mirroring_twice_gives_the_original_class():
    for name in ALLOWED["perspective"]:
        once = mirror_class("perspective", name, ALLOWED)
        assert mirror_class("perspective", once, ALLOWED) == name


def test_a_swap_that_is_not_a_valid_class_is_discarded():
    """The swap must never invent a class: a one-sided classes source comes back unchanged."""
    one_sided = {"perspective": ["angle_left", "head_on"]}
    assert mirror_class("perspective", "angle_left", one_sided) == "angle_left"


def test_an_unknown_dimension_simply_changes_nothing():
    assert mirror_class("not_a_dimension", "angle_left", ALLOWED) == "angle_left"


def test_both_words_swap_in_one_pass():
    """A name holding both words must not collapse into one of them."""
    both = {"perspective": ["left_to_right", "right_to_left"]}
    assert mirror_class("perspective", "left_to_right", both) == "right_to_left"


def test_mirrorable_dimensions_names_only_the_dimension_that_can_change():
    assert mirrorable_dimensions(ALLOWED) == ("perspective",)


def test_a_classes_source_with_no_left_or_right_mirrors_nothing():
    assert mirrorable_dimensions({"perspective": ["head_on", "top_down"]}) == ()


# --- mirrored_rows ------------------------------------------------------------------------------------------


def test_only_the_rows_whose_label_changes_are_mirrored(frame):
    mirrored = mirrored_rows(frame, ALLOWED, NO_TEXT)
    assert sorted(mirrored["image_id"]) == ["img_1", "img_2"]


def test_a_head_on_shot_is_not_mirrored(frame):
    """Nothing is learned from flipping a symmetric view, so it would only duplicate a row."""
    assert "img_3" not in set(mirrored_rows(frame, ALLOWED, NO_TEXT)["image_id"])


def test_an_image_carrying_text_is_never_mirrored(frame):
    """Mirrored text is nonsense, and no real product photo looks like that."""
    assert "img_4" not in set(mirrored_rows(frame, ALLOWED, NO_TEXT)["image_id"])


def test_a_model_prelabel_is_not_trusted(frame):
    assert "img_5" not in set(mirrored_rows(frame, ALLOWED, NO_TEXT)["image_id"])


def test_a_model_prelabelled_text_cell_is_not_trusted_either(frame):
    """'Carries no text' has to be a person's judgement too, or the text rule rests on a guess."""
    assert "img_6" not in set(mirrored_rows(frame, ALLOWED, NO_TEXT)["image_id"])


def test_the_mirrored_row_carries_the_swapped_label(frame):
    mirrored = mirrored_rows(frame, ALLOWED, NO_TEXT).set_index("image_id")
    assert mirrored.loc["img_1", "perspective"] == "angle_right"
    assert mirrored.loc["img_2", "perspective"] == "angle_left"


def test_the_mirrored_rows_are_flagged(frame):
    assert mirrored_rows(frame, ALLOWED, NO_TEXT)[MIRRORED_COLUMN].all()


def test_the_input_frame_is_not_changed(frame):
    before = frame.copy()
    mirrored_rows(frame, ALLOWED, NO_TEXT)
    pd.testing.assert_frame_equal(frame, before)


def test_mirroring_the_mirrored_rows_gives_the_original_labels_back(frame):
    once = mirrored_rows(frame, ALLOWED, NO_TEXT)
    twice = mirrored_rows(once.drop(columns=[MIRRORED_COLUMN]), ALLOWED, NO_TEXT).set_index("image_id")
    original = frame.set_index("image_id")
    for image_id in twice.index:
        assert twice.loc[image_id, "perspective"] == original.loc[image_id, "perspective"]


def test_everything_else_in_the_row_is_kept(frame):
    mirrored = mirrored_rows(frame, ALLOWED, NO_TEXT).set_index("image_id")
    assert mirrored.loc["img_1", "product_id"] == "p1"
    assert mirrored.loc["img_1", "text_classification"] == NO_TEXT


def test_a_no_text_class_that_is_not_a_class_is_refused(frame):
    with pytest.raises(ValueError, match="not a class of text_classification"):
        mirrored_rows(frame, ALLOWED, "not_a_class")


def test_a_frame_without_the_label_metadata_is_refused():
    bare = pd.DataFrame([{"image_id": "img_1", "perspective": "angle_left", "text_classification": NO_TEXT}])
    with pytest.raises(KeyError, match="source"):
        mirrored_rows(bare, ALLOWED, NO_TEXT)


def test_an_unknown_dimension_is_refused(frame):
    with pytest.raises(KeyError, match="not a dimension"):
        mirrored_rows(frame, ALLOWED, NO_TEXT, dimension="not_a_dimension")


def test_no_rows_qualify_is_an_empty_frame_not_an_error():
    only_symmetric = pd.DataFrame([row("img_1", "head_on"), row("img_2", "top_down")])
    assert mirrored_rows(only_symmetric, ALLOWED, NO_TEXT).empty


# --- with_mirrored_rows -------------------------------------------------------------------------------------


def test_the_combined_frame_marks_which_copy_to_serve(frame):
    combined = with_mirrored_rows(frame, ALLOWED, NO_TEXT)
    assert len(combined) == len(frame) + 2
    assert (~combined[MIRRORED_COLUMN].head(len(frame))).all()
    assert combined[MIRRORED_COLUMN].tail(2).all()


def test_the_combined_frame_leaves_the_input_alone(frame):
    before = frame.copy()
    with_mirrored_rows(frame, ALLOWED, NO_TEXT)
    pd.testing.assert_frame_equal(frame, before)


# --- swap_mirrored_confidence -------------------------------------------------------------------------------


def test_confidences_come_back_in_the_original_orientation():
    mirrored = {"angle_left": 0.7, "angle_right": 0.2, "head_on": 0.1}
    assert swap_mirrored_confidence(mirrored, "perspective", ALLOWED) == {
        "angle_right": 0.7,
        "angle_left": 0.2,
        "head_on": 0.1,
    }


def test_the_dimension_is_worked_out_from_the_class_names():
    assert swap_mirrored_confidence({"angle_left": 1.0}, allowed=ALLOWED) == {"angle_right": 1.0}


def test_swapping_twice_returns_the_original_mapping():
    mirrored = {"angle_left": 0.6, "angle_right": 0.4}
    once = swap_mirrored_confidence(mirrored, "perspective", ALLOWED)
    assert swap_mirrored_confidence(once, "perspective", ALLOWED) == mirrored


def test_an_empty_mapping_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="empty"):
        swap_mirrored_confidence({}, allowed=ALLOWED)


def test_class_names_that_match_no_dimension_are_refused():
    with pytest.raises(ValueError, match="cannot tell which dimension"):
        swap_mirrored_confidence({"not_a_class": 1.0}, allowed=ALLOWED)


def test_a_swap_that_would_drop_a_probability_is_refused():
    """A one-sided classes source makes two answers land on one name, which would lose probability mass.

    ``angle_right`` swaps to the valid ``angle_left``, while ``angle_left`` has no valid swap and stays put — so
    both end up as ``angle_left``. That is a broken classes source (#11), and it must not pass silently.
    """
    one_sided = {"perspective": ["angle_left"]}
    with pytest.raises(ValueError, match="drop a probability"):
        swap_mirrored_confidence({"angle_left": 0.4, "angle_right": 0.6}, "perspective", one_sided)
