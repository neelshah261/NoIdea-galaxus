"""The label side of mirroring: mirrored training rows, and perspective answers mapped back (issue #29).

``images.py`` already mirrors pixels. Mirroring only helps if the *label* follows: a photo taken from the left is a
photo from the right once flipped. The deep-dive notes (#17) say left versus right is where the hosts' own model is
weakest, so this is the lane's contribution to the one innovation the model lane is building (#48, #49).

No class name is written in this module. The swap works on the words ``left`` and ``right`` inside whatever names
the classes source returns, and a swapped name is used only if it is itself a valid class — so a dimension with no
left/right classes comes back untouched, and nothing can invent a class that does not exist.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from galaxus.data.classes import allowed_classes, dimensions, no_overlay_class, overlay_dimension

#: The column that says a row is served from the mirrored copy of its image (``prepared_jpeg(..., mirrored=True)``).
MIRRORED_COLUMN = "mirrored"

#: ``<dimension>_source`` holds this when a person labelled the cell; anything else is a model prelabel.
HUMAN_SOURCE = "human"

#: The two orientation words the swap works on. **Not class names**: the classes are names like ``angle_left``,
#: and these are the substrings inside them that a flip exchanges. A swapped name is used only when it is itself a
#: class of that dimension, so a classes source whose names do not contain them simply mirrors nothing. Which
#: dimension carries the text overlay, and which of its classes means "no text", are *names* and so come from
#: ``contracts/classes.json`` through :mod:`galaxus.data.classes` (#81, AGENTS.md "Hard rules").
LEFT = "left"
RIGHT = "right"
#: A character no class name contains, so the two words swap in one pass instead of overwriting each other.
_PLACEHOLDER = "\x00"


def _swap_left_right(name: str) -> str:
    """``..._left`` ↔ ``..._right`` in a single pass."""
    return name.replace(LEFT, _PLACEHOLDER).replace(RIGHT, LEFT).replace(_PLACEHOLDER, RIGHT)


def mirror_class(dimension: str, cls: object, allowed: Mapping[str, Iterable[str]] | None = None) -> str:
    """The class the same picture shows once it is mirrored.

    Left and right are swapped inside the name; every other class, and every other dimension, comes back unchanged.
    A swapped name that is not itself a valid class of ``dimension`` is discarded rather than returned, so this can
    never produce a class the competition does not have.
    """
    names = frozenset((allowed if allowed is not None else allowed_classes()).get(dimension, ()))
    name = str(cls)
    swapped = _swap_left_right(name)
    if swapped == name or swapped not in names:
        return name
    return swapped


def _dimension_of(names: Iterable[str], allowed: Mapping[str, Iterable[str]]) -> str:
    """Which dimension a set of class names belongs to, so a confidence dict needs no dimension passed in."""
    wanted = {str(name) for name in names}
    if not wanted:
        raise ValueError("an empty confidence mapping says nothing about which dimension it is for")
    found = [dimension for dimension, classes in allowed.items() if wanted <= frozenset(classes)]
    if len(found) != 1:
        raise ValueError(
            f"cannot tell which dimension {sorted(wanted)} belongs to ({len(found)} match); pass dimension="
        )
    return found[0]


def swap_mirrored_confidence(
    confidence: Mapping[str, float],
    dimension: str | None = None,
    allowed: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, float]:
    """Per-class probabilities of the *mirrored* image, mapped back to the original orientation.

    This is what makes mirror test-time augmentation comparable: predict the flipped image, bring the answer back
    here, then combine. ``dimension`` is worked out from the class names when it is not given.
    """
    resolved = dict(allowed) if allowed is not None else allowed_classes()
    name = dimension if dimension is not None else _dimension_of(confidence.keys(), resolved)
    swapped = {mirror_class(name, cls, resolved): value for cls, value in confidence.items()}
    if len(swapped) != len(confidence):
        raise ValueError(
            f"two classes of {name} mirror onto the same name, which would drop a probability: {sorted(confidence)}"
        )
    return swapped


def _check_dimension(dimension: str, allowed: Mapping[str, Iterable[str]]) -> None:
    if dimension not in allowed:
        raise KeyError(f"{dimension!r} is not a dimension of the classes source; it has {sorted(allowed)}")


def mirrorable_dimension(allowed: Mapping[str, Iterable[str]] | None = None) -> str:
    """The one dimension whose classes change under a flip, worked out from the names rather than typed.

    Mirroring applies to whichever dimension has left/right classes; :func:`mirrorable_dimensions` finds them all
    and this returns the single one, raising when there is none or more than one. That keeps the dimension's name
    out of this module (#81) and makes a classes source without such a dimension an explicit error instead of a
    silent no-op on a name that happens not to be there.

    Public because a caller that wants the name and not the list would otherwise have to index
    ``mirrorable_dimensions()[0]`` blindly or repeat the two checks below. #81 removed
    ``PERSPECTIVE_DIMENSION`` and left only the plural accessor, which is what `galaxus.serve.results_tables`
    imported (#85); this is the replacement for that one-value use.

    Raises:
        ValueError: if no dimension of the classes source can be mirrored, or if more than one can.
    """
    resolved = dict(allowed) if allowed is not None else allowed_classes()
    found = mirrorable_dimensions(resolved)
    if not found:
        raise ValueError(
            "no dimension of the classes source has left/right classes, so there is nothing to mirror; "
            f"it has {sorted(resolved)}. Pass dimension= to say which one to use"
        )
    if len(found) > 1:
        raise ValueError(f"more than one dimension could be mirrored ({list(found)}); pass dimension= to choose")
    return found[0]


def mirrored_rows(
    frame: pd.DataFrame,
    allowed: Mapping[str, Iterable[str]] | None = None,
    no_text_class: str | None = None,
    dimension: str | None = None,
    text_dimension: str | None = None,
) -> pd.DataFrame:
    """The training rows worth adding a second time, mirrored, with the swapped label.

    A row qualifies only when all three hold:

    1. **The label actually changes when mirrored.** A head-on or top-down shot gains nothing from a flip.
    2. **The label is a person's.** A model prelabel is not evidence, so mirroring it would multiply a guess
       (the same rule as :func:`galaxus.data.split.scored_mask`). The text label must be a person's too.
    3. **The image carries no text.** Mirrored text is unreadable nonsense, and a flipped overlay would teach the
       model something no real product photo looks like.

    Returns:
        Only the new rows, each a copy of its original with the mirrored label and ``mirrored=True``. The input is
        not changed: the caller concatenates, e.g.
        ``pd.concat([frame.assign(mirrored=False), mirrored_rows(frame)], ignore_index=True)``.
    """
    resolved = dict(allowed) if allowed is not None else allowed_classes()
    # Resolved lazily, never at import: a caller that passes its own names (every test here does) must not need
    # the contract to be present at all.
    dimension = dimension if dimension is not None else mirrorable_dimension(resolved)
    text_dimension = text_dimension if text_dimension is not None else overlay_dimension()
    _check_dimension(dimension, resolved)
    _check_dimension(text_dimension, resolved)

    no_text = no_text_class if no_text_class is not None else no_overlay_class()
    if no_text not in frozenset(resolved[text_dimension]):
        raise ValueError(
            f"{no_text!r} is not a class of {text_dimension}, so 'carries no text' cannot be decided; "
            f"{text_dimension} has {sorted(resolved[text_dimension])}. Pass no_text_class= if it was renamed"
        )

    needed = [dimension, text_dimension, f"{dimension}_source", f"{text_dimension}_source"]
    absent = [column for column in needed if column not in frame.columns]
    if absent:
        raise KeyError(f"{', '.join(absent)} missing: mirrored rows need the train label metadata")

    swapped = frame[dimension].map(lambda cls: mirror_class(dimension, cls, resolved))
    worth_it = swapped != frame[dimension].astype(str)
    human = (frame[f"{dimension}_source"] == HUMAN_SOURCE) & (frame[f"{text_dimension}_source"] == HUMAN_SOURCE)
    text_free = frame[text_dimension].astype(str) == no_text

    rows = frame[worth_it & human & text_free].copy()
    rows[dimension] = swapped[worth_it & human & text_free]
    rows[MIRRORED_COLUMN] = True
    return rows


def with_mirrored_rows(
    frame: pd.DataFrame,
    allowed: Mapping[str, Iterable[str]] | None = None,
    no_text_class: str | None = None,
    dimension: str | None = None,
    text_dimension: str | None = None,
) -> pd.DataFrame:
    """The training frame plus its mirrored rows, with ``mirrored`` saying which copy of the image to serve."""
    original = frame.copy()
    original[MIRRORED_COLUMN] = False
    extra = mirrored_rows(
        frame,
        allowed=allowed,
        no_text_class=no_text_class,
        dimension=dimension,
        text_dimension=text_dimension,
    )
    return pd.concat([original, extra], ignore_index=True)


def mirrorable_dimensions(allowed: Mapping[str, Iterable[str]] | None = None) -> tuple[str, ...]:
    """The dimensions that have at least one class whose name changes when mirrored.

    Useful as a check rather than a guess: if this does not contain the perspective dimension, the classes source
    has no left/right classes and mirroring cannot teach anything about orientation.
    """
    resolved = dict(allowed) if allowed is not None else allowed_classes()
    found = []
    for dimension in dimensions() if allowed is None else tuple(resolved):
        names = frozenset(resolved.get(dimension, ()))
        if any(_swap_left_right(name) != name and _swap_left_right(name) in names for name in names):
            found.append(dimension)
    return tuple(found)
