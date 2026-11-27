"""The hero-shot rules: which image of a product a judge should see first, and what to hide or flag.

Issue #31. The rules live in ``hero_rules.json`` beside this module, never in this code, because only one of them
is actually known: the deep-dive talk (#17 point 1) says mice are shot ``top_down``. Everything else is ours and is
carried through the page marked as an assumption, with the ``TODO(owner)`` that would settle it.

No class name is written in this module. They come from the config, and the config is checked against
``galaxus.data.classes`` when it loads: :func:`load_rules` collects every name that is not a valid class instead of
raising, because the classes source itself is not on ``main`` yet (``contracts/classes.json``, issue #11). The
problems travel with the rules so the page and the CLI can show them, and ``strict=True`` turns them into an error
once there is a contract to check against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RULES_FILE = Path(__file__).with_name("hero_rules.json")

#: The two directions the mirror swaps, as words that appear inside a class name, never as class names.
LEFT = "left"
RIGHT = "right"

KNOWN = "known"
ASSUMPTION = "assumption"

_PREFERENCE_SECTIONS = ("shop_languages", "subject_preference", "background_preference", "default")


@dataclass(frozen=True)
class Rules:
    """The loaded rule set, carrying whatever could not be verified against the classes source."""

    raw: dict[str, Any]
    problems: tuple[str, ...] = ()

    @property
    def match_context_keys(self) -> tuple[str, ...]:
        return tuple(self.raw.get("match_context_keys", ()))

    @property
    def no_text_class(self) -> str | None:
        value = self.raw.get("no_text_class")
        return None if value is None else str(value)

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        return dict(value) if isinstance(value, dict) else {}

    def classes_of(self, name: str) -> tuple[str, ...]:
        return tuple(str(item) for item in self.section(name).get("classes", ()))

    def _entries(self) -> list[tuple[str, dict[str, Any]]]:
        named = [(name, self.section(name)) for name in _PREFERENCE_SECTIONS]
        for entry in self.raw.get("product_rules", []):
            named.append((f"product {entry.get('match')}", dict(entry)))
        return named

    def assumptions(self) -> list[dict[str, str]]:
        """Every rule we chose rather than were told, so the page can list them in one place."""
        return [
            {"rule": name, "todo": str(entry.get("todo", ""))}
            for name, entry in self._entries()
            if entry.get("status") == ASSUMPTION
        ]

    def sources(self) -> list[dict[str, str]]:
        """Every rule the hosts actually gave us, with where it came from."""
        return [
            {"rule": name, "source": str(entry.get("source", ""))}
            for name, entry in self._entries()
            if entry.get("status") == KNOWN
        ]


def _every_named_class(raw: dict[str, Any]) -> list[str]:
    """Each class name the config mentions, so the loader can check all of them in one pass."""
    names: list[str] = []
    for section in ("shop_languages", "subject_preference", "background_preference"):
        entry = raw.get(section) or {}
        names.extend(str(name) for name in entry.get("classes", ()))
    default = raw.get("default") or {}
    names.extend(str(name) for name in default.get("perspective_order", ()))
    for entry in raw.get("product_rules", []):
        names.extend(str(name) for name in entry.get("perspective_order", ()))
    if raw.get("no_text_class") is not None:
        names.append(str(raw["no_text_class"]))
    return names


def load_rules(path: str | Path | None = None, strict: bool = False) -> Rules:
    """The rule set, with every class name it mentions checked against ``galaxus.data.classes``.

    Args:
        path: a rule file to load instead of the one shipped beside this module.
        strict: raise instead of recording problems. Turn this on once ``contracts/classes.json`` is on ``main``.
    """
    raw = json.loads(Path(path or RULES_FILE).read_text(encoding="utf-8"))
    problems: list[str] = []
    valid = _valid_classes()
    if valid is None:
        problems.append("class names were not checked: the classes source is unavailable (contracts/classes.json, #11)")
    else:
        problems.extend(
            f"{name} in the rules is not a class in any dimension"
            for name in _every_named_class(raw)
            if name not in valid
        )
    if problems and strict:
        raise ValueError("; ".join(problems))
    return Rules(raw=raw, problems=tuple(problems))


def _valid_classes() -> set[str] | None:
    """Every valid class name across the dimensions, or ``None`` when there is no classes source yet."""
    try:
        from galaxus.data.classes import allowed_classes

        resolved = allowed_classes()
    except Exception:  # noqa: BLE001 - no contract and no $GALAXUS_DATA is normal before #11 lands
        return None
    valid: set[str] = set()
    for classes in resolved.values():
        valid |= {str(name) for name in classes}
    return valid


def product_rule(rules: Rules, context: dict[str, str] | None) -> dict[str, Any]:
    """The rule for one product, matching its context against each rule's ``match`` words.

    The deep dive (#17 point 9) says only the title, type and category tell you what a product is, so those are the
    context keys the config looks at. Falls back to the default rule, which is an assumption.
    """
    keys = rules.match_context_keys
    haystack = " ".join(str(value).lower() for key, value in (context or {}).items() if not keys or key in keys)
    for entry in rules.raw.get("product_rules", []):
        if any(str(word).lower() in haystack for word in entry.get("match", ())):
            return dict(entry)
    return {**rules.section("default"), "match": None}


def mean_confidence(confidence: dict[str, float] | None) -> float:
    """The mean of a per-dimension confidence map, or 0.0 when there is none."""
    if not confidence:
        return 0.0
    numbers = [float(value) for value in confidence.values()]
    return sum(numbers) / len(numbers) if numbers else 0.0


def hero_sort_key(
    rules: Rules,
    rule: dict[str, Any],
    labels: dict[str, str],
    confidence: dict[str, float] | None = None,
) -> tuple[int, int, int, float]:
    """How good a hero shot an image is; lower sorts first.

    Perspective first, because that is the business priority (#17 point 1); then the preferred background, since a
    colour background is not used as a hero shot (#17 point 11); then the preferred subject; then confidence, so the
    order is stable rather than arbitrary when the rules tie.
    """
    order = [str(name) for name in rule.get("perspective_order", ())]
    perspective = labels.get("perspective")
    rank = order.index(perspective) if perspective in order else len(order)
    return (
        rank,
        0 if labels.get("background") in rules.classes_of("background_preference") else 1,
        0 if labels.get("subject") in rules.classes_of("subject_preference") else 1,
        -mean_confidence(confidence),
    )


def hidden_reason(rules: Rules, labels: dict[str, str]) -> str | None:
    """Why the shop would hide this image, or ``None`` to show it.

    The deep dive (#17 point 11) says ``text_language`` exists to hide images carrying foreign text, and named
    Chinese and Polish as examples. Which languages are kept is an assumption in the config, so this demonstrates
    the rule; it is not a statement about what Galaxus does.
    """
    language = labels.get("text_language")
    if language is None:
        return None
    if rules.no_text_class is not None and language == rules.no_text_class:
        return None
    kept = rules.classes_of("shop_languages")
    if kept and language not in kept:
        return f"text_language is {language}, which the configured shop languages do not include"
    return None


def text_overlay(rules: Rules, labels: dict[str, str]) -> str | None:
    """The ``text_classification`` value when the image carries text, otherwise ``None``."""
    value = labels.get("text_classification")
    if value is None or (rules.no_text_class is not None and value == rules.no_text_class):
        return None
    return value


def mirror_class(name: str, valid: set[str] | None) -> str:
    """``name`` with the words left and right swapped, when the result is also a valid class.

    The words are swapped inside the name and the result is checked against the classes source, so no class name is
    written here. This is the same rule as the data lane's issue #29, which owns it properly.
    TODO(data): import the ``galaxus.data`` version once #29 is merged, and delete this.
    """
    placeholder = "\x00"
    swapped = name.replace(LEFT, placeholder).replace(RIGHT, LEFT).replace(placeholder, RIGHT)
    if swapped == name:
        return name
    return swapped if valid is None or swapped in valid else name


def mirrored_labels(labels: dict[str, str], valid: set[str] | None, dimension: str = "perspective") -> dict[str, str]:
    """The labels of a mirrored image, read back in the original orientation."""
    if dimension not in labels:
        return dict(labels)
    return {**labels, dimension: mirror_class(str(labels[dimension]), valid)}
