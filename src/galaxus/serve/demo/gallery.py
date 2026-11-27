"""The offline HTML gallery: per product, the images ordered with the chosen hero shot first.

Issue #31. One self-contained file, in the design/STYLE.md look, that a judge can open with no network and no server.

Two deliberate choices:

- **The images are embedded.** Each one is encoded by ``galaxus.data.images.prepared_jpeg``, so the page shows the
  pixels the model was actually served, and the file keeps working when it is moved. The page therefore belongs in
  ``$GALAXUS_DATA/out/demo/`` and never in git (AGENTS.md product rule 3).
- **No webfont is fetched.** STYLE.md asks for Geist from Google Fonts, but this page has to run with no network, so
  it names Geist first and falls back to the system stack. The colours are STYLE.md's exactly.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.serve.demo.rules import (
    Rules,
    hero_sort_key,
    hidden_reason,
    load_rules,
    mean_confidence,
    mirrored_labels,
    product_rule,
    text_overlay,
)

PRODUCT_COLUMN = "product_id"
ID_COLUMN = "image_id"
IMAGE_FILE_COLUMN = "image_file"
#: Thumbnail edge in pixels. A multiple of the patch size, so the data lane will prepare it.
THUMB_SIZE = 256

# STYLE.md "Colours". One accent, baselines blue, us orange.
PALETTE = {
    "paper": "#f2efe9",
    "paper2": "#f8f6f1",
    "white": "#ffffff",
    "ink": "#141414",
    "ink2": "#2e2b27",
    "muted": "#55524c",
    "muted2": "#6e6a63",
    "rule": "#dcd6cc",
    "rule2": "#c4bcae",
    "accent": "#c93c05",
    "green": "#157a43",
    "red": "#d1173a",
    "amber": "#e3a008",
}


def _data_uri(path: str | Path, thumb_size: int = THUMB_SIZE) -> str | None:
    """The image as a JPEG data URI, prepared exactly as the model sees it, or ``None`` if it cannot be read."""
    try:
        from galaxus.data.images import prepared_jpeg

        encoded = base64.b64encode(prepared_jpeg(Path(path), image_size=thumb_size)).decode("ascii")
    except Exception:  # noqa: BLE001 - a missing or unreadable image must not take the whole page down
        return None
    return f"data:image/jpeg;base64,{encoded}"


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def products_from_frame(
    frame: pd.DataFrame, predictions: dict[str, dict[str, Any]], context_keys: list[str] | None = None
) -> list[dict[str, Any]]:
    """Group a table read by ``galaxus.data.load.read_part`` into products, keeping only predicted images.

    Images whose ``product_id`` is missing each become their own product, so a part without that column (dev and
    final have no products) still renders one card per image.
    """
    if context_keys is None:
        from galaxus.serve.items import context_columns

        context_keys = context_columns(frame)
    grouped: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(frame.to_dict(orient="records")):
        image_id = str(row.get(ID_COLUMN, ""))
        record = predictions.get(image_id)
        if record is None:
            continue
        product = str(row.get(PRODUCT_COLUMN) or f"{image_id} (no {PRODUCT_COLUMN})")
        context = {key: row[key] for key in context_keys if key in row and str(row[key]).strip()}
        bucket = grouped.setdefault(product, {"product_id": product, "context": context, "images": []})
        bucket["images"].append(
            {
                "image_id": image_id,
                "path": row.get(IMAGE_FILE_COLUMN),
                "labels": dict(record.get("labels") or {}),
                "confidence": dict(record.get("confidence") or {}) or None,
                "position": position,
            }
        )
    return list(grouped.values())


def order_product(rules: Rules, product: dict[str, Any]) -> dict[str, Any]:
    """Sort one product's images, hero first, and attach what the rules say about each."""
    rule = product_rule(rules, product.get("context"))
    images = sorted(
        product["images"],
        key=lambda image: (hero_sort_key(rules, rule, image["labels"], image["confidence"]), image["position"]),
    )
    for image in images:
        image["hidden"] = hidden_reason(rules, image["labels"])
        image["overlay"] = text_overlay(rules, image["labels"])
    return {**product, "images": images, "rule": rule}


def _meter_band(meter: dict[str, Any] | None) -> str:
    if not meter:
        return ""
    cells = []
    for label, value, note in (
        ("images/s", meter.get("images_per_second"), meter.get("hardware", "")),
        ("cost / 100k images", meter.get("cost_per_100k"), meter.get("cost_note", "")),
        ("images", meter.get("images"), meter.get("seconds_note", "")),
    ):
        if value is None:
            continue
        shown = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        cells.append(
            f'<div class="figure"><div class="eyebrow">{_escape(label)}</div>'
            f'<div class="number">{_escape(shown)}</div><div class="note">{_escape(note)}</div></div>'
        )
    if not cells:
        return ""
    return f'<section class="band"><div class="figures">{"".join(cells)}</div></section>'


def _mirror_panel(rules: Rules, mirror: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> str:
    """Original against mirrored-and-swapped, with the disagreements called out."""
    from galaxus.serve.demo.rules import _valid_classes

    valid = _valid_classes()
    rows = []
    agreed = 0
    for image_id, record in sorted(mirror.items()):
        original = (predictions.get(image_id) or {}).get("labels") or {}
        swapped = mirrored_labels(dict(record.get("labels") or {}), valid)
        before, after = original.get("perspective"), swapped.get("perspective")
        if before is None and after is None:
            continue
        same = before == after
        agreed += same
        mark = "agrees" if same else "disagrees"
        colour = "ok" if same else "bad"
        rows.append(
            f"<tr><td class=mono>{_escape(image_id)}</td><td class=mono>{_escape(before)}</td>"
            f"<td class=mono>{_escape(after)}</td><td class='mono {colour}'>{mark}</td></tr>"
        )
    if not rows:
        return ""
    total = len(rows)
    return (
        '<section class="band"><div class="eyebrow">MIRROR TEST-TIME AUGMENTATION</div>'
        f"<h2>{total - agreed} of {total} images change their perspective answer when mirrored</h2>"
        "<p class=lead>The mirrored image is classified, then left and right are swapped back. A disagreement is a "
        "perspective the model is not sure of, which is where the deep dive says the hosts' errors live "
        "(issue #17, slide 22).</p>"
        "<table><thead><tr><th>image</th><th>original</th><th>mirrored, swapped back</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _image_card(image: dict[str, Any], dimensions: tuple[str, ...], is_hero: bool, thumb_size: int) -> str:
    uri = _data_uri(image["path"], thumb_size) if image.get("path") else None
    frame = (
        f'<img src="{uri}" alt="prepared image {_escape(image["image_id"])}">'
        if uri
        else '<div class="missing">image not available</div>'
    )
    if image.get("hidden"):
        frame = f'<div class="hidden-wrap">{frame}<div class="veil">hidden in the shop</div></div>'
    badges = []
    if is_hero:
        badges.append('<span class="badge hero">hero shot</span>')
    if image.get("overlay"):
        badges.append(f'<span class="badge warn">text: {_escape(image["overlay"])}</span>')
    if image.get("hidden"):
        badges.append(f'<span class="badge bad">{_escape(image["hidden"])}</span>')
    answers = "".join(
        f'<div class="answer"><span class="dim">{_escape(dimension)}</span>'
        f'<span class="mono">{_escape(image["labels"].get(dimension, "-"))}</span></div>'
        for dimension in dimensions
    )
    confidence = image.get("confidence")
    footer = f'<div class="note mono">mean confidence {mean_confidence(confidence):.2f}</div>' if confidence else ""
    return (
        f'<article class="card{" hero" if is_hero else ""}">{frame}'
        f'<div class="badges">{"".join(badges)}</div>'
        f'<div class="answers">{answers}</div>{footer}'
        f'<div class="note mono">{_escape(image["image_id"])}</div></article>'
    )


def _product_section(product: dict[str, Any], dimensions: tuple[str, ...], thumb_size: int) -> str:
    rule = product.get("rule") or {}
    status = str(rule.get("status", ""))
    label = "from the hosts" if status == "known" else "our assumption"
    tone = "ok" if status == "known" else "warn"
    order = ", ".join(str(name) for name in rule.get("perspective_order", ())) or "no preference"
    context = " · ".join(f"{_escape(key)}: {_escape(value)}" for key, value in (product.get("context") or {}).items())
    cards = "".join(
        _image_card(image, dimensions, index == 0, thumb_size) for index, image in enumerate(product["images"])
    )
    return (
        '<section class="product"><div class="eyebrow">PRODUCT</div>'
        f"<h2 class=mono>{_escape(product['product_id'])}</h2>"
        f'<p class="note">{context}</p>'
        f'<p class="rule-line">hero rule: <span class=mono>{_escape(order)}</span> '
        f'<span class="badge {tone}">{label}</span></p>'
        f'<div class="gallery">{cards}</div></section>'
    )


def _css() -> str:
    p = PALETTE
    return f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {p["paper"]}; color: {p["ink"]};
  font-family: Geist, "Segoe UI", system-ui, -apple-system, sans-serif; line-height: 1.45; }}
.mono {{ font-family: "Geist Mono", ui-monospace, "Cascadia Mono", Consolas, monospace; }}
main {{ max-width: 1400px; margin: 0 auto; padding: 48px 32px 96px; }}
h1 {{ font-size: 44px; font-weight: 700; letter-spacing: -0.03em; margin: 0 0 8px; }}
h2 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 6px; }}
.eyebrow {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.08em; color: {p["muted"]}; border-bottom: 1px solid {p["ink"]}; padding-bottom: 6px;
  margin-bottom: 14px; }}
.lead {{ font-size: 17px; color: {p["ink2"]}; max-width: 70ch; }}
.note {{ font-size: 12px; color: {p["muted2"]}; }}
.band, .product {{ background: {p["paper2"]}; border: 1px solid {p["rule"]}; border-radius: 4px;
  padding: 24px; margin: 24px 0; }}
.figures {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; border-top: 2px solid {p["ink"]};
  padding-top: 16px; }}
.number {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: 30px; color: {p["accent"]}; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 18px; margin-top: 16px; }}
.card {{ background: {p["white"]}; border: 1px solid {p["rule"]}; border-radius: 3px; padding: 12px; }}
.card.hero {{ border: 2px solid {p["accent"]}; }}
.card img {{ width: 100%; height: auto; display: block; background: {p["white"]}; }}
.missing {{ height: 140px; display: grid; place-items: center; color: {p["muted2"]}; font-size: 12px;
  border: 1px dashed {p["rule2"]}; }}
.hidden-wrap {{ position: relative; }}
.hidden-wrap img {{ filter: grayscale(1) contrast(0.6) blur(3px); }}
.veil {{ position: absolute; inset: 0; display: grid; place-items: center; font-size: 12px; font-weight: 700;
  color: {p["ink"]}; background: rgba(242,239,233,0.72); }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 8px; }}
.badge {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: 11px; padding: 2px 7px; border-radius: 2px;
  border: 1px solid {p["rule2"]}; color: {p["ink2"]}; }}
.badge.hero {{ border-color: {p["accent"]}; color: {p["accent"]}; }}
.badge.warn {{ border-color: {p["amber"]}; color: #8a6200; }}
.badge.bad {{ border-color: {p["red"]}; color: {p["red"]}; }}
.badge.ok {{ border-color: {p["green"]}; color: {p["green"]}; }}
.answers {{ display: grid; gap: 2px; margin-bottom: 8px; }}
.answer {{ display: flex; justify-content: space-between; gap: 8px; font-size: 12px;
  border-bottom: 1px solid {p["rule"]}; padding: 2px 0; }}
.dim {{ color: {p["muted"]}; }}
.rule-line {{ font-size: 13px; color: {p["ink2"]}; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid {p["rule"]}; }}
th {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.06em; color: {p["muted"]}; }}
tbody tr:nth-child(even) {{ background: {p["paper"]}; }}
.ok {{ color: {p["green"]}; }}
.bad {{ color: {p["red"]}; }}
.caveat {{ border-left: 3px solid {p["amber"]}; padding: 10px 14px; background: {p["paper2"]};
  margin: 10px 0; font-size: 13px; }}
.caveat.problem {{ border-left-color: {p["red"]}; }}
ul {{ margin: 8px 0 0 18px; padding: 0; font-size: 13px; }}
li {{ margin-bottom: 6px; }}
"""


def build_page(
    products: list[dict[str, Any]],
    rules: Rules | None = None,
    dimensions: tuple[str, ...] | None = None,
    meter: dict[str, Any] | None = None,
    mirror: dict[str, dict[str, Any]] | None = None,
    predictions: dict[str, dict[str, Any]] | None = None,
    title: str = "Hero-shot gallery",
    thumb_size: int = THUMB_SIZE,
) -> str:
    """The whole page as one HTML string. Nothing is fetched when it is opened."""
    rules = rules if rules is not None else load_rules()
    if dimensions is None:
        try:
            from galaxus.data.classes import dimensions as resolved_dimensions

            dimensions = resolved_dimensions()
        except Exception:  # noqa: BLE001 - fall back to whatever the predictions actually carry
            seen: list[str] = []
            for product in products:
                for image in product["images"]:
                    seen.extend(key for key in image["labels"] if key not in seen)
            dimensions = tuple(seen)

    ordered = [order_product(rules, product) for product in products]
    sections = "".join(_product_section(product, dimensions, thumb_size) for product in ordered)
    mirror_html = _mirror_panel(rules, mirror, predictions or {}) if mirror else ""

    assumptions = rules.assumptions()
    assumption_html = ""
    if assumptions:
        items = "".join(
            f"<li><span class=mono>{_escape(entry['rule'])}</span> — {_escape(entry['todo'])}</li>"
            for entry in assumptions
        )
        assumption_html = (
            '<section class="band"><div class="eyebrow">ASSUMPTIONS, NOT FACTS</div>'
            "<h2>Only one hero rule on this page came from the hosts</h2>"
            "<p class=lead>The deep dive gave us exactly one: mice are shot top down. Every rule below is ours, and "
            "needs confirming before anyone treats it as how Galaxus works.</p>"
            f"<ul>{items}</ul></section>"
        )
    sourced = rules.sources()
    source_html = ""
    if sourced:
        items = "".join(
            f"<li><span class=mono>{_escape(entry['rule'])}</span> — {_escape(entry['source'])}</li>"
            for entry in sourced
        )
        source_html = f'<section class="band"><div class="eyebrow">FROM THE HOSTS</div><ul>{items}</ul></section>'
    problem_html = "".join(f'<div class="caveat problem">{_escape(problem)}</div>' for problem in rules.problems)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_css()}</style></head>
<body><main>
<div class="eyebrow">HACK APERTUS ZURICH · DIGITEC GALAXUS · TEAM NOIDEA</div>
<h1>{_escape(title)}</h1>
<p class="lead">Each product below is ordered the way a shop would show it: the hero shot first, chosen from the
predicted perspective, background and subject. Images the <span class=mono>text_language</span> rule would hide are
veiled, and images carrying text are flagged.</p>
{problem_html}
{_meter_band(meter)}
{sections}
{mirror_html}
{assumption_html}
{source_html}
<p class="note">Generated offline from a predictions file. No competition image is committed to git
(AGENTS.md product rule 3); the images here are the prepared pixels the model was served.</p>
</main></body></html>
"""


def write_page(html_text: str, path: str | Path) -> Path:
    """Write the page, creating its directory. The caller picks a path under ``$GALAXUS_DATA/out/``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_text, encoding="utf-8")
    return target
