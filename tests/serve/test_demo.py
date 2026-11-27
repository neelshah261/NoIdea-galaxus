"""Issue #31: the hero-shot gallery. Synthetic images and synthetic classes only, and never the network."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from PIL import Image as PillowImage
from serve_helpers import DIMENSIONS

from galaxus.serve.demo import gallery
from galaxus.serve.demo.rules import hidden_reason, load_rules, mirror_class, product_rule, text_overlay

# The fixture's classes are synthetic (``<dimension>_a`` and so on), so the rules used in these tests name those
# instead of real competition classes. The shipped hero_rules.json is checked separately.
TEST_RULES = {
    "match_context_keys": ["title"],
    "dimensions": {"view": "perspective", "backdrop": "background", "content": "subject", "language": "text_language",
                   "overlay": "text_classification"},
    "no_text_class": "none",
    "shop_languages": {
        "classes": ["text_language_a"],
        "status": "assumption",
        "todo": "TODO(owner): which languages the shop keeps",
    },
    "subject_preference": {
        "classes": ["subject_a"],
        "status": "assumption",
        "todo": "TODO(owner): confirm the hero subject",
    },
    "background_preference": {
        "classes": ["background_a"],
        "status": "known",
        "source": "issue #17 point 11",
    },
    "default": {
        "perspective_order": ["perspective_b"],
        "status": "assumption",
        "todo": "TODO(owner): confirm the default hero perspective",
    },
    "product_rules": [
        {
            "match": ["mice"],
            "perspective_order": ["perspective_a"],
            "status": "known",
            "source": "issue #17 point 1",
        }
    ],
}


@pytest.fixture
def rules_file(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(TEST_RULES), encoding="utf-8")
    return path


@pytest.fixture
def rules(rules_file):
    return load_rules(rules_file)


def labels(perspective, background="background_a", subject="subject_a", language="none", text="none"):
    return {
        "subject": subject,
        "background": background,
        "perspective": perspective,
        "individuals": "none",
        "text_classification": text,
        "text_language": language,
    }


@pytest.fixture
def product(tmp_path):
    """One product with two images on disk: a side view first in the file, and the hero angle second."""
    frame_rows, predictions = [], {}
    for image_id, perspective in (("img_1", "perspective_b"), ("img_2", "perspective_a")):
        path = tmp_path / f"{image_id}.png"
        PillowImage.new("RGB", (64, 48), (200, 200, 200)).save(path)
        frame_rows.append({"image_id": image_id, "product_id": "p1", "image_file": str(path), "title": "wireless mice"})
        predictions[image_id] = {"image_id": image_id, "labels": labels(perspective), "confidence": None}
    return pd.DataFrame(frame_rows), predictions


def test_the_known_mice_rule_beats_the_default(rules):
    assert product_rule(rules, {"title": "wireless mice"})["status"] == "known"
    assert product_rule(rules, {"title": "a kettle"})["status"] == "assumption"


def test_only_the_title_type_and_category_decide_the_product_rule(rules):
    """Deep dive #17 point 9: those columns are what tell you what a product is."""
    assert product_rule(rules, {"description": "mice"})["match"] is None


def test_the_hero_shot_is_first_even_though_it_is_second_in_the_file(product, rules):
    frame, predictions = product
    ordered = gallery.order_product(rules, gallery.products_from_frame(frame, predictions)[0])
    assert [image["image_id"] for image in ordered["images"]] == ["img_2", "img_1"]


def test_the_hero_card_is_marked_and_only_once(product, rules):
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert page.count('class="badge hero"') == 1
    assert page.count('class="card hero"') == 1


def test_the_page_never_reaches_the_network(product, rules):
    """It has to open offline, so no webfont, no CDN, no remote image."""
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "http://" not in page
    assert "https://" not in page


def test_the_images_are_embedded_so_the_file_stands_alone(product, rules):
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert page.count("data:image/jpeg;base64,") == 2


def test_an_unreadable_image_does_not_take_the_page_down(tmp_path, rules):
    frame = pd.DataFrame(
        [{"image_id": "img_1", "product_id": "p1", "image_file": str(tmp_path / "gone.png"), "title": "x"}]
    )
    predictions = {"img_1": {"image_id": "img_1", "labels": labels("perspective_a")}}
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "image not available" in page


def test_a_foreign_text_language_hides_the_image(rules):
    assert hidden_reason(rules, labels("perspective_a", language="text_language_b")) is not None


def test_the_kept_language_and_no_text_are_both_shown(rules):
    assert hidden_reason(rules, labels("perspective_a", language="text_language_a")) is None
    assert hidden_reason(rules, labels("perspective_a", language="none")) is None


def test_text_on_the_image_is_flagged(rules):
    assert text_overlay(rules, labels("perspective_a", text="text_classification_b")) == "text_classification_b"
    assert text_overlay(rules, labels("perspective_a")) is None


def test_a_hidden_image_is_veiled_on_the_page(tmp_path, rules):
    path = tmp_path / "img_1.png"
    PillowImage.new("RGB", (32, 32), (10, 10, 10)).save(path)
    frame = pd.DataFrame([{"image_id": "img_1", "product_id": "p1", "image_file": str(path), "title": "x"}])
    predictions = {"img_1": {"image_id": "img_1", "labels": labels("perspective_a", language="text_language_b")}}
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "hidden in the shop" in page


def test_every_assumption_is_shown_as_an_assumption(product, rules):
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "ASSUMPTIONS, NOT FACTS" in page
    assert "TODO(owner): confirm the default hero perspective" in page
    assert "our assumption" in page or "from the hosts" in page


def test_the_known_rule_is_shown_with_its_source(product, rules):
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "issue #17 point 1" in page


def test_product_context_is_escaped(tmp_path, rules):
    """The context comes from a competition csv, so it is data, not markup."""
    path = tmp_path / "img_1.png"
    PillowImage.new("RGB", (32, 32), (0, 0, 0)).save(path)
    frame = pd.DataFrame(
        [
            {
                "image_id": "img_1",
                "product_id": "p1",
                "image_file": str(path),
                "title": "<script>alert(1)</script>",
            }
        ]
    )
    predictions = {"img_1": {"image_id": "img_1", "labels": labels("perspective_a")}}
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_images_without_a_prediction_are_left_out(product):
    frame, predictions = product
    products = gallery.products_from_frame(frame, {"img_1": predictions["img_1"]})
    assert [image["image_id"] for image in products[0]["images"]] == ["img_1"]


def test_a_part_without_product_id_gives_one_product_per_image(tmp_path):
    frame = pd.DataFrame([{"image_id": "img_1", "image_file": str(tmp_path / "x.png"), "title": "t"}])
    predictions = {"img_1": {"image_id": "img_1", "labels": labels("perspective_a")}}
    products = gallery.products_from_frame(frame, predictions)
    assert len(products) == 1
    assert "no product_id" in products[0]["product_id"]


def test_the_meter_shows_images_per_second_and_cost(product, rules):
    frame, predictions = product
    meter = {"images_per_second": 12.5, "cost_per_100k": 4.2, "images": 3, "cost_note": "at $1.00/hour"}
    page = gallery.build_page(
        gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS, meter=meter
    )
    assert "12.50" in page
    assert "cost / 100k images" in page


def test_no_meter_band_without_timing(product, rules):
    frame, predictions = product
    page = gallery.build_page(
        gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS, meter=None
    )
    assert "images/s" not in page


def test_the_mirror_panel_flags_a_disagreement(product, rules):
    """A mirrored prediction that disagrees once left and right are swapped back is the interesting case."""
    frame, predictions = product
    mirror = {"img_2": {"image_id": "img_2", "labels": labels("perspective_b")}}
    page = gallery.build_page(
        gallery.products_from_frame(frame, predictions),
        rules=rules,
        dimensions=DIMENSIONS,
        mirror=mirror,
        predictions=predictions,
    )
    assert "MIRROR TEST-TIME AUGMENTATION" in page
    assert "disagrees" in page


def test_there_is_no_mirror_panel_without_mirrored_predictions(product, rules):
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    assert "MIRROR TEST-TIME AUGMENTATION" not in page


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("front_angle_left", "front_angle_right"),
        ("front_angle_right", "front_angle_left"),
        ("top_down", "top_down"),
    ],
)
def test_left_and_right_swap_without_any_class_name_in_the_code(name, expected):
    valid = {"front_angle_left", "front_angle_right", "top_down"}
    assert mirror_class(name, valid) == expected


def test_a_swap_that_is_not_a_valid_class_is_left_alone():
    """Swapping must never invent a class: if the swapped name is unknown, keep the original."""
    assert mirror_class("front_angle_left", {"front_angle_left"}) == "front_angle_left"


def test_the_shipped_rules_load_and_carry_their_sources():
    """The real hero_rules.json: exactly one rule came from the hosts, the rest are marked assumptions."""
    shipped = load_rules()
    assert any("mice" in entry["rule"] for entry in shipped.sources())
    assert shipped.assumptions()


def test_unknown_class_names_are_reported_against_the_classes_source(tmp_path, galaxus_data):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**TEST_RULES, "no_text_class": "not_a_class"}), encoding="utf-8")
    assert any("not_a_class" in problem for problem in load_rules(bad).problems)


def test_strict_loading_refuses_unknown_class_names(tmp_path, galaxus_data):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**TEST_RULES, "no_text_class": "not_a_class"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_class"):
        load_rules(bad, strict=True)


def test_the_page_is_written_where_it_is_asked_for(tmp_path, product, rules):
    frame, predictions = product
    page = gallery.build_page(gallery.products_from_frame(frame, predictions), rules=rules, dimensions=DIMENSIONS)
    written = gallery.write_page(page, tmp_path / "out" / "demo" / "page.html")
    assert written.is_file()
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")
