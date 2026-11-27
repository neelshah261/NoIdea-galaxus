"""The hero-shot gallery simulator (issue #31): an offline page a judge can open.

``python -m galaxus.serve.demo --exp EXP-0001 --part dev`` writes it under ``$GALAXUS_DATA/out/demo/``.
"""

from __future__ import annotations

from galaxus.serve.demo.gallery import (
    PALETTE,
    THUMB_SIZE,
    build_page,
    order_product,
    products_from_frame,
    write_page,
)
from galaxus.serve.demo.rules import (
    Rules,
    hero_sort_key,
    hidden_reason,
    load_rules,
    mirror_class,
    mirrored_labels,
    product_rule,
    text_overlay,
)

__all__ = [
    "PALETTE",
    "THUMB_SIZE",
    "Rules",
    "build_page",
    "hero_sort_key",
    "hidden_reason",
    "load_rules",
    "mirror_class",
    "mirrored_labels",
    "order_product",
    "product_rule",
    "products_from_frame",
    "text_overlay",
    "write_page",
]
