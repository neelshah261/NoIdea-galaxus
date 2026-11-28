"""A (#211): the catalog text as features for the heads, next to the image features. **Skeleton: the contract only.**

System V builds this on branch ``model/ad-a-context`` (draft PR, #211).

Why it should help:

* Product text *inside the prompt* hurt every time (docs/method.md: EXP-0021 0.343 vs EXP-0025 0.360): the language
  model leaned on the text instead of the image.
* The catalog text *alone* (TF-IDF + logistic regression, docs/method.md) scores 0.425, so it carries signal the
  image lacks, mostly for ``subject``: the deep dive (#17) says the context decides accessory vs product.
* In a trained head the text is weighed against the image, class by class, with the class weights of #51.

The contract (fixed by #211 so A and D fit together):

* **Fields:** :data:`CONTEXT_FIELDS`, the catalog columns of challenge_description.md "The task".
* **Training features are out of fold:** :meth:`ContextFeatures.out_of_fold` gives every training row a vector made
  by a text model that never saw that row's product (``helper.product_folds``), so the heads cannot learn a leak.
* **Prediction features:** :meth:`ContextFeatures.transform` maps an item's ``ImageItem.context`` (holdout, dev, final)
  to the vector of the text model fitted on all training rows. A missing context gives the "no context" vector.
* **Context dropout:** a seeded share (:data:`DROPOUT`) of training rows gets the "no context" vector, so a catalog
  that is missing in production still works (challenge_description.md "The task").
* **Wiring (the only change in helper.py):** ``helper.factory(..., context=ContextFeatures())`` concatenates the
  vector *after* the image features; ``HELPER_CONTEXT=1`` turns it on in ``from_env`` and ``fused_from_env``; eval
  names ``helper-siglip-text`` and ``helper-fused-text``.
* **Only numpy, no new dependency:** for example hashed unigrams and bigrams per field into a few thousand columns,
  then one head per dimension (``heads.fit_head``) whose log-probabilities are the features.
* **With D (#211):** D's teacher rows are dev images, which carry the catalog fields, so they get their vectors from
  :meth:`ContextFeatures.transform`; the text model itself trains on human-labelled train rows only.
* **Determinism:** seeded dropout, seeded heads, a fixed hashing function (not Python's salted ``hash``).
* **Never:** a class name typed here (``galaxus.data.classes``), competition data in a test, a holdout row in fit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

#: The catalog columns of challenge_description.md "The task", as the competition csvs name them.
CONTEXT_FIELDS = ("product_title", "product_type", "brand", "category", "subcategory")
#: The share of training rows that get the "no context" vector (seeded), so a missing catalog still works.
DROPOUT = 0.3
#: ``HELPER_CONTEXT=1`` turns the text features on in ``helper.from_env`` and ``helper.fused_from_env``.
CONTEXT_ENV = "HELPER_CONTEXT"

_TODO = "A (#211) is not built yet: system V builds it on branch model/ad-a-context"


class ContextFeatures:
    """Catalog-text features for the heads. See the module docstring for the contract."""

    def fit(self, rows: pd.DataFrame, data: Any, seed: int = 0) -> ContextFeatures:
        """Fit the text model on ``rows`` (train minus the holdout, or every labelled row with HELPER_FIT_ALL)."""
        raise NotImplementedError(_TODO)

    def out_of_fold(self, rows: pd.DataFrame, data: Any, seed: int = 0) -> dict[str, np.ndarray]:
        """``{image_id: vector}`` for the training rows, each from a model that never saw that row's product."""
        raise NotImplementedError(_TODO)

    def transform(self, contexts: Mapping[str, Mapping[str, str] | None]) -> dict[str, np.ndarray]:
        """``{image_id: vector}`` for items to predict, from their catalog context (None: the "no context" vector)."""
        raise NotImplementedError(_TODO)
