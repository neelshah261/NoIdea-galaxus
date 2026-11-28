"""A (#211): the catalog text as features for the heads, next to the image features.

System V builds this on branch ``model/ad-a-context`` (draft PR #212, issue #211).

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
* **Only numpy, no new dependency:** hashed unigrams and bigrams per field into :data:`COLUMNS` columns, then one
  head per dimension (``heads.fit_head``) whose log-probabilities are the features.
* **With D (#211):** D's teacher rows are dev images, which carry the catalog fields, so they get their vectors from
  :meth:`ContextFeatures.transform`; the text model itself trains on human-labelled train rows only.
* **Determinism:** seeded dropout, seeded heads, a fixed hashing function (not Python's salted ``hash``).
* **Never:** a class name typed here (``galaxus.data.classes``), competition data in a test, a holdout row in fit.

How it is built, in two layers:

1. **The hashed text.** Per field, the lowercase word unigrams and bigrams of that field, each prefixed with the
   field's own name (``product_type=kettle``), so the same word in two fields stays two different features. Each
   token is hashed with ``zlib.crc32`` -- fixed for all time and across processes, unlike Python's salted ``hash``
   -- into one of :data:`COLUMNS` columns, counted, put through ``log1p``, multiplied by the idf of the rows the
   model was fitted on, and the row divided by its own L2 norm. One extra column is the "no context" flag, 1 exactly
   when there is no text at all, so a head can answer differently when the catalog is missing.
2. **One text head per dimension**, ``heads.fit_head`` on that dimension's scored rows (#51's class weights). The
   vector handed to the image heads is every dimension's log-probabilities concatenated, in
   :attr:`ContextFeatures.layout`'s order, plus the flag.

Why the log-probabilities and not the hashed columns themselves: 4,096 sparse columns would be five times the image
features, and the image head would need far more rows to make use of them. The text heads compress the text to one
number per class, each trained with the same macro-F1 class weights, so the image head sees a small dense summary of
what the catalog implies and cannot drown in the hashed columns.
"""

from __future__ import annotations

import itertools
import re
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from galaxus.model import heads

#: The catalog columns of challenge_description.md "The task", as the competition csvs name them.
CONTEXT_FIELDS = ("product_title", "product_type", "brand", "category", "subcategory")
#: The share of training rows that get the "no context" vector (seeded), so a missing catalog still works.
DROPOUT = 0.3
#: ``HELPER_CONTEXT=1`` turns the text features on in ``helper.from_env`` and ``helper.fused_from_env``.
CONTEXT_ENV = "HELPER_CONTEXT"
#: Hashed columns for the text, plus one more for the "no context" flag (#211's contract).
COLUMNS = 4096
#: Log-probabilities are clipped here, and a class a fold never saw gets this value, so one absent rare class cannot
#: put a huge number next to the image features. ``exp(-30)`` is 9e-14: "this class is out", on a bounded scale.
LOG_FLOOR = -30.0
ID_COLUMN = "image_id"
#: Words for the hashing: runs of letters and digits, so punctuation and separators never become features.
WORD = re.compile(r"[0-9a-z]+")


def tokens(context: Mapping[str, Any] | None, fields: Sequence[str] = CONTEXT_FIELDS) -> list[str]:
    """The hashing tokens of one catalog context: per field, its word unigrams and bigrams, field-prefixed.

    ``context`` is any mapping from the catalog field names to their values, so an ``ImageItem.context`` and a csv
    row (D's teacher rows, #211) both work. A missing, empty or all-empty context gives no tokens at all, which is
    what makes it the "no context" vector.
    """
    if context is None:
        return []
    out: list[str] = []
    for field in fields:
        value = context.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        words = WORD.findall(str(value).lower())
        out.extend(f"{field}={word}" for word in words)
        out.extend(f"{field}={first} {second}" for first, second in itertools.pairwise(words))
    return out


def column_of(token: str, columns: int = COLUMNS) -> int:
    """``token``'s hashed column. ``zlib.crc32`` is fixed for all time; Python's ``hash`` is salted per process."""
    return zlib.crc32(token.encode("utf-8")) % columns


def document_frequency(token_lists: Iterable[Sequence[str]], columns: int = COLUMNS) -> np.ndarray:
    """The idf of each hashed column over the rows a model is fitted on (smoothed, so an unseen column is finite)."""
    df = np.zeros(columns, dtype=np.float64)
    documents = 0
    for row in token_lists:
        if not row:
            continue
        documents += 1
        df[np.unique([column_of(token, columns) for token in row])] += 1.0
    return (np.log((1.0 + documents) / (1.0 + df)) + 1.0).astype(np.float32)


def hashed_matrix(token_lists: Sequence[Sequence[str]], idf: np.ndarray, columns: int = COLUMNS) -> np.ndarray:
    """``(n, columns + 1)``: log1p counts times idf at unit length, and the "no context" flag in the last column."""
    x = np.zeros((len(token_lists), columns + 1), dtype=np.float32)
    for r, row in enumerate(token_lists):
        if not row:
            x[r, columns] = 1.0  # no catalog text at all: the flag is the only thing this row says
            continue
        counts = np.bincount([column_of(token, columns) for token in row], minlength=columns)
        values = np.log1p(counts).astype(np.float32) * idf
        norm = float(np.linalg.norm(values))
        x[r, :columns] = values / norm if norm else values
    return x


def dropped(image_id: str, seed: int = 0, share: float = DROPOUT) -> bool:
    """Whether this training row's context is dropped: a fixed hash of ``seed`` and the id, so the choice is seeded,
    reproducible in any process, and independent of the order the rows arrive in."""
    if share <= 0:
        return False
    return zlib.crc32(f"{seed}:{image_id}".encode()) % 10_000 < round(share * 10_000)


@dataclass(frozen=True)
class TextModel:
    """One fitted text model: the idf of its own rows, the columns it may use, and one head per dimension.

    ``keep`` is the heart of it. ``heads.fit_head`` standardises every column by the variance of the rows it was
    fitted on, and a hashed column those rows never used has no variance at all: its scale falls back to 1e-6, so
    one new word in a row being predicted would arrive as its value divided by 1e-6 and saturate the head -- the
    answer then says more about which words are new than about the product. Measured while building this (#212): the
    out-of-fold text answers were right 48.3% of the time, chance for two classes, with every vector pinned at the
    floor. So a column no fit row varied on stays zero everywhere, which is also what a TF-IDF vocabulary learnt on
    the training rows does: a word the model never saw carries no weight it could have learnt.
    """

    idf: np.ndarray
    keep: np.ndarray
    heads: dict[str, heads.Head]


class ContextFeatures:
    """Catalog-text features for the heads. See the module docstring for the contract.

    :meth:`fit` gives the model :meth:`transform` uses for new items; :meth:`out_of_fold` gives the training rows
    their vectors from fold models that never saw their product. Both produce vectors of the same width and column
    order (:attr:`layout`), which is what lets the image heads train on one and predict on the other.
    """

    def __init__(self, fields: Sequence[str] = CONTEXT_FIELDS, columns: int = COLUMNS, dropout: float = DROPOUT,
                 epochs: int = heads.EPOCHS) -> None:
        self.fields = tuple(fields)
        self.columns = int(columns)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        #: Per dimension, its classes in the order their log-probabilities sit in the vector. Set by :meth:`fit`.
        self.layout: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self.model: TextModel | None = None

    # ---------------------------------------------------------------- the hashed text

    def row_tokens(self, rows: pd.DataFrame) -> list[list[str]]:
        """One token list per row of a competition csv, from whichever catalog fields the table carries."""
        present = [field for field in self.fields if field in rows.columns]
        if not present:
            return [[] for _ in range(len(rows))]
        return [tokens(context, present) for context in rows[present].to_dict("records")]

    def _fit_text_model(self, rows: pd.DataFrame, data: Any, seed: int) -> TextModel:
        """A text model on ``rows``: the idf of these rows and one head per dimension on their scored human labels.

        Written to hold as few copies of the hashed matrix as it can. On the real training rows it is 9,432 by 4,097
        float32, 155 MB, and ``fit_head`` makes several temporaries of that size of its own; the masking is in place
        and the rows of one dimension are taken by position, so this adds one copy rather than three.
        """
        token_lists = self.row_tokens(rows)
        idf = document_frequency(token_lists, self.columns)
        x = hashed_matrix(token_lists, idf, self.columns)
        keep = x.std(axis=0) > 0  # see TextModel.keep: a column these rows never varied on would saturate the head
        x *= keep
        position = {image_id: k for k, image_id in enumerate(rows[ID_COLUMN].astype(str))}
        fitted: dict[str, heads.Head] = {}
        for dimension in data.dimensions():
            scored = rows[data.scored_mask(rows, dimension)]
            if scored.empty:
                continue  # a fold without one scored row for this dimension: its classes stay at the floor
            take = [position[image_id] for image_id in scored[ID_COLUMN].astype(str)]
            fitted[dimension] = heads.fit_head(
                x[take],
                scored[dimension].astype(str).tolist(),
                epochs=self.epochs,
                seed=seed,
            )
        return TextModel(idf=idf, keep=keep, heads=fitted)

    # ---------------------------------------------------------------- the vectors the image heads see

    @property
    def width(self) -> int:
        """The number of columns :meth:`transform` adds after the image features."""
        return sum(len(classes) for _, classes in self.layout) + 1

    def _hashed(self, token_lists: Sequence[Sequence[str]], model: TextModel) -> np.ndarray:
        """The hashed rows weighted by ``model``'s own idf. :meth:`_vectors` applies ``model.keep`` to them."""
        return hashed_matrix(token_lists, model.idf, self.columns)

    def _vectors(self, x: np.ndarray, model: TextModel) -> np.ndarray:
        """``(n, width)``: every dimension's log-probabilities in :attr:`layout`'s order, then the flag.

        ``x`` is the raw hashed matrix. The heads see it through ``model.keep``, so a word they never saw cannot
        saturate them; the flag in the vector is the row's real flag, not the masked one.
        """
        masked = x * model.keep
        parts: list[np.ndarray] = []
        for dimension, classes in self.layout:
            block = np.full((len(x), len(classes)), LOG_FLOOR, dtype=np.float32)
            head = model.heads.get(dimension)
            if head is not None:
                log_probs = np.clip(head.log_probs(masked), LOG_FLOOR, 0.0)
                index = {name: k for k, name in enumerate(classes)}
                for k, name in enumerate(head.classes):
                    if name in index:  # a class this fold never saw keeps the floor
                        block[:, index[name]] = log_probs[:, k]
            parts.append(block)
        parts.append(x[:, self.columns : self.columns + 1])  # the "no context" flag, as the text heads also saw it
        return np.hstack(parts).astype(np.float32)

    def _no_context(self, model: TextModel) -> np.ndarray:
        """The vector of a row with no catalog text at all, through ``model``."""
        return self._vectors(self._hashed([[]], model), model)[0]

    # ---------------------------------------------------------------- the contract (#211)

    def fit(self, rows: pd.DataFrame, data: Any, seed: int = 0) -> ContextFeatures:
        """Fit the text model on ``rows`` (train minus the holdout, or every labelled row with HELPER_FIT_ALL).

        This also fixes :attr:`layout`, the column order every later vector follows, from the classes these rows
        carry per dimension.
        """
        layout: list[tuple[str, tuple[str, ...]]] = []
        for dimension in data.dimensions():
            scored = rows[data.scored_mask(rows, dimension)]
            if scored.empty:
                raise ValueError(f"no scored human label for {dimension}: the text head has nothing to learn from")
            layout.append((dimension, tuple(sorted(set(scored[dimension].astype(str))))))
        self.layout = tuple(layout)
        self.model = self._fit_text_model(rows, data, seed)
        return self

    def out_of_fold(self, rows: pd.DataFrame, data: Any, seed: int = 0) -> dict[str, np.ndarray]:
        """``{image_id: vector}`` for the training rows, each from a model that never saw that row's product.

        The folds are ``helper.product_folds(seed)``, the ones the heads calibrate on, so a row's text vector and its
        calibration come from the same split. A seeded :data:`DROPOUT` share gets the "no context" vector instead, so
        the image heads also learn what to do when the catalog is missing.
        """
        from galaxus.model.helper import product_folds

        if self.model is None:
            self.fit(rows, data, seed)
        out: dict[str, np.ndarray] = {}
        for fit_part, held in product_folds(seed)(rows):
            if fit_part.empty or held.empty:
                continue
            fold = self._fit_text_model(fit_part, data, seed)
            vectors = self._vectors(self._hashed(self.row_tokens(held), fold), fold)
            no_context = self._no_context(fold)
            for image_id, vector in zip(held[ID_COLUMN].astype(str), vectors, strict=True):
                out[image_id] = no_context if dropped(image_id, seed, self.dropout) else vector
        missing = [str(i) for i in rows[ID_COLUMN] if str(i) not in out]
        if missing:
            raise KeyError(f"{len(missing)} training row(s) are in no fold, first: {missing[:3]}; a row without an "
                           "out-of-fold vector would have to borrow one from a model that saw its product")
        return out

    def transform(self, contexts: Mapping[str, Mapping[str, Any] | None]) -> dict[str, np.ndarray]:
        """``{image_id: vector}`` for items to predict, from their catalog context (None: the "no context" vector).

        The values are an ``ImageItem.context`` or a csv row's catalog fields (D's teacher rows, #211); None, empty,
        or a context whose catalog fields are all empty all give the "no context" vector.
        """
        if self.model is None:
            raise RuntimeError("fit the text model before transform: it has no heads and no idf yet")
        keys = list(contexts)
        token_lists = [tokens(contexts[key], self.fields) for key in keys]
        vectors = self._vectors(self._hashed(token_lists, self.model), self.model)
        return dict(zip(keys, vectors, strict=True))

    def config(self) -> dict[str, Any]:
        """What the EXP row records about A, so a row says exactly which text features it used."""
        return {
            "fields": list(self.fields),
            "hashed_columns": self.columns,
            "hash": "zlib.crc32",
            "tokens": "lowercase word unigrams and bigrams, field-prefixed",
            "weighting": "log1p counts * idf, unit length, plus a no-context flag",
            "features": "one head per dimension, its log-probabilities concatenated, plus the flag",
            "width": self.width,
            "dropout": self.dropout,
            "out_of_fold": "product folds (helper.product_folds)",
            "epochs": self.epochs,
        }
