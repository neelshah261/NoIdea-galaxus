"""The official local holdout and the scored-cell mask, so our local score is exactly the official one.

Both follow [challenge_description.md, "Scoring
locally"](../../../challenge_description.md#scoring-locally) literally: the split is by ``product_id`` through
``drop_duplicates().sample(frac=0.2, random_state=0)``, and a cell is unscored when its ``{dimension}_source`` is
``model`` or its ``{dimension}_relevant`` flag is false. Unscored cells are ``None`` in the solution, which is what
``metric.py`` expects.

The ``sample`` call is pandas' own RNG, so the split is reproducible for a given pandas version; the holdout is a
fixed seed-0 split and must not be re-rolled (AGENTS.md "Galaxus product rules" 1).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from galaxus.data.classes import ID_COLUMN, dimensions

PRODUCT_COLUMN = "product_id"
#: Fraction of products held out, and the seed, both fixed by the challenge description.
HOLDOUT_FRAC = 0.2
HOLDOUT_SEED = 0

#: Cross-validation on train minus the holdout (BASE_ARCHITECTURE.md B1).
FOLD_COUNT = 5
FOLD_SEED = 0

_TRUE = frozenset({"true", "t", "yes", "y", "1"})
_FALSE = frozenset({"false", "f", "no", "n", "0", ""})


def _as_bool(column: pd.Series) -> pd.Series:
    """A ``{dimension}_relevant`` column as booleans, whether the CSV was read as strings or as ``bool``."""
    if column.dtype == bool:
        return column

    def one(value: object) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"{column.name} holds {value!r}, which is neither true nor false")

    return column.map(one).astype(bool)


def holdout_product_ids(train: pd.DataFrame, frac: float = HOLDOUT_FRAC, random_state: int = HOLDOUT_SEED) -> pd.Series:
    """The products held out, exactly as in challenge_description.md "Scoring locally"."""
    return train[PRODUCT_COLUMN].drop_duplicates().sample(frac=frac, random_state=random_state)


def split_holdout(
    train: pd.DataFrame, frac: float = HOLDOUT_FRAC, random_state: int = HOLDOUT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``train`` split by product into (train minus holdout, holdout).

    No product appears in both parts, so a product's other images cannot leak into the holdout.
    """
    if PRODUCT_COLUMN not in train.columns:
        raise KeyError(f"{PRODUCT_COLUMN} is not a column of this table: the split must be by product, not by image")
    held_out = train[PRODUCT_COLUMN].isin(holdout_product_ids(train, frac, random_state))
    return train[~held_out], train[held_out]


def product_folds(frame: pd.DataFrame, n_folds: int = FOLD_COUNT, random_state: int = FOLD_SEED) -> pd.Series:
    """A fold number per row, grouped so that one product's images never straddle two folds.

    BASE_ARCHITECTURE.md B1 asks for these "on the rest of train", so pass ``split_holdout(train)[0]``:
    the fixed holdout stays out of cross-validation, or a model tuned on these folds would be tuned on
    the holdout it is later scored against.

    Folds are balanced by image count rather than by product count, because products carry very
    different numbers of images. Products are permuted with ``random_state``, then taken largest
    first and each put in the fold holding the fewest images, which is deterministic for a given
    seed and set of products.
    """
    if PRODUCT_COLUMN not in frame.columns:
        raise KeyError(f"{PRODUCT_COLUMN} is not a column of this table: folds must be grouped by product")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2 to cross-validate, got {n_folds}")
    sizes = frame[PRODUCT_COLUMN].value_counts()
    if n_folds > len(sizes):
        raise ValueError(f"{n_folds} folds asked for, but only {len(sizes)} product(s) to spread over them")

    permuted = np.random.default_rng(random_state).permutation(sizes.index.to_numpy())
    sizes = sizes.reindex(permuted).sort_values(ascending=False, kind="stable")

    images_per_fold = [0] * n_folds
    fold_of_product: dict[object, int] = {}
    for product, size in sizes.items():
        fold = min(range(n_folds), key=lambda index: (images_per_fold[index], index))
        fold_of_product[product] = fold
        images_per_fold[fold] += int(size)
    return frame[PRODUCT_COLUMN].map(fold_of_product).astype(int)


def iter_product_folds(
    frame: pd.DataFrame, n_folds: int = FOLD_COUNT, random_state: int = FOLD_SEED
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """``(fit, validate)`` once per fold: ``validate`` is that fold, ``fit`` is every other fold.

    Predicting each ``validate`` part with a model fitted on its ``fit`` part gives one out-of-fold
    prediction per row, which is what the label-error finder and the feature heads need.
    """
    folds = product_folds(frame, n_folds, random_state)
    for index in range(n_folds):
        held = folds == index
        yield frame[~held], frame[held]


def scored_mask(holdout: pd.DataFrame, dimension: str) -> pd.Series:
    """Which cells of ``dimension`` are scored: a human label in a cell where the dimension is meaningful.

    A model prelabel never counts, and neither does a cell the labeling rules mark as not relevant.
    """
    source = f"{dimension}_source"
    relevant = f"{dimension}_relevant"
    missing = [column for column in (source, relevant) if column not in holdout.columns]
    if missing:
        raise KeyError(f"{', '.join(missing)} missing: the scored-cell mask needs the train label metadata")
    unscored = (holdout[source] == "model") | ~_as_bool(holdout[relevant])
    return ~unscored


def solution_frame(holdout: pd.DataFrame, dimension_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The solution ``metric.py`` scores against: the human label of every scored cell, ``None`` everywhere else."""
    names = tuple(dimension_names) if dimension_names is not None else dimensions()
    solution = holdout[[ID_COLUMN, *names]].copy()
    solution = solution.astype({name: object for name in names})
    for dimension in names:
        solution.loc[~scored_mask(holdout, dimension), dimension] = None
    return solution
