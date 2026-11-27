"""Issue #5: the official holdout and the scored-cell mask. Synthetic labels only."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import DIMENSIONS

from galaxus.data.split import (
    FOLD_COUNT,
    HOLDOUT_FRAC,
    HOLDOUT_SEED,
    PRODUCT_COLUMN,
    holdout_product_ids,
    iter_product_folds,
    product_folds,
    scored_mask,
    solution_frame,
    split_holdout,
)


def test_no_product_is_in_both_parts(train_table):
    rest, holdout = split_holdout(train_table)
    assert set(rest[PRODUCT_COLUMN]) & set(holdout[PRODUCT_COLUMN]) == set()
    assert len(rest) + len(holdout) == len(train_table)
    assert set(rest["image_id"]) | set(holdout["image_id"]) == set(train_table["image_id"])


def test_a_product_is_never_split_across_the_two_parts(train_table):
    _, holdout = split_holdout(train_table)
    for product_id in set(holdout[PRODUCT_COLUMN]):
        in_train = train_table[train_table[PRODUCT_COLUMN] == product_id]
        assert set(in_train["image_id"]) <= set(holdout["image_id"])


def test_the_split_is_deterministic(train_table):
    first = split_holdout(train_table)[1]["image_id"].tolist()
    second = split_holdout(train_table)[1]["image_id"].tolist()
    assert first == second


def test_the_split_depends_on_the_row_order_of_the_file(train_table):
    """``drop_duplicates()`` keeps first-occurrence order, so the holdout follows the row order of train.csv.

    This is the official recipe's own behaviour, not ours. It means the fixed seed-0 holdout is only reproducible
    while train.csv is read as Kaggle ships it: never sort, shuffle or filter rows before splitting.
    """
    shuffled = train_table.sample(frac=1.0, random_state=7)
    assert set(split_holdout(shuffled)[1]["image_id"]) != set(split_holdout(train_table)[1]["image_id"])


def test_the_split_follows_the_challenge_description_literally(train_table):
    """The same two lines as challenge_description.md, "Scoring locally", must give the same holdout."""
    expected_products = train_table[PRODUCT_COLUMN].drop_duplicates().sample(frac=0.2, random_state=0)
    expected = train_table[train_table[PRODUCT_COLUMN].isin(expected_products)]
    _, holdout = split_holdout(train_table)
    assert holdout["image_id"].tolist() == expected["image_id"].tolist()


def test_the_defaults_are_the_fixed_seed_zero_split():
    assert (HOLDOUT_FRAC, HOLDOUT_SEED) == (0.2, 0)


def test_about_a_fifth_of_the_products_is_held_out():
    train = pd.DataFrame({PRODUCT_COLUMN: [f"p{index}" for index in range(100)] * 2})
    train["image_id"] = [f"img_{index}" for index in range(len(train))]
    assert len(holdout_product_ids(train)) == 20


def test_a_table_without_product_id_is_refused(train_table):
    with pytest.raises(KeyError, match=PRODUCT_COLUMN):
        split_holdout(train_table.drop(columns=[PRODUCT_COLUMN]))


def test_model_prelabels_and_irrelevant_cells_are_not_scored(train_table):
    scored = scored_mask(train_table, "subject")
    by_image = dict(zip(train_table["image_id"], scored, strict=True))
    assert by_image["img_1"] is True or by_image["img_1"]
    assert not by_image["img_3"], "a model prelabel is never scored"
    assert not by_image["img_4"], "an irrelevant cell is never scored"
    assert not by_image["img_6"], "a model prelabel in an irrelevant cell is never scored"


def test_unscored_cells_are_none_in_the_solution(train_table):
    solution = solution_frame(train_table, DIMENSIONS)
    assert list(solution.columns) == ["image_id", *DIMENSIONS]
    by_image = dict(zip(solution["image_id"], solution["subject"], strict=True))
    assert by_image["img_1"] == "alpha"
    assert by_image["img_2"] == "beta"
    assert by_image["img_3"] is None
    assert by_image["img_4"] is None
    assert by_image["img_6"] is None
    assert by_image["img_7"] == "alpha"


def test_the_relevant_flag_is_read_as_a_boolean_whichever_way_the_csv_was_parsed(train_table):
    """``_relevant`` is ``true``/``false`` text when read as strings and ``bool`` with pandas' default parsing."""
    as_text = scored_mask(train_table, "subject").tolist()
    typed = train_table.copy()
    for dimension in DIMENSIONS:
        typed[f"{dimension}_relevant"] = typed[f"{dimension}_relevant"] == "true"
    assert scored_mask(typed, "subject").tolist() == as_text


def test_an_unreadable_relevant_flag_is_an_error(train_table):
    broken = train_table.copy()
    broken.loc[0, "subject_relevant"] = "maybe"
    with pytest.raises(ValueError, match="neither true nor false"):
        scored_mask(broken, "subject")


def test_missing_label_metadata_is_an_error(train_table):
    with pytest.raises(KeyError, match="subject_source"):
        scored_mask(train_table.drop(columns=["subject_source"]), "subject")


# Grouped cross-validation folds: BASE_ARCHITECTURE.md B1, "folds grouped by product_id for
# cross-validation on the rest of train". The feature heads and the label-error finder both need
# one out-of-fold prediction per row.


def catalog(n_products: int = 20) -> pd.DataFrame:
    """A synthetic catalog where products carry different numbers of images, as the real one does."""
    rows = []
    for product in range(n_products):
        for image in range(1 + product % 4):
            rows.append((f"img_{product}_{image}", f"p{product}"))
    frame = pd.DataFrame(rows, columns=["image_id", PRODUCT_COLUMN])
    for dimension in DIMENSIONS:
        frame[dimension] = "alpha"
    return frame


def test_a_product_never_straddles_two_folds():
    frame = catalog()
    folds = product_folds(frame)
    per_product = frame.assign(fold=folds).groupby(PRODUCT_COLUMN)["fold"].nunique()
    assert per_product.max() == 1


def test_every_row_lands_in_exactly_one_fold():
    frame = catalog()
    folds = product_folds(frame)
    assert len(folds) == len(frame)
    assert folds.index.equals(frame.index)
    assert set(folds.unique()) == set(range(FOLD_COUNT))


def test_the_folds_are_deterministic_for_a_seed():
    frame = catalog()
    assert product_folds(frame).equals(product_folds(frame))


def test_the_seed_is_a_parameter():
    frame = catalog(40)
    assert not product_folds(frame, random_state=0).equals(product_folds(frame, random_state=1))


def test_the_fold_count_is_a_parameter():
    folds = product_folds(catalog(), n_folds=3)
    assert set(folds.unique()) == {0, 1, 2}


def test_the_folds_are_balanced_by_image_count_not_product_count():
    """Products carry 1 to 4 images, so balancing by product would leave lopsided folds."""
    frame = catalog(40)
    counts = product_folds(frame).value_counts()
    assert counts.max() - counts.min() <= 4


def test_a_table_without_product_id_is_refused_by_the_folds():
    with pytest.raises(KeyError, match=PRODUCT_COLUMN):
        product_folds(catalog().drop(columns=[PRODUCT_COLUMN]))


@pytest.mark.parametrize("n_folds", [0, 1])
def test_fewer_than_two_folds_is_refused(n_folds):
    with pytest.raises(ValueError, match="at least 2"):
        product_folds(catalog(), n_folds=n_folds)


def test_more_folds_than_products_is_refused():
    with pytest.raises(ValueError, match="only 3 product"):
        product_folds(catalog(3), n_folds=5)


def test_each_fold_yields_a_fit_part_and_a_validate_part():
    frame = catalog()
    pairs = list(iter_product_folds(frame))
    assert len(pairs) == FOLD_COUNT
    for fit, validate in pairs:
        assert len(fit) + len(validate) == len(frame)
        assert not validate.empty
        assert set(fit[PRODUCT_COLUMN]) & set(validate[PRODUCT_COLUMN]) == set()


def test_the_validate_parts_cover_every_row_exactly_once():
    """This is what makes the predictions out-of-fold: every row predicted by a model that did not fit it."""
    frame = catalog()
    seen = pd.concat([validate for _, validate in iter_product_folds(frame)])
    assert sorted(seen["image_id"]) == sorted(frame["image_id"])


def test_folds_on_the_rest_of_train_never_touch_the_holdout(train_table):
    """B1 asks for folds on train minus the holdout, so a model tuned on them is not tuned on the holdout."""
    rest, holdout = split_holdout(train_table)
    for _, validate in iter_product_folds(rest, n_folds=2):
        assert set(validate[PRODUCT_COLUMN]) & set(holdout[PRODUCT_COLUMN]) == set()
