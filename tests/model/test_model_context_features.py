"""A (#211): the catalog text as features for the heads (``galaxus.model.context_features``).

Synthetic rows only: invented dimension and class names, invented catalog words, image features drawn from a
generator. No competition data, no image file, no encoder, no network.

What the tests pin, in the order #212 asks for them:

1. text that decides the class raises accuracy over image-only features;
2. an out-of-fold vector never comes from a text model that saw that row's product;
3. the dropout share is seeded, reproducible and order-independent;
4. a missing catalog works, in training and at prediction time;
5. two runs give bit-identical vectors, and the hashing is fixed rather than Python's salted ``hash``;
6. both eval names are registered.
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from galaxus.model import context_features as cf
from galaxus.model import helper
from galaxus.model.eval import PREDICTORS
from galaxus.types import ImageItem

DIMENSION = "kind"  # invented: a test never types a real dimension or class name (AGENTS.md "Hard rules")
CLASSES = ("left_thing", "right_thing")
#: Small and quick: far fewer hashed columns than the contract's 4,096, but the real number of epochs, because
#: a text head that has not converged is the one thing these tests must not hide (the gate is a measured score).
SMALL = {"columns": 256, "epochs": 300}

DATA = SimpleNamespace(dimensions=lambda: (DIMENSION,), scored_mask=lambda frame, d: frame[f"{d}_source"] == "human")


def synthetic_rows(products: int = 60, per_product: int = 2, seed: int = 0, decisive: bool = True) -> pd.DataFrame:
    """Rows whose class is decided by their title, two images per product, as a competition csv looks."""
    rng = np.random.default_rng(seed)
    records = []
    for p in range(products):
        label = CLASSES[p % 2]
        word = "alpha" if label == CLASSES[0] else "beta"
        for k in range(per_product):
            records.append({
                "image_id": f"img_{p:03d}_{k}",
                "image_file": "unused",
                "product_id": f"prod_{p:03d}",
                "product_title": f"{word} widget number {p}" if decisive else f"widget number {p}",
                "product_type": word if decisive else "widget",
                "brand": f"brand_{p % 7}",
                "category": "things",
                "subcategory": "small things",
                DIMENSION: label,
                f"{DIMENSION}_source": "human",
            })
    frame = pd.DataFrame(records)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def noise_embeddings(width: int = 4, seed: int = 7):
    """Image features that say nothing about the class, so only the text can lift the score."""

    def embed(pairs, _name=None):
        return {
            str(image_id): np.random.default_rng(seed + zlib.crc32(str(image_id).encode())).normal(0, 1, width)
            .astype(np.float32)
            for image_id, _ in pairs
        }

    return embed


def items_of(rows: pd.DataFrame, with_context: bool = True) -> list[ImageItem]:
    fields = [f for f in cf.CONTEXT_FIELDS if f in rows.columns]
    return [
        ImageItem(
            image_id=str(row["image_id"]),
            path="unused",
            context={f: row[f] for f in fields} if with_context else None,
        )
        for _, row in rows.iterrows()
    ]


def accuracy(predictor, rows: pd.DataFrame, with_context: bool = True) -> float:
    answers = predictor.predict(items_of(rows, with_context))
    truth = dict(zip(rows["image_id"].astype(str), rows[DIMENSION], strict=True))
    return float(np.mean([a.labels[DIMENSION] == truth[str(a.image_id)] for a in answers]))


# ---------------------------------------------------------------- 1. the text has to earn its place


def test_text_that_decides_the_class_beats_image_only_features():
    """The point of A: with the image saying nothing, the catalog alone must carry the dimension."""
    rows = synthetic_rows()
    embed = noise_embeddings()
    image_only = helper.factory(embed=embed)(rows, DATA, 0)
    with_text = helper.factory(embed=embed, context=cf.ContextFeatures(**SMALL))(rows, DATA, 0)
    assert accuracy(with_text, rows) > 0.95
    assert accuracy(with_text, rows) - accuracy(image_only, rows) > 0.3


def test_the_text_vector_sits_after_the_image_features():
    """#211's contract: concatenated *after* the image features, so a saved head's columns keep their meaning."""
    rows = synthetic_rows(products=20)
    features = cf.ContextFeatures(**SMALL)
    predictor = helper.factory(embed=noise_embeddings(width=5), context=features)(rows, DATA, 0)
    assert features.layout == ((DIMENSION, CLASSES),)
    assert features.width == len(CLASSES) + 1  # the log-probabilities, plus the "no context" flag
    assert predictor.heads_model.heads[DIMENSION].weights.shape[0] == 5 + features.width
    assert predictor.heads_model.key.endswith("+context")  # the EXP row says these are not the plain features


def test_a_vector_is_log_probabilities_and_the_flag():
    rows = synthetic_rows(products=20)
    features = cf.ContextFeatures(**SMALL).fit(rows, DATA, 0)
    vector = features.transform({"x": {"product_title": "alpha widget", "product_type": "alpha"}})["x"]
    assert vector.shape == (features.width,)
    assert (vector[:-1] <= 0).all() and (vector[:-1] >= cf.LOG_FLOOR).all()
    assert np.exp(vector[:-1]).sum() == pytest.approx(1.0, abs=1e-5)  # one head's log-probabilities over its classes
    assert vector[-1] == 0.0  # a context is present


# ---------------------------------------------------------------- 2. out of fold, really


def test_an_out_of_fold_vector_cannot_come_from_a_model_that_saw_the_row_product():
    """Every product gets a word of its own, so only a model that saw the product can be certain about it.

    Fitted on all rows, the text model is sure of each row (it has seen that word with that label). Out of fold the
    word is new, so the vector must be markedly less certain. Equal vectors would mean the training features came
    from the full model, which is exactly the leak ``out_of_fold`` exists to prevent.
    """
    rows = synthetic_rows(products=40)
    rows["product_title"] = [f"token{p}" for p in rows["product_id"]]  # a word that appears in one product only
    rows["product_type"] = "widget"
    column = {name: k for k, name in enumerate(CLASSES)}
    features = cf.ContextFeatures(dropout=0.0, **SMALL)
    out_of_fold = features.out_of_fold(rows, DATA, 0)
    inside = features.transform({str(r["image_id"]): r for _, r in rows.iterrows()})

    def truth_log_prob(vectors):
        return np.mean([vectors[str(r["image_id"])][column[r[DIMENSION]]] for _, r in rows.iterrows()])

    assert truth_log_prob(inside) > truth_log_prob(out_of_fold) + 0.2
    assert all(not np.array_equal(out_of_fold[k], inside[k]) for k in out_of_fold)


def test_every_training_row_gets_exactly_one_out_of_fold_vector():
    rows = synthetic_rows(products=30)
    vectors = cf.ContextFeatures(**SMALL).out_of_fold(rows, DATA, 0)
    assert set(vectors) == set(rows["image_id"].astype(str))
    assert {v.shape for v in vectors.values()} == {(len(CLASSES) + 1,)}


def test_the_fit_rows_are_the_only_rows_the_text_model_sees():
    """``fit`` is given train minus the holdout by the factory; a held-back product must not reach its heads."""
    rows = synthetic_rows(products=30)
    kept, held = rows[rows["product_id"] != "prod_000"], rows[rows["product_id"] == "prod_000"]
    features = cf.ContextFeatures(**SMALL).fit(kept, DATA, 0)
    unseen = features.transform({str(r["image_id"]): r for _, r in held.iterrows()})
    assert len(unseen) == len(held)  # it answers them, from words it learnt elsewhere, and never refits


# ---------------------------------------------------------------- 3. seeded dropout


def test_the_dropout_share_is_seeded_and_reproducible():
    ids = [f"img_{k:05d}" for k in range(4000)]
    share = np.mean([cf.dropped(i, 0, 0.3) for i in ids])
    assert share == pytest.approx(0.3, abs=0.02)
    assert [cf.dropped(i, 0, 0.3) for i in ids] == [cf.dropped(i, 0, 0.3) for i in ids]  # same seed, same choice
    assert [cf.dropped(i, 1, 0.3) for i in ids] != [cf.dropped(i, 0, 0.3) for i in ids]  # another seed, another set
    assert not any(cf.dropped(i, 0, 0.0) for i in ids)  # dropout off means off


def test_the_dropped_training_rows_are_exactly_the_seeded_ones():
    """The flag column says it: a dropped row carries the "no context" vector, whatever its catalog said."""
    rows = synthetic_rows(products=30)
    vectors = cf.ContextFeatures(dropout=0.3, **SMALL).out_of_fold(rows, DATA, 0)
    flagged = {image_id for image_id, vector in vectors.items() if vector[-1] == 1.0}
    assert flagged == {i for i in vectors if cf.dropped(i, 0, 0.3)}
    assert 0 < len(flagged) < len(vectors)


def test_dropout_does_not_touch_prediction():
    """Dropout teaches the heads to cope with a missing catalog; it never throws away a catalog we do have."""
    rows = synthetic_rows(products=20)
    features = cf.ContextFeatures(dropout=1.0, **SMALL).fit(rows, DATA, 0)
    present = features.transform({str(r["image_id"]): r for _, r in rows.iterrows()})
    assert all(vector[-1] == 0.0 for vector in present.values())


# ---------------------------------------------------------------- 4. a missing catalog


def test_a_missing_context_gives_the_no_context_vector():
    rows = synthetic_rows(products=20)
    features = cf.ContextFeatures(**SMALL).fit(rows, DATA, 0)
    vectors = features.transform({"none": None, "empty": {}, "blank": {"product_title": "", "brand": None}})
    assert all(vector[-1] == 1.0 for vector in vectors.values())
    assert np.array_equal(vectors["none"], vectors["empty"])
    assert np.array_equal(vectors["none"], vectors["blank"])  # a catalog of empty strings is a missing catalog


def test_an_item_without_a_context_is_predicted_rather_than_refused():
    """In production the catalog can be missing (challenge_description.md "The task"), so this must not raise."""
    rows = synthetic_rows(products=20)
    predictor = helper.factory(embed=noise_embeddings(), context=cf.ContextFeatures(**SMALL))(rows, DATA, 0)
    answers = predictor.predict(items_of(rows.head(4), with_context=False))
    assert [a.labels[DIMENSION] for a in answers] and all(a.labels[DIMENSION] in CLASSES for a in answers)


def test_a_table_without_catalog_columns_still_fits():
    """D's rows and some stubs carry no catalog at all: then every row is a "no context" row, and nothing breaks."""
    rows = synthetic_rows(products=20).drop(columns=list(cf.CONTEXT_FIELDS))
    features = cf.ContextFeatures(**SMALL)
    vectors = features.out_of_fold(rows, DATA, 0)
    assert all(vector[-1] == 1.0 for vector in vectors.values())


def test_a_csv_row_works_as_a_context():
    """#211: D's teacher rows come from a csv, so a pandas row must be as good as an ImageItem's context mapping."""
    rows = synthetic_rows(products=20)
    features = cf.ContextFeatures(**SMALL).fit(rows, DATA, 0)
    row = rows.iloc[0]
    from_row = features.transform({"a": row})["a"]
    from_mapping = features.transform({"a": {f: row[f] for f in cf.CONTEXT_FIELDS}})["a"]
    assert np.array_equal(from_row, from_mapping)


# ---------------------------------------------------------------- 5. determinism


def test_two_runs_give_bit_identical_vectors():
    rows = synthetic_rows(products=30)
    first = cf.ContextFeatures(**SMALL)
    second = cf.ContextFeatures(**SMALL)
    assert all(np.array_equal(v, second.out_of_fold(rows, DATA, 0)[k])
               for k, v in first.out_of_fold(rows, DATA, 0).items())
    contexts = {str(r["image_id"]): r for _, r in rows.iterrows()}
    assert all(np.array_equal(v, second.transform(contexts)[k]) for k, v in first.transform(contexts).items())


def test_a_row_vector_does_not_depend_on_what_else_is_in_the_batch():
    """D transforms its teacher rows in its own order and in its own batches (#211), so a row's vector must be its
    own: it may not shift because of the other rows beside it.

    This is about :meth:`transform` on a fitted model. The *training* vectors cannot promise the same, because the
    folds come from ``galaxus.data.split.product_folds``, whose ``value_counts`` breaks ties in the order the rows
    arrive, so a reordered csv gives a different (equally valid) split. Every run reads ``read_train()``, one fixed
    order, which is what makes an EXP row reproducible.
    """
    rows = synthetic_rows(products=30)
    features = cf.ContextFeatures(**SMALL).fit(rows, DATA, 0)
    contexts = {str(r["image_id"]): r for _, r in rows.iterrows()}
    together = features.transform(contexts)
    one_at_a_time = {key: features.transform({key: value})[key] for key, value in contexts.items()}
    reversed_order = features.transform(dict(reversed(list(contexts.items()))))
    # float32 matrix products are blocked differently for a batch of 60 and a batch of 1, so last-bit equal
    assert all(np.allclose(together[k], one_at_a_time[k], atol=1e-5) for k in together)
    assert all(np.allclose(together[k], reversed_order[k], atol=1e-5) for k in together)


def test_the_hash_is_fixed_for_all_time_not_pythons_salted_one():
    """``hash`` is salted per process (PYTHONHASHSEED), so a cached vector would differ between runs."""
    assert cf.column_of("product_type=kettle", 4096) == zlib.crc32(b"product_type=kettle") % 4096 == 2263
    assert cf.column_of("brand=acme", 4096) == 1196
    assert cf.column_of("product_title=electric kettle", 4096) == 810


def test_the_tokens_are_field_prefixed_unigrams_and_bigrams():
    assert cf.tokens({"product_type": "Electric Kettle"}, ("product_type",)) == [
        "product_type=electric", "product_type=kettle", "product_type=electric kettle"]
    assert cf.tokens({"brand": "ACME"}, ("brand", "product_type")) == ["brand=acme"]
    # the same word in two fields is two features, so "kettle" the type and "kettle" the title cannot be confused
    assert cf.tokens({"brand": "kettle"}, ("brand",)) != cf.tokens({"product_type": "kettle"}, ("product_type",))
    assert cf.tokens(None) == [] and cf.tokens({}) == []


def test_transform_before_fit_says_so_instead_of_answering():
    with pytest.raises(RuntimeError, match="fit the text model"):
        cf.ContextFeatures().transform({"a": None})


def test_a_dimension_without_a_scored_row_is_refused_rather_than_skipped():
    rows = synthetic_rows(products=20)
    rows[f"{DIMENSION}_source"] = "model"  # a prelabel, not a human label
    with pytest.raises(ValueError, match=DIMENSION):
        cf.ContextFeatures(**SMALL).fit(rows, DATA, 0)


# ---------------------------------------------------------------- 6. the wiring


def test_both_eval_names_are_registered():
    assert "helper-siglip-text" in PREDICTORS
    assert "helper-fused-text" in PREDICTORS


def test_the_env_flag_is_what_turns_a_on():
    assert helper._context({}) is None
    assert helper._context({cf.CONTEXT_ENV: "0"}) is None
    assert isinstance(helper._context({cf.CONTEXT_ENV: "1"}), cf.ContextFeatures)


def test_the_exp_row_records_which_text_features_were_used():
    rows = synthetic_rows(products=20)
    features = cf.ContextFeatures(**SMALL)
    predictor = helper.factory(embed=noise_embeddings(), context=features)(rows, DATA, 0)
    recorded = predictor.config()["context"]
    assert recorded["hash"] == "zlib.crc32"
    assert recorded["hashed_columns"] == SMALL["columns"]
    assert recorded["dropout"] == cf.DROPOUT
    assert recorded["width"] == features.width


def test_the_mirror_head_and_the_text_features_are_refused_together_not_silently_mixed():
    """The mirror head compares an image with its mirror, and both inputs must have the same columns (#211: A alone)."""
    rows = synthetic_rows(products=20)
    build = helper.factory(embed=noise_embeddings(), mirror=True, context=cf.ContextFeatures(**SMALL))
    with pytest.raises(ValueError, match="not combined yet"):
        build(rows, DATA, 0)
