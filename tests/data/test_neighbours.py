"""Issue #112 part A: the similar-image index.

Synthetic throughout: solid-colour images written here, a stub encoder that turns a colour into a vector, invented
classes from ``conftest``. No network, no model download, no competition data. The stub is deliberately a colour
reader rather than a random generator, because it makes similarity something the test chooses: two images of the
same colour are neighbours, two of opposite colours are not.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import write_image

from galaxus.data.label_errors import find_label_errors, out_of_fold_frame
from galaxus.data.neighbours import (
    DEFAULT_ENCODER,
    EMBEDDING_DTYPE,
    Index,
    build_index,
    embed_rows,
    embedding_dir,
    encoder_slug,
    main,
    nearest,
    neighbour_votes,
    unit_rows,
    write_votes,
)

DIM = "subject"
OTHER = "background"


class ColourEncoder:
    """Returns each image's top-left RGB as its vector, so the test decides what is similar to what."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        return np.asarray([[float(channel) for channel in image.getpixel((0, 0))[:3]] for image in images])


def _rows(colours: dict[str, tuple[int, int, int]], folder, products: dict[str, str] | None = None):
    """(id, path) pairs for solid-colour images, written under ``folder``."""
    pairs = []
    for image_id, colour in colours.items():
        path = folder / f"{image_id}.png"
        write_image(path, "RGB", (40, 30), color=colour)
        pairs.append((image_id, path))
    return pairs


def _train(labels: dict[str, str], products: dict[str, str], sources: dict[str, str] | None = None):
    """A train table with one dimension, its source column, and a product per image."""
    sources = sources or {image_id: "human" for image_id in labels}
    return pd.DataFrame(
        {
            "image_id": list(labels),
            "product_id": [products[image_id] for image_id in labels],
            DIM: [labels[image_id] for image_id in labels],
            f"{DIM}_source": [sources[image_id] for image_id in labels],
        }
    )


def _index(vectors: dict[str, tuple[float, ...]], products: dict[str, str]) -> Index:
    ids = tuple(vectors)
    return Index(ids, tuple(products[i] for i in ids), unit_rows(np.asarray([vectors[i] for i in ids])))


# --- the small pieces -----------------------------------------------------------------------------------------


def test_the_encoder_name_becomes_one_folder_not_two():
    assert "/" not in encoder_slug("google/siglip-base-patch16-224")
    assert encoder_slug("google/siglip-base-patch16-224") == "google-siglip-base-patch16-224"


def test_the_embedding_folder_names_the_encoder(galaxus_data):
    """Two encoders produce incomparable vectors, so they must not share a folder."""
    first = embedding_dir("org/one")
    second = embedding_dir("org/two")
    assert first != second
    assert first.parent == second.parent == galaxus_data / "cache" / "embeddings"


def test_rows_are_scaled_to_length_one_and_a_zero_row_survives():
    scaled = unit_rows(np.asarray([[3.0, 4.0], [0.0, 0.0]]))
    assert np.isclose(np.linalg.norm(scaled[0]), 1.0)
    assert not np.isnan(scaled[1]).any()


def test_a_one_dimensional_array_is_refused():
    with pytest.raises(ValueError, match="one row per image"):
        unit_rows(np.asarray([1.0, 2.0, 3.0]))


# --- embedding ------------------------------------------------------------------------------------------------


def test_embeddings_are_cached_once_per_image_as_float16(galaxus_data, tmp_path):
    pairs = _rows({"a": (255, 0, 0), "b": (0, 255, 0)}, tmp_path)
    encoder = ColourEncoder()

    counts = embed_rows(pairs, encoder)

    assert counts["written"] == 2
    assert counts["skipped"] == 0
    stored = np.load(embedding_dir(DEFAULT_ENCODER) / "a.npy")
    assert stored.dtype == EMBEDDING_DTYPE


def test_a_second_run_embeds_nothing_so_an_interrupted_run_resumes(galaxus_data, tmp_path):
    pairs = _rows({"a": (255, 0, 0), "b": (0, 255, 0)}, tmp_path)
    encoder = ColourEncoder()
    embed_rows(pairs, encoder)

    again = embed_rows(pairs, encoder)

    assert again["written"] == 0
    assert again["skipped"] == 2
    assert encoder.calls == 1, "the encoder ran again for images that were already cached"


def test_an_image_whose_file_is_gone_is_counted_not_raised(galaxus_data, tmp_path):
    pairs = _rows({"a": (255, 0, 0)}, tmp_path) + [("b", tmp_path / "not_here.png")]

    counts = embed_rows(pairs, ColourEncoder())

    assert counts["written"] == 1
    assert counts["missing"] == 1


def test_an_encoder_returning_the_wrong_number_of_rows_is_refused(galaxus_data, tmp_path):
    class Short:
        def encode(self, images):
            return np.asarray([[1.0, 2.0, 3.0]])

    pairs = _rows({"a": (1, 2, 3), "b": (4, 5, 6)}, tmp_path)
    with pytest.raises(ValueError, match="returned 1 row"):
        embed_rows(pairs, Short())


def test_the_prepared_pixels_are_what_gets_embedded(galaxus_data, tmp_path):
    """The image cache is filled as a side effect, which is what makes the embedding describe the served pixels."""
    from galaxus.data.images import image_cache_dir

    pairs = _rows({"a": (10, 20, 30)}, tmp_path)
    embed_rows(pairs, ColourEncoder())

    assert (image_cache_dir() / "a.jpg").is_file()


# --- the index ------------------------------------------------------------------------------------------------


def test_the_index_follows_the_tables_rows(galaxus_data, tmp_path):
    pairs = _rows({"a": (255, 0, 0), "b": (0, 255, 0)}, tmp_path)
    embed_rows(pairs, ColourEncoder())
    train = _train({"a": "alpha", "b": "beta"}, {"a": "p1", "b": "p2"})

    index = build_index(train)

    assert index.image_ids == ("a", "b")
    assert index.products == ("p1", "p2")
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


def test_a_table_without_product_id_is_refused(galaxus_data):
    with pytest.raises(KeyError, match="product_id"):
        build_index(pd.DataFrame({"image_id": ["a"]}))


def test_the_holdout_can_be_kept_out_of_the_index(galaxus_data, tmp_path):
    """The leak guard: a holdout image's neighbour must not be a row the model is about to be scored on."""
    pairs = _rows({"a": (255, 0, 0), "b": (0, 255, 0)}, tmp_path)
    embed_rows(pairs, ColourEncoder())
    train = _train({"a": "alpha", "b": "beta"}, {"a": "p1", "b": "p2"})

    index = build_index(train, exclude_ids=["b"])

    assert index.image_ids == ("a",)


def test_an_index_with_nothing_cached_says_which_command_fills_it(galaxus_data):
    train = _train({"a": "alpha"}, {"a": "p1"})
    with pytest.raises(FileNotFoundError, match="neighbours embed"):
        build_index(train)


# --- the neighbour search -------------------------------------------------------------------------------------


def test_the_closest_neighbour_comes_first():
    vectors = {"q": (1.0, 0.0), "near": (0.9, 0.1), "far": (0.3, 0.9)}
    index = _index(vectors, {"q": "p1", "near": "p2", "far": "p3"})

    found = nearest(index, index, k=2, min_similarity=0.0)

    assert [neighbour for neighbour, _ in found["q"]] == ["near", "far"]


def test_a_row_is_never_its_own_neighbour():
    index = _index({"a": (1.0, 0.0), "b": (0.0, 1.0)}, {"a": "p1", "b": "p2"})
    found = nearest(index, index, k=5, min_similarity=-1.0)
    assert "a" not in [neighbour for neighbour, _ in found["a"]]


def test_another_image_of_the_same_product_is_never_a_neighbour():
    """A product's own images share its labels, so they would agree with it whatever the label is."""
    vectors = {"q": (1.0, 0.0), "sibling": (1.0, 0.0), "other": (0.8, 0.2)}
    index = _index(vectors, {"q": "p1", "sibling": "p1", "other": "p2"})

    found = nearest(index, index, k=5, min_similarity=0.0)

    assert [neighbour for neighbour, _ in found["q"]] == ["other"]


def test_a_distant_neighbour_is_dropped_rather_than_returned():
    vectors = {"q": (1.0, 0.0), "far": (0.0, 1.0)}
    index = _index(vectors, {"q": "p1", "far": "p2"})

    assert nearest(index, index, k=1, min_similarity=0.5) == {}


def test_k_limits_the_neighbours_kept():
    vectors = {"q": (1.0, 0.0), "a": (0.99, 0.01), "b": (0.98, 0.02), "c": (0.97, 0.03)}
    index = _index(vectors, {"q": "p1", "a": "p2", "b": "p3", "c": "p4"})

    assert len(nearest(index, index, k=2, min_similarity=0.0)["q"]) == 2


def test_the_same_answer_comes_back_however_the_queries_are_chunked():
    vectors = {f"i{n}": (1.0, n / 100) for n in range(10)}
    index = _index(vectors, {f"i{n}": f"p{n}" for n in range(10)})

    assert nearest(index, index, k=3, min_similarity=0.0, chunk=2) == nearest(
        index, index, k=3, min_similarity=0.0, chunk=1000
    )


def test_k_below_one_and_an_empty_index_are_refused():
    index = _index({"a": (1.0, 0.0)}, {"a": "p1"})
    with pytest.raises(ValueError, match="at least 1"):
        nearest(index, index, k=0)
    with pytest.raises(ValueError, match="index is empty"):
        nearest(index, Index((), (), np.zeros((0, 2), dtype=np.float32)))


# --- the votes ------------------------------------------------------------------------------------------------


def test_the_neighbours_majority_label_is_the_suggestion_and_their_agreement_is_the_confidence():
    train = _train(
        {"q": "alpha", "n1": "beta", "n2": "beta", "n3": "gamma"},
        {"q": "p1", "n1": "p2", "n2": "p3", "n3": "p4"},
    )
    neighbours = {"q": [("n1", 0.9), ("n2", 0.8), ("n3", 0.7)]}

    records = neighbour_votes(train, neighbours, (DIM,))

    assert records == [{"image_id": "q", "labels": {DIM: "beta"}, "confidence": {DIM: 2 / 3}}]


def test_a_model_prelabel_never_votes():
    """The prelabels are what this is checking, so letting them vote would have the pipeline confirm itself."""
    train = _train(
        {"q": "alpha", "human": "beta", "machine": "gamma"},
        {"q": "p1", "human": "p2", "machine": "p3"},
        sources={"q": "human", "human": "human", "machine": "model"},
    )
    neighbours = {"q": [("human", 0.9), ("machine", 0.9)]}

    records = neighbour_votes(train, neighbours, (DIM,))

    assert records[0]["labels"][DIM] == "beta"
    assert records[0]["confidence"][DIM] == 1.0, "the model prelabel was counted in the denominator"


def test_a_dimension_no_neighbour_has_a_human_label_for_is_left_out():
    train = _train({"q": "alpha", "n": "beta"}, {"q": "p1", "n": "p2"}, sources={"q": "human", "n": "model"})
    assert neighbour_votes(train, {"q": [("n", 0.9)]}, (DIM,)) == []


def test_a_tie_is_broken_by_name_so_two_runs_write_the_same_file():
    train = _train({"q": "alpha", "n1": "zeta", "n2": "beta"}, {"q": "p1", "n1": "p2", "n2": "p3"})
    records = neighbour_votes(train, {"q": [("n1", 0.9), ("n2", 0.9)]}, (DIM,))
    assert records[0]["labels"][DIM] == "beta"


def test_a_missing_dimension_column_is_named():
    train = _train({"q": "alpha", "n": "beta"}, {"q": "p1", "n": "p2"})
    with pytest.raises(KeyError, match=OTHER):
        neighbour_votes(train, {"q": [("n", 0.9)]}, (OTHER,))


# --- writing, and the hand-off to #7 --------------------------------------------------------------------------


def test_the_votes_are_written_where_the_label_error_finder_reads_them(galaxus_data):
    records = [{"image_id": "q", "labels": {DIM: "beta"}, "confidence": {DIM: 0.75}}]

    written = write_votes(records)

    assert written == galaxus_data / "out" / "neighbour_votes.jsonl"
    frame = out_of_fold_frame(written)
    assert frame.to_dict(orient="records") == [
        {"image_id": "q", "dimension": DIM, "prediction": "beta", "confidence": 0.75}
    ]


def test_the_votes_refuse_to_land_in_a_git_work_tree(galaxus_data, tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="never enter git"):
        write_votes([{"image_id": "q", "labels": {DIM: "beta"}, "confidence": {DIM: 0.9}}], tmp_path / "votes.jsonl")


def test_the_votes_become_a_label_error_report(galaxus_data):
    """The point of part A: the neighbours' disagreement reaches #7's csv as a second signal."""
    train = _train({"q": "alpha", "n1": "beta", "n2": "beta"}, {"q": "p1", "n1": "p2", "n2": "p3"})
    records = neighbour_votes(train, {"q": [("n1", 0.9), ("n2", 0.8)]}, (DIM,))

    report = find_label_errors(write_votes(records), train, dimension_names=(DIM,))

    assert report.to_dict(orient="records") == [
        {"image_id": "q", "dimension": DIM, "label": "alpha", "suggested_label": "beta"}
    ]


# --- the command line -----------------------------------------------------------------------------------------


@pytest.fixture
def data_with_embeddings(galaxus_data):
    """$GALAXUS_DATA with a train.csv and an embedding per row, written straight into the cache."""
    train = _train(
        {"q": "alpha", "n1": "beta", "n2": "beta"},
        {"q": "p1", "n1": "p2", "n2": "p3"},
    )
    train.to_csv(galaxus_data / "train.csv", index=False)
    folder = embedding_dir(DEFAULT_ENCODER)
    folder.mkdir(parents=True, exist_ok=True)
    for image_id, vector in (("q", (1.0, 0.0)), ("n1", (0.99, 0.01)), ("n2", (0.98, 0.02))):
        np.save(folder / f"{image_id}.npy", np.asarray(vector, dtype=EMBEDDING_DTYPE))
    return galaxus_data


def test_the_cli_writes_the_votes_and_prints_no_image_id(data_with_embeddings, capsys):
    assert main(["label-errors", "--k", "2", "--min-similarity", "0.0", "--dimensions", DIM]) == 0

    printed = capsys.readouterr().out
    assert "never commit it" in printed
    assert "3 image(s) in the index" in printed
    for image_id in ("q", "n1", "n2"):
        assert f" {image_id} " not in printed
    written = data_with_embeddings / "out" / "neighbour_votes.jsonl"
    assert written.is_file()
    record = json.loads(written.read_text(encoding="utf-8").splitlines()[0])
    assert set(record) == {"image_id", "labels", "confidence"}


def test_the_cli_refuses_a_k_below_one(data_with_embeddings):
    with pytest.raises(SystemExit):
        main(["label-errors", "--k", "0"])


def test_the_cli_reports_an_empty_cache_without_a_traceback(galaxus_data, capsys):
    _train({"q": "alpha"}, {"q": "p1"}).to_csv(galaxus_data / "train.csv", index=False)

    assert main(["label-errors", "--dimensions", DIM]) == 1
    assert capsys.readouterr().err.startswith("cannot work out the neighbours:")


def test_the_default_encoder_is_never_downloaded_by_these_tests():
    """A guard on the tests themselves: #112 asks for no network and no model download."""
    import galaxus.data.neighbours as module

    assert "transformers" not in dir(module), "transformers must stay inside load_encoder, imported lazily"


def test_a_duplicated_image_id_under_another_product_is_still_not_its_own_neighbour():
    """Why `nearest` excludes the row itself as well as its product, which looks redundant and nearly is.

    A row's own product contains the row, so the product mask already hides it; removing the self-exclusion
    leaves every other test in this file passing. It bites only on an inconsistent table, where the same image
    id carries two different `product_id` values - a duplicate row, or a merge that went wrong. Then the product
    mask hides one copy and not the other, and an image would be its own evidence.
    """
    index = Index(("q", "q", "other"), ("p1", "p2", "p3"), unit_rows(np.asarray([[1.0, 0.0]] * 3)))
    queries = Index(("q",), ("p1",), unit_rows(np.asarray([[1.0, 0.0]])))

    found = nearest(queries, index, k=3, min_similarity=0.0)

    assert [neighbour for neighbour, _ in found["q"]] == ["other"]
