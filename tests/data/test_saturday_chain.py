"""The data lane in the order Saturday runs it, composed instead of unit-tested.

Every piece here has its own tests. What none of them covers is the *chain*: on Saturday the final part is
loaded, its pixels prepared, answers merged over ``sample_submission.csv``, the file validated, the holdout
scored through the hosts' ``metric.py``, and the label-error csv written outside the repository - in that order,
once, against data nobody has seen. A break between two of those steps passes every unit test and would be
found at 13:00 on the day (challenge_description.md, Timeline).

Synthetic throughout: invented ids, the invented classes from ``conftest``, images generated here, and a
stand-in ``metric.py`` with the semantics the real one is documented to have. No competition data, no network.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import DIMENSIONS, FAKE_CLASSES, write_image

from galaxus.data.images import PATCH_SIZE, fill_image_cache, image_cache_dir
from galaxus.data.label_errors import main as label_errors_main
from galaxus.data.load import read_final
from galaxus.data.metric import score
from galaxus.data.split import solution_frame, split_holdout
from galaxus.data.submission import write_submission
from galaxus.data.validate import read_submission

SIZE = PATCH_SIZE * 2
FINAL_IDS = ["img_00001", "img_00002", "img_00003"]
#: What a predictor answers: the *second* class everywhere, so every cell differs from the sample's constant.
ANSWER = {dimension: FAKE_CLASSES[dimension][1] for dimension in DIMENSIONS}

METRIC_STANDIN = '''
"""Stand-in for the hosts metric.py: macro F1 per dimension over the scored cells, then the mean."""


def _macro_f1(truth, predicted):
    classes = sorted({c for c in list(truth) + list(predicted) if c is not None})
    total = 0.0
    for name in classes:
        tp = sum(1 for t, p in zip(truth, predicted) if t == name and p == name)
        fp = sum(1 for t, p in zip(truth, predicted) if t != name and p == name)
        fn = sum(1 for t, p in zip(truth, predicted) if t == name and p != name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return total / len(classes) if classes else 0.0


def _per_dimension(solution, submission, id_column):
    merged = solution.merge(submission, on=id_column, suffixes=("_true", "_pred"))
    out = {}
    for column in solution.columns:
        if column == id_column:
            continue
        keep = merged[merged[f"{column}_true"].notna()]
        out[column] = _macro_f1(list(keep[f"{column}_true"]), list(keep[f"{column}_pred"]))
    return out


def score(solution, submission, id_column):
    per = _per_dimension(solution, submission, id_column)
    return sum(per.values()) / len(per) if per else 0.0


def evaluate(solution, submission, id_column):
    per = _per_dimension(solution, submission, id_column)
    return {"per_dimension": per, "overall": {"f1_macro": score(solution, submission, id_column)}}
'''


def _train_rows() -> pd.DataFrame:
    """A train table big enough to split: 20 products x 2 images, with the source and relevance columns."""
    rows = []
    for product in range(20):
        for shot in range(2):
            index = product * 2 + shot
            row = {"image_id": f"t_{index:03d}", "product_id": f"p_{product:02d}"}
            for position, dimension in enumerate(DIMENSIONS):
                names = FAKE_CLASSES[dimension]
                row[dimension] = names[(index + position) % len(names)]
                # Both a prelabel and an unscored cell exist whichever products the split picks, so the
                # scored-cell mask has something to do.
                row[f"{dimension}_source"] = "model" if (index + position) % 3 == 0 else "human"
                row[f"{dimension}_relevant"] = "false" if (index + position) % 5 == 0 else "true"
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def saturday_data(galaxus_data, sample_submission):
    """``$GALAXUS_DATA`` as it will be at 13:00 Saturday: a decrypted final part, the sample, train, metric.py."""
    sample_submission.to_csv(galaxus_data / "sample_submission.csv", index=False)
    _train_rows().to_csv(galaxus_data / "train.csv", index=False)
    (galaxus_data / "metric.py").write_text(METRIC_STANDIN, encoding="utf-8")

    # The archive decides how deep final.csv lands, so nest it the way a real release would.
    release = galaxus_data / "final" / "final_images"
    (release / "images").mkdir(parents=True)
    for image_id in FINAL_IDS:
        write_image(release / "images" / f"{image_id}.png", "RGBA", (40, 30), color=(20, 90, 140, 255))
    pd.DataFrame({"image_id": FINAL_IDS, "image_path": [f"images/{i}.png" for i in FINAL_IDS]}).to_csv(
        release / "final.csv", index=False
    )
    return galaxus_data


def test_the_final_part_becomes_a_submission_that_validates(saturday_data):
    """Load the final part, answer every row, merge over the sample, validate. The whole Saturday path."""
    final = read_final()
    assert final["image_id"].tolist() == FINAL_IDS

    written = write_submission(
        {image_id: dict(ANSWER) for image_id in final["image_id"]},
        saturday_data / "out" / "submission.csv",
        validate=True,  # the gate product rule 2 requires before anything reaches Kaggle
    )

    # Header and row order are the sample's, which is what Kaggle checks first.
    submitted = read_submission(written)
    assert list(submitted.columns) == ["image_id", *DIMENSIONS]
    assert submitted["image_id"].tolist() == FINAL_IDS
    # Every dimension, not just the first: a merge that quietly stops early is exactly the kind of break a
    # chain test exists to catch, and checking one column would miss it.
    for dimension in DIMENSIONS:
        assert submitted[dimension].tolist() == [ANSWER[dimension]] * 3, dimension


def test_images_the_model_failed_on_keep_the_samples_answer(saturday_data):
    """The realistic Saturday case: some images error out. The file must still be submittable."""
    written = write_submission(
        {FINAL_IDS[0]: dict(ANSWER)},  # two of the three got no answer at all
        saturday_data / "out" / "partial.csv",
        validate=True,
    )
    submitted = read_submission(written)
    first = FAKE_CLASSES["subject"][0]
    assert submitted["image_id"].tolist() == FINAL_IDS
    assert submitted["subject"].tolist() == [ANSWER["subject"], first, first]


def test_the_prepared_cache_covers_every_final_image(saturday_data):
    """The run is served cache files, so a final part whose pixels never cached would be a run of nothing."""
    counts = fill_image_cache(read_final(), image_size=SIZE)
    assert counts["written"] == len(FINAL_IDS)
    assert counts["missing"] == 0
    assert sorted(path.stem for path in image_cache_dir(SIZE).glob("*.jpg")) == FINAL_IDS


def test_the_holdout_score_goes_through_the_hosts_metric(saturday_data):
    """Split, mask the unscored cells, score. A float out of metric.py rather than arithmetic of ours."""
    train = pd.read_csv(saturday_data / "train.csv", dtype=str, keep_default_na=False)
    _, holdout = split_holdout(train)
    assert not holdout.empty

    solution = solution_frame(holdout, DIMENSIONS)
    submission = pd.DataFrame({"image_id": solution["image_id"], **{d: ANSWER[d] for d in DIMENSIONS}})
    value = score(solution, submission)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
    # The mask is doing something: at least one cell is withheld from scoring.
    assert solution[list(DIMENSIONS)].isna().to_numpy().any()


def test_the_label_error_csv_lands_outside_every_git_work_tree(saturday_data, capsys):
    """Product rule 3, end to end: the bonus csv holds labels, so it must land outside the repository."""
    folder = saturday_data / "out" / "predictions" / "EXP-0001"
    folder.mkdir(parents=True)
    train = pd.read_csv(saturday_data / "train.csv", dtype=str, keep_default_na=False)
    (folder / "holdout.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "image_id": image_id,
                    "labels": {"subject": ANSWER["subject"]},
                    "confidence": {"subject": 0.97},
                }
            )
            for image_id in train["image_id"]
        ),
        encoding="utf-8",
    )

    assert label_errors_main(["--predictions", str(folder), "--dimensions", "subject"]) == 0
    written = saturday_data / "out" / "label_errors.csv"
    assert written.is_file()
    assert list(pd.read_csv(written).columns) == ["image_id", "dimension", "label", "suggested_label"]

    # Nothing on the path from the csv up to the filesystem root is a git work tree.
    for folder_up in (written.resolve(), *written.resolve().parents):
        assert not (folder_up / ".git").exists(), f"{written} is inside the work tree at {folder_up}"

    printed = capsys.readouterr().out
    assert "never commit it" in printed
    for image_id in train["image_id"]:
        assert image_id not in printed


def test_the_bonus_csv_can_be_made_without_the_gateway(saturday_data, capsys):
    """The 14:30 deliverable has a second source: neighbour votes (#112 part A), so no gateway and no GPU.

    Every other test of the label-error finder feeds it a ``holdout.jsonl`` written by hand or by the model lane,
    which is the path that needs a gateway run with logprobs. This walks the other one end to end with the real
    commands - embeddings in the cache, ``neighbours label-errors``, then ``label_errors`` over its output - so
    the fallback stays verified rather than being something that worked once.

    The table has three clusters of four images, one per class, and one planted mislabel: an image sitting in the
    second cluster while carrying the first cluster's label. That is the row the csv must name.
    """
    import numpy as np

    from galaxus.data.neighbours import DEFAULT_ENCODER, EMBEDDING_DTYPE, embedding_dir
    from galaxus.data.neighbours import main as neighbours_main

    dimension = DIMENSIONS[0]
    classes = FAKE_CLASSES[dimension]
    # One cluster per class, however many the fixture's dimension has, so this does not assume a class count.
    per_cluster = 12 // len(classes)
    planted = "t_07"

    rows, vectors = [], {}
    for position in range(12):
        image_id = f"t_{position:02d}"
        cluster = position // per_cluster
        label = classes[0] if image_id == planted else classes[cluster]
        row = {"image_id": image_id, "product_id": f"p_{position:02d}", dimension: label}
        row[f"{dimension}_source"] = "human"
        row[f"{dimension}_relevant"] = "true"
        rows.append(row)
        vector = np.zeros(8, dtype=np.float32)
        vector[cluster] = 1.0
        vector[2 + (position % 6)] = 0.05  # a little spread inside the cluster, so neighbours have an order
        vectors[image_id] = vector
    pd.DataFrame(rows).to_csv(saturday_data / "train.csv", index=False)

    folder = embedding_dir(DEFAULT_ENCODER)
    folder.mkdir(parents=True, exist_ok=True)
    for image_id, vector in vectors.items():
        np.save(folder / f"{image_id}.npy", vector.astype(EMBEDDING_DTYPE))

    assert neighbours_main(["label-errors", "--k", "3", "--min-similarity", "0.5", "--dimensions", dimension]) == 0
    votes = saturday_data / "out" / "neighbour_votes.jsonl"
    assert votes.is_file()

    # 0.75 rather than the default 0.9: a vote's confidence is the neighbours' agreement, which at k=3 is 1.0,
    # 0.67 or 0.33, so the default would keep only a unanimous three. The two flags interact and the chain is
    # where that is visible.
    assert label_errors_main(["--predictions", str(votes), "--dimensions", dimension, "--threshold", "0.75"]) == 0

    written = saturday_data / "out" / "label_errors.csv"
    reported = pd.read_csv(written).to_dict(orient="records")
    assert {
        "image_id": planted,
        "dimension": dimension,
        "label": classes[0],
        "suggested_label": classes[1],
    } in reported, reported

    # Same two rules as the other path: outside every work tree, and no id on stdout.
    for folder_up in (written.resolve(), *written.resolve().parents):
        assert not (folder_up / ".git").exists()
    printed = capsys.readouterr().out
    assert planted not in printed
