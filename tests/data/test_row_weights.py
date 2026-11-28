"""#136: per-row training weights from two independent signals.

The behaviour worth protecting is the asymmetry: the same model disagreement becomes a *down-weight* when the
neighbours back the model and an *oversample* when they back the human. A test suite that only checked "disputed
rows get a low weight" would pass against a version that threw away every hard example.

Every table here is invented. ``train_table`` carries four scored human cells (img_1, img_2, img_5, img_7); img_3
and img_6 are model prelabels and img_4 is marked not relevant, so none of those three may ever appear.
"""

from __future__ import annotations

import pandas as pd
import pytest

from galaxus.data.row_weights import (
    DEFAULT_WEIGHT,
    DROP_REASON,
    DROP_WEIGHT,
    HARD_REASON,
    HARD_WEIGHT,
    SUSPECT_REASON,
    SUSPECT_WEIGHT,
    WEIGHT_COLUMNS,
    build_row_weights,
    judge_cells,
    main,
    per_image_weights,
    suggested_relabels,
    write_row_weights,
)

DIM = ("subject",)


def record(image_id: str, value: str, confidence: float | None) -> dict:
    """One prediction in the shape both signals share (``galaxus.data.label_errors.out_of_fold_frame``)."""
    return {
        "image_id": image_id,
        "labels": {"subject": value},
        "confidence": None if confidence is None else {"subject": confidence},
    }


def judged(train, model, neighbours, **kwargs):
    return judge_cells(train, model, neighbours, dimension_names=DIM, **kwargs)


def weight_of(frame: pd.DataFrame, image_id: str) -> float:
    return float(frame.loc[frame["image_id"] == image_id, "weight"].iloc[0])


def reason_of(frame: pd.DataFrame, image_id: str) -> str:
    return str(frame.loc[frame["image_id"] == image_id, "reason"].iloc[0])


# --- the asymmetry, which is the whole point ------------------------------------------------------------------


def test_both_signals_naming_the_same_replacement_is_a_suspected_label_error(train_table):
    """img_1's human label is alpha; the model and the neighbours both say beta, so the label is in doubt."""
    model = [record("img_1", "beta", 0.95)]
    neighbours = [record("img_1", "beta", 1.0)]
    frame = build_row_weights(judged(train_table, model, neighbours))
    assert reason_of(frame, "img_1") == SUSPECT_REASON
    assert weight_of(frame, "img_1") == SUSPECT_WEIGHT


def test_the_neighbours_backing_the_human_makes_it_a_hard_example_not_noise(train_table):
    """The same model disagreement, with the neighbours on the human's side, must RAISE the weight.

    This is the test that fails against a version treating every disagreement as noise - the failure mode that
    would quietly discard the rows most worth training on.
    """
    model = [record("img_1", "beta", 0.95)]
    neighbours = [record("img_1", "alpha", 1.0)]
    frame = build_row_weights(judged(train_table, model, neighbours))
    assert reason_of(frame, "img_1") == HARD_REASON
    assert weight_of(frame, "img_1") == HARD_WEIGHT
    assert HARD_WEIGHT > DEFAULT_WEIGHT, "a hard example has to be seen more often, not less"


def test_two_kinds_of_doubt_are_not_one_finding(train_table):
    """Both signals disagree with the human but name DIFFERENT replacements, so nothing is established.

    "Not alpha" from the model and "not alpha either, but gamma rather than beta" from the neighbours is two
    disagreements, not corroboration. Dropping such a cell would act on a consensus that does not exist.
    """
    model = [record("img_1", "beta", 0.99)]
    neighbours = [record("img_1", "gamma", 1.0)]
    frame = build_row_weights(judged(train_table, model, neighbours))
    assert weight_of(frame, "img_1") == DEFAULT_WEIGHT
    assert reason_of(frame, "img_1") == ""


def test_one_signal_alone_changes_nothing(train_table):
    """The neighbours have no opinion on this cell, so the model's disagreement is not evidence."""
    model = [record("img_1", "beta", 0.99)]
    frame = build_row_weights(judged(train_table, model, []))
    assert weight_of(frame, "img_1") == DEFAULT_WEIGHT


def test_a_model_below_its_threshold_is_not_evidence_either(train_table):
    model = [record("img_1", "beta", 0.4)]
    neighbours = [record("img_1", "beta", 1.0)]
    frame = build_row_weights(judged(train_table, model, neighbours, model_threshold=0.9))
    assert weight_of(frame, "img_1") == DEFAULT_WEIGHT


def test_weak_neighbour_agreement_is_not_evidence_either(train_table):
    """At k=3 one vote of three is 0.33: a single neighbour is not a second source."""
    model = [record("img_1", "beta", 0.99)]
    neighbours = [record("img_1", "beta", 0.33)]
    frame = build_row_weights(judged(train_table, model, neighbours, neighbour_threshold=0.67))
    assert weight_of(frame, "img_1") == DEFAULT_WEIGHT


def test_both_signals_certain_drops_the_row_outright(train_table):
    model = [record("img_1", "beta", 0.999)]
    neighbours = [record("img_1", "beta", 1.0)]
    frame = build_row_weights(judged(train_table, model, neighbours, drop_threshold=0.98))
    assert reason_of(frame, "img_1") == DROP_REASON
    assert weight_of(frame, "img_1") == DROP_WEIGHT


def test_a_model_that_agrees_with_the_human_leaves_the_row_alone(train_table):
    model = [record("img_1", "alpha", 0.99)]
    neighbours = [record("img_1", "alpha", 1.0)]
    frame = build_row_weights(judged(train_table, model, neighbours))
    assert weight_of(frame, "img_1") == DEFAULT_WEIGHT


# --- which cells are eligible at all --------------------------------------------------------------------------


def test_only_scored_human_cells_appear(train_table):
    """A model prelabel is the thing being checked and an irrelevant cell is never trained on."""
    model = [record(name, "beta", 0.99) for name in ("img_1", "img_3", "img_4", "img_6")]
    neighbours = [record(name, "beta", 1.0) for name in ("img_1", "img_3", "img_4", "img_6")]
    frame = build_row_weights(judged(train_table, model, neighbours))
    assert sorted(frame["image_id"]) == ["img_1", "img_2", "img_5", "img_7"]
    for excluded in ("img_3", "img_4", "img_6"):
        assert excluded not in set(frame["image_id"])


def test_the_file_covers_every_scored_cell_not_only_the_changed_ones(train_table):
    """A weight file is complete: a consumer can join on it without filling gaps itself."""
    frame = build_row_weights(judged(train_table, [record("img_1", "beta", 0.99)], [record("img_1", "beta", 1.0)]))
    assert len(frame) == 4
    assert list(frame.columns) == list(WEIGHT_COLUMNS)
    assert (frame.loc[frame["image_id"] != "img_1", "weight"] == DEFAULT_WEIGHT).all()


def test_a_train_table_with_no_scored_human_cell_is_refused(train_table):
    blanked = train_table.copy()
    blanked["subject_source"] = "model"
    with pytest.raises(ValueError, match="no scored human cells"):
        judged(blanked, [], [])


# --- the bonus csv --------------------------------------------------------------------------------------------


def test_the_bonus_csv_holds_only_cells_both_signals_dispute(train_table):
    """#7's csv is a suggestion to a human, so an unsupported one wastes their time."""
    model = [record("img_1", "beta", 0.99), record("img_2", "alpha", 0.95), record("img_5", "beta", 0.99)]
    neighbours = [record("img_1", "beta", 1.0), record("img_2", "beta", 1.0), record("img_5", "alpha", 1.0)]
    relabels = judged(train_table, model, neighbours)
    report = suggested_relabels(relabels)
    assert list(report.columns) == ["image_id", "dimension", "label", "suggested_label"]
    assert report["image_id"].tolist() == ["img_1"], "img_2 was two kinds of doubt; img_5 was a hard example"
    assert report["label"].iloc[0] == "alpha"
    assert report["suggested_label"].iloc[0] == "beta"


def test_the_bonus_csv_reads_strongest_first(train_table):
    model = [record("img_1", "beta", 0.95), record("img_5", "beta", 0.99)]
    neighbours = [record("img_1", "beta", 1.0), record("img_5", "beta", 1.0)]
    report = suggested_relabels(judged(train_table, model, neighbours))
    assert report["image_id"].tolist() == ["img_5", "img_1"]


# --- collapsing to one weight per image -----------------------------------------------------------------------


def test_per_image_min_treats_an_image_with_one_bad_cell_as_bad():
    """The LoRA trains on whole images, so it cannot apply a weight per cell: min is the cautious collapse."""
    weights = pd.DataFrame(
        {
            "image_id": ["img_1", "img_1", "img_2"],
            "dimension": ["subject", "background", "subject"],
            "weight": [0.0, 1.0, 1.0],
            "reason": [DROP_REASON, "", ""],
        }
    )
    collapsed = per_image_weights(weights, how="min")
    assert collapsed.loc[collapsed["image_id"] == "img_1", "weight"].iloc[0] == 0.0


def test_per_image_mean_dilutes_it_instead():
    weights = pd.DataFrame(
        {
            "image_id": ["img_1", "img_1"],
            "dimension": ["subject", "background"],
            "weight": [0.0, 1.0],
            "reason": [DROP_REASON, ""],
        }
    )
    collapsed = per_image_weights(weights, how="mean")
    assert collapsed["weight"].iloc[0] == pytest.approx(0.5)


def test_an_unknown_collapse_is_refused():
    weights = pd.DataFrame({"image_id": ["img_1"], "dimension": ["subject"], "weight": [1.0], "reason": [""]})
    with pytest.raises(ValueError, match="min.*mean"):
        per_image_weights(weights, how="median")


# --- writing it, which must never reach git -------------------------------------------------------------------


def test_writing_refuses_a_path_inside_a_work_tree(tmp_path):
    """These rows carry competition labels: product rule 3. A mistyped path is the one way they get committed."""
    repository = tmp_path / "checkout"
    (repository / ".git").mkdir(parents=True)
    frame = pd.DataFrame({"image_id": ["img_1"], "dimension": ["subject"], "weight": [1.0], "reason": [""]})
    with pytest.raises(ValueError):
        write_row_weights(frame, path=repository / "weights.csv")


def test_writing_refuses_the_wrong_columns(galaxus_data):
    with pytest.raises(ValueError, match="exactly the columns"):
        write_row_weights(pd.DataFrame({"image_id": ["img_1"], "weight": [1.0]}))


def test_writing_lands_in_the_galaxus_data_out_dir(galaxus_data):
    frame = pd.DataFrame({"image_id": ["img_1"], "dimension": ["subject"], "weight": [0.25], "reason": [""]})
    written = write_row_weights(frame)
    assert written.parent == galaxus_data / "out"
    assert pd.read_csv(written)["weight"].iloc[0] == 0.25


# --- the command ----------------------------------------------------------------------------------------------


@pytest.fixture
def signal_files(galaxus_data, train_table):
    """$GALAXUS_DATA with train.csv and both signals on disk, as the command reads them."""
    import json

    train_table.to_csv(galaxus_data / "train.csv", index=False)
    model = galaxus_data / "model.jsonl"
    neighbours = galaxus_data / "neighbours.jsonl"
    model.write_text(json.dumps(record("img_1", "beta", 0.99)), encoding="utf-8")
    neighbours.write_text(json.dumps(record("img_1", "beta", 1.0)), encoding="utf-8")
    return model, neighbours


def test_the_command_prints_counts_and_never_an_image_id(signal_files, capsys):
    model, neighbours = signal_files
    assert (
        main(["--model-predictions", str(model), "--neighbour-votes", str(neighbours), "--dimensions", "subject"]) == 0
    )
    printed = capsys.readouterr().out
    assert "scored cell" in printed
    assert "img_1" not in printed, "product rule 3: counts and class names only"
    assert "never commit it" in printed


def test_the_command_dry_run_writes_nothing(signal_files, galaxus_data, capsys):
    model, neighbours = signal_files
    assert (
        main(
            [
                "--model-predictions",
                str(model),
                "--neighbour-votes",
                str(neighbours),
                "--dimensions",
                "subject",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "nothing written" in capsys.readouterr().out
    assert not (galaxus_data / "out").exists()


def test_the_command_reports_a_missing_signal_without_a_traceback(signal_files, galaxus_data, capsys):
    model, _ = signal_files
    code = main(
        [
            "--model-predictions",
            str(model),
            "--neighbour-votes",
            str(galaxus_data / "nope.jsonl"),
            "--dimensions",
            "subject",
        ]
    )
    assert code == 1
    assert capsys.readouterr().err.startswith("cannot work out the row weights:")
