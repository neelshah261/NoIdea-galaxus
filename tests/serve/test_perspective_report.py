"""Issue #34: the perspective error report. Synthetic labels and predictions in ``tmp_path``, never the network.

The class names here are invented and only self-consistent, as in ``test_demo.py``: real names are read at runtime
from ``galaxus.data.classes``, never written down (AGENTS.md "Hard rules"). Two of them carry the words left and
right so the mirror swap has something to swap; they are not competition class names.

``bench/`` is a script folder, not a package, so the module under test is loaded from its file the way
``galaxus.data.metric`` loads ``metric.py``.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
import pytest

from galaxus.data.split import split_holdout

MODULE_PATH = Path(__file__).resolve().parents[2] / "bench" / "perspective_report.py"
MODULE_NAME = "galaxus_bench_perspective_report"

DIMENSION = "perspective"
# Invented classes. LEFT_POSE and RIGHT_POSE differ only in the words left and right, which is what mirror_class
# swaps; FLAT_POSE and ANGLED_POSE have neither word, so the mirror must leave them alone.
LEFT_POSE = "pose_left_edge"
RIGHT_POSE = "pose_right_edge"
FLAT_POSE = "pose_flat"
ANGLED_POSE = "pose_angled"

HUMAN = "human"
MODEL = "model"


def _load_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


report_module = _load_module()


@pytest.fixture
def pr():
    """The module under test."""
    return report_module


# ------------------------------------------------------------------------------------------------ synthetic inputs


def train_with_holdout(cells: list[tuple[str, str, bool]], products: int = 60) -> tuple[pd.DataFrame, list[str]]:
    """A ``train.csv``-shaped frame whose holdout carries exactly ``cells``, and the image ids that carry them.

    ``cells`` is ``(label, source, relevant)`` per holdout image, in sorted image-id order. Every other held-out
    image is given a model prelabel, which the data lane's ``scored_mask`` drops, so the scored cells of the
    holdout are exactly the ones this returns.
    """
    rows = [
        {
            "image_id": f"img_{index:03d}",
            "product_id": f"prod_{index:03d}",
            DIMENSION: FLAT_POSE,
            f"{DIMENSION}_source": HUMAN,
            f"{DIMENSION}_relevant": "true",
        }
        for index in range(products)
    ]
    frame = pd.DataFrame(rows)
    held = sorted(str(image_id) for image_id in split_holdout(frame)[1]["image_id"])
    assert len(cells) <= len(held), f"only {len(held)} products are held out of {products}"
    for image_id in held:
        frame.loc[frame["image_id"] == image_id, f"{DIMENSION}_source"] = MODEL
    used = []
    for image_id, (label, source, relevant) in zip(held, cells, strict=False):
        selector = frame["image_id"] == image_id
        frame.loc[selector, DIMENSION] = label
        frame.loc[selector, f"{DIMENSION}_source"] = source
        frame.loc[selector, f"{DIMENSION}_relevant"] = "true" if relevant else "false"
        used.append(image_id)
    return frame, used


def write_run(path: Path, labels: dict[str, str], confidence: dict[str, float] | None = None) -> Path:
    """A predictions jsonl in the shape ``galaxus.serve.runner`` writes."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for image_id, label in labels.items():
            record = {
                "image_id": image_id,
                "labels": {DIMENSION: label},
                "confidence": None if confidence is None else {DIMENSION: confidence[image_id]},
            }
            handle.write(json.dumps(record) + "\n")
    return path


def pairs_file(path: Path, dimension: str = DIMENSION) -> Path:
    """A pair file in the shape of the shipped one, with the synthetic classes of this test module."""
    path.write_text(
        json.dumps(
            {
                "dimension": dimension,
                "source": "synthetic pairs for the tests",
                "pairs": [
                    {
                        "true": RIGHT_POSE,
                        "predicted": LEFT_POSE,
                        "host_share": 0.12,
                        "open_for_hosts": True,
                        "status": "known",
                        "source": "test",
                    },
                    {
                        "true": FLAT_POSE,
                        "predicted": ANGLED_POSE,
                        "host_share": 0.07,
                        "open_for_hosts": False,
                        "status": "known",
                        "source": "test",
                    },
                ],
                "combine": {"rule": "higher_mean_confidence", "status": "assumption", "todo": "TODO(owner): test"},
            }
        ),
        encoding="utf-8",
    )
    return path


# -------------------------------------------------------------------------------------------------- the pair config


def test_shipped_pairs_load_and_carry_their_source(pr):
    pairs = pr.load_pairs()
    assert pairs.dimension and "#17" in pairs.source
    assert len(pairs.pairs()) >= 2
    assert all(pair.source for pair in pairs.pairs())
    assert all(0.0 < pair.host_share < 1.0 for pair in pairs.pairs())
    assert pairs.combine_rule["status"] == "assumption" and pairs.combine_rule["todo"].startswith("TODO(")


def test_the_open_pairs_are_the_two_the_hosts_named(pr):
    assert len(pr.load_pairs().pairs(only_open=True)) == 2


def test_no_class_name_is_written_in_the_module(pr):
    """Every class name lives in the JSON beside the module, as in ``demo/hero_rules.json``."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for pair in pr.load_pairs().pairs():
        assert pair.true not in source
        assert pair.predicted not in source


def test_a_missing_classes_source_is_recorded_not_raised(pr, tmp_path, monkeypatch):
    """Without ``contracts/classes.json`` (#11) or ``$GALAXUS_DATA`` nothing can be checked, and that is recorded."""
    monkeypatch.setattr(pr, "valid_class_names", lambda: None)
    path = pairs_file(tmp_path / "pairs.json")
    assert any("classes source is unavailable" in problem for problem in pr.load_pairs(path).problems)
    with pytest.raises(ValueError, match="classes source"):
        pr.load_pairs(path, strict=True)


def test_names_that_are_not_classes_are_reported(pr, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "valid_class_names", lambda: {FLAT_POSE})
    problems = pr.load_pairs(pairs_file(tmp_path / "pairs.json")).problems
    assert len(problems) == 3  # both names of the first pair, and the second pair's prediction
    assert all("is not a class in any dimension" in problem for problem in problems)


def test_known_names_leave_no_problem(pr, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "valid_class_names", lambda: {LEFT_POSE, RIGHT_POSE, FLAT_POSE, ANGLED_POSE})
    assert pr.load_pairs(pairs_file(tmp_path / "pairs.json")).problems == ()


# --------------------------------------------------------------------------------------- labels, runs and confusion


def test_only_scored_cells_are_compared(pr, tmp_path):
    frame, held = train_with_holdout(
        [
            (RIGHT_POSE, HUMAN, True),
            (RIGHT_POSE, MODEL, True),  # a model prelabel never counts
            (LEFT_POSE, HUMAN, False),  # the dimension is not relevant for this image
            (FLAT_POSE, HUMAN, True),
        ]
    )
    labels = pr.labels_of(pr.holdout_solution(DIMENSION, frame), DIMENSION)
    assert set(labels) == {held[0], held[3]}

    run = pr.read_run(write_run(tmp_path / "run.jsonl", {image_id: RIGHT_POSE for image_id in held}), DIMENSION)
    table = pr.confuse(labels, run, DIMENSION)
    assert table.scored == 2
    assert table.compared == 2
    assert table.count(RIGHT_POSE, RIGHT_POSE) == 1
    assert table.count(FLAT_POSE, RIGHT_POSE) == 1
    assert table.unlabelled == len(held) - 2
    assert table.missing == 0
    assert table.accuracy == 0.5


def test_a_human_label_without_a_prediction_is_counted_as_missing(pr, tmp_path):
    frame, held = train_with_holdout([(RIGHT_POSE, HUMAN, True), (FLAT_POSE, HUMAN, True)])
    labels = pr.labels_of(pr.holdout_solution(DIMENSION, frame), DIMENSION)
    run = pr.read_run(write_run(tmp_path / "run.jsonl", {held[0]: RIGHT_POSE}), DIMENSION)
    table = pr.confuse(labels, run, DIMENSION)
    assert (table.compared, table.missing, table.unlabelled) == (1, 1, 0)


def test_read_run_never_repairs_the_file_it_reads(pr, tmp_path):
    """A report is read-only: a run cut off by a crash stays exactly as it is on disk."""
    path = tmp_path / "run.jsonl"
    write_run(path, {"img_000": RIGHT_POSE})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"image_id": "img_001", "labels": {"persp')
    before = path.read_bytes()
    run = pr.read_run(path, DIMENSION)
    assert run.labels == {"img_000": RIGHT_POSE}
    assert path.read_bytes() == before


def test_per_class_f1_and_the_mean_over_the_classes_present(pr, tmp_path):
    frame, held = train_with_holdout(
        [(RIGHT_POSE, HUMAN, True), (RIGHT_POSE, HUMAN, True), (FLAT_POSE, HUMAN, True), (FLAT_POSE, HUMAN, True)]
    )
    labels = pr.labels_of(pr.holdout_solution(DIMENSION, frame), DIMENSION)
    answers = {held[0]: RIGHT_POSE, held[1]: LEFT_POSE, held[2]: FLAT_POSE, held[3]: FLAT_POSE}
    table = pr.confuse(labels, pr.read_run(write_run(tmp_path / "run.jsonl", answers), DIMENSION), DIMENSION)
    scores = {score.name: score for score in pr.per_class_f1(table)}

    assert scores[RIGHT_POSE].support == 2 and scores[RIGHT_POSE].predicted == 1
    assert scores[RIGHT_POSE].precision == 1.0
    assert scores[RIGHT_POSE].recall == 0.5
    assert scores[RIGHT_POSE].f1 == pytest.approx(2 / 3)
    assert scores[FLAT_POSE].f1 == 1.0
    assert scores[LEFT_POSE].f1 == 0.0  # predicted once, never the human label
    assert pr.macro_f1(list(scores.values())) == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)


# ---------------------------------------------------------------------------------------------------- mirror TTA


def test_unmirror_uses_the_data_lanes_swap(pr, tmp_path):
    """The mirrored run is read back through ``galaxus.serve.demo.rules.mirror_class``, nothing of our own."""
    path = write_run(tmp_path / "mirror.jsonl", {"a": LEFT_POSE, "b": FLAT_POSE})
    turned = pr.unmirror(pr.read_run(path, DIMENSION), DIMENSION, None)
    assert turned.labels == {"a": RIGHT_POSE, "b": FLAT_POSE}
    assert turned.swapped == 1 and turned.mirrored


def test_combine_mirror_takes_the_more_confident_answer(pr, tmp_path):
    base = pr.read_run(
        write_run(tmp_path / "b.jsonl", {"a": LEFT_POSE, "b": FLAT_POSE}, {"a": 0.4, "b": 0.9}), DIMENSION
    )
    other = pr.unmirror(
        pr.read_run(
            write_run(tmp_path / "m.jsonl", {"a": LEFT_POSE, "b": ANGLED_POSE}, {"a": 0.8, "b": 0.1}), DIMENSION
        ),
        DIMENSION,
        None,
    )
    combined = pr.combine_mirror(base, other)
    assert combined.labels == {"a": RIGHT_POSE, "b": FLAT_POSE}


def test_combine_mirror_keeps_the_base_answer_without_confidences(pr, tmp_path):
    base = pr.read_run(write_run(tmp_path / "b.jsonl", {"a": LEFT_POSE}), DIMENSION)
    other = pr.unmirror(pr.read_run(write_run(tmp_path / "m.jsonl", {"a": FLAT_POSE}), DIMENSION), DIMENSION, None)
    assert pr.combine_mirror(base, other).labels == {"a": LEFT_POSE}


def test_before_and_after_show_the_pair_the_mirror_fixes(pr, tmp_path, monkeypatch):
    # No classes source in the tests, and these classes are invented anyway: mirror_class then swaps unchecked.
    monkeypatch.setattr(pr, "valid_class_names", lambda: None)
    frame, held = train_with_holdout([(RIGHT_POSE, HUMAN, True)] * 4)
    # The base run reads every right-side image as the left-side class: the host pair, at 100%.
    base = write_run(tmp_path / "base.jsonl", {image_id: LEFT_POSE for image_id in held}, dict.fromkeys(held, 0.3))
    # The mirrored run answers the left-side class too; swapped back that is the right-side class, more confident.
    mirrored = write_run(
        tmp_path / "mirror.jsonl", {image_id: LEFT_POSE for image_id in held}, dict.fromkeys(held, 0.9)
    )

    report = pr.build_report(
        dimension=DIMENSION,
        predictions=base,
        mirrored=mirrored,
        train=frame,
        pairs_file=pairs_file(tmp_path / "pairs.json"),
    )
    assert report.has_mirror and report.mirror_swapped == 4
    row = next(row for row in report.pair_rows() if row["pair"].open_for_hosts)
    assert row["before_share"] == 1.0
    assert row["after_share"] == 0.0
    assert report.before.accuracy == 0.0
    assert report.after.accuracy == 1.0


def test_without_a_mirrored_run_the_comparison_is_left_out(pr, tmp_path):
    frame, held = train_with_holdout([(RIGHT_POSE, HUMAN, True)])
    report = pr.build_report(
        dimension=DIMENSION,
        predictions=write_run(tmp_path / "base.jsonl", {held[0]: LEFT_POSE}),
        train=frame,
        pairs_file=pairs_file(tmp_path / "pairs.json"),
    )
    assert not report.has_mirror and report.after is None
    assert any("no mirrored run" in note for note in report.notes)
    assert "MIRROR TEST-TIME AUGMENTATION" not in pr.render_text(report)


def test_build_report_needs_a_predictions_file(pr):
    with pytest.raises(ValueError, match="predictions file"):
        pr.build_report(dimension=DIMENSION)


# --------------------------------------------------------------------------------------------------- the text report


@pytest.fixture
def report(pr, tmp_path):
    frame, held = train_with_holdout([(RIGHT_POSE, HUMAN, True), (RIGHT_POSE, HUMAN, True), (FLAT_POSE, HUMAN, True)])
    answers = {held[0]: LEFT_POSE, held[1]: RIGHT_POSE, held[2]: ANGLED_POSE}
    return pr.build_report(
        dimension=DIMENSION,
        predictions=write_run(tmp_path / "base.jsonl", answers, dict.fromkeys(held, 0.5)),
        train=frame,
        pairs_file=pairs_file(tmp_path / "pairs.json"),
    )


def test_the_report_says_where_every_number_came_from(pr, report):
    text = pr.render_text(report)
    assert "WHERE EACH NUMBER COMES FROM" in text
    assert "results/metrics.json" in text
    assert "galaxus.data.split" in text
    assert "the hosts' own model, not ours" in text
    assert "synthetic pairs for the tests" in text  # the pair file's own source line


def test_the_report_shows_the_pairs_the_f1_and_the_table(pr, report):
    text = pr.render_text(report)
    assert f"{RIGHT_POSE} -> {LEFT_POSE}" in text
    assert "PER-CLASS F1 (before)" in text
    assert "CONFUSION TABLE" in text
    assert " 50.0%" in text  # one of the two right-side cells was read as the left-side class
    assert " 12.0%" in text  # the hosts' own share for that pair, from the pair file


def test_the_report_only_ever_writes_the_one_figure(pr):
    """AGENTS.md product rule 1: ``results/metrics.json`` is the model lane's file; this report only points at it."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "results/metrics.json" in source
    assert source.count("savefig") == 1
    for writer in (".write_text(", ".write_bytes(", ".to_csv(", ".to_json(", '"w"', "'w'"):
        assert writer not in source


# ------------------------------------------------------------------------------------------------------- the figure


def test_figure_path_is_under_the_out_directory(pr, tmp_path):
    assert pr.figure_path(tmp_path) == tmp_path / pr.FIGURE_DIR / pr.FIGURE_NAME


def test_the_figure_step_skips_cleanly_without_matplotlib(pr, report, tmp_path, monkeypatch):
    """matplotlib is not a dependency and a worker may not add one: the table still has to come out."""
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    result = pr.save_figure(report, tmp_path)
    assert not result.saved and result.path is None
    assert "matplotlib is not installed" in result.reason and "pyproject.toml" in result.reason
    assert list(tmp_path.glob("**/*.png")) == []


@pytest.mark.skipif(find_spec("matplotlib") is None, reason="matplotlib is not installed here")
def test_the_figure_is_one_file_under_the_out_directory(pr, report, tmp_path):
    result = pr.save_figure(report, tmp_path)
    assert result.saved
    assert result.path == pr.figure_path(tmp_path)
    assert result.path.stat().st_size > 0
    assert len(list(tmp_path.glob("**/*.png"))) == 1


def test_the_figure_skips_when_no_host_pair_occurs(pr, tmp_path):
    frame, held = train_with_holdout([(FLAT_POSE, HUMAN, True)])
    report = pr.build_report(
        dimension=DIMENSION,
        predictions=write_run(tmp_path / "base.jsonl", {held[0]: FLAT_POSE}),
        train=frame,
        pairs_file=pairs_file(tmp_path / "only.json"),
    )
    # Keep only the first pair, whose human label never occurs in this slice, so no pair has a scored cell.
    report.pairs = pr.HostPairs(raw={**report.pairs.raw, "pairs": report.pairs.raw["pairs"][:1]})
    result = pr.save_figure(report, tmp_path)
    assert not result.saved and "nothing to plot" in result.reason


# ------------------------------------------------------------------------------------------------------ the command


def test_the_command_needs_a_run(pr, capsys):
    with pytest.raises(SystemExit):
        pr.main([])
    assert "--predictions" in capsys.readouterr().err
