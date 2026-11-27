"""Issue #34: the perspective error report, and the one figure behind deck slide 8.

What it does: take a predictions jsonl written by :mod:`galaxus.serve.runner`, score it against the holdout's
**human** labels through the data lane (``galaxus.data.split.split_holdout`` → ``scored_mask`` → ``solution_frame``),
and print

1. the confusion table of one dimension (perspective) over the scored cells;
2. the hosts' open confusion pairs, highlighted, with our rate next to the hosts' own percentage;
3. per-class precision, recall and F1;
4. before vs after mirror test-time augmentation, when both runs are on disk.

It then saves **one** figure for deck slide 8 under ``$GALAXUS_DATA/out/`` in the design/STYLE.md colours.
Nothing it writes is ever committed (AGENTS.md "Galaxus product rules" 3), and the figure step skips with a reason
when matplotlib is not installed, so the table still comes out.

Where the numbers come from, which the report repeats in its own header because AGENTS.md
"Galaxus product rules" 1 says every number shown to judges comes from ``results/metrics.json``:

- **our counts, rates and F1**: computed here, from the predictions file plus the holdout human labels. They are an
  error analysis, not a score. Nothing from this report goes on a slide as our result unless the same number is in
  ``results/metrics.json`` (model lane, eval script). This file never writes that file.
- **the host percentages**: the hosts' model on their slides, read from ``bench/perspective_pairs.json``, which
  carries the source of every one of them (issue #17, slide 22).

No class name is written in this module: the pairs come from the JSON beside it, which is checked against
``galaxus.data.classes``, and the left/right swap of the mirrored run is the data lane's rule, imported from
``galaxus.serve.demo.rules.mirror_class`` (a stand-in for the data lane's issue #29, which owns it).

As a command line (``$GALAXUS_DATA`` set, data lane installed):

    python bench/perspective_report.py --exp EXP-0001
    python bench/perspective_report.py --predictions <run.jsonl> --mirrored <mirrored_run.jsonl>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.data.classes import ID_COLUMN
from galaxus.data.split import solution_frame, split_holdout
from galaxus.serve.demo.rules import mirror_class
from galaxus.serve.runner import predictions_path, read_predictions

#: The pairs the hosts showed, as data. Never a class name in this module.
PAIRS_FILE = Path(__file__).with_name("perspective_pairs.json")
#: Where the slide 8 figure goes, under ``$GALAXUS_DATA/out/`` and never in git.
FIGURE_DIR = "perspective"
FIGURE_NAME = "slide8_perspective_pairs.png"
#: STYLE.md "Colours", the tokens this figure uses.
PALETTE = {
    "paper": "#f2efe9",
    "white": "#ffffff",
    "ink": "#141414",
    "muted": "#55524c",
    "muted2": "#6e6a63",
    "rule": "#dcd6cc",
    "rule2": "#c4bcae",
    "green": "#157a43",
    "red": "#d1173a",
}
BEFORE = "before"
AFTER = "after"


# --------------------------------------------------------------------------------------------------- the host pairs


@dataclass(frozen=True)
class Pair:
    """One confusion pair the hosts reported: a human label read as another class."""

    true: str
    predicted: str
    host_share: float | None
    open_for_hosts: bool
    source: str

    @property
    def name(self) -> str:
        return f"{self.true} -> {self.predicted}"


@dataclass(frozen=True)
class HostPairs:
    """The loaded pair file, carrying whatever could not be checked against the classes source."""

    raw: dict[str, Any]
    problems: tuple[str, ...] = ()

    @property
    def dimension(self) -> str:
        return str(self.raw["dimension"])

    @property
    def source(self) -> str:
        return str(self.raw.get("source", ""))

    @property
    def combine_rule(self) -> dict[str, str]:
        value = self.raw.get("combine")
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}

    def pairs(self, only_open: bool = False) -> list[Pair]:
        """The pairs, newest-first as the file lists them; ``only_open`` keeps the ones the hosts called open."""
        result = []
        for entry in self.raw.get("pairs", []):
            share = entry.get("host_share")
            pair = Pair(
                true=str(entry["true"]),
                predicted=str(entry["predicted"]),
                host_share=None if share is None else float(share),
                open_for_hosts=bool(entry.get("open_for_hosts", False)),
                source=str(entry.get("source", "")),
            )
            if pair.open_for_hosts or not only_open:
                result.append(pair)
        return result


def valid_class_names() -> set[str] | None:
    """Every valid class name across the dimensions, or ``None`` when there is no classes source here.

    ``contracts/classes.json`` is not on ``main`` yet (issue #11) and ``$GALAXUS_DATA`` is absent in the tests, so a
    missing source is normal and must not take the report down: the names are then simply not checked.
    """
    try:
        from galaxus.data.classes import allowed_classes

        resolved = allowed_classes()
    except Exception:  # noqa: BLE001 - any failure of the classes source means "cannot check", never "invalid"
        return None
    names: set[str] = set()
    for classes in resolved.values():
        names |= {str(name) for name in classes}
    return names


def load_pairs(path: str | Path | None = None, strict: bool = False) -> HostPairs:
    """The host pairs, with every class name they mention checked against ``galaxus.data.classes``.

    Follows ``galaxus.serve.demo.rules.load_rules``: problems are collected rather than raised, because the classes
    source itself is not on ``main`` yet. ``strict=True`` turns them into an error once there is a contract.
    """
    raw = json.loads(Path(path or PAIRS_FILE).read_text(encoding="utf-8"))
    problems: list[str] = []
    valid = valid_class_names()
    if valid is None:
        problems.append("class names were not checked: the classes source is unavailable (contracts/classes.json, #11)")
    else:
        for entry in raw.get("pairs", []):
            problems.extend(
                f"{entry[side]} in {PAIRS_FILE.name} is not a class in any dimension"
                for side in ("true", "predicted")
                if str(entry[side]) not in valid
            )
    if problems and strict:
        raise ValueError("; ".join(problems))
    return HostPairs(raw=raw, problems=tuple(problems))


# ------------------------------------------------------------------------------------------------- labels and runs


def holdout_solution(dimension: str, train: pd.DataFrame | None = None) -> pd.DataFrame:
    """The scored human labels of one dimension on the official holdout: ``image_id`` plus the label or ``None``.

    The split, the scored-cell mask and the solution shape are all the data lane's
    (``galaxus.data.split``), so this report scores against exactly what the metric scores against.
    ``train`` is for the tests, which pass a synthetic table instead of reading ``$GALAXUS_DATA``.
    """
    if train is None:
        from galaxus.data.load import read_train

        train = read_train()
    holdout = split_holdout(train)[1]
    return solution_frame(holdout, (dimension,))


def labels_of(solution: pd.DataFrame, dimension: str) -> dict[str, str]:
    """``{image_id: human label}`` for the scored cells only.

    An unscored cell is empty in the solution: ``None``, or the ``NaN`` pandas may put in its place, or the empty
    string the CSV reader keeps. All three are dropped, so what is left is the cells the metric scores.
    """
    column = solution[dimension]
    keep = column.notna() & (column.astype(str).str.strip() != "")
    return {
        str(image_id): str(value) for image_id, value in zip(solution.loc[keep, ID_COLUMN], column[keep], strict=True)
    }


@dataclass(frozen=True)
class Run:
    """One predictions file, reduced to the one dimension this report looks at."""

    path: str
    labels: dict[str, str]
    confidence: dict[str, float]
    mirrored: bool = False
    swapped: int = 0

    @property
    def name(self) -> str:
        return Path(self.path).name


def read_run(path: str | os.PathLike[str], dimension: str) -> Run:
    """One run's prediction for ``dimension``, per image, read through ``galaxus.serve.runner.read_predictions``.

    ``repair=False``: a report never edits the run it is reading.
    """
    records = read_predictions(path, repair=False)
    labels: dict[str, str] = {}
    confidence: dict[str, float] = {}
    for image_id, record in records.items():
        value = (record.get("labels") or {}).get(dimension)
        if value is None or not str(value).strip():
            continue
        labels[image_id] = str(value)
        scores = record.get("confidence") or {}
        if dimension in scores:
            confidence[image_id] = float(scores[dimension])
    return Run(path=str(path), labels=labels, confidence=confidence)


def unmirror(run: Run, dimension: str, valid: set[str] | None = None) -> Run:
    """A mirrored run read back in the original orientation: left and right swapped in its answers.

    The swap is the data lane's rule (its issue #29, not built yet), reused from
    ``galaxus.serve.demo.rules.mirror_class``: it swaps the words left and right inside the class name and keeps the
    result only when it is a class itself. Without a classes source nothing is checked, which that function handles.
    """
    if valid is None:
        valid = valid_class_names()
    swapped: dict[str, str] = {}
    changed = 0
    for image_id, name in run.labels.items():
        turned = mirror_class(name, valid)
        swapped[image_id] = turned
        changed += turned != name
    return Run(path=run.path, labels=swapped, confidence=dict(run.confidence), mirrored=True, swapped=changed)


def combine_mirror(base: Run, unmirrored: Run) -> Run:
    """Mirror test-time augmentation: the two answers per image, combined into one.

    The rule is the one named in ``perspective_pairs.json`` (``higher_mean_confidence``) and it is **ours**, not the
    hosts': where the two runs disagree, the answer with the higher confidence for this dimension wins, and the base
    run wins ties and any image where a confidence is missing. The deep dive only says mirroring helped
    (#17 slide 19), so the combination itself is an assumption the config records with its ``TODO(owner)``.
    """
    labels = dict(base.labels)
    confidence = dict(base.confidence)
    for image_id, mirrored_label in unmirrored.labels.items():
        if image_id not in labels:
            labels[image_id] = mirrored_label
            if image_id in unmirrored.confidence:
                confidence[image_id] = unmirrored.confidence[image_id]
            continue
        if mirrored_label == labels[image_id]:
            continue
        mine = base.confidence.get(image_id)
        theirs = unmirrored.confidence.get(image_id)
        if mine is not None and theirs is not None and theirs > mine:
            labels[image_id] = mirrored_label
            confidence[image_id] = theirs
    return Run(path=f"{base.name} + {unmirrored.name} (mirror TTA)", labels=labels, confidence=confidence)


# ------------------------------------------------------------------------------------------------- the error tables


@dataclass(frozen=True)
class Confusion:
    """The confusion table of one dimension over the scored holdout cells."""

    dimension: str
    run: str
    counts: dict[tuple[str, str], int]
    scored: int
    predicted: int
    compared: int
    unlabelled: int
    missing: int

    @property
    def classes(self) -> tuple[str, ...]:
        names = {true for true, _ in self.counts} | {predicted for _, predicted in self.counts}
        return tuple(sorted(names))

    @property
    def correct(self) -> int:
        return sum(count for (true, predicted), count in self.counts.items() if true == predicted)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.compared if self.compared else None

    def count(self, true: str, predicted: str) -> int:
        return self.counts.get((true, predicted), 0)

    def support(self, name: str) -> int:
        """How many scored cells carry ``name`` as the human label."""
        return sum(count for (true, _), count in self.counts.items() if true == name)

    def predicted_as(self, name: str) -> int:
        return sum(count for (_, predicted), count in self.counts.items() if predicted == name)

    def pair_share(self, pair: Pair) -> tuple[int, int, float | None]:
        """``(count, support of the true class, share)`` for one pair; the share is ``None`` with no support."""
        support = self.support(pair.true)
        count = self.count(pair.true, pair.predicted)
        return count, support, (count / support if support else None)


def confuse(labels: dict[str, str], run: Run, dimension: str) -> Confusion:
    """The confusion table of ``run`` against the human ``labels``, over the images both have."""
    counts: dict[tuple[str, str], int] = {}
    for image_id, true in labels.items():
        predicted = run.labels.get(image_id)
        if predicted is None:
            continue
        key = (true, predicted)
        counts[key] = counts.get(key, 0) + 1
    compared = sum(counts.values())
    return Confusion(
        dimension=dimension,
        run=run.name,
        counts=counts,
        scored=len(labels),
        predicted=len(run.labels),
        compared=compared,
        unlabelled=len([image_id for image_id in run.labels if image_id not in labels]),
        missing=len(labels) - compared,
    )


@dataclass(frozen=True)
class ClassScore:
    """Precision, recall and F1 of one class, computed in this report (never a deck number)."""

    name: str
    support: int
    predicted: int
    correct: int

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def per_class_f1(table: Confusion) -> list[ClassScore]:
    """One :class:`ClassScore` per class that occurs as a human label or as a prediction, by class name."""
    return [
        ClassScore(
            name=name, support=table.support(name), predicted=table.predicted_as(name), correct=table.count(name, name)
        )
        for name in table.classes
    ]


def macro_f1(scores: list[ClassScore]) -> float | None:
    """The mean F1 over the classes present here.

    Not the leaderboard number: ``metric.py`` averages over every valid class of the dimension, and the official
    number for a run is the one in ``results/metrics.json``. This is an error-analysis figure on the classes this
    holdout slice actually contains.
    """
    return sum(score.f1 for score in scores) / len(scores) if scores else None


# ------------------------------------------------------------------------------------------------------- the report


@dataclass
class Report:
    """Everything the text and the figure show, with the provenance of each block."""

    dimension: str
    pairs: HostPairs
    before: Confusion
    after: Confusion | None = None
    mirror_swapped: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_mirror(self) -> bool:
        return self.after is not None

    def pair_rows(self) -> list[dict[str, Any]]:
        """One row per host pair: our share before, our share after when there is a mirrored run, and theirs."""
        rows = []
        for pair in self.pairs.pairs():
            count, support, share = self.before.pair_share(pair)
            row: dict[str, Any] = {
                "pair": pair,
                "before_count": count,
                "before_support": support,
                "before_share": share,
                "after_count": None,
                "after_support": None,
                "after_share": None,
            }
            if self.after is not None:
                after_count, after_support, after_share = self.after.pair_share(pair)
                row.update(after_count=after_count, after_support=after_support, after_share=after_share)
            rows.append(row)
        return rows


def build_report(
    dimension: str | None = None,
    predictions: str | os.PathLike[str] | None = None,
    mirrored: str | os.PathLike[str] | None = None,
    train: pd.DataFrame | None = None,
    pairs_file: str | Path | None = None,
) -> Report:
    """Read the runs and the holdout labels and assemble the report.

    Args:
        dimension: the dimension to analyse; defaults to the one named in the pair file.
        predictions: the base run's jsonl.
        mirrored: the run over mirrored images, whose answers are swapped back before they are compared.
        train: a table to split instead of reading ``train.csv`` from ``$GALAXUS_DATA`` (the tests pass one).
        pairs_file: a pair file to load instead of the one beside this module.
    """
    pairs = load_pairs(pairs_file)
    dimension = dimension or pairs.dimension
    if predictions is None:
        raise ValueError("a predictions file is needed: pass --predictions, or --exp to find it under $GALAXUS_DATA")
    labels = labels_of(holdout_solution(dimension, train), dimension)
    base = read_run(predictions, dimension)
    report = Report(dimension=dimension, pairs=pairs, before=confuse(labels, base, dimension))
    if mirrored is not None:
        valid = valid_class_names()
        turned = unmirror(read_run(mirrored, dimension), dimension, valid)
        report.after = confuse(labels, combine_mirror(base, turned), dimension)
        report.mirror_swapped = turned.swapped
        if valid is None:
            report.notes.append(
                "the mirrored run's left/right swap was not checked against the classes source: a swapped name is "
                "kept only when it is a class itself, and without contracts/classes.json (#11) that check is off"
            )
    else:
        report.notes.append("no mirrored run given: the before/after mirror TTA comparison is left out")
    report.notes.extend(pairs.problems)
    if dimension not in (pairs.dimension,):
        report.notes.append(f"the pair file describes {pairs.dimension}, this report was asked for {dimension}")
    return report


def _percent(share: float | None) -> str:
    return "     -" if share is None else f"{share * 100:5.1f}%"


def render_text(report: Report) -> str:
    """The report as text, provenance first. Everything here is an error analysis, not a score."""
    before, after = report.before, report.after
    lines = [
        f"PERSPECTIVE ERROR REPORT - dimension {report.dimension} - issue #34",
        "",
        "WHERE EACH NUMBER COMES FROM",
        f"  our counts, shares, F1   computed in this report, from {before.run} and the holdout's human labels",
        "                           (galaxus.data.split: split_holdout -> scored_mask -> solution_frame)",
        f"  the host percentages     the hosts' own model, not ours: {report.pairs.source}",
        "                           carried in bench/perspective_pairs.json with the source of each one",
        "  a number for the deck    results/metrics.json only (AGENTS.md 'Galaxus product rules' 1). This report",
        "                           does not write that file and none of its numbers is a score to show a judge.",
        "",
        "SCORED CELLS (the data lane's mask: a human label in a cell where the dimension is meaningful)",
        f"  holdout cells with a human label   {before.scored}",
        f"  images in the run                  {before.predicted}",
        f"  compared                           {before.compared}",
        f"  human-labelled but not predicted   {before.missing}",
        f"  predicted but not scored here      {before.unlabelled} (ignored)",
    ]
    if before.accuracy is not None:
        lines.append(
            f"  agreement with the human label     {before.correct}/{before.compared} {_percent(before.accuracy)}"
        )
    lines += [
        "",
        f"HOST PAIRS ({report.pairs.source})",
        "  a human label read as another class; share of that label's scored cells",
    ]
    header = f"  {'open':<5} {'pair':<40} {'ours ' + BEFORE:>13}"
    if report.has_mirror:
        header += f" {'ours ' + AFTER:>13}"
    lines += [header + f" {'hosts':>7}", f"  {'-' * (len(header) + 6)}"]
    for row in report.pair_rows():
        pair: Pair = row["pair"]
        text = (
            f"  {'yes' if pair.open_for_hosts else '':<5} {pair.name:<40} "
            f"{row['before_count']:>3}/{row['before_support']:<3} {_percent(row['before_share'])}"
        )
        if report.has_mirror:
            text += f" {row['after_count']:>3}/{row['after_support']:<3} {_percent(row['after_share'])}"
        text += f" {_percent(pair.host_share)}"
        lines.append(text)
    lines.append("  'open' marks the pairs the hosts said are still open for them.")

    if after is not None:
        combine = report.pairs.combine_rule
        rule = f"{combine.get('rule', '?')} ({combine.get('status', '?')}: ours, not the hosts'; see the pair file)"
        lines += [
            "",
            "MIRROR TEST-TIME AUGMENTATION",
            f"  base run                 {before.run}",
            f"  combined run             {after.run}",
            f"  labels swapped back      {report.mirror_swapped} (galaxus.serve.demo.rules.mirror_class, the data",
            "                           lane's rule from its issue #29; the swap is theirs, not ours)",
            f"  agreement before         {before.correct}/{before.compared} {_percent(before.accuracy)}",
            f"  agreement after          {after.correct}/{after.compared} {_percent(after.accuracy)}",
            f"  combination rule         {rule}",
        ]

    for title, table in ((BEFORE, before), (AFTER, after)):
        if table is None:
            continue
        scores = per_class_f1(table)
        mean = macro_f1(scores)
        lines += [
            "",
            f"PER-CLASS F1 ({title}) - computed here from the run and the human labels, not a leaderboard number",
            f"  {'class':<30} {'support':>7} {'pred':>6} {'precision':>10} {'recall':>7} {'F1':>6}",
        ]
        for score in sorted(scores, key=lambda item: item.name):
            lines.append(
                f"  {score.name:<30} {score.support:>7} {score.predicted:>6} "
                f"{score.precision:>10.3f} {score.recall:>7.3f} {score.f1:>6.3f}"
            )
        lines.append(f"  mean F1 over the {len(scores)} classes present: {'-' if mean is None else f'{mean:.3f}'}")
        lines.append("  metric.py averages over every valid class, so this is not the dimension's macro F1.")

    # Class names are long, so the columns are numbered and the numbers are spelled out under the table; a column
    # per full name would be wider than any terminal once a dimension has ten classes.
    names = before.classes
    lines += ["", "CONFUSION TABLE (rows: human label, columns: prediction, base run)"]
    width = max([len(name) for name in names] + [10])
    lines.append("  " + " " * (width + 4) + " ".join(f"{number:>5}" for number in range(1, len(names) + 1)))
    for number, true in enumerate(names, start=1):
        cells = " ".join(f"{before.count(true, predicted):>5}" for predicted in names)
        lines.append(f"  {number:>2}. {true:<{width}} {cells}")
    lines.append("  columns are the rows, in the same order.")

    if report.notes:
        lines += ["", "NOTES"]
        lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------------------- the figure


@dataclass(frozen=True)
class FigureResult:
    """Whether the slide 8 figure was written, and why not when it was not."""

    path: Path | None
    reason: str = ""

    @property
    def saved(self) -> bool:
        return self.path is not None


def figure_path(out_root: str | os.PathLike[str] | None = None) -> Path:
    """``$GALAXUS_DATA/out/perspective/slide8_perspective_pairs.png``, or the same name under ``out_root``."""
    if out_root is None:
        from galaxus.data.paths import out_dir

        out_root = out_dir()
    return Path(out_root) / FIGURE_DIR / FIGURE_NAME


def save_figure(report: Report, out_root: str | os.PathLike[str] | None = None) -> FigureResult:
    """The one figure for deck slide 8: the host pairs, before in ``red`` and after in ``green`` (STYLE.md).

    matplotlib is not a dependency of this repo and a worker may not add one (AGENTS.md "Lanes"), so a missing
    matplotlib is a clean skip with a reason: the tables above are the deliverable, the figure is the nice version
    of them.
    """
    rows = [row for row in report.pair_rows() if row["before_support"]]
    if not rows:
        return FigureResult(None, "none of the host pairs has a scored cell in this holdout slice: nothing to plot")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as missing:
        return FigureResult(
            None,
            f"matplotlib is not installed ({missing}). The tables above are complete; to get the slide 8 figure, "
            "ask design for matplotlib in the dev dependency group (a worker never edits pyproject.toml) and run "
            "this script again.",
        )

    # STYLE.md "Charts and figures": report figures use these rcParams; Geist first, system fallback after, because
    # the font may not be installed where this runs.
    # STYLE.md asks for Geist. It is a webfont and may not be installed where this runs, so it is only named when
    # it is there; otherwise matplotlib's default is used, as the demo page falls back to the system stack.
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = ["Geist", "DejaVu Sans"] if "Geist" in installed else ["DejaVu Sans"]
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["white"],
            "axes.edgecolor": PALETTE["rule"],
            "axes.grid": True,
            "grid.color": PALETTE["rule"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": PALETTE["muted"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
        }
    )
    figure, axes = plt.subplots(figsize=(11, 0.95 * len(rows) + 3.0))
    height = 0.26 if report.has_mirror else 0.38
    positions = list(range(len(rows)))[::-1]

    before = [100 * (row["before_share"] or 0.0) for row in rows]
    offset = height / 2 if report.has_mirror else 0.0
    axes.barh([p + offset for p in positions], before, height=height, color=PALETTE["red"], label="this run")
    if report.has_mirror:
        after = [100 * (row["after_share"] or 0.0) for row in rows]
        axes.barh(
            [p - offset for p in positions], after, height=height, color=PALETTE["green"], label="after mirror TTA"
        )

    for position, row in zip(positions, rows, strict=True):
        axes.text(
            100 * (row["before_share"] or 0.0) + 0.6,
            position + offset,
            f"{100 * (row['before_share'] or 0.0):.1f}%  ({row['before_count']}/{row['before_support']})",
            va="center",
            fontsize=9,
            color=PALETTE["ink"],
        )
        if report.has_mirror:
            axes.text(
                100 * (row["after_share"] or 0.0) + 0.6,
                position - offset,
                f"{100 * (row['after_share'] or 0.0):.1f}%  ({row['after_count']}/{row['after_support']})",
                va="center",
                fontsize=9,
                color=PALETTE["ink"],
            )
        host = row["pair"].host_share
        if host is not None:
            axes.plot(
                [100 * host],
                [position],
                marker="|",
                markersize=22,
                markeredgewidth=2,
                color=PALETTE["rule2"],
                linestyle="none",
            )

    axes.plot(
        [],
        [],
        marker="|",
        markersize=12,
        markeredgewidth=2,
        color=PALETTE["rule2"],
        linestyle="none",
        label="the hosts' model (#17 slide 22)",
    )
    axes.set_yticks(positions)
    axes.set_yticklabels([row["pair"].name.replace("->", "→") for row in rows], color=PALETTE["ink"], fontsize=11)
    axes.set_ylim(-0.65, len(rows) - 0.35)
    axes.set_xlabel("share of that human label's scored cells read as the other class (%)")
    axes.xaxis.grid(True)
    axes.yaxis.grid(False)
    headline = "Mirror TTA on the pairs the hosts left open" if report.has_mirror else "The pairs the hosts left open"
    axes.set_title(headline, loc="left", color=PALETTE["ink"], fontsize=15, pad=16)
    axes.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        labelcolor=PALETTE["muted"],
    )
    figure.text(
        0.012,
        0.015,
        f"dimension {report.dimension} - our shares: this run against the holdout's human labels "
        f"(galaxus.data.split), error analysis only.\nEvery number on a slide comes from results/metrics.json "
        f"(AGENTS.md product rule 1). Host percentages: {report.pairs.source}.",
        fontsize=8,
        color=PALETTE["muted2"],
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))

    path = figure_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return FigureResult(path)


# ------------------------------------------------------------------------------------------------------ the command


def main(argv: list[str] | None = None) -> int:
    """``python bench/perspective_report.py``: print the tables, save the one slide 8 figure."""
    parser = argparse.ArgumentParser(prog="python bench/perspective_report.py", description=__doc__.split("\n")[0])
    parser.add_argument("--predictions", help="the base run's jsonl; or give --exp")
    parser.add_argument("--mirrored", help="the run over mirrored images; its answers are swapped back")
    parser.add_argument("--exp", help="experiment id, to find <exp>/holdout.jsonl under $GALAXUS_DATA/out")
    parser.add_argument("--mirrored-exp", help="experiment id of the mirrored run, same layout")
    parser.add_argument("--part", default="holdout", help="which part of the run to read (default: holdout)")
    parser.add_argument("--dimension", default=None, help="default: the dimension named in perspective_pairs.json")
    parser.add_argument("--out", default=None, help="where the figure goes (default: $GALAXUS_DATA/out)")
    parser.add_argument("--no-figure", action="store_true", help="tables only")
    arguments = parser.parse_args(argv)

    predictions = arguments.predictions
    if predictions is None and arguments.exp:
        predictions = str(predictions_path(arguments.exp, arguments.part))
    mirrored = arguments.mirrored
    if mirrored is None and arguments.mirrored_exp:
        mirrored = str(predictions_path(arguments.mirrored_exp, arguments.part))
    if predictions is None:
        parser.error("give --predictions <file.jsonl> or --exp <EXP-####>")

    report = build_report(dimension=arguments.dimension, predictions=predictions, mirrored=mirrored, pairs_file=None)
    print(render_text(report))
    if arguments.no_figure:
        print("[figure] skipped: --no-figure")
        return 0
    result = save_figure(report, arguments.out)
    if result.saved:
        print(f"[figure] slide 8 figure written to {result.path} (never committed: AGENTS.md product rule 3)")
    else:
        print(f"[figure] skipped: {result.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
