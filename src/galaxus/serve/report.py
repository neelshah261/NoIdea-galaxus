"""Section 7 of technical_report.md ("Galaxus challenge: results and resources"), generated, never typed.

AGENTS.md product rule 1: every number shown to judges comes from ``results/metrics.json`` (written only by the eval
script) or another generated file, and names its source. This module reads those files and writes the section as
Markdown; anything that has not been measured yet is written as ``TODO(...)``, never guessed.

Sources, each optional:

- EXP rows: ``results/metrics.json`` (``--results`` to point elsewhere): scores per dimension, weighted F1, the model,
  the commit, the hardware and the cost per 100k.
- Benchmarks: ``$GALAXUS_DATA/out/bench/*.jsonl`` from ``galaxus.serve.bench``: images/s, peak memory, cost per 100k.
- Final-run drills: ``$GALAXUS_DATA/out/drill/*/report.json`` from ``final_run --drill`` (#32, #181): the speed of the
  final method end to end, on the machine that runs it, for the method named by ``--exp``.
- The method's members: an ensemble or type-prior row names its members' EXP ids in its config; the models, revisions
  and gateway tokens of the final method are read from those rows, recursively, not from every row ever written.
- MODELS.md's "Development only" table: the names of the models used while building, never at runtime.
- Label errors: ``$GALAXUS_DATA/out/label_errors.csv``: only the row count is reported, never the rows.
- Kaggle: the public score of a submission (read-only CLI) when ``--kaggle`` is given; otherwise TODO(owner).

    python -m galaxus.serve.report --exp EXP-0010 --out docs/report_section7.md
    python -m galaxus.serve.report --exp EXP-0061 --weights "<HF repo> @ <revision>"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def dimension_order(rows: Sequence[Mapping[str, Any]] = ()) -> tuple[str, ...]:
    """The six dimensions in submission order, from ``galaxus.data.classes`` (#82: never typed here).

    Without a classes source, the order the EXP rows give them in.
    """
    try:
        from galaxus.data.classes import dimensions

        return tuple(dimensions())
    except Exception:  # noqa: BLE001 - no contracts/classes.json and no $GALAXUS_DATA
        seen: dict[str, None] = {}
        for row in rows:
            seen.update(dict.fromkeys(row.get("macro_f1") or {}))
        return tuple(seen)
COMPETITION = "hackapertus-galaxus"
#: The hosts' baselines live in challenge_description.md "Baselines"; they are linked, never restated here.
BASELINES_LINK = "[challenge_description.md, Baselines](../challenge_description.md#baselines)"


def _f(value: Any, digits: int = 3) -> str:
    return "TODO(model): not measured" if value is None else f"{value:.{digits}f}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows if isinstance(rows, list) else []


def pick(rows: Sequence[Mapping[str, Any]], exp: str | None) -> Mapping[str, Any] | None:
    """The row named ``exp``, or the best holdout score when none is named."""
    if exp:
        return next((row for row in rows if row.get("id") == exp), None)
    scored = [row for row in rows if isinstance(row.get("score"), (int, float))]
    return max(scored, key=lambda row: row["score"]) if scored else None


def read_bench(directory: Path) -> list[dict[str, Any]]:
    """Benchmark rows, fake ones (label ``fake-...``) left out."""
    rows = []
    for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not str(row.get("label", "")).startswith("fake-"):
                    rows.append(row)
    return rows


def count_label_errors(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def kaggle_public_score(run: Any = None) -> str | None:
    """The newest scored submission's public score, read-only (``kaggle competitions submissions``)."""
    command = [sys.executable, "-m", "kaggle", "competitions", "submissions", COMPETITION, "--csv"]
    try:
        output = (run or _run)(command)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = list(csv.DictReader(output.splitlines()))
    scored = [row for row in rows if (row.get("publicScore") or "").strip() not in ("", "None")]
    return scored[0]["publicScore"] if scored else None


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=60, check=False).stdout


#: The final images, from challenge_description.md "Timeline" (Sat 28 Nov, 13:00): "the 5,292 images".
FINAL_IMAGES = 5292
EXP_ID = re.compile(r"^EXP-\d{4}$")


def method_rows(rows: Sequence[Mapping[str, Any]], selected: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """``selected`` and every row it is built from: an ensemble's ``config.members`` and a prior's ``config.base``,
    followed while they name EXP ids, so the final method's models come from its own rows only."""
    if not selected:
        return []
    by_id = {str(row.get("id")): row for row in rows}
    found, todo, seen = [], [selected], {str(selected.get("id"))}
    while todo:
        row = todo.pop(0)
        found.append(row)
        config = row.get("config") or {}
        named = list((config.get("members") or {}).values()) + [config.get("base")]
        for exp in (str(e) for e in named if e):
            if EXP_ID.match(exp) and exp not in seen and exp in by_id:
                seen.add(exp)
                todo.append(by_id[exp])
    return found


def read_drills(directory: Path) -> list[dict[str, Any]]:
    """Every ``final_run --drill`` report under ``directory`` (``$GALAXUS_DATA/out/drill``)."""
    drills = []
    for path in sorted(directory.glob("*/report.json")) if directory.is_dir() else []:
        try:
            drills.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return drills


def development_models(models_file: Path) -> list[str]:
    """The first column of MODELS.md's "Development only" table, in its order."""
    if not models_file.is_file():
        return []
    names, inside = [], False
    for line in models_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip().lower().startswith("## development only")
            continue
        if not (inside and line.startswith("|")):
            continue
        first = line.strip().strip("|").split("|")[0].strip()
        if first and first != "Model" and not set(first) <= {"-", " ", ":"}:
            names.append(first)
    return names


def gateway_tokens(members: Sequence[Mapping[str, Any]]) -> list[str]:
    """Prompt and answer tokens per image of each member that called a model over the gateway, from its config."""
    lines = []
    for row in members:
        config = row.get("config") or {}
        prompt, answer, images = config.get("prompt_tokens"), config.get("completion_tokens"), config.get("images")
        if isinstance(prompt, (int, float)) and isinstance(images, (int, float)) and images:
            per_image = prompt / images
            answers = f" and {answer / images:,.0f} answer" if isinstance(answer, (int, float)) else ""
            lines.append(f"{row.get('model') or row.get('id')} ({row.get('id')}): {per_image:,.0f} prompt{answers} "
                         f"tokens per image, about {per_image * 100_000 / 1e6:,.0f}M prompt tokens per 100,000 images")
    return lines


def section7(
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    bench: Sequence[Mapping[str, Any]],
    label_errors: int | None,
    public_score: str | None = None,
    team: str | None = None,
    results_name: str = "results/metrics.json",
    drills: Sequence[Mapping[str, Any]] = (),
    development: Sequence[str] = (),
    weights: str | None = None,
) -> str:
    """The section as Markdown, in the template's order."""
    out = ["## 7. Galaxus challenge: results and resources", ""]
    source = f"`{results_name}`"

    kaggle = f"public score (dev) **{public_score}**" if public_score else "public score: TODO(owner) after the upload"
    team_text = team or "TODO(owner): Kaggle team name (#25)"
    out += [f"- **Kaggle:** team {team_text}; {kaggle}. The private score is Kaggle's.", ""]

    if selected:
        images = selected.get("holdout", {}).get("images", "?")
        heading = f"- **Per dimension** ({selected['id']}, official holdout of {images} images, metric.py; {source}):"
        out += [heading, ""]
        out += ["| Dimension | Macro F1 | Weighted F1 |", "| --- | --- | --- |"]
        macro = selected.get("macro_f1", {})
        weighted = selected.get("weighted_f1", {})
        order = dimension_order([selected])
        for dimension in [d for d in order if d in macro] + [d for d in macro if d not in order]:
            out.append(f"| {dimension} | {_f(macro.get(dimension))} | {_f(weighted.get(dimension))} |")
        out.append(f"| **mean** | **{_f(selected.get('score'))}** | **{_f(selected.get('overall_weighted_f1'))}** |")
        out += ["", f"  The bar to beat is the hosts' baselines: {BASELINES_LINK}.", ""]
    else:
        out += ["- **Per dimension:** TODO(model): no EXP row yet.", ""]

    out += [f"- **What mattered** (every EXP row, holdout macro F1; source {source}):", ""]
    if rows:
        order = dimension_order(rows)
        out += ["| EXP | Predictor | Mean | " + " | ".join(order) + " | img/s | $ / 100k | Note |",
                "| --- | --- | --- | " + " | ".join("---" for _ in order) + " | --- | --- | --- |"]
        for row in rows:
            macro = row.get("macro_f1", {})
            out.append(
                f"| {row.get('id')} | {row.get('predictor', '')} | {_f(row.get('score'))} | "
                + " | ".join(_f(macro.get(d)) for d in order)
                + f" | {_f(row.get('images_per_second'), 1)} | {_f(row.get('cost_per_100k_usd'), 2)} | "
                f"{str(row.get('note', '')).replace('|', '/')} |"
            )
        out += ["", "  What did not work: TODO(model): one line per EXP that did not help, and why.", ""]
    else:
        out += ["  TODO(model): no EXP rows yet.", ""]

    members = method_rows(rows, selected)
    sources = members or rows
    models = sorted({str(r["model"]) + (f" @ {r['model_revision']}" if r.get("model_revision") else "")
                     for r in sources if r.get("model")})
    helpers = sorted({str(h) for r in sources for h in r.get("helpers", [])})
    built_from = ", ".join(str(r.get("id")) for r in members[1:]) or "none"
    scope = f" (the final method {selected['id']} and the rows it is built from: {built_from})" if members else ""
    out += [f"- **Models**{scope}:", ""]
    out += [f"  - Runtime: {', '.join(models) if models else 'TODO(model): Apertus checkpoint and revision'}"
            + (f"; helpers: {', '.join(helpers)}" if helpers else "; no helper models")]
    if development:
        out += [f"  - Development only, never at runtime (source `MODELS.md`): {'; '.join(development)}", ""]
    else:
        out += [("  - Development only: TODO(owner): every tool and model used while building (coding assistants, "
                 "any model used for targets), with version and role."), ""]

    real = [b for b in bench if b.get("device")]
    mine = [d for d in drills if selected and d.get("exp") == selected.get("id") and d.get("images_per_second")]
    if mine:
        out += resources_from_drill(max(mine, key=lambda d: str(d.get("finished_at", ""))), members)
    elif real:
        out += ["- **Resources** (source `$GALAXUS_DATA/out/bench/`, `galaxus.serve.bench`):", ""]
        best = max(real, key=lambda b: b.get("images_per_second") or 0)
        per_day = 100_000 / best["images_per_second"] / 3600 if best.get("images_per_second") else None
        memory = f"{_f(best.get('peak_memory_gb'), 1)} GB ({best.get('memory_note', '')})"
        speed = f"**{_f(best.get('images_per_second'), 2)} images/s**"
        price = f"**${_f(best.get('cost_per_100k'), 2)}** at ${_f(best.get('dollars_per_hour'), 2)}/h"
        out += [
            f"  - Hardware: {best.get('device')}; peak memory {memory}",
            f"  - Batch size {best.get('batch_size')}, workers {best.get('workers', 1)}: {speed}",
            f"  - 100,000 images a day: {_f(per_day, 2)} GPU hours, {price}",
            "",
        ]
    else:
        out += ["- **Resources:**", "",
                ("  TODO(serve): no drill of the final method (`final_run --drill`) and no benchmark "
                 "(`python -m galaxus.serve.bench`) yet."), ""]

    commit = selected.get("commit") if selected else None
    dirty = " (dirty checkout: re-run from a clean commit)" if selected and selected.get("dirty") else ""
    commit_text = commit[:12] if commit else "TODO(owner)"
    weights = weights or "TODO(owner): Hugging Face repository and revision, published by 14:30"
    out += [f"- **Commit and weights:** commit `{commit_text}`{dirty}; weights: {weights}.", ""]

    if label_errors is None:
        errors = "label errors: TODO(data): not generated yet (`python -m galaxus.model.heads label-errors`)."
    else:
        errors = f"{label_errors} suspected label errors, sent to the hosts as a csv by Discord DM (not committed)."
    out += [f"- **Optional:** {errors}", ""]
    return "\n".join(out)


def resources_from_drill(drill: Mapping[str, Any], members: Sequence[Mapping[str, Any]]) -> list[str]:
    """The final method's speed from its newest drill; the machine from its rows; gateway tokens from its members."""
    speed = float(drill["images_per_second"])
    hardware = next((r.get("hardware") for r in members if r.get("hardware")), None) or {}
    machine = " ".join(str(hardware[k]) for k in ("system", "machine") if hardware.get(k)) or "TODO(serve): machine"
    verdict = "clears" if drill.get("meets_floor") else "is BELOW"
    end_to_end = (f"  - The final method end to end ({drill.get('predictor')}, {drill.get('images')} "
                  f"{drill.get('part')} images, kill and resume included, {str(drill.get('finished_at', ''))[:16]} "
                  f"UTC): **{speed:.3f} images/s**, which {verdict} the {drill.get('floor_images_per_second', 2.0)} "
                  f"images/s floor; the {FINAL_IMAGES:,} final images take about {FINAL_IMAGES / speed / 60:.0f} min "
                  "at that speed")
    machine_line = (f"  - Hardware: {machine}, {hardware.get('gpu') or 'no GPU'}; 100,000 images a day at this speed: "
                    f"{100_000 / speed / 3600:.1f} hours on that machine, no GPU hours bought")
    lines = ["- **Resources** (source `$GALAXUS_DATA/out/drill/`, `final_run --drill`, and the method's rows):", "",
             end_to_end, machine_line]
    lines += [f"  - Gateway: {line}" for line in gateway_tokens(members)]
    return lines + [""]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.report", description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--exp", default=None, help="the EXP of the final submission (default: best holdout score)")
    parser.add_argument("--team", default=None, help="Kaggle team name")
    parser.add_argument("--kaggle", action="store_true", help="read the public score from Kaggle (read-only)")
    parser.add_argument("--public-score", default=None, help="the final submission's public score, as Kaggle shows it")
    parser.add_argument("--weights", default=None, help="the published weights: <Hugging Face repo> @ <revision>")
    parser.add_argument("--models-file", type=Path, default=Path("MODELS.md"))
    parser.add_argument("--out", type=Path, default=None, help="write here (default: print)")
    arguments = parser.parse_args(argv)

    from galaxus.data.paths import out_dir

    out = out_dir()
    rows = read_rows(arguments.results)
    text = section7(
        rows,
        pick(rows, arguments.exp),
        read_bench(out / "bench"),
        count_label_errors(out / "label_errors.csv"),
        arguments.public_score or (kaggle_public_score() if arguments.kaggle else None),
        arguments.team,
        str(arguments.results).replace("\\", "/"),
        read_drills(out / "drill"),
        development_models(arguments.models_file),
        arguments.weights,
    )
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(text + "\n", encoding="utf-8")
        print(f"written to {arguments.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
