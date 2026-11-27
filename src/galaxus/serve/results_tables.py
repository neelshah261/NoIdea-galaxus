"""#57: results and ablation tables, plus one SVG chart, generated from ``results/metrics.json`` and nothing else.

The insights part of the assessment asks for "clear evidence of what mattered and what did not"; design/JUDGING.md
names the artifact: an ablation table generated from ``results/metrics.json``. This writes ``docs/results.md`` and
``docs/results_chart.svg`` so the report (#54) and the deck (#52) never carry a typed number.

- **All runs:** one row per EXP: predictor, mean macro F1, the six dimensions, weighted F1, images/s, cost per 100k,
  commit.
- **Ablations:** each EXP against the run it changes. The parent is ``config.parent`` (or ``parent`` on the row) when
  the eval recorded one; otherwise the previous EXP **on the same holdout**. Rows whose ``holdout.sha256`` differ
  scored different images and are never compared; the table says so.
- **Perspective** on its own line: the hosts' business priority and a bonus criterion (#17).
- **Chart:** per-dimension macro F1 for the chosen EXPs, in the design/STYLE.md colours, as plain SVG (no new
  dependency; ``ci`` rejects other image files).

    python -m galaxus.serve.results_tables --exps EXP-0001 EXP-0010
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from galaxus.serve.report import DIMENSIONS_ORDER, read_rows

PAPER, WHITE, INK, MUTED, RULE, RULE2, ACCENT, ACCENT2 = (
    "#f2efe9", "#ffffff", "#141414", "#55524c", "#dcd6cc", "#c4bcae", "#c93c05", "#a86400")
OTHER_COLOURS = (ACCENT2, MUTED, "#8a8276")


def colours(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """STYLE.md: orange only for our best run, grey for the majority baseline, quiet tones for the rest."""
    scores = [r.get("score") for r in rows]
    best = max(range(len(rows)), key=lambda i: scores[i] if scores[i] is not None else -1) if rows else -1
    result, other = [], 0
    for i, row in enumerate(rows):
        if i == best:
            result.append(ACCENT)
        elif row.get("predictor") == "majority":
            result.append(RULE2)
        else:
            result.append(OTHER_COLOURS[other % len(OTHER_COLOURS)])
            other += 1
    return result


def _num(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _sha(row: Mapping[str, Any]) -> str | None:
    return (row.get("holdout") or {}).get("sha256")


def _dims(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = {d for row in rows for d in (row.get("macro_f1") or {})}
    return [d for d in DIMENSIONS_ORDER if d in seen] + sorted(seen - set(DIMENSIONS_ORDER))


def parent_of(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, str]:
    """(the run this EXP changes, why). Never a run scored on other holdout images."""
    by_id = {r.get("id"): r for r in rows}
    named = (row.get("config") or {}).get("parent") or row.get("parent")
    if named:
        parent = by_id.get(named)
        if parent is None:
            return None, f"parent {named} not in the file"
        if _sha(parent) != _sha(row):
            return None, f"parent {named} scored a different holdout: not compared"
        return parent, "config.parent"
    earlier = [r for r in rows[: rows.index(row)] if _sha(r) == _sha(row)] if row in rows else []
    return (earlier[-1], "previous run, same holdout") if earlier else (None, "first run on this holdout")


def all_runs_table(rows: Sequence[Mapping[str, Any]]) -> str:
    dims = _dims(rows)
    head = "| EXP | Predictor | Mean | " + " | ".join(dims) + " | Weighted | img/s | $ / 100k | Commit |"
    rule = "| --- | --- | --- | " + " | ".join("---" for _ in dims) + " | --- | --- | --- | --- |"
    lines = [head, rule]
    for row in rows:
        macro = row.get("macro_f1") or {}
        lines.append(
            f"| {row.get('id')} | {row.get('predictor', '')} | **{_num(row.get('score'))}** | "
            + " | ".join(_num(macro.get(d)) for d in dims)
            + f" | {_num(row.get('overall_weighted_f1'))} | {_num(row.get('images_per_second'), 1)} | "
            f"{_num(row.get('cost_per_100k_usd'), 2)} | `{str(row.get('commit') or '-')[:8]}` |"
        )
    return "\n".join(lines)


def ablation_table(rows: Sequence[Mapping[str, Any]]) -> str:
    dims = _dims(rows)
    lines = ["| EXP | Changes | Basis | Mean gain | " + " | ".join(dims) + " |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in dims) + " |"]
    for row in rows:
        parent, why = parent_of(row, rows)
        if parent is None:
            lines.append(f"| {row.get('id')} | - | {why} | - | " + " | ".join("-" for _ in dims) + " |")
            continue
        gains = [(row.get("macro_f1") or {}).get(d) for d in dims]
        bases = [(parent.get("macro_f1") or {}).get(d) for d in dims]
        cells = [("-" if g is None or b is None else f"{g - b:+.3f}") for g, b in zip(gains, bases)]
        mean = (None if row.get("score") is None or parent.get("score") is None
                else f"{row['score'] - parent['score']:+.3f}")
        lines.append(f"| {row.get('id')} | {parent.get('id')} | {why} | **{mean or '-'}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def perspective_line(rows: Sequence[Mapping[str, Any]]) -> str:
    scored = [r for r in rows if (r.get("macro_f1") or {}).get("perspective") is not None]
    if not scored:
        return "Perspective: no run has a perspective score yet."
    best = max(scored, key=lambda r: r["macro_f1"]["perspective"])
    runs = ", ".join(f"{r.get('id')} {r['macro_f1']['perspective']:.3f}" for r in scored)
    return f"**Perspective** (the hosts' priority and a bonus): best {best.get('id')} at " \
           f"**{best['macro_f1']['perspective']:.3f}**. All runs: {runs}."


def chart_svg(rows: Sequence[Mapping[str, Any]], title: str = "Macro F1 per dimension, official holdout") -> str:
    """Grouped bars, one group per dimension, one bar per EXP; numbers written on the bars."""
    dims = _dims(rows) or list(DIMENSIONS_ORDER)
    width, height, left, right, top, bottom = 1600, 700, 90, 40, 120, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    group = plot_w / len(dims)
    bar = min(48, (group - 30) / max(1, len(rows)))
    palette = colours(rows)
    style = (
        '<style>text{font-family:Geist,"Segoe UI",system-ui,sans-serif;fill:' + INK + '}'
        '.m{font-family:"Geist Mono",Consolas,monospace;font-size:13px;fill:' + MUTED + '}'
        '.v{font-family:"Geist Mono",Consolas,monospace;font-size:12px}</style>'
    )
    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img"><title>{escape(title)}</title>'
    )
    out = [opening, style,
           f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
           f'<text x="{left}" y="56" font-size="30" font-weight="700">{escape(title)}</text>',
           f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="{WHITE}"/>']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - tick)
        out.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{RULE}"/>')
        out.append(f'<text x="{left - 10}" y="{y + 4:.1f}" class="m" text-anchor="end">{tick:.2f}</text>')
    for i, dim in enumerate(dims):
        x0 = left + i * group + (group - bar * len(rows)) / 2
        for j, row in enumerate(rows):
            value = (row.get("macro_f1") or {}).get(dim)
            if value is None:
                continue
            h = plot_h * max(0.0, min(1.0, value))
            x, y = x0 + j * bar, top + plot_h - h
            colour = palette[j]
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar - 4:.1f}" height="{h:.1f}" fill="{colour}"/>')
            out.append(f'<text x="{x + (bar - 4) / 2:.1f}" y="{y - 6:.1f}" class="v" text-anchor="middle">'
                       f'{value:.2f}</text>')
        label_x = left + i * group + group / 2
        out.append(f'<text x="{label_x:.1f}" y="{top + plot_h + 28}" class="m" text-anchor="middle">{escape(dim)}</text>')
    for j, row in enumerate(rows):
        x = left + j * 260
        out.append(f'<rect x="{x}" y="{height - 46}" width="16" height="16" fill="{palette[j]}"/>')
        label = f"{row.get('id')} {row.get('predictor', '')} (mean {_num(row.get('score'))})"
        out.append(f'<text x="{x + 24}" y="{height - 33}" class="m">{escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def results_markdown(rows: Sequence[Mapping[str, Any]], source: str, chart: str | None) -> str:
    if not rows:
        return f"# Results\n\nNo EXP rows in `{source}` yet.\n"
    shas = {_sha(r) for r in rows}
    holdout_note = ("All rows scored the same holdout." if len(shas) == 1
                    else f"{len(shas)} different holdouts appear: only rows on the same holdout are compared.")
    parts = [
        "# Results",
        "",
        f"Generated by `python -m galaxus.serve.results_tables` from `{source}`; do not edit by hand. "
        "Scores are macro F1 on the official holdout (metric.py, human labels only). " + holdout_note,
        "",
        "## All runs",
        "",
        all_runs_table(rows),
        "",
        "## What each change did",
        "",
        ablation_table(rows),
        "",
        perspective_line(rows),
        "",
    ]
    if chart:
        parts += ["## Chart", "", f"![Macro F1 per dimension]({chart})", ""]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.results_tables", description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--exps", nargs="*", default=None, help="EXPs in the chart (default: the best four)")
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"))
    parser.add_argument("--chart", type=Path, default=Path("docs/results_chart.svg"))
    arguments = parser.parse_args(argv)

    rows = read_rows(arguments.results)
    if arguments.exps:
        chosen = [r for e in arguments.exps for r in rows if r.get("id") == e]
    else:
        chosen = sorted([r for r in rows if r.get("score") is not None], key=lambda r: -r["score"])[:4]
    chart_name = None
    if chosen:
        arguments.chart.parent.mkdir(parents=True, exist_ok=True)
        arguments.chart.write_text(chart_svg(chosen), encoding="utf-8")
        chart_name = arguments.chart.name
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(results_markdown(rows, str(arguments.results).replace("\\", "/"), chart_name),
                             encoding="utf-8")
    print(f"{len(rows)} runs -> {arguments.out}" + (f", chart -> {arguments.chart}" if chart_name else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
