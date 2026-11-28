"""#52: the pitch deck. Fills ``presentation/deck.html`` from ``results/metrics.json`` and prints it to PDF.

    uv run python -m galaxus.serve.deck --best EXP-0025 --method "Apertus 8B, observe first, no context"

- **Placeholders** (design/PITCH_DECK.md), in braces:
  - ``EXP-0025.score``, ``EXP-0025.macro_f1.subject``, ``best.images_per_second``: a field of a row; ``best`` is
    ``--best``.
  - ``delta EXP-0027.score EXP-0021.score``: the signed difference of two fields.
  - ``raw EXP-0025.score``: the bare number, for a bar's width; 0 when missing.
  - ``rows EXP-0001 EXP-0002 ...``: table rows (id, predictor, model, score, against the first id, img/s), best first.
  - ``chart EXP-0002 EXP-0025``: ``results_tables.chart_svg`` of those rows, per dimension, inline.
  - ``method``: the ``--method`` text, the final method's name once #137 picks it.
  - ``label_errors``: the row count of ``$GALAXUS_DATA/out/label_errors.csv`` (only the count, never the rows).
- **No made-up numbers:** a missing row or a null field shows as "pending" (amber), never as a number (#52).
- **Output:** ``$GALAXUS_DATA/out/deck/deck.html`` and ``deck.pdf``. The PDF goes to sprintd, never into git.
- **PDF:** headless Chrome prints the 1920 x 1080 pages (``--chrome`` or ``$CHROME`` for another browser binary).
"""

from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from galaxus.serve.report import count_label_errors, read_rows
from galaxus.serve.results_tables import chart_svg, no_model

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "presentation" / "deck.html"
RESULTS = REPO / "results" / "metrics.json"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PENDING = "<span class=pending>pending</span>"  # no quotes, so it can't end an attribute it lands in
#: Decimals per field; every other number gets 3, as the eval script prints scores.
DECIMALS = {"images_per_second": 1, "cost_per_100k_usd": 2, "peak_gpu_memory_gb": 1}
PLACEHOLDER = re.compile(
    r"\{(EXP-\d{4}\.[\w.]+|best\.[\w.]+|(?:delta|raw|rows|chart) [^{}]+|method|label_errors)\}"
)

Rows = Mapping[str, Mapping[str, Any]]


def value(rows: Rows, ref: str, best: str | None) -> Any:
    """The field ``EXP-####.a.b`` (or ``best.a.b``) of a row, None when the row or the field is missing."""
    exp, _, path = ref.partition(".")
    node: Any = rows.get(best or "") if exp == "best" else rows.get(exp)
    for key in path.split("."):
        node = node.get(key) if isinstance(node, Mapping) else None
    return node


def number(field: str, x: Any) -> str:
    """A field's value as text: floats rounded per ``DECIMALS``, ints with thousands separators, None pending."""
    if x is None:
        return PENDING
    if isinstance(x, (bool, str)):
        return html.escape(str(x))
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{DECIMALS.get(field.rsplit('.', 1)[-1], 3)}f}"


def table_rows(rows: Rows, ids: Sequence[str]) -> str:
    """``<tr>`` per id that has a row, best score first; the gain column is against the first id's score."""
    base = rows.get(ids[0], {}).get("score") if ids else None
    present = sorted((rows[i] for i in ids if i in rows), key=lambda r: -(r.get("score") or 0))
    lines = []
    for index, row in enumerate(present):
        score = row.get("score")
        gain = f"{score - base:+.3f}" if isinstance(score, (int, float)) and isinstance(base, (int, float)) else PENDING
        speed = "—" if no_model(row) else number("images_per_second", row.get("images_per_second"))
        lines.append(
            f'<tr{" class=best" if index == 0 else ""}><td class=mono>{row["id"]}</td>'
            f'<td class=mono>{html.escape(str(row.get("predictor", "")))}</td>'
            f'<td class=mono>{html.escape(str(row.get("model") or "—"))}</td>'
            f'<td class=num>{number("score", score)}</td><td class=num>{gain}</td><td class=num>{speed}</td></tr>'
        )
    return "\n".join(lines)


def fill(template: str, rows: Rows, best: str | None = None, method: str | None = None,
         label_errors: int | None = None) -> str:
    """The template with every placeholder replaced; see the module docstring for the forms."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "method":
            return html.escape(method) if method else PENDING
        if token == "label_errors":
            return number("count", label_errors)
        kind, _, rest = token.partition(" ")
        ids = [best or "" if i == "best" else i for i in rest.split()]
        if kind == "delta":
            a, b = (value(rows, ref, best) for ref in rest.split())
            numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
            return f"{a - b:+.3f}" if numeric else PENDING
        if kind == "raw":
            x = value(rows, rest.strip(), best)
            return f"{x:.4f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "0"
        if kind == "rows":
            return table_rows(rows, ids)
        if kind == "chart":
            chosen = [rows[i] for i in dict.fromkeys(ids) if i in rows]
            return chart_svg(chosen) if chosen else PENDING
        return number(token, value(rows, token, best))

    return PLACEHOLDER.sub(replace, template)


def print_pdf(page: Path, pdf: Path, chrome: str) -> None:
    """Headless Chrome prints ``page`` to ``pdf``; the web fonts get a few seconds to load."""
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
         f"--print-to-pdf={pdf}", page.as_uri()],
        check=True, capture_output=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--best", help="the EXP row of the final method (#137); its fields fill best.*")
    parser.add_argument("--method", help="the final method's name for the method slide (#137)")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--chrome", default=os.environ.get("CHROME", CHROME))
    parser.add_argument("--no-pdf", action="store_true", help="write deck.html only")
    args = parser.parse_args(argv)

    from galaxus.data.paths import out_dir

    rows = {row["id"]: row for row in read_rows(args.results) if "id" in row}
    if args.best and args.best not in rows:
        parser.error(f"--best {args.best} has no row in {args.results}")
    target = out_dir() / "deck"
    target.mkdir(exist_ok=True)
    page = target / "deck.html"
    errors = count_label_errors(out_dir() / "label_errors.csv")
    page.write_text(fill(TEMPLATE.read_text(encoding="utf-8"), rows, args.best, args.method, errors), encoding="utf-8")
    print(f"[deck] {page}")
    if not args.no_pdf:
        print_pdf(page, target / "deck.pdf", args.chrome)
        print(f"[deck] {target / 'deck.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
