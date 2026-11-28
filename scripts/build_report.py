"""Build the technical report: section 7 regenerated from results/metrics.json, then <Team>_Report.pdf (#54).

    just report                  # or: uv run --extra report python scripts/build_report.py [--exp EXP-0004]

1. Section 7 comes from ``galaxus.serve.report`` (numbers only from ``results/metrics.json`` and generated files,
   ``TODO`` for anything unmeasured) and replaces the text between the ``section7`` markers of technical_report.md.
   Nothing in that section is ever typed by hand (AGENTS.md product rule 1).
2. technical_report.md is rendered to HTML in the design/STYLE.md look and printed to PDF by a headless
   Chromium-family browser (Chrome, Edge, Brave or Chromium; ``REPORT_BROWSER`` overrides), to ``<Team>_Report.pdf``
   in the repository root, as README.md "Technical Report (pdf)" asks.

Without ``$GALAXUS_DATA`` the benchmark and label-error lines of section 7 stay TODO (an empty folder is used).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "technical_report.md"
START = re.compile(r"<!-- section7:start[^>]*-->")
END = "<!-- section7:end -->"
PAGE_LIMIT = 6  # README.md "Technical Report (pdf)": max. 6 pages w/o references

BROWSERS = (
    "chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge", "brave",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)

STYLE = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Geist', Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #141414; background: #ffffff; }
h1 { font-size: 19pt; letter-spacing: -0.03em; margin: 0 0 6pt; }
h1 + p { color: #55524c; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; padding-top: 4pt; border-top: 1.5px solid #141414; }
h3 { font-size: 10.5pt; margin: 9pt 0 3pt; color: #2e2b27; }
code { font-family: 'Geist Mono', 'Courier New', monospace; font-size: 8.6pt; color: #2e2b27; }
a { color: #c93c05; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt; font-size: 8.4pt; }
th { text-align: left; border-bottom: 1.5px solid #141414; padding: 2pt 4pt; }
td { border-bottom: 1px solid #dcd6cc; padding: 2pt 4pt; vertical-align: top; }
strong { color: #141414; }
ul { margin: 3pt 0 6pt; padding-left: 15pt; }
li { margin: 1pt 0; }
"""


def section7(team: str, exp: str | None, weights: str | None = None) -> str:
    """The generated section 7, links made relative to the repository root."""
    env = dict(os.environ)
    empty = None
    if not env.get("GALAXUS_DATA"):
        empty = tempfile.mkdtemp(prefix="galaxus-empty-")
        env["GALAXUS_DATA"] = empty
    command = [sys.executable, "-m", "galaxus.serve.report", "--team", team]
    if exp:
        command += ["--exp", exp]
    if weights:
        command += ["--weights", weights]
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", check=False)
    finally:
        if empty:
            shutil.rmtree(empty, ignore_errors=True)
    if result.returncode != 0:
        raise SystemExit(f"galaxus.serve.report failed:\n{result.stderr}")
    return result.stdout.strip().replace("](../", "](")


def splice(text: str, generated: str) -> str:
    match = START.search(text)
    if not match or END not in text[match.end():]:
        raise SystemExit("technical_report.md has no section7 start/end markers")
    end = text.index(END, match.end())
    return text[: match.end()] + "\n" + generated + "\n" + text[end:]


def find_browser() -> str | None:
    override = os.environ.get("REPORT_BROWSER")
    for candidate in ([override] if override else []) + list(BROWSERS):
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    return None


def to_html(markdown_text: str) -> str:
    try:
        import markdown
    except ImportError as missing:
        raise SystemExit("the PDF needs the 'report' extra: uv run --extra report python scripts/build_report.py") from missing
    body = markdown.markdown(re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL), extensions=["tables", "fenced_code"])
    fonts = "https://fonts.googleapis.com/css2?family=Geist:wght@400;600;700&family=Geist+Mono:wght@400&display=swap"
    return (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{fonts}">'
            f"<style>{STYLE}</style></head><body>{body}</body></html>")


def print_pdf(html: str, pdf: Path) -> int:
    browser = find_browser()
    if not browser:
        raise SystemExit("no Chrome, Edge, Brave or Chromium found; set REPORT_BROWSER to one")
    with tempfile.TemporaryDirectory() as folder:
        page = Path(folder) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
             f"--user-data-dir={Path(folder) / 'profile'}", "--virtual-time-budget=8000",
             f"--print-to-pdf={pdf}", page.as_uri()],
            check=True, capture_output=True,
        )
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="NoIdea", help="team name, also the PDF's file name")
    parser.add_argument("--exp", default=None, help="the EXP of the final submission (default: best holdout score)")
    parser.add_argument("--weights", default=None, help="the published weights: <Hugging Face repo> @ <revision>")
    parser.add_argument("--no-pdf", action="store_true", help="only regenerate section 7")
    args = parser.parse_args(argv)

    REPORT.write_text(splice(REPORT.read_text(encoding="utf-8"), section7(args.team, args.exp, args.weights)), encoding="utf-8")
    print(f"section 7 regenerated in {REPORT.name}")
    if args.no_pdf:
        return 0
    pdf = ROOT / f"{re.sub(r'[^A-Za-z0-9_-]', '', args.team)}_Report.pdf"
    pages = print_pdf(to_html(REPORT.read_text(encoding="utf-8")), pdf)
    print(f"{pdf.name}: {pages} pages (limit {PAGE_LIMIT} without references)")
    if pages > PAGE_LIMIT + 1:
        print("warning: over the page limit; shorten the report before the hand-in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
