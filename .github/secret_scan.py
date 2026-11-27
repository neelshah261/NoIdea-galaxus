"""Secret scan for ci: no key, token, credential file or passphrase enters git (AGENTS.md product rule 3).

    python3 .github/secret_scan.py --diff origin/main...HEAD   # pull request: added lines and added files
    python3 .github/secret_scan.py --tree                      # push to main: every tracked file

A hit prints ``path:line: rule`` and never the matched value, so the ci log cannot leak it a second time. Exit code 1
on any hit. A deliberately fake value in a test can carry the marker ``secret-scan: allow`` on its line.
Standard library only, so it runs before ``uv sync``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ALLOW_MARKER = "secret-scan: allow"

#: Content rules: (name, pattern). Patterns look for the shape of a real credential, not for the word "key".
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Kaggle credentials", re.compile(r'"username"\s*:\s*"[^"]+"\s*,\s*"key"\s*:\s*"[0-9a-f]{32}"')),
    (
        "key in an env line",
        re.compile(
            r"^\s*(?:export\s+)?(?:LLM_API_KEY(?:_\d+)?|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|KAGGLE_KEY|OPENAI_API_KEY)"
            r"\s*=\s*['\"]?[^\s'\"<>$]{8,}"
        ),
    ),
    (
        "secret assigned in code",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passphrase)\b['\"]?\s*[:=]\s*"
            r"['\"][A-Za-z0-9_\-./+=]{16,}['\"]"
        ),
    ),
)

#: File rules: credential files have no business in the repository, whatever they contain.
CREDENTIAL_FILE = re.compile(
    r"(^|/)(\.env(\.(?!example$)[^/]+)?|kaggle\.json|credentials[^/]*\.json|[^/]+\.(pem|key|p12|pfx)"
    r"|id_(rsa|dsa|ecdsa|ed25519)[^/]*|\.netrc|\.pgpass)$"
)


def scan_line(line: str) -> list[str]:
    """The names of the rules a line breaks; empty when it is clean or carries the allow marker."""
    if ALLOW_MARKER in line:
        return []
    return [name for name, pattern in RULES if pattern.search(line)]


def scan_lines(path: str, numbered: Iterable[tuple[int, str]]) -> Iterator[str]:
    for number, line in numbered:
        for rule in scan_line(line):
            yield f"{path}:{number}: {rule}"


def scan_path_name(path: str) -> list[str]:
    return [f"{path}: credential file"] if CREDENTIAL_FILE.search(path.replace("\\", "/")) else []


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def _readable_text(path: Path) -> list[str] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def scan_tree(root: Path) -> list[str]:
    hits: list[str] = []
    for name in _git("-C", str(root), "ls-files").splitlines():
        hits += scan_path_name(name)
        lines = _readable_text(root / name)
        if lines is not None:
            hits += scan_lines(name, enumerate(lines, 1))
    return hits


def added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """``(path, line number, text)`` for every added line of a ``git diff --unified=0`` output."""
    path, number = "", 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            number = int(match.group(1)) if match else 0
        elif line.startswith("+") and path:
            yield path, number, line[1:]
            number += 1


def scan_diff(revision_range: str) -> list[str]:
    hits: list[str] = []
    for name in _git("diff", "--name-only", "--diff-filter=AR", revision_range).splitlines():
        hits += scan_path_name(name)
    diff = _git("diff", "--unified=0", "--no-color", "--no-ext-diff", revision_range)
    for path, number, text in added_lines(diff):
        hits += [f"{path}:{number}: {rule}" for rule in scan_line(text)]
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diff", metavar="RANGE", help="scan what a git range adds, for example origin/main...HEAD")
    mode.add_argument("--tree", action="store_true", help="scan every tracked file")
    args = parser.parse_args(argv)
    hits = scan_diff(args.diff) if args.diff else scan_tree(Path.cwd())
    for hit in hits:
        print(f"::error::possible secret: {hit}")
    if hits:
        print(f"{len(hits)} possible secret(s). Remove them, rotate any real key, and never paste keys in issues.")
        return 1
    print("secret scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
