"""The final method's own weights as files to publish, with a model card (#176, Swarup's task 3). A person uploads.

challenge_description.md:86-90: the report names the commit and "the revision of the weights that produced your final
submission", published by 14:30, and the hosts may rerun a team's code from that commit and those weights. Apertus is
used unchanged, so for Apertus the commit is enough. What we trained ourselves is the heads: small softmax heads on
the embeddings of open image encoders. This module finds every fitted heads model inside the final predictor (also
inside the live ensemble's members), saves each with ``HeadsModel.save`` (plain numpy, loadable without pickle),
and writes a model card naming the encoders and their pinned revisions, the EXP row, and the commit.

    uv run --extra local --env-file .env python -m galaxus.serve.publish \\
        --predictor galaxus.model.ensemble:from_env --exp EXP-0061 --repo <hf user>/noidea-galaxus-heads

It writes ``$GALAXUS_DATA/out/publish/<EXP>/`` and prints the ``hf upload`` command; it never uploads (AGENTS.md
product rule 7). **Only weights leave:** the predictor's cached image embeddings are never written, and the files
are checked for anything that looks like an image id before the command is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: An image id of the competition data (``img_00033``): it must never appear in a published file.
IMAGE_ID = re.compile(r"img_\d{3,}")
MAX_DEPTH = 4


@dataclass(frozen=True)
class FoundHeads:
    """One fitted heads model inside the predictor, and what produced its features."""

    name: str
    model: Any  # galaxus.model.heads.HeadsModel
    encoder: str | None
    revision: str | None


def find_heads(predictor: Any, name: str = "heads", depth: int = 0) -> list[FoundHeads]:
    """Every ``heads_model`` in ``predictor``: itself, a live ensemble's ``predictors``, or a wrapper's ``inner``."""
    if predictor is None or depth > MAX_DEPTH:
        return []
    found = []
    model = getattr(predictor, "heads_model", None)
    if model is not None and hasattr(model, "save"):
        found.append(FoundHeads(name, model, getattr(predictor, "encoder_name", None),
                                getattr(predictor, "model_revision", None)))
    members = getattr(predictor, "predictors", None)
    if isinstance(members, Mapping):
        for member, inner in sorted(members.items()):
            found += find_heads(inner, str(member), depth + 1)
    inner = getattr(predictor, "inner", None)
    if inner is not None:
        found += find_heads(inner, name, depth + 1)
    return found


def check_no_data(folder: Path) -> list[str]:
    """Problems that would make a file unsafe to publish: anything that looks like an image id, in any file."""
    problems = []
    for path in sorted(folder.iterdir()):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    values = data[key]
                    if values.dtype.kind in "US" and any(IMAGE_ID.search(str(v)) for v in values.reshape(-1)):
                        problems.append(f"{path.name}: {key} holds an image id")
        elif IMAGE_ID.search(path.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"{path.name} holds an image id")
    return problems


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_card(exp: str, row: Mapping[str, Any] | None, commit: str, found: Sequence[FoundHeads],
               files: Mapping[str, str]) -> str:
    """The README.md of the published folder: what the files are, how to load them, and what is not in them."""
    score = row.get("score") if row else None
    lines = [
        "---", "license: cc-by-4.0", "tags: [hack-apertus, galaxus, apertus]", "---", "",
        f"# Team NoIdea: the heads of {exp} (Hack Apertus x Galaxus)", "",
        (f"The weights we trained for {exp}, the method behind our final Kaggle submission. Apertus v1.5 itself is "
         "used unchanged, so for Apertus the code commit is enough; these files are what is ours."), "",
        f"- **Code:** https://github.com/neelshah261/NoIdea-galaxus at commit `{commit}`",
        f"- **Holdout score of {exp}:** " + (f"{score:.4f} macro F1 (official holdout, metric.py; "
                                              "`results/metrics.json`)" if isinstance(score, (int, float))
                                              else "see `results/metrics.json`"),
        "", "| File | Member | Encoder of its features | Encoder revision | sha256 |", "| --- | --- | --- | --- | --- |",
    ]
    for item in found:
        file = f"{item.name}.heads.npz"
        lines.append(f"| `{file}` | {item.name} | `{item.encoder or 'see the code'}` | `{item.revision or '-'}` | "
                     f"`{files[file][:16]}` |")
    lines += [
        "", "## Load", "", "```python", "from galaxus.model.heads import HeadsModel",
        f"model = HeadsModel.load(\"{found[0].name}.heads.npz\")  # numpy only, no pickle" if found else "",
        "```", "",
        ("Each file holds, per dimension, a softmax head on standardised features: `weights`, `intercept`, `mean`, "
         "`scale`, the calibration `bias`, and the class names. The features are the encoder's pooled image "
         "embedding at the revision above (`galaxus.model.helper`)."), "",
        "## Not in these files", "",
        ("No image, image id, label or prediction from the competition data: only the fitted weights and the class "
         "names, which are the competition's public label set. Checked before publishing (`galaxus.serve.publish`)."),
        "",
    ]
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_folder(folder: Path, exp: str, row: Mapping[str, Any] | None, commit: str,
                 found: Sequence[FoundHeads]) -> dict[str, str]:
    """The heads files, manifest.json and README.md in ``folder``; returns ``{file: sha256}``."""
    folder.mkdir(parents=True, exist_ok=True)
    names = [item.name for item in found]
    if len(set(names)) != len(names):
        raise ValueError(f"two heads share a name: {names}")
    files = {}
    for item in found:
        path = item.model.save(folder / f"{item.name}.heads.npz")
        files[path.name] = sha256(path)
    manifest = {"exp": exp, "commit": commit, "files": files,
                "encoders": {item.name: {"encoder": item.encoder, "revision": item.revision} for item in found},
                "score": row.get("score") if row else None}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "README.md").write_text(model_card(exp, row, commit, found, files), encoding="utf-8")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.publish", description=__doc__.split("\n")[0])
    parser.add_argument("--predictor", required=True, help="package.module:name of the final method")
    parser.add_argument("--exp", required=True, help="its EXP id")
    parser.add_argument("--repo", default="<hf user>/noidea-galaxus-heads", help="the Hugging Face model repo")
    parser.add_argument("--results", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--out", type=Path, default=None, help="default: $GALAXUS_DATA/out/publish/<EXP>")
    arguments = parser.parse_args(argv)

    from galaxus.data.paths import out_dir
    from galaxus.serve.items import load_predictor

    predictor = load_predictor(arguments.predictor)
    found = find_heads(predictor, str(getattr(predictor, "name", None) or "heads"))
    if not found:
        print(f"{arguments.predictor} holds no fitted heads: nothing of ours to publish (the commit is enough)",
              file=sys.stderr)
        return 1
    rows = json.loads(arguments.results.read_text(encoding="utf-8")) if arguments.results.is_file() else []
    row = next((r for r in rows if r.get("id") == arguments.exp), None)
    folder = arguments.out or out_dir() / "publish" / arguments.exp
    commit = git_commit()
    files = write_folder(folder, arguments.exp, row, commit, found)
    problems = check_no_data(folder)
    if problems:
        print("NOT safe to publish: " + "; ".join(problems), file=sys.stderr)
        return 1
    for name, digest in files.items():
        print(f"{name}: {(folder / name).stat().st_size / 1024:.0f} KB, sha256 {digest[:16]}")
    print(f"\nwritten to {folder}; checked: no image id in any file.")
    print("\nA person runs this (agents never publish, AGENTS.md product rule 7):\n")
    print(f"  hf upload {arguments.repo} {folder} . --repo-type model --commit-message \"{arguments.exp} heads, "
          f"code {commit[:12]}\"")
    print("\nThen put the repo and the revision it prints into the report:"
          f"\n  python scripts/build_report.py --exp {arguments.exp} --weights \"{arguments.repo} @ <revision>\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
