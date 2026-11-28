"""#218: the feature job's launcher, run for real against a fake ``uvx`` on PATH, so nothing reaches the network.

macOS ships bash 3.2, where expanding an EMPTY array under ``set -u`` is an "unbound variable" error: every real
(non-dry-run) start failed on Neel's Mac until the arrays were expanded as ``${A[@]+"${A[@]}"}``. CI runs a newer
bash, so the guard is also pinned statically below.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
START = ROOT / "train" / "hf_job_apertus_features_start.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def launch(tmp_path, *arguments, extra_env=None):
    """Run the launcher with a fake uvx that prints the command it was given; returns (exit code, the arguments)."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "uvx").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    (fake / "uvx").chmod(0o755)
    env = {"PATH": f"{fake}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(tmp_path),
           "HF_TOKEN": "hf-test", "KAGGLE_API_TOKEN": "kaggle-test", "GH_TOKEN": "gh-test",  # secret-scan: allow
           **(extra_env or {})}
    done = subprocess.run(["bash", str(START), *arguments], env=env, capture_output=True, text=True, cwd=tmp_path)
    return done.returncode, done.stdout.splitlines(), done.stderr


def test_a_real_start_runs_with_no_dry_run_flag(tmp_path):
    code, argv, err = launch(tmp_path, "features")
    assert code == 0, err
    assert "jobs" in argv and "run" in argv and "--dry-run" not in argv
    assert "--expose" in argv and "8000" in argv
    for secret in ("HF_TOKEN", "KAGGLE_API_TOKEN", "GH_TOKEN"):
        assert argv[argv.index(secret) - 1] == "--secrets"  # by name: the value is not on the command line
    assert "hf-test" not in argv and "gh-test" not in argv
    assert "STAGE=features" in argv and "REF=main" in argv


def test_the_final_start_passes_the_passphrase_by_name_only(tmp_path):
    code, argv, err = launch(tmp_path, "final", extra_env={"PASSPHRASE": "not-the-real-one"})  # secret-scan: allow
    assert code == 0, err
    assert argv[argv.index("PASSPHRASE") - 1] == "--secrets" and "not-the-real-one" not in argv


def test_a_dry_run_still_says_so(tmp_path):
    code, argv, err = launch(tmp_path, "features", "--dry-run")
    assert code == 0, err
    assert "--dry-run" in argv


def test_no_array_is_expanded_bare_for_bash_3_2():
    """A bare "${A[@]}" fails on macOS's bash 3.2 when A is empty under set -u."""
    text = re.sub(r"(?m)#.*$", "", START.read_text())
    assert not re.search(r'(?<!\+)"\$\{[A-Za-z_]+\[@\]\}"', text)
    assert "${#DRY[@]}" not in text
