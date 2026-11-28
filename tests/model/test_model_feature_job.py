"""#218 (umbrella #214): the GPU feature job's scripts, and the fp32 vision tokenizer of the local backend.

No model, no torch, no network: the scripts are read and checked with ``bash -n``; the backend gets a stand-in.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from galaxus.model.apertus.local import keep_vision_tokenizer_fp32

ROOT = Path(__file__).resolve().parents[2]


class Module:
    def __init__(self):
        self.floated = False

    def float(self):
        self.floated = True
        return self


def test_the_vision_tokenizer_is_kept_in_float32():
    tokenizer, layer = Module(), Module()
    network = SimpleNamespace(named_modules=lambda: [("model.vision_tokenizer", tokenizer), ("model.layers.0", layer)])
    assert keep_vision_tokenizer_fp32(network) == 1
    assert tokenizer.floated and not layer.floated
    assert keep_vision_tokenizer_fp32(SimpleNamespace()) == 0  # a stand-in without named_modules



# ---------------------------------------------------------------- the GPU feature job's scripts


JOB = ROOT / "train" / "hf_job_apertus_features.sh"
START = ROOT / "train" / "hf_job_apertus_features_start.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", [JOB, START], ids=["job", "start"])
def test_the_job_scripts_are_valid_bash(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_the_job_passes_layers_with_an_equals_sign():
    """argparse reads ``--layers -1,-9`` as two options and fails; ``--layers=-1,-9`` works."""
    text = JOB.read_text()
    assert '"--layers=$LAYERS"' in text
    assert not re.search(r"--layers\s+-", text)


def test_the_features_leave_the_job_only_through_its_protected_port():
    job, start = JOB.read_text(), START.read_text()
    assert "http.server 8000" in job and "--expose 8000" in start
    assert "hf upload" not in job and "upload" not in re.sub(r"#.*", "", job)  # no repo, no dataset
    for secret in ("HF_TOKEN", "KAGGLE_API_TOKEN", "PASSPHRASE", "GH_TOKEN"):
        assert f"--secrets {secret}" in start  # by name only: the values never enter the command line
    assert "echo \"$PASSPHRASE" not in job and "print(os.environ[\"PASSPHRASE\"]" not in job


def test_the_private_repo_is_cloned_with_the_token_in_a_header_never_in_a_url():
    """#215 review: the repo is private until the hand-in, so an anonymous clone fails in the job's first minute."""
    job, start = JOB.read_text(), START.read_text()
    assert "http.https://github.com/.extraHeader=Authorization: Basic" in job
    assert "GIT_TERMINAL_PROMPT=0" in job
    assert not re.search(r"https://[^\s\"']*GH_TOKEN", job)  # never user:token@github.com
    assert "ls-remote --exit-code" in start  # without GH_TOKEN, an anonymous check stops before any paid time


def test_the_image_cache_step_skips_a_broken_file_instead_of_stopping_the_job():
    """#217: one broken final image must not stop the job before Apertus extracts anything."""
    assert "python -m galaxus.data.images --part \"$part\" --workers 8 --skip-missing" in JOB.read_text()
