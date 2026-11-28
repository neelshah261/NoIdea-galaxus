"""#43 on top of #145 S1: FP8 and the base model in the serving job. Checks the scripts as text and with `bash -n`."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "bench"
JOB, START = BENCH / "hf_job_serve.sh", BENCH / "hf_job_serve_start.sh"


@pytest.mark.parametrize("script", [JOB, START])
def test_the_scripts_parse(script):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    subprocess.run([bash, "-n", str(script)], check=True)


def test_fp8_reaches_every_vllm_start_and_renames_the_served_model():
    text = JOB.read_text(encoding="utf-8")
    assert text.count('"${QUANT_ARGS[@]}"') == 3  # base, multi-adapter and merged
    assert "fp8) QUANT_ARGS=(--quantization fp8); SUFFIX=\"-fp8\"" in text
    assert 'SERVED="${SERVED:-apertus-8b$SUFFIX}"' in text and 'SERVED="${SERVED:-apertus-lora$SUFFIX}"' in text


def test_base_mode_serves_the_published_model_before_any_adapter_download():
    text = JOB.read_text(encoding="utf-8")
    assert text.index('if [[ "$SERVE_MODE" == "base" ]]') < text.index('log "1/3 download the adapter(s)"')


def test_the_start_script_passes_quant_and_no_kaggle_token():
    text = START.read_text(encoding="utf-8")
    assert '-e QUANT="$QUANT"' in text and "kaggle" not in text.lower()
    assert "ZERO_SHOT_SERVED_MODEL=apertus-8b" in text


def test_the_default_served_names_are_the_ones_the_predictors_ask_for():
    from galaxus.model.lora import DEFAULT_SERVED_NAME

    assert DEFAULT_SERVED_NAME == "apertus-lora"
