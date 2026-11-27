"""One-command GPU session: order, resume, gpu-only stop, cost summary. Nothing is executed for real."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "bench" / "gpu_session.py"
spec = importlib.util.spec_from_file_location("galaxus_bench_gpu_session", MODULE_PATH)
session = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = session
spec.loader.exec_module(session)


def steps():
    return session.plan("EXP-0042", "/data/out/heads/heads-x.npz", 100, 50)


def test_plan_order_gpu_steps_and_heads_file():
    plan = steps()
    assert [s.name for s in plan] == ["cache", "extract-train", "extract-dev", "fit", "dev-file", "label-errors"]
    assert [s.name for s in plan if s.gpu] == list(session.GPU_STEPS)
    dev_file = next(s for s in plan if s.name == "dev-file")
    assert dev_file.env == {"HEADS_MODEL": "/data/out/heads/heads-x.npz"} and "EXP-0042" in dev_file.command
    assert "submit" in " ".join(dev_file.command) and "kaggle" not in " ".join(dev_file.command)


def test_runs_in_order_and_stops_at_a_failure():
    ran = []
    results = session.run_session(steps(), execute=lambda s: ran.append(s.name) or (2 if s.name == "fit" else 0),
                                  log=lambda m: None)
    assert ran == ["cache", "extract-train", "extract-dev", "fit"] and results[-1].returncode == 2


def test_resume_from_a_step_and_gpu_only_stop():
    ran = []
    session.run_session(steps(), start="extract-dev", execute=lambda s: ran.append(s.name) or 0, log=lambda m: None)
    assert ran == ["extract-dev", "fit", "dev-file", "label-errors"]
    ran.clear()
    messages = []
    session.run_session(steps(), gpu_only=True, execute=lambda s: ran.append(s.name) or 0, log=messages.append)
    assert ran == ["cache", "extract-train", "extract-dev"] and "STOP THE GPU" in messages[-1]
    with pytest.raises(ValueError, match="unknown step"):
        session.run_session(steps(), start="nope", execute=lambda s: 0)


def test_summary_speed_and_cost():
    results = [
        session.StepResult("cache", 60, 0),
        session.StepResult("extract-train", 1000, 0, 5000),
        session.StepResult("extract-dev", 500, 0, 2500),
        session.StepResult("fit", 40, 0),
    ]
    report = session.summary(results, dollars_per_hour=1.8)
    assert report["apertus_images_per_second"] == 5.0
    assert report["estimated_cost_usd"] == round(1.8 * 1600 / 3600, 2)
    assert session.summary(results, None)["estimated_cost_usd"] is None


def test_preconditions_name_what_is_missing_without_reading_the_token():
    env = {"GALAXUS_DATA": "/data", "HF_TOKEN": ""}
    problems = session.preconditions(env.get)
    assert any("HF_TOKEN" in p for p in problems)
    assert not any("GALAXUS_DATA" in p for p in problems)
