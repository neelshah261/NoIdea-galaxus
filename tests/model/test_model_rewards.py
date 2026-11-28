"""#134: the GRPO reward and data side. Synthetic answers only; no torch, no TRL."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from galaxus.model import lora, rewards

CLASSES = {"subject": ["alpha", "beta"], "background": ["plain", "scene"]}
TARGET = {"subject": "beta", "background": "plain"}
WEIGHTS = {"subject": {"alpha": 0.8, "beta": 2.0}, "background": {"plain": 1.0, "scene": 1.5}}

PATH = Path(__file__).resolve().parents[2] / "train" / "grpo_apertus.py"
spec = importlib.util.spec_from_file_location("galaxus_train_grpo_apertus", PATH)
grpo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = grpo
spec.loader.exec_module(grpo)


def answer(subject: str, background: str = "plain") -> str:
    return json.dumps({"subject": subject, "background": background})


def test_invalid_answers_earn_nothing():
    for bad in ("not json", '{"subject": "beta"}', answer("gamma"), ""):
        assert rewards.answer_reward(bad, TARGET, ["subject", "background"], CLASSES) == 0.0


def test_each_right_trained_dimension_pays_its_class_weight():
    trained = ["subject", "background"]
    both = rewards.answer_reward(answer("beta"), TARGET, trained, CLASSES, WEIGHTS)
    rare_only = rewards.answer_reward(answer("beta", "scene"), TARGET, trained, CLASSES, WEIGHTS)
    common_only = rewards.answer_reward(answer("alpha"), TARGET, trained, CLASSES, WEIGHTS)
    wrong = rewards.answer_reward(answer("alpha", "scene"), TARGET, trained, CLASSES, WEIGHTS)
    assert both == pytest.approx(rewards.FORMAT_BONUS + (2.0 + 1.0) / 2)
    assert rare_only > common_only > wrong == pytest.approx(rewards.FORMAT_BONUS)  # the rare class pays more


def test_prelabelled_cells_pay_nothing_and_no_weights_means_one_each():
    only_background = rewards.answer_reward(answer("alpha"), TARGET, ["background"], CLASSES)
    assert only_background == pytest.approx(rewards.FORMAT_BONUS + 1.0)  # subject is wrong but not trained
    assert rewards.answer_reward(answer("alpha"), TARGET, [], CLASSES) == rewards.FORMAT_BONUS


def test_the_trl_reward_function_reads_chat_completions_and_the_dataset_columns():
    reward = rewards.reward_function(CLASSES, WEIGHTS)
    out = reward(prompts=["p", "p"], completions=[[{"role": "assistant", "content": answer("beta")}], "junk"],
                 target=[TARGET, TARGET], trained=[["subject"], ["subject"]], image=[None, None])
    assert out == [pytest.approx(rewards.FORMAT_BONUS + 2.0), 0.0]
    assert reward.__name__ == "human_label_reward"


def example(i: int, subject: str, mirrored: bool = False) -> lora.Example:
    return lora.Example(f"i{i}", f"/x/{i}.jpg", mirrored, None, {"subject": subject, "background": "plain"},
                        ("subject", "background"))


def test_picking_favours_rare_classes_takes_distinct_originals_and_is_seeded():
    examples = [example(i, "alpha") for i in range(900)] + [example(900 + i, "beta") for i in range(100)]
    examples += [example(0, "alpha", mirrored=True), example(1, "alpha")]  # a mirror and a repeat
    weights = lora.class_weights(examples)
    chosen = rewards.pick_examples(examples, 100, weights, seed=1)
    assert len(chosen) == len({e.image_id for e in chosen}) == 100 and not any(e.mirrored for e in chosen)
    assert sum(e.target["subject"] == "beta" for e in chosen) > 10  # 10% of the pool, more than 10% of the pick
    assert [e.image_id for e in rewards.pick_examples(examples, 100, weights, seed=1)] == [e.image_id for e in chosen]


def test_rows_carry_what_the_prompt_and_the_reward_need():
    row = rewards.grpo_rows([example(3, "beta")])[0]
    assert row == {"image_id": "i3", "image_file": "/x/3.jpg", "context": None,
                   "target": {"subject": "beta", "background": "plain"}, "trained": ["subject", "background"]}


def test_the_plan_is_slower_than_the_lora_by_the_generations_and_fits_the_budget():
    plan = rewards.GrpoPlan(1000, 4, 1.8)
    assert plan.images_per_second == pytest.approx(1.13 / 6)
    n = rewards.images_for_budget(15, 4, 1.8)
    assert rewards.GrpoPlan(n, 4, 1.8).dollars <= 15 < rewards.GrpoPlan(n + 1, 4, 1.8).dollars
    assert rewards.images_for_budget(0) == 0


def test_the_script_asks_for_a_budget_and_the_starting_adapter():
    parser = grpo.build_parser()
    assert parser.parse_args([]).generations == rewards.GENERATIONS
    with pytest.raises(SystemExit):
        grpo.main(["--plan"])
    with pytest.raises(SystemExit):
        grpo.main(["--budget", "15"])  # a real run needs --model and --adapter
