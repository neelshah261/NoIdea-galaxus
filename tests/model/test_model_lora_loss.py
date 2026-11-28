"""#135: class weights and focal loss for the LoRA (the hosts' idea 5, #17). Synthetic examples only.

The arithmetic is tested with numpy, so ci needs no torch; the torch side (collator, Trainer loss) is skipped where
torch is not installed.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from galaxus.model import lora

DIMS = ("subject", "background", "perspective")

PATH = Path(__file__).resolve().parents[2] / "train" / "lora_apertus.py"
spec = importlib.util.spec_from_file_location("galaxus_train_lora_apertus_loss", PATH)
la = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = la
spec.loader.exec_module(la)


def char_offsets(text):
    ids = [b + 1 for b in text.encode("utf-8")]
    return ids, [(i, i + 1) for i in range(len(ids))]


def example(subject: str, trained=DIMS, i: int = 0) -> lora.Example:
    return lora.Example(f"i{i}", "", False, None, {"subject": subject, "background": "plain", "perspective": "front"},
                        tuple(trained))


def imbalanced(common: int = 90, rare: int = 10) -> list[lora.Example]:
    return [example("common", i=i) for i in range(common)] + [example("rare", i=common + i) for i in range(rare)]


# ---------------------------------------------------------------- class weights


def test_rare_classes_weigh_more_and_the_average_weight_stays_one():
    examples = imbalanced()
    weights = lora.class_weights(examples)["subject"]
    assert weights["rare"] > 1 > weights["common"]
    assert weights["rare"] / weights["common"] == pytest.approx(math.sqrt(90 / 10))  # power 0.5
    assert (90 * weights["common"] + 10 * weights["rare"]) / 100 == pytest.approx(1.0)


def test_power_zero_is_no_weighting_and_a_single_class_weighs_one():
    weights = lora.class_weights(imbalanced(), power=0.0)
    assert weights["subject"] == pytest.approx({"common": 1.0, "rare": 1.0})
    assert lora.class_weights(imbalanced())["background"] == pytest.approx({"plain": 1.0})


def test_only_trained_cells_are_counted():
    examples = imbalanced() + [example("rare", trained=("background",), i=500 + i) for i in range(80)]
    weights = lora.class_weights(examples)["subject"]  # the 80 prelabelled 'rare' cells must not count
    assert weights["rare"] / weights["common"] == pytest.approx(3.0)


def test_loss_options_default_to_the_plain_loss_and_check_the_flags():
    assert lora.loss_options(imbalanced()).plain
    assert lora.loss_options(imbalanced()).describe() == "plain cross-entropy"
    tempered = lora.loss_options(imbalanced(), "tempered", 2.0)
    assert not tempered.plain and tempered.class_weights["subject"]["rare"] > 1
    assert "focal gamma 2" in tempered.describe()
    assert not lora.loss_options(imbalanced(), "none", 1.0).plain
    with pytest.raises(ValueError, match="not one of"):
        lora.loss_options(imbalanced(), "balanced")
    with pytest.raises(ValueError, match=">= 0"):
        lora.loss_options(imbalanced(), "none", -1.0)


# ---------------------------------------------------------------- per-token weights


def test_class_value_tokens_carry_the_class_weight_and_nothing_else_changes():
    target = example("rare", trained=("subject", "background"))
    ids = char_offsets("<img> prompt\n" + target.answer)[0]
    weights = {"subject": {"rare": 3.0}, "background": {"plain": 0.5}, "perspective": {"front": 7.0}}
    out = lora.loss_weights(ids, target, char_offsets, weights)
    _, positions = lora.value_positions(ids, target.answer, char_offsets)
    assert len(out["loss_weights"]) == len(out["value_mask"]) == len(ids)
    assert all(out["loss_weights"][p] == 3.0 and out["value_mask"][p] == 1 for p in positions["subject"])
    assert all(out["loss_weights"][p] == 0.5 for p in positions["background"])
    # the prelabelled perspective keeps weight 1 and no focal mark (its labels are -100 anyway)
    assert all(out["loss_weights"][p] == 1.0 and out["value_mask"][p] == 0 for p in positions["perspective"])
    marked = {p for d in ("subject", "background") for p in positions[d]}
    assert all(w == 1.0 for i, w in enumerate(out["loss_weights"]) if i not in marked)
    assert sum(out["value_mask"]) == len(marked)


def test_without_class_weights_every_weight_is_one_but_the_values_are_still_marked():
    target = example("rare")
    ids = char_offsets("p\n" + target.answer)[0]
    out = lora.loss_weights(ids, target, char_offsets, None)
    assert set(out["loss_weights"]) == {1.0} and sum(out["value_mask"]) == len("rare") + len("plain") + len("front")


# ---------------------------------------------------------------- the loss arithmetic (numpy here, torch in training)


def test_gamma_zero_and_unit_weights_give_the_plain_mean():
    token_loss = np.array([[0.0, 0.7, 1.2, 0.3, 2.0]])
    valid = np.array([[0.0, 1.0, 1.0, 1.0, 1.0]])
    total, count = lora.combine_token_losses(token_loss, valid, np.ones_like(token_loss),
                                             np.array([[0, 0, 1, 1, 0]]), 0.0, np)
    assert total / count == pytest.approx(token_loss[0, 1:].mean())


def test_a_weighted_loss_puts_more_weight_on_the_rare_class_token():
    token_loss = np.array([[1.0, 1.0]])  # the same error on a common and a rare class token
    valid, mask = np.ones((1, 2)), np.ones((1, 2))
    plain, _ = lora.combine_token_losses(token_loss, valid, np.ones((1, 2)), mask, 0.0, np)
    weighted, _ = lora.combine_token_losses(token_loss, valid, np.array([[0.5, 3.0]]), mask, 0.0, np)
    assert weighted > plain and weighted == pytest.approx(0.5 + 3.0)


def test_focal_loss_shrinks_confident_tokens_on_class_values_only():
    p = np.array([[0.9, 0.9, 0.2]])
    token_loss = -np.log(p)
    valid, weights = np.ones((1, 3)), np.ones((1, 3))
    mask = np.array([[1, 0, 1]])  # the second token is JSON format, not a class value
    total, _ = lora.combine_token_losses(token_loss, valid, weights, mask, 2.0, np)
    expected = (0.1**2) * token_loss[0, 0] + token_loss[0, 1] + (0.8**2) * token_loss[0, 2]
    assert total == pytest.approx(expected)


def test_masked_positions_carry_no_loss_whatever_their_weight():
    total, count = lora.combine_token_losses(np.array([[5.0, 1.0]]), np.array([[0.0, 1.0]]), np.array([[9.0, 1.0]]),
                                             np.array([[1, 1]]), 2.0, np)
    assert count == 1 and total == pytest.approx((1 - math.exp(-1.0)) ** 2 * 1.0)


# ---------------------------------------------------------------- the script's flags


def test_the_flags_default_off_so_the_recipe_is_unchanged():
    defaults = la.build_parser().parse_args([])
    assert defaults.class_weighting == "none" and defaults.focal_gamma == 0.0
    chosen = la.build_parser().parse_args(["--class-weighting", "tempered", "--focal-gamma", "2"])
    assert chosen.class_weighting == "tempered" and chosen.focal_gamma == 2.0
    with pytest.raises(SystemExit):
        la.build_parser().parse_args(["--class-weighting", "balanced"])


def test_a_negative_gamma_is_refused():
    with pytest.raises(SystemExit):
        la.main(["--smoke", "--focal-gamma", "-1"])


# ---------------------------------------------------------------- torch side (skipped without torch)


def test_the_collator_pads_the_weights_with_zero():
    torch = pytest.importorskip("torch")
    batch = la.make_collator(0)([
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "loss_weights": [1.0, 2.0, 3.0], "value_mask": [0, 1, 1]},
        {"input_ids": [5], "labels": [-100], "loss_weights": [1.0], "value_mask": [0]},
    ])
    assert batch["loss_weights"].dtype == torch.float32 and batch["value_mask"].dtype == torch.long
    assert batch["loss_weights"][1].tolist() == [1.0, 0.0, 0.0] and batch["value_mask"][1].tolist() == [0, 0, 0]


def test_the_weighted_loss_with_both_off_equals_the_model_loss():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(config).eval()
    ids = torch.randint(1, 50, (2, 12))
    labels = ids.clone()
    labels[:, :5] = -100
    with torch.no_grad():
        outputs = model(input_ids=ids, labels=labels)
        total, count = la.weighted_lm_loss(outputs.logits, labels, torch.ones_like(ids, dtype=torch.float32),
                                           torch.zeros_like(ids), 0.0)
    assert torch.allclose(total / count, outputs.loss, atol=1e-5)
