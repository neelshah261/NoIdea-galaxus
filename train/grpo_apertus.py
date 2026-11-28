"""#134: GRPO from our LoRA (RL with verifiable rewards, the hosts' idea 4). Reward and data side: galaxus.model.rewards.

    python train/grpo_apertus.py --plan --budget 15                   # images, generations, estimated cost; CPU
    python train/grpo_apertus.py --smoke                              # TRL's GRPOTrainer on a tiny random model, CPU
    python train/grpo_apertus.py --model swiss-ai/Apertus-v1.5-8B --adapter <EXP-0004 adapter dir> --budget 15

The real run is a human's (AGENTS.md product rule 7: GPU money needs the owner's OK; this one is meant for the team's
$20 pool, #134 R4). It starts from the LoRA adapter, samples ``--generations`` answers per image with the LoRA's own
prompt and 512 px pixels, rewards each against the human labels (class-weighted, #135), and saves a new adapter that
``bench/hf_job_serve.sh`` serves and ``python -m galaxus.model.eval lora`` scores, exactly like EXP-0004.

**Not verified here** (no torch, no TRL on the machine this was written on): TRL's multimodal GRPO path with Apertus
1.5's processor. Run ``--smoke`` first, then the real run with ``--max-steps 5``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxus.model.lora import IMAGE_SIZE, build_training_set, class_weights, system_prompt, user_prompt
from galaxus.model.rewards import (
    GENERATIONS,
    GrpoPlan,
    grpo_rows,
    images_for_budget,
    pick_examples,
    reward_function,
)

LEARNING_RATE, BETA, SEED = 1e-6, 0.04, 3407
MAX_COMPLETION_TOKENS = 160   # the LoRA predictor's max_tokens: the JSON answer fits


def data(budget: float | None, images: int | None, generations: int, price: float):
    """Train minus the official holdout, the class weights, and the images GRPO uses (rare classes favoured)."""
    from galaxus.data import classes, load, split

    dimensions = tuple(classes.dimensions())
    rows, _holdout = split.split_holdout(load.read_train())
    examples = build_training_set(rows, dimensions).examples
    weights = class_weights(examples)
    count = images or images_for_budget(budget or 0, generations, price)
    chosen = pick_examples(examples, count, weights, seed=SEED)
    allowed = {d: sorted(v) for d, v in classes.allowed_classes().items()}
    return allowed, weights, chosen


def plan(arguments) -> int:
    _, weights, chosen = data(arguments.budget, arguments.images, arguments.generations, arguments.dollars_per_hour)
    p = GrpoPlan(len(chosen), arguments.generations, arguments.dollars_per_hour)
    print(f"{p.images} images (rare classes favoured), {p.generations} answers each")
    print(f"ESTIMATE on one L40S: {p.images_per_second:.3f} images/s, {p.hours:.1f} h, ${p.dollars:.2f} at "
          f"${p.dollars_per_hour}/h (from run 3's measured LoRA rate; the run prints the real one)")
    print(json.dumps({d: round(max(v.values()), 2) for d, v in weights.items()}, indent=2), "<- largest class weight")
    return 0


def train(arguments) -> int:
    import torch
    import transformers
    from datasets import Dataset
    from peft import PeftModel
    from PIL import Image as PillowImage
    from trl import GRPOConfig, GRPOTrainer

    from galaxus.data.images import cached_jpeg

    classes, weights, chosen = data(arguments.budget, arguments.images, arguments.generations,
                                    arguments.dollars_per_hour)
    system = system_prompt(classes)
    rows = []
    for row in grpo_rows(chosen):
        image = PillowImage.open(io.BytesIO(cached_jpeg(row["image_id"], row["image_file"], image_size=IMAGE_SIZE)))
        rows.append({"prompt": [{"role": "system", "content": system},
                                {"role": "user", "content": [{"type": "image"},
                                                             {"type": "text", "text": user_prompt(row["context"])}]}],
                     "image": image.convert("RGB"), "target": row["target"], "trained": row["trained"]})
    from galaxus.data.paths import out_dir

    out = Path(arguments.output or out_dir() / "grpo" / f"grpo-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    processor = transformers.AutoProcessor.from_pretrained(arguments.model)
    model_class = next(getattr(transformers, n) for n in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
                                                          "AutoModelForCausalLM") if hasattr(transformers, n))
    base = model_class.from_pretrained(arguments.model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, arguments.adapter, is_trainable=True)
    print(f"[grpo] {len(rows)} images x {arguments.generations} answers, from adapter {arguments.adapter}", flush=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function(classes, weights)],
        args=GRPOConfig(output_dir=str(out), learning_rate=LEARNING_RATE, beta=BETA, seed=SEED,
                        num_generations=arguments.generations, per_device_train_batch_size=arguments.generations,
                        gradient_accumulation_steps=4, max_completion_length=MAX_COMPLETION_TOKENS,
                        max_steps=arguments.max_steps or -1, num_train_epochs=1, logging_steps=5,
                        save_steps=50, save_total_limit=2, bf16=torch.cuda.is_available(), report_to="none",
                        remove_unused_columns=False),
        train_dataset=Dataset.from_list(rows),
        processing_class=processor,
    )
    trainer.train()
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    print(f"[grpo] adapter in {out / 'adapter'}: serve it like EXP-0004's. Stop the GPU if nothing else runs.")
    return 0


def smoke(steps: int = 3) -> int:
    """TRL's GRPOTrainer with our reward on a tiny random Llama and an in-memory character tokenizer (no download)."""
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    torch.manual_seed(SEED)
    vocab = {"<pad>": 0, "<eos>": 1, **{chr(c): i + 2 for i, c in enumerate(range(32, 127))}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    core.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", eos_token="<eos>")
    tokenizer.padding_side = "left"
    classes = {"subject": ["alpha", "beta"], "background": ["plain", "scene"]}
    rows = [{"prompt": f"image {i}: answer in JSON\n", "target": {"subject": "beta" if i % 3 == 0 else "alpha",
                                                                  "background": "plain"},
             "trained": ["subject", "background"]} for i in range(8)]
    weights = {"subject": {"alpha": 0.8, "beta": 1.6}, "background": {"plain": 1.0}}
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                         pad_token_id=0, eos_token_id=1)
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        trainer = GRPOTrainer(
            model=LlamaForCausalLM(config),
            reward_funcs=[reward_function(classes, weights)],
            args=GRPOConfig(output_dir=folder, num_generations=2, per_device_train_batch_size=2, max_steps=steps,
                            max_completion_length=24, logging_steps=1, report_to="none", use_cpu=True,
                            remove_unused_columns=False, save_strategy="no"),
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
        )
        trainer.train()
    print(f"[smoke] OK: GRPOTrainer ran {steps} steps with the human-label reward (a random model earns ~0; the point "
          "is that the path runs)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python train/grpo_apertus.py", description=__doc__.split("\n")[0])
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None, help="the LoRA adapter to start from (EXP-0004's)")
    parser.add_argument("--budget", type=float, default=None, help="dollars for the run; sets the image count")
    parser.add_argument("--images", type=int, default=None, help="the image count, instead of --budget")
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--dollars-per-hour", type=float, default=1.8)
    parser.add_argument("--max-steps", type=int, default=None, help="a short first run, e.g. 5")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        return smoke()
    if arguments.budget is None and arguments.images is None:
        parser.error("give --budget (dollars) or --images")
    if arguments.plan:
        return plan(arguments)
    if not arguments.model or not arguments.adapter:
        parser.error("--model and --adapter are needed for a real run (or use --plan / --smoke)")
    return train(arguments)


if __name__ == "__main__":
    sys.exit(main())
