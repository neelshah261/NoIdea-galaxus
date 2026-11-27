"""#49: LoRA on Apertus v1.5 8B (EXP-0004). The data side is ``galaxus.model.lora`` (tested in ci, no torch).

Three modes:

    python train/lora_apertus.py --plan --dollars-per-hour 1.8      # counts, steps, time and cost estimate; CPU
    python train/lora_apertus.py --smoke                            # the same masking + LoRA + Trainer path on a
                                                                     # tiny random model, CPU, a minute
    python train/lora_apertus.py --model swiss-ai/Apertus-v1.5-8B    # the real run: GPU, Neel's OK first

Real run, on the GPU machine (AGENTS.md product rule 7: a human starts it, with the owner's OK and budget):

    export HF_TOKEN=...   GALAXUS_DATA=...        # gated model; data outside the repo
    python -m galaxus.data.prep --part train --size 512 --workers 8 && \
    python -m galaxus.data.prep --part train --size 512 --workers 8 --mirror
    uv run --with peft --with "transformers>=5.24" --with accelerate --with bitsandbytes \
        python train/lora_apertus.py --model swiss-ai/Apertus-v1.5-8B --epochs 2 --dollars-per-hour <price> --merge

After about 50 optimizer steps it prints the **measured** examples/s and the projected total time and cost; stop it
(Ctrl+C) if the projection is over budget. Adapters are saved every 200 steps and at the end under
``$GALAXUS_DATA/out/lora/<run>/``; ``--resume`` continues from the last checkpoint. ``--merge`` writes merged bf16
weights for vLLM (``MODEL=<dir> bash bench/vllm_apertus.sh``).

**Not verified here** (no GPU, no access to the gated model): that ``AutoProcessor`` of Apertus 1.5 accepts the chat
messages below and returns tensors the model's ``forward`` takes (Q-0006: the transformers version). The first run
should be ``--max-steps 5`` to check that, and the printed vision-token count per image (Q-0002).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxus.model.lora import (
    IMAGE_SIZE,
    MAX_REPEATS,
    RARE_CLASS_TARGET,
    Example,
    Plan,
    build_training_set,
    system_prompt,
    tokens_per_example,
    training_labels,
    user_prompt,
)

#: Recipe (reference notebook, deep-dive slide 18): LoRA r32/a32 on all linear layers, lr 2e-4, cosine, 5% warmup,
#: effective batch 16, 2-4 epochs (their holdout loss was flat after about 2).
RANK, ALPHA, LEARNING_RATE, WARMUP, WEIGHT_DECAY, EPOCHS, SEED = 32, 32, 2e-4, 0.05, 0.01, 2, 3407
#: One image per step: images of other aspect ratios give pixel tensors of other shapes; 1 x 16 = the recipe's 16.
BATCH, ACCUMULATION = 1, 16
#: Language-model linear layers of Apertus (attention + the xIELU MLP). The vision tokenizer is never adapted.
TARGET_MODULES = r".*(?<!vision_tokenizer)\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$"
#: Reference throughput: the hosts' Qwen 3.5 9B recipe trained 3 epochs in 73 min on one H100 with ~700 tokens per
#: example (256 image tokens). An L40S is taken as 2.5x slower. Replaced by the measured rate after 50 steps.
REFERENCE_TOKENS_PER_SECOND_L40S = 14_000 * 700 * 3 / (73 * 60) / 2.5


def training_set():
    """Train minus the official holdout, with the data lane's mirrored rows (#29)."""
    from galaxus.data import classes, load, mirror, split

    dimensions = tuple(classes.dimensions())
    rows, _holdout = split.split_holdout(load.read_train())
    mirrored = mirror.mirrored_rows(rows)
    return dimensions, build_training_set(rows, dimensions, mirrored), classes.allowed_classes()


def plan(arguments) -> int:
    _, data, _ = training_set()
    p = Plan(data.stats["examples_per_epoch"], arguments.epochs, BATCH, ACCUMULATION, tokens_per_example(),
             REFERENCE_TOKENS_PER_SECOND_L40S, arguments.dollars_per_hour or 1.8)
    print(json.dumps(data.stats, indent=2))
    print(f"\nexamples per epoch {p.examples_per_epoch} x {p.epochs} epochs, {p.steps} optimizer steps of "
          f"{BATCH}x{ACCUMULATION}, ~{p.tokens_per_example} tokens per example (512 px at 16 px per token + prompt)")
    print(f"ESTIMATE on one L40S: {p.hours:.1f} h, ${p.dollars:.2f} at ${p.dollars_per_hour}/h "
          "(from the hosts' Qwen timing scaled by tokens; the run prints the measured rate after 50 steps)")
    print(f"recipe: LoRA r{RANK}/a{ALPHA}, lr {LEARNING_RATE}, cosine, warmup {WARMUP}, rare classes towards "
          f"{RARE_CLASS_TARGET} (max {MAX_REPEATS}x), mirrored rows, human labels only, holdout left out")
    return 0


# ---------------------------------------------------------------- torch side (imported lazily)


def make_collator(pad_id: int):
    import torch

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        longest = max(len(f["input_ids"]) for f in features)
        batch: dict[str, Any] = {
            "input_ids": torch.tensor([f["input_ids"] + [pad_id] * (longest - len(f["input_ids"])) for f in features]),
            "labels": torch.tensor([f["labels"] + [-100] * (longest - len(f["labels"])) for f in features]),
            "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (longest - len(f["input_ids"]))
                                            for f in features]),
        }
        extra = [k for k in features[0] if k not in ("input_ids", "labels", "attention_mask")]
        for key in extra:  # pixel / image-token tensors from the processor; batch 1 keeps shapes simple
            values = [f[key] for f in features]
            batch[key] = torch.cat(values) if hasattr(values[0], "shape") else values
        return batch

    return collate


class LazyDataset:
    """Encodes one example when the Trainer asks for it, so 30k images never sit in memory."""

    def __init__(self, examples: list[Example], encode):
        self.examples, self.encode = examples, encode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.encode(self.examples[index])


def apertus_encoder(processor, classes):
    """Example -> model inputs + labels with Apertus' processor. Unverified until the first GPU run (see top)."""
    from PIL import Image as PillowImage

    from galaxus.data.images import cached_jpeg

    system = system_prompt(classes)
    tokenizer = getattr(processor, "tokenizer", processor)

    def with_offsets(text: str):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(encoded["input_ids"]), [tuple(o) for o in encoded["offset_mapping"]]

    def encode(example: Example) -> dict[str, Any]:
        image = PillowImage.open(io.BytesIO(cached_jpeg(example.image_id, example.image_file, mirrored=example.mirrored,
                                                        image_size=IMAGE_SIZE))).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": user_prompt(example.context)}]},
            {"role": "assistant", "content": [{"type": "text", "text": example.answer}]},
        ]
        inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        features = {"input_ids": ids, "labels": training_labels(ids, example.answer, example.trained_dimensions,
                                                                with_offsets)}
        for key, value in inputs.items():
            if key not in ("input_ids", "attention_mask"):
                features[key] = value
        return features

    return encode


class ProjectionCallback:
    """After 50 optimizer steps: the measured examples/s and the projected total time and cost."""

    def __init__(self, total_examples: int, dollars_per_hour: float | None, at_step: int = 50):
        self.total, self.price, self.at = total_examples, dollars_per_hour, at_step
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.at and self.started:
            seconds = time.perf_counter() - self.started
            done = state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
            rate = done / seconds
            hours = self.total / rate / 3600
            cost = f", ${hours * self.price:.2f} at ${self.price}/h" if self.price else ""
            print(f"[lora] measured {rate:.2f} examples/s -> about {hours:.1f} h for the whole run{cost}. "
                  "Ctrl+C now if that is over budget; the adapters so far are saved at the next checkpoint.",
                  flush=True)


def trainer_callbacks(total: int, price: float | None):
    from transformers import TrainerCallback

    class _Projection(ProjectionCallback, TrainerCallback):
        pass

    return [_Projection(total, price)]


def lora_wrap(model, rank: int, alpha: int, target: str | list[str]):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=target,
                        task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def keep_vision_tokenizer_fp32(model) -> int:
    """The vision tokenizer must stay float32 and frozen (vllm#50496 review; transformers docs)."""
    count = 0
    for name, module in model.named_modules():
        if name.endswith(("vision_tokenizer", "image_tokenizer")):
            module.float()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            count += 1
    return count


def train(arguments) -> int:
    import torch
    import transformers
    from transformers import Trainer, TrainingArguments

    _dimensions, data, classes = training_set()
    out = Path(arguments.output or _out_dir() / f"lora-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    print(f"[lora] {data.stats['examples_per_epoch']} examples per epoch; output {out}; transformers "
          f"{transformers.__version__}, torch {torch.__version__}, cuda {torch.cuda.is_available()}", flush=True)
    processor = transformers.AutoProcessor.from_pretrained(arguments.model)
    model_class = next(getattr(transformers, n) for n in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
                                                          "AutoModelForCausalLM") if hasattr(transformers, n))
    model = model_class.from_pretrained(arguments.model, dtype=torch.bfloat16)
    print(f"[lora] {model_class.__name__}; vision tokenizers kept fp32: {keep_vision_tokenizer_fp32(model)}")
    if arguments.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = lora_wrap(model, arguments.rank, arguments.alpha, TARGET_MODULES)
    model.print_trainable_parameters()
    encode = apertus_encoder(processor, classes)
    first = encode(data.examples[0])
    print(f"[lora] first example: {len(first['input_ids'])} tokens, "
          f"{sum(1 for x in first['labels'] if x != -100)} in the loss", flush=True)
    pad = getattr(getattr(processor, "tokenizer", processor), "pad_token_id", None) or 0
    total = data.stats["examples_per_epoch"] * arguments.epochs
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out), per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUMULATION,
            num_train_epochs=arguments.epochs, max_steps=arguments.max_steps or -1, learning_rate=arguments.lr,
            lr_scheduler_type="cosine", warmup_ratio=WARMUP, weight_decay=WEIGHT_DECAY, bf16=torch.cuda.is_available(),
            logging_steps=10, save_steps=200, save_total_limit=2, seed=SEED, report_to="none",
            remove_unused_columns=False, dataloader_num_workers=4,
            optim="adamw_bnb_8bit" if _has("bitsandbytes") else "adamw_torch",
        ),
        train_dataset=LazyDataset(data.examples, encode),
        data_collator=make_collator(pad),
        callbacks=trainer_callbacks(total, arguments.dollars_per_hour),
    )
    trainer.train(resume_from_checkpoint=True if arguments.resume else None)
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    (out / "run.json").write_text(json.dumps({"model": arguments.model, "epochs": arguments.epochs,
                                              "stats": data.stats, "transformers": transformers.__version__}, indent=2))
    if arguments.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged")
        processor.save_pretrained(out / "merged")
        print(f"[lora] merged weights in {out / 'merged'}: serve with MODEL={out / 'merged'} bash bench/vllm_apertus.sh")
    print(f"[lora] adapters in {out / 'adapter'}. Stop the GPU if nothing else runs.")
    return 0


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _out_dir() -> Path:
    from galaxus.data.paths import out_dir

    return out_dir() / "lora"


# ---------------------------------------------------------------- smoke: the same path on a tiny model, CPU


class CharTokenizer:
    """A byte-level stand-in tokenizer with offsets, so the smoke run needs no download."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids = [b + 1 for b in text.encode("utf-8")]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]  # ASCII-only answers
        return result


def smoke_examples(dimensions: tuple[str, ...]) -> list[Example]:
    examples = []
    for i in range(24):
        target = {d: ("alpha" if i % 2 else "beta") for d in dimensions}
        trained = dimensions if i % 3 else dimensions[:3]  # some prelabels masked
        examples.append(Example(f"s{i}", "", False, {"product_title": f"thing {i % 2}"}, target, trained))
    return examples


def smoke(steps: int = 30) -> int:
    """Masking + LoRA + Trainer on a tiny random Llama (no download): loss must fall, prelabels must be masked."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    torch.manual_seed(SEED)
    dimensions = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
    tokenizer = CharTokenizer()

    def with_offsets(text):
        encoded = tokenizer(text, return_offsets_mapping=True)
        return encoded["input_ids"], encoded["offset_mapping"]

    def encode(example: Example) -> dict[str, Any]:
        prompt = f"<img> {user_prompt(example.context)}\n"
        ids = tokenizer(prompt + example.answer)["input_ids"]
        return {"input_ids": ids, "labels": training_labels(ids, example.answer, example.trained_dimensions, with_offsets)}

    examples = smoke_examples(dimensions)
    masked = encode(examples[0])  # i=0: only the first three dimensions are human
    value_pos = [p for d in dimensions[3:] for p in __import__("galaxus.model.lora", fromlist=["x"]).value_positions(
        masked["input_ids"], examples[0].answer, with_offsets)[1][d]]
    assert all(masked["labels"][p] == -100 for p in value_pos), "a prelabel value is in the loss"
    config = LlamaConfig(vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024)
    model = lora_wrap(LlamaForCausalLM(config), 8, 16, ["q_proj", "v_proj", "up_proj", "down_proj"])
    losses: list[float] = []

    from transformers import TrainerCallback

    class Record(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                losses.append(logs["loss"])

    with __import__("tempfile").TemporaryDirectory() as folder:
        Trainer(
            model=model,
            args=TrainingArguments(output_dir=folder, per_device_train_batch_size=4, max_steps=steps,
                                   learning_rate=5e-3, logging_steps=5, report_to="none", save_strategy="no",
                                   remove_unused_columns=False, use_cpu=True),
            train_dataset=LazyDataset(examples, encode),
            data_collator=make_collator(0),
            callbacks=[Record()],
        ).train()
    print(f"[smoke] losses {[round(x, 3) for x in losses]}")
    ok = len(losses) >= 2 and losses[-1] < losses[0]
    print("[smoke] OK: loss falls, prelabel values masked, LoRA + Trainer path runs" if ok else "[smoke] FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python train/lora_apertus.py", description=__doc__.split("\n")[0])
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--alpha", type=int, default=ALPHA)
    parser.add_argument("--max-steps", type=int, default=None, help="a short first run, e.g. 5")
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge", action="store_true", help="also write merged bf16 weights for vLLM")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dollars-per-hour", type=float, default=None)
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        return smoke()
    if arguments.plan:
        return plan(arguments)
    if not arguments.model:
        parser.error("--model is needed for a real run (or use --plan / --smoke)")
    if not os.environ.get("HF_TOKEN"):
        print("note: HF_TOKEN is not set; the Apertus v1.5 models are gated", file=sys.stderr)
    return train(arguments)


if __name__ == "__main__":
    sys.exit(main())
