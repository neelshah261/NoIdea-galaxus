"""Is EXP-0004's 0.343 the adapter or the serving? Score the trained adapter through the training code's own path.

EXP-0004 was trained with transformers (train/lora_apertus.py: AutoProcessor + chat_messages) but scored through the
Swiss AI vLLM build, which tokenizes the image its own way. BYMO scored their comparable LoRA (0.73) with transformers
and the same image tokens as in training. This runs the adapter exactly as training saw its inputs: the same
processor, the same 512 px cached image, the same system prompt and user text, and a generation prompt checked to be,
token for token and pixel for pixel, the start of a training example, greedy, bf16 autocast as the Trainer used.

    python bench/lora_parity.py --adapter <dir with adapter_config.json> [--limit 300] [--both | --no-context]
                                [--deadline <unix seconds>] [--device cuda]

Runs on a GPU (inside the HF job: STAGE=parity of bench/hf_job_lora.sh). Prints only aggregate numbers: the macro F1
on a fixed sample of the official holdout, per dimension, invalid answers, prompt tokens and the share of the most
common answer per dimension (a collapse shows there). No image ids, no labels, no product text. The sample is
``holdout.sample(n=limit, random_state=seed)``, so the vLLM answers of EXP-0004 can be scored on the same images on
the laptop (``--vllm-predictions``, no GPU). Both modes print the same ``sample`` fingerprint (a hash of the ids, not
an id) to show it is the same sample.

The job has a hard timeout, so the GPU passes keep to ``--deadline``: the first pass may use half the time left, the
second the rest, and a pass that runs out stops and scores the images it has. A sample of n is the first n rows of any
larger sample with the same seed (numpy's choice without replacement is a prefix of one permutation), so a pass cut at
n images is compared with ``--vllm-predictions ... --limit n``. ``--device cpu`` runs the same path without autocast,
for the end-to-end test on a tiny random model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

#: The training answer ends with this token, and it is in the loss (training_labels keeps everything from the answer
#: on), so the adapter learned to stop there: generation ends at it instead of running to max_new_tokens.
END_OF_ANSWER = "<|assistant_end|>"
PROGRESS_EVERY = 25


def holdout_sample(limit: int, seed: int):
    from galaxus.data import load, split

    holdout = split.split_holdout(load.read_train())[1]
    return holdout.sample(n=min(limit, len(holdout)), random_state=seed)


def fingerprint(sample) -> str:
    """A short hash of the sample's image ids in order: equal in the job and on the laptop iff the sample is."""
    return hashlib.sha256("\n".join(sample["image_id"].astype(str)).encode()).hexdigest()[:12]


def report(name: str, sample, answers: list[dict[str, str]], extra: dict) -> dict:
    """Macro F1 of ``answers`` (one dict per row of ``sample``, in order) with the competition metric.

    ``sample`` may be longer than ``answers`` (a pass stopped by the deadline): its first ``len(answers)`` rows are
    scored, which is the sample ``--limit len(answers)`` gives.
    """
    import pandas as pd

    from galaxus.data import classes, metric, split

    dims = classes.dimensions()
    scored = sample.iloc[: len(answers)]
    submission = pd.DataFrame([{"image_id": str(i), **a} for i, a in zip(scored["image_id"], answers)])
    solution = split.solution_frame(scored)
    evaluation = metric.evaluate(solution, submission)
    per_dim = {d: round(float((evaluation.get(d) or {}).get("f1_macro") or 0.0), 3) for d in dims}
    top_share = {}
    for d in dims:
        value, count = Counter(a[d] for a in answers).most_common(1)[0]
        top_share[d] = f"{count / len(answers):.0%} {value}"
    out = {"what": name, "images": len(answers), "sample": fingerprint(scored),
           "macro_f1": round(metric.score(solution, submission), 4), "per_dimension": per_dim,
           "most_common_answer": top_share, **extra}
    print(json.dumps(out, indent=2), flush=True)
    return out


def same_inputs(prompt, trained) -> bool:
    """True if every processor output of the generation prompt is the start of the training example's.

    ``input_ids`` must be a prefix; the image tensors (``pixel_values``, ``image_sizes``) must be equal in shape and
    value; a tensor along the sequence (an attention mask, token types) is compared on the prompt's length. A key the
    training example lacks, a shape that differs or a value that is not a tensor and not equal makes it False: never
    an exception, because a False is reported and stops the run with a clear message.
    """
    import torch

    prompt_ids, train_ids = prompt["input_ids"][0].tolist(), trained["input_ids"][0].tolist()
    if train_ids[: len(prompt_ids)] != prompt_ids:
        return False
    length = len(prompt_ids)
    for key in prompt:
        if key == "input_ids":
            continue
        if key not in trained:
            return False
        a, b = prompt[key], trained[key]
        if isinstance(a, torch.Tensor) != isinstance(b, torch.Tensor):
            return False
        if not isinstance(a, torch.Tensor):
            if a != b:
                return False
            continue
        if a.dim() >= 2 and a.shape[-1] == length and b.shape[-1] == len(train_ids) and a.shape[:-1] == b.shape[:-1]:
            b = b[..., :length]
        if a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            return False
    return True


def image_code_agreement(model, records, prompt_inputs, device: str) -> str:
    """Share of the image's VQ codes that are the same in fp32 and under bf16 autocast, on a few images.

    The vision tokenizer picks each code by an argmax over 131,072 codebook logits. The Trainer ran it under bf16
    autocast, a server may run it in fp32: where the two disagree, the adapter is served image tokens it never trained
    on. Diagnostic only (a share, never a code): any failure here is printed and the run goes on.
    """
    import torch

    try:
        inner = model.get_base_model().model
        equal = total = 0
        for record in records:
            inputs = prompt_inputs(record, True).to(device)
            pixels, sizes = inputs["pixel_values"], inputs["image_sizes"]
            with torch.inference_mode():
                exact = inner.get_image_tokens(pixels, sizes)
                with torch.autocast(device, dtype=torch.bfloat16):
                    mixed = inner.get_image_tokens(pixels, sizes)
            for x, y in zip(exact, mixed):
                equal += int((x == y).sum())
                total += x.numel()
        return f"{equal / max(1, total):.1%} of {total} codes on {len(records)} images"
    except Exception as error:  # noqa: BLE001 - a diagnostic must not end the paid run
        return f"not measured ({type(error).__name__}: {error})"


def pass_plan(contexts: list[bool], deadline: float | None) -> list[tuple[bool, float | None]]:
    """(context, share of the time left at the pass's start) per pass: the first of two passes gets half, the last all."""
    return [(c, None if deadline is None else 1.0 / (len(contexts) - i)) for i, c in enumerate(contexts)]


def run_transformers(arguments) -> int:
    import lora_apertus as T
    import torch
    import transformers
    from peft import PeftModel
    from PIL import Image

    from galaxus.data import classes
    from galaxus.data.images import cached_jpeg
    from galaxus.model import lora
    from galaxus.model.baselines import MajorityPredictor

    device = arguments.device
    sample = holdout_sample(arguments.limit, arguments.seed)
    allowed = classes.allowed_classes()
    dims = classes.dimensions()
    system = lora.system_prompt(allowed)
    answer_classes = {d: sorted(allowed[d]) for d in dims}
    # The vLLM predictor's fallback for an invalid answer (LoraPredictor via from_env): the majority answer of train
    # minus the holdout, so an invalid answer here costs what it cost there.
    fallback = dict(MajorityPredictor().answer)
    print(f"[parity] {len(sample)} holdout images, sample {fingerprint(sample)}, device {device}, transformers "
          f"{transformers.__version__}, torch {torch.__version__}", flush=True)

    processor = transformers.AutoProcessor.from_pretrained(arguments.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    model_class = next(getattr(transformers, n) for n in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
                                                          "AutoModelForCausalLM") if hasattr(transformers, n))
    started = time.perf_counter()
    model = model_class.from_pretrained(arguments.model, dtype=torch.bfloat16)
    print(f"[parity] {model_class.__name__}; vision tokenizers fp32: {T.keep_vision_tokenizer_fp32(model)}", flush=True)
    model = PeftModel.from_pretrained(model, arguments.adapter).to(device).eval()
    print(f"[parity] model and adapter loaded in {time.perf_counter() - started:.0f} s", flush=True)

    end_ids = [tokenizer.convert_tokens_to_ids(END_OF_ANSWER), tokenizer.eos_token_id]
    end_ids = sorted({i for i in end_ids if isinstance(i, int) and i != tokenizer.unk_token_id})
    # The longest answer the classes allow, plus room: max_new_tokens must never cut a valid answer.
    longest = lora.serialize_answer({d: max(answer_classes[d], key=len) for d in dims})[0]
    max_new = max(arguments.max_new_tokens, len(tokenizer(longest, add_special_tokens=False)["input_ids"]) + 16)
    print(f"[parity] greedy, at most {max_new} new tokens, stops at {len(end_ids)} end tokens", flush=True)

    dummy = json.dumps({d: answer_classes[d][0] for d in dims})

    def prompt_inputs(record, use_context, with_answer=False):
        picture = Image.open(io.BytesIO(cached_jpeg(record["image_id"], record["image_file"],
                                                    image_size=lora.IMAGE_SIZE))).convert("RGB")
        messages = T.chat_messages(system, picture, lora._context(record) if use_context else None, dummy)
        if with_answer:  # the training encoding (apertus_encoder): the whole conversation, no generation prompt
            return processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        return processor.apply_chat_template(messages[:2], tokenize=True, return_dict=True, return_tensors="pt",
                                             add_generation_prompt=True)

    contexts = [False] if arguments.no_context else [True, False] if arguments.both else [True]
    first = sample.iloc[0].to_dict()
    # The prompt must be token for token and pixel for pixel the start of a training example, for each prompt this
    # run sends (with and without context), checked on the first image; else the check would prove nothing.
    for use_context in contexts:
        prompt, trained = prompt_inputs(first, use_context), prompt_inputs(first, use_context, with_answer=True)
        same = same_inputs(prompt, trained)
        print(f"[parity] context {'on' if use_context else 'off'}: prompt = start of the training example, ids and "
              f"pixels: {same} ({prompt['input_ids'].shape[1]} tokens)", flush=True)
        if not same:
            raise SystemExit("[parity] the prompt is not what training saw: the check would prove nothing")
    print(f"[parity] image codes equal in fp32 and bf16 autocast: "
          f"{image_code_agreement(model, [r.to_dict() for _, r in sample.head(4).iterrows()], prompt_inputs, device)}",
          flush=True)

    autocast = torch.autocast(device, dtype=torch.bfloat16) if device.startswith("cuda") else None
    for use_context, share in pass_plan(contexts, arguments.deadline):
        label = f"context {'on' if use_context else 'off'}"
        pass_started = time.time()
        stop_at = None if share is None else pass_started + share * max(0.0, arguments.deadline - pass_started)
        answers, invalid, tokens, new_tokens, ended = [], 0, [], [], 0
        for _, row in sample.iterrows():
            if stop_at is not None and time.time() >= stop_at:
                break
            inputs = prompt_inputs(row.to_dict(), use_context).to(device)
            length = int(inputs["input_ids"].shape[1])
            tokens.append(length)
            with torch.inference_mode():
                if autocast is not None:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        output = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                                eos_token_id=end_ids, pad_token_id=tokenizer.pad_token_id)
                else:
                    output = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                            eos_token_id=end_ids, pad_token_id=tokenizer.pad_token_id)
            generated = output[0][length:].tolist()
            new_tokens.append(len(generated))
            ended += bool(generated) and generated[-1] in end_ids
            labels = lora.parse_answer(tokenizer.decode(generated, skip_special_tokens=True), answer_classes)
            if labels is None:
                invalid += 1
                labels = dict(fallback)
            answers.append(labels)
            if len(answers) % PROGRESS_EVERY == 0:
                print(f"[parity] {label}: {len(answers)}/{len(sample)} images, "
                      f"{len(answers) / (time.time() - pass_started):.2f} images/s", flush=True)
        seconds = time.time() - pass_started
        if not answers:
            print(f"[parity] {label}: no time left for a single image, nothing to score", flush=True)
            continue
        stopped = len(answers) < len(sample)
        if stopped:
            print(f"[parity] {label}: the time budget stopped the pass at {len(answers)} of {len(sample)} images; "
                  f"score vLLM on the same images with --vllm-predictions ... --limit {len(answers)}", flush=True)
        report(f"transformers + PEFT, training path, {label}", sample, answers,
               {"invalid_answers": invalid, "ended_at_end_token": ended,
                "mean_prompt_tokens": round(sum(tokens) / len(tokens)),
                "mean_new_tokens": round(sum(new_tokens) / len(new_tokens), 1),
                "images_per_second": round(len(answers) / seconds, 2), "stopped_by_deadline": stopped,
                "prompt_equals_training": True})
    return 0


def run_vllm_file(arguments) -> int:
    """The same sample scored from EXP-0004's vLLM answers (holdout.jsonl), on the laptop."""
    sample = holdout_sample(arguments.limit, arguments.seed)
    by_id = {}
    for line in Path(arguments.vllm_predictions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_id[str(record["image_id"])] = record["labels"]
    missing = [i for i in sample["image_id"].astype(str) if i not in by_id]
    if missing:
        raise SystemExit(f"[parity] {len(missing)} sample images are not in the vLLM file")
    report("vLLM (EXP-0004 as scored)", sample, [by_id[str(i)] for i in sample["image_id"]], {})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adapter", help="folder with adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-context", action="store_true", help="only the pass without product context")
    parser.add_argument("--both", action="store_true", help="a pass with, then one without product context")
    parser.add_argument("--device", default="cuda", help="cuda (bf16 autocast, as the Trainer) or cpu (no autocast)")
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="an answer is ~50 tokens; raised to the longest valid answer + 16 if that is more")
    parser.add_argument("--deadline", type=float, default=None,
                        help="unix time by which the passes must have stopped (the job's timeout minus a margin)")
    parser.add_argument("--vllm-predictions", help="score this holdout.jsonl on the same sample instead (no GPU)")
    arguments = parser.parse_args(argv)
    if arguments.vllm_predictions:
        return run_vllm_file(arguments)
    if not arguments.adapter:
        parser.error("--adapter is required")
    return run_transformers(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
