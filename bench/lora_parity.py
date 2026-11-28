"""Is EXP-0004's 0.343 the adapter or the serving? Score the trained adapter through the training code's own path.

EXP-0004 was trained with transformers (train/lora_apertus.py: AutoProcessor + chat_messages) but scored through the
Swiss AI vLLM build, which tokenizes the image its own way. BYMO scored their comparable LoRA (0.73) with transformers
and the same image tokens as in training. This runs the adapter exactly as training saw its inputs: the same
processor, the same 512 px cached image, the same system prompt and user text, and a generation prompt checked to be,
token for token and pixel for pixel, the start of a training example, greedy, bf16 autocast as the Trainer used.

    python bench/lora_parity.py --adapter <dir with adapter_config.json> [--limit 300] [--no-context]

Runs on a GPU (inside the HF job: STAGE=parity of bench/hf_job_lora.sh). Prints only aggregate numbers: the macro F1
on a fixed sample of the official holdout, per dimension, invalid answers, prompt tokens and the share of the most
common answer per dimension (a collapse shows there). No image ids, no labels, no product text. The sample is
``holdout.sample(n=limit, random_state=seed)``, so the vLLM answers of EXP-0004 can be scored on the same images on
the laptop (``--vllm-predictions``, no GPU).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))


def holdout_sample(limit: int, seed: int):
    from galaxus.data import load, split

    holdout = split.split_holdout(load.read_train())[1]
    return holdout.sample(n=min(limit, len(holdout)), random_state=seed)


def report(name: str, sample, answers: list[dict[str, str]], extra: dict) -> dict:
    """Macro F1 of ``answers`` (one dict per sample row, in order) with the competition metric."""
    import pandas as pd

    from galaxus.data import classes, metric, split

    dims = classes.dimensions()
    submission = pd.DataFrame([{"image_id": str(i), **a} for i, a in zip(sample["image_id"], answers)])
    solution = split.solution_frame(sample)
    evaluation = metric.evaluate(solution, submission)
    per_dim = {d: round(float((evaluation.get(d) or {}).get("f1_macro") or 0.0), 3) for d in dims}
    top_share = {}
    for d in dims:
        value, count = Counter(a[d] for a in answers).most_common(1)[0]
        top_share[d] = f"{count / len(answers):.0%} {value}"
    out = {"what": name, "images": len(answers), "macro_f1": round(metric.score(solution, submission), 4),
           "per_dimension": per_dim, "most_common_answer": top_share, **extra}
    print(json.dumps(out, indent=2), flush=True)
    return out


def run_transformers(arguments) -> int:
    import lora_apertus as T
    import torch
    import transformers
    from peft import PeftModel
    from PIL import Image

    from galaxus.data import classes
    from galaxus.data.images import cached_jpeg
    from galaxus.model import lora

    sample = holdout_sample(arguments.limit, arguments.seed)
    allowed = classes.allowed_classes()
    dims = classes.dimensions()
    system = lora.system_prompt(allowed)
    answer_classes = {d: sorted(allowed[d]) for d in dims}

    processor = transformers.AutoProcessor.from_pretrained(arguments.model)
    model_class = next(getattr(transformers, n) for n in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
                                                          "AutoModelForCausalLM") if hasattr(transformers, n))
    model = model_class.from_pretrained(arguments.model, dtype=torch.bfloat16)
    print(f"[parity] {model_class.__name__}; vision tokenizers fp32: {T.keep_vision_tokenizer_fp32(model)}", flush=True)
    model = PeftModel.from_pretrained(model, arguments.adapter).to("cuda").eval()
    tokenizer = getattr(processor, "tokenizer", processor)

    # The prompt must be token for token the start of a training example: checked on the first image, else stop.
    dummy = json.dumps({d: answer_classes[d][0] for d in dims})

    def prompt_inputs(record, use_context):
        picture = Image.open(io.BytesIO(cached_jpeg(record["image_id"], record["image_file"],
                                                    image_size=lora.IMAGE_SIZE))).convert("RGB")
        messages = T.chat_messages(system, picture, lora._context(record) if use_context else None, dummy)
        inputs = processor.apply_chat_template(messages[:2], tokenize=True, return_dict=True, return_tensors="pt",
                                               add_generation_prompt=True)
        return messages, inputs

    messages, inputs = prompt_inputs(sample.iloc[0].to_dict(), True)
    trained = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
    prompt_ids, train_ids = inputs["input_ids"][0].tolist(), trained["input_ids"][0].tolist()
    same = train_ids[:len(prompt_ids)] == prompt_ids and all(
        (inputs[k] == trained[k]).all() for k in inputs if k not in ("input_ids", "attention_mask"))
    print(f"[parity] prompt = start of the training example, ids and pixels: {same} ({len(prompt_ids)} tokens)",
          flush=True)
    if not same:
        raise SystemExit("[parity] the prompt is not what training saw: the check would prove nothing")

    results = []
    for use_context in ([False] if arguments.no_context else [True, False][: 1 + arguments.both]):
        answers, invalid, tokens, started = [], 0, [], time.perf_counter()
        for _, row in sample.iterrows():
            _, inputs = prompt_inputs(row.to_dict(), use_context)
            inputs = inputs.to("cuda")
            tokens.append(int(inputs["input_ids"].shape[1]))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            labels = lora.parse_answer(reply, answer_classes)
            if labels is None:
                invalid += 1
                labels = {d: answer_classes[d][0] for d in dims}
            answers.append(labels)
        seconds = time.perf_counter() - started
        results.append(report(
            f"transformers + PEFT, training path, context {'on' if use_context else 'off'}", sample, answers,
            {"invalid_answers": invalid, "mean_prompt_tokens": round(sum(tokens) / len(tokens)),
             "images_per_second": round(len(answers) / seconds, 2),
             "prompt_equals_training": same}))
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
    parser.add_argument("--both", action="store_true", help="a pass with and one without product context")
    parser.add_argument("--vllm-predictions", help="score this holdout.jsonl on the same sample instead (no GPU)")
    arguments = parser.parse_args(argv)
    if arguments.vllm_predictions:
        return run_vllm_file(arguments)
    if not arguments.adapter:
        parser.error("--adapter is required")
    return run_transformers(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
