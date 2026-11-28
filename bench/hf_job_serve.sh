#!/usr/bin/env bash
# Runs INSIDE a Hugging Face Job: serve the trained LoRA (EXP-0004, #76) with vLLM on one L40S, on port 8000.
#
# Started by a human with bench/hf_job_serve_start.sh, which exposes the port through HF's jobs proxy (only an HF token
# with read access to this namespace gets through). The laptop then scores and submits against it:
#   LORA_BASE_URL=<exposed url>/v1 LORA_API_KEY=$HF_TOKEN python -m galaxus.model.eval lora      (EXP-0004 holdout)
#   ... python -m galaxus.serve.final_run --predictor galaxus.model.lora:from_env ...           (Saturday)
# In order:
#   1. download the adapter $ADAPTER_PATH (default $RUN/adapter; or a checkpoint such as $RUN/checkpoint-500, of which
#      only adapter_config.json and adapter_model.* are fetched, not the optimizer state) from the private repo
#      $CKPT_REPO (LoRA weights only; no data ever here);
#   2. merge it into Apertus in a clean venv with the training's own library versions (transformers 5.24.x, peft),
#      weights saved in the dtypes Apertus ships, and every non-weight file (config, tokenizer, processor, chat
#      template) copied from the base model, so vLLM sees exactly the base model's layout;
#   3. vllm serve the merged folder as "apertus-lora", with the image processor's defaults: training used
#      AutoProcessor's defaults on the same 512 px prepared images, so serving sees the same pixels (no min/max_pixels
#      override, unlike bench/vllm_apertus.sh, which is for zero-shot on our own machine).
# The merged weights stay on this job's disk and are deleted with it (only the adapter is stored, Neel, #76).
#
# SERVE_MODE=lora (optional, untested): skip the merge and serve the BASE model with several adapters at once
# (vLLM --enable-lora), named in ADAPTERS="name=path,name=path" (e.g.
# "final=exp0004-e1/adapter,ck500=exp0004-e1/checkpoint-500"). Each name is a model the client picks with
# LORA_SERVED_MODEL=<name>, so several checkpoints are compared in one session with no merge. It only works if the
# Swiss AI vLLM build supports LoRA for Apertus 1.5 (not known): if vLLM refuses to start, use the default (merge).
set -euo pipefail

: "${HF_TOKEN:?the HF_TOKEN secret is missing}"
: "${CKPT_REPO:?CKPT_REPO (the private checkpoint repo) is missing}"
RUN="${RUN:-exp0004-e1}"
export ADAPTER_PATH="${ADAPTER_PATH:-$RUN/adapter}"   # which save to serve: the final adapter or a checkpoint-N
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-8B}"
SERVED="${SERVED:-apertus-lora}"
SERVE_MODE="${SERVE_MODE:-merge}"   # merge (default, safer) | lora (multi-adapter, no merge)
export ADAPTERS="${ADAPTERS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
WORK=/tmp/serve
ADAPTER="$WORK/pulled/$ADAPTER_PATH"
MERGED="$WORK/merged"
log() { echo "[serve $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$WORK"
log "run=$RUN adapter=$ADAPTER_PATH model=$MODEL served as $SERVED, from $CKPT_REPO (private)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import vllm; print('vllm', vllm.__version__)" || true

log "1/3 download the adapter(s)"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
pairs = [p.split("=", 1) for p in os.environ.get("ADAPTERS", "").split(",") if "=" in p]
paths = [path for _, path in pairs] if os.environ.get("SERVE_MODE") == "lora" and pairs else [os.environ["ADAPTER_PATH"]]
# A checkpoint-N folder also holds the optimizer and RNG state (hundreds of MB): only the adapter files are needed.
patterns = [f"{p}/{name}" for p in paths for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")]
snapshot_download(os.environ["CKPT_REPO"], repo_type="model", token=os.environ["HF_TOKEN"], local_dir="/tmp/serve/pulled",
                  allow_patterns=patterns)
PY

if [[ "$SERVE_MODE" == "lora" ]]; then
  [[ -n "$ADAPTERS" ]] || ADAPTERS="$SERVED=$ADAPTER_PATH"
  MODULES=()
  IFS=',' read -ra PAIRS <<< "$ADAPTERS"
  for pair in "${PAIRS[@]}"; do
    name="${pair%%=*}"; path="${pair#*=}"
    ls "$WORK/pulled/$path/adapter_config.json" >/dev/null
    MODULES+=("$name=$WORK/pulled/$path")
    log "adapter $name: $path ($(du -sh "$WORK/pulled/$path" | cut -f1))"
  done
  log "2/3 no merge (SERVE_MODE=lora); 3/3 vllm serve the base model with ${#MODULES[@]} adapter(s) on :8000"
  exec vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
    --chat-template-content-format string --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --max-num-seqs "$MAX_NUM_SEQS" --limit-mm-per-prompt '{"image": 1}' \
    --enable-lora --max-lora-rank 32 --max-loras "${#MODULES[@]}" --lora-modules "${MODULES[@]}"
fi

ls "$ADAPTER/adapter_config.json" >/dev/null
log "adapter: $(du -sh "$ADAPTER" | cut -f1)"

log "2/3 merge it into $MODEL at the file level (base files kept as they are, LoRA added in place)"
# Why not PeftModel.merge_and_unload + save_pretrained: transformers re-saves under generic shard names
# (model-0000X-of-...), while vLLM's Apertus loader reads Apertus' own files (model-apertus-model-*,
# model-vision_tokenizer-model, model-wavtokenizer-model): the first serving run failed with "Cannot find any model
# weights". So: copy every base file unchanged, then in each shard add scale * B @ A to the weights the LoRA trained,
# under the same tensor names. Runs in the vLLM image's own python (torch + safetensors).
python3 - <<'PY'
import json, os, re, shutil
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

adapter = Path("/tmp/serve/pulled") / os.environ["ADAPTER_PATH"]
merged = Path("/tmp/serve/merged")
base_dir = Path(snapshot_download(os.environ.get("MODEL", "swiss-ai/Apertus-v1.5-8B"), token=os.environ["HF_TOKEN"]))
config = json.loads((adapter / "adapter_config.json").read_text())
scale = float(config["lora_alpha"]) / float(config["r"])
assert not config.get("use_rslora") and not config.get("use_dora"), "only plain LoRA is merged here"
pairs = {}
with safe_open(adapter / "adapter_model.safetensors", framework="pt") as handle:
    for key in handle.keys():
        m = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
        if not m:
            raise SystemExit(f"[merge] unexpected adapter tensor {key}")
        pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = handle.get_tensor(key)
index = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
missing = sorted(k for k in pairs if k not in index)
if missing:
    raise SystemExit(f"[merge] {len(missing)} adapter targets are not base tensors, e.g. {missing[:3]}")
merged.mkdir(parents=True, exist_ok=True)
for f in base_dir.iterdir():
    if f.is_file():
        shutil.copy2(f, merged / f.name)
device = "cuda" if torch.cuda.is_available() else "cpu"
done = 0
for shard in sorted({index[k] for k in pairs}):
    with safe_open(base_dir / shard, framework="pt") as handle:
        meta = handle.metadata() or {}
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
    for key in [k for k in pairs if index[k] == shard]:
        w = tensors[key]
        delta = (pairs[key]["B"].to(device, torch.float32) @ pairs[key]["A"].to(device, torch.float32)) * scale
        if delta.shape != w.shape:
            raise SystemExit(f"[merge] shape mismatch for {key}: {tuple(delta.shape)} vs {tuple(w.shape)}")
        tensors[key] = (w.to(device, torch.float32) + delta).to(w.dtype).cpu()
        done += 1
    save_file(tensors, merged / shard, metadata={"format": "pt", **meta})
    print(f"[merge] {shard}: rewritten", flush=True)
print(f"[merge] {done} of {len(pairs)} LoRA weights added (scale {scale}); every other file is the base model's; done",
      flush=True)
PY
log "merged: $(du -sh "$MERGED" | cut -f1)"

log "3/3 vllm serve on :8000 (prefix caching, up to $MAX_NUM_SEQS images at once)"
exec vllm serve "$MERGED" --served-model-name "$SERVED" --host 0.0.0.0 --port 8000 \
  --chat-template-content-format string --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --max-num-seqs "$MAX_NUM_SEQS" --limit-mm-per-prompt '{"image": 1}'
