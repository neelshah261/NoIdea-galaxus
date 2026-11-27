#!/usr/bin/env bash
# Runs INSIDE a Hugging Face Job: serve the trained LoRA (EXP-0004, #76) with vLLM on one L40S, on port 8000.
#
# Started by a human with bench/hf_job_serve_start.sh, which exposes the port through HF's jobs proxy (only an HF token
# with read access to this namespace gets through). The laptop then scores and submits against it:
#   LORA_BASE_URL=<exposed url>/v1 LORA_API_KEY=$HF_TOKEN python -m galaxus.model.eval lora      (EXP-0004 holdout)
#   ... python -m galaxus.serve.final_run --predictor galaxus.model.lora:from_env ...           (Saturday)
# In order:
#   1. download the adapter $RUN/adapter from the private repo $CKPT_REPO (LoRA weights only; no data ever here);
#   2. merge it into Apertus in a clean venv with the training's own library versions (transformers 5.24.x, peft),
#      weights saved in the dtypes Apertus ships, and every non-weight file (config, tokenizer, processor, chat
#      template) copied from the base model, so vLLM sees exactly the base model's layout;
#   3. vllm serve the merged folder as "apertus-lora", with the image processor's defaults: training used
#      AutoProcessor's defaults on the same 512 px prepared images, so serving sees the same pixels (no min/max_pixels
#      override, unlike bench/vllm_apertus.sh, which is for zero-shot on our own machine).
# The merged weights stay on this job's disk and are deleted with it (only the adapter is stored, Neel, #76).
set -euo pipefail

: "${HF_TOKEN:?the HF_TOKEN secret is missing}"
: "${CKPT_REPO:?CKPT_REPO (the private checkpoint repo) is missing}"
RUN="${RUN:-exp0004-e1}"
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-8B}"
SERVED="${SERVED:-apertus-lora}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
WORK=/tmp/serve
ADAPTER="$WORK/pulled/$RUN/adapter"
MERGED="$WORK/merged"
log() { echo "[serve $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$WORK"
log "run=$RUN model=$MODEL served as $SERVED, adapter from $CKPT_REPO (private)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import vllm; print('vllm', vllm.__version__)" || true

log "1/3 download the adapter"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["CKPT_REPO"], repo_type="model", allow_patterns=[f"{os.environ.get('RUN', 'exp0004-e1')}/adapter/*"],
                  local_dir="/tmp/serve/pulled", token=os.environ["HF_TOKEN"])
PY
ls "$ADAPTER/adapter_config.json" >/dev/null
log "adapter: $(du -sh "$ADAPTER" | cut -f1)"

log "2/3 merge it into $MODEL (clean venv, the training's versions)"
python3 -m pip install --quiet uv
python3 -m uv venv --quiet --python 3.12 "$WORK/mergeenv"
python3 -m uv pip install --quiet --python "$WORK/mergeenv/bin/python" torch "transformers==5.24.*" peft accelerate safetensors
"$WORK/mergeenv/bin/python" - <<'PY'
import json, os, shutil
from pathlib import Path
import torch, transformers
from huggingface_hub import snapshot_download
from peft import PeftModel

model_id, adapter, merged = os.environ.get("MODEL", "swiss-ai/Apertus-v1.5-8B"), "/tmp/serve/pulled/" + os.environ.get("RUN", "exp0004-e1") + "/adapter", Path("/tmp/serve/merged")
base_dir = Path(snapshot_download(model_id, token=os.environ["HF_TOKEN"]))
cls = next(getattr(transformers, n) for n in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM") if hasattr(transformers, n))
print("[merge] transformers", transformers.__version__, "| class", cls.__name__, flush=True)
base = cls.from_pretrained(base_dir, dtype="auto", device_map="cuda")
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
del model, base
torch.cuda.empty_cache()
# Every non-weight file of the base model wins, so vLLM sees the base layout (config, processor, tokenizer, template).
for f in base_dir.iterdir():
    if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and f.name != "model.safetensors.index.json":
        shutil.copy2(f, merged / f.name)
base_keys = set(json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]) if (base_dir / "model.safetensors.index.json").exists() else None
idx = merged / "model.safetensors.index.json"
merged_keys = set(json.loads(idx.read_text())["weight_map"]) if idx.exists() else None
if base_keys is not None and merged_keys is not None:
    print(f"[merge] tensors: base {len(base_keys)}, merged {len(merged_keys)}, same names: {base_keys == merged_keys}", flush=True)
    if base_keys != merged_keys:
        print("[merge] WARNING names differ, e.g.", sorted(base_keys ^ merged_keys)[:4], flush=True)
print("[merge] done", flush=True)
PY
log "merged: $(du -sh "$MERGED" | cut -f1)"

log "3/3 vllm serve on :8000 (prefix caching, up to $MAX_NUM_SEQS images at once)"
exec vllm serve "$MERGED" --served-model-name "$SERVED" --host 0.0.0.0 --port 8000 \
  --chat-template-content-format string --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --max-num-seqs "$MAX_NUM_SEQS" --limit-mm-per-prompt '{"image": 1}'
