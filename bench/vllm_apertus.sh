#!/usr/bin/env bash
# Serve Apertus v1.5 8B with vLLM on one GPU, images at 512 x 512 pixels of area (plan #40, option 4).
#
# Upstream vLLM support for Apertus 1.5 is still an open PR (vllm-project/vllm#50496, last activity 25 Nov 2026), so
# this uses Swiss AI's release image, as the Apertus docs and the model card recommend.
#
# Usage, on the GPU machine (Docker with the NVIDIA runtime; the model is gated, so a Hugging Face token that has
# accepted its terms is needed in the environment, never on the command line or in git):
#
#   export HF_TOKEN=...            # read from your shell, never echoed
#   bash bench/vllm_apertus.sh               # bf16
#   QUANT=fp8 bash bench/vllm_apertus.sh     # FP8 weights for the ablation (#43)
#   MODEL=<run>/merged bash bench/vllm_apertus.sh   # our merged LoRA (#49): a local folder, served as "apertus-lora"
#
# Then, from the machine with the data:
#
#   python -m galaxus.model.apertus.smoke --backend vllm --base-url http://<gpu-host>:8000/v1
#   python -m galaxus.serve.bench --predictor <package.module:name> --part dev --limit 256 --workers 8 \
#       --dollars-per-hour <price of this GPU>
#
# STOP THE GPU WHEN DONE: an idle GPU costs as much as a busy one.
#
# Untested: written without GPU access on 27 Nov. Check on first use that the smoke test reports about
# (512 / 16)^2 = 1024 image tokens per 512 x 512 image (Apertus' vision tokenizer uses 16 px per token); if not, the
# max_pixels setting below is not reaching the image processor (Q-0002).
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-amd64}"
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-8B}"
PORT="${PORT:-8000}"
QUANT="${QUANT:-none}"                    # none | fp8
MAX_PIXELS="${MAX_PIXELS:-262144}"        # 512 * 512: Swarup's condition, every image at 512 px of area
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"        # images decoded together; raise while memory allows
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"    # Apertus docs' recommended start; one 512 px image is ~1k tokens
GPU_UTIL="${GPU_UTIL:-0.90}"
PREFIX_CACHING="${PREFIX_CACHING:-on}"   # on | off (speed matrix #33)
SERVED_NAME="${SERVED_NAME:-}"            # the name clients ask for; a local folder defaults to apertus-lora

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set: the Apertus v1.5 models are gated on Hugging Face (FAQ.md, 'Do we need a Hugging Face account?')." >&2
  exit 1
fi

quant_args=()
cache_args=(--enable-prefix-caching)
[[ "$PREFIX_CACHING" == "off" ]] && cache_args=(--no-enable-prefix-caching)
case "$QUANT" in
  none) ;;
  # vLLM's online FP8 quantises the language model's linear layers. The vision tokenizer must stay full precision
  # (vllm#50496 review: quality drops without FP32); compare its image codes against bf16 before trusting this row.
  fp8) quant_args=(--quantization fp8) ;;
  *) echo "QUANT must be none or fp8, got $QUANT" >&2; exit 1 ;;
esac

# A local folder (the merged LoRA) is not visible inside the container: mount it read-only at /model.
mount_args=()
model_ref="$MODEL"
if [[ -d "$MODEL" ]]; then
  mount_args=(-v "$(cd "$MODEL" && pwd):/model:ro")
  model_ref=/model
  SERVED_NAME="${SERVED_NAME:-apertus-lora}"   # what galaxus.model.lora ($LORA_SERVED_MODEL) asks for
fi
name_args=()
if [[ -n "$SERVED_NAME" ]]; then name_args=(--served-model-name "$SERVED_NAME"); fi

echo "serving $MODEL on port $PORT (quant=$QUANT, max_pixels=$MAX_PIXELS, max_num_seqs=$MAX_NUM_SEQS, prefix_caching=$PREFIX_CACHING, name=${SERVED_NAME:-$MODEL})"
exec docker run --rm --gpus all --ipc=host -p "$PORT:8000" \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  "${mount_args[@]}" \
  "$IMAGE" \
  vllm serve "$model_ref" \
    --host 0.0.0.0 --port 8000 \
    --chat-template-content-format string \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    "${cache_args[@]}" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS, \"min_pixels\": $MAX_PIXELS}" \
    "${name_args[@]}" \
    "${quant_args[@]}"
