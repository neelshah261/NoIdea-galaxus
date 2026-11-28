#!/usr/bin/env bash
# Start (or find) the serving job for the trained LoRA (#76). A human runs this: it starts paid GPU time.
#
#   bash bench/hf_job_serve_start.sh --dry-run    # print the job, start nothing (free)
#   bash bench/hf_job_serve_start.sh              # start it: one L40S ($1.80/h), stops itself after $TIMEOUT
#   ADAPTER_PATH=exp0004-e1/checkpoint-500 bash bench/hf_job_serve_start.sh   # serve an earlier checkpoint
#   SERVE_MODE=lora ADAPTERS="final=exp0004-e1/adapter,ck500=exp0004-e1/checkpoint-500" bash bench/hf_job_serve_start.sh
#        # optional, untested: several adapters in one session without a merge; pick one with LORA_SERVED_MODEL=<name>
#   bash bench/hf_job_serve_start.sh url <id>     # once it runs: print the two lines to export on this laptop
#
# Then, with those exported (the token is read from .env, never printed):
#   curl -s -H "Authorization: Bearer $LORA_API_KEY" "$LORA_BASE_URL/models"          # is it up?
#   python -m galaxus.model.eval lora --exp-id EXP-0004 --usd-per-hour 1.80             # holdout score
#   python -m galaxus.serve.bench --predictor galaxus.model.lora:from_env --workers 8   # img/s
# and STOP IT when done (an idle GPU costs as much as a busy one): hf jobs cancel <id>
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '\r')"
fi
: "${HF_TOKEN:?set HF_TOKEN in .env}"
export HF_TOKEN

if [[ "${1:-}" == "url" ]]; then
  JOB="${2:?usage: bash bench/hf_job_serve_start.sh url <job id>}"
  python - "$JOB" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
job_id = sys.argv[1].split("/")[-1]
job = api.inspect_job(job_id=job_id)
urls = getattr(job.status, "expose_urls", None) or []
print(f"# job {job_id}: {job.status.stage}")
if not urls:
    print("# no exposed URL yet: wait until the stage is RUNNING and the log says vLLM is listening")
for url in urls:
    print(f'export LORA_BASE_URL="{url.rstrip("/")}/v1"')
print('export LORA_API_KEY="$(grep -E \'^HF_TOKEN=\' .env | cut -d= -f2- | tr -d \'\\r\')"')
print('export LORA_SERVED_MODEL=apertus-lora')
PY
  exit 0
fi

DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)
USER_NAME="$(python -c "import os; from huggingface_hub import whoami; print(whoami(token=os.environ['HF_TOKEN'])['name'])")"
CKPT_REPO="${CKPT_REPO:-$USER_NAME/noidea-lora-checkpoints}"
RUN="${RUN:-exp0004-e1}"
ADAPTER_PATH="${ADAPTER_PATH:-$RUN/adapter}"   # or $RUN/checkpoint-N to serve an earlier save
TIMEOUT="${TIMEOUT:-60m}"     # one unit only; 60 min x $1.80 = $1.80 at most (cancel as soon as you are done)
SERVE_MODE="${SERVE_MODE:-merge}"
ADAPTERS="${ADAPTERS:-}"
IMAGE="${IMAGE:-ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-amd64}"

echo "serve ($SERVE_MODE): ${ADAPTERS:-$ADAPTER_PATH} from $CKPT_REPO on l40sx1, timeout $TIMEOUT, port 8000 behind the HF jobs proxy"
exec hf jobs run "${DRY[@]}" --detach --flavor l40sx1 --timeout "$TIMEOUT" --name "noidea-lora-serve" \
  --expose 8000 --secrets HF_TOKEN -e CKPT_REPO="$CKPT_REPO" -e RUN="$RUN" -e ADAPTER_PATH="$ADAPTER_PATH" \
  -e SERVE_MODE="$SERVE_MODE" -e ADAPTERS="$ADAPTERS" \
  "$IMAGE" bash -c "$(tr -d '\r' < bench/hf_job_serve.sh)"
