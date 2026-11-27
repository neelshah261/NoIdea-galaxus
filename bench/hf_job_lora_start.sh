#!/usr/bin/env bash
# Start the LoRA job on Hugging Face (#76). A human runs this: it starts paid GPU time (AGENTS.md product rule 7).
#
#   bash bench/hf_job_lora_start.sh setup --dry-run   # print the job, start nothing (free)
#   bash bench/hf_job_lora_start.sh setup             # rehearsal on cpu-upgrade ($0.03/h): all but the GPU
#   bash bench/hf_job_lora_start.sh check             # 5 optimizer steps on one L40S ($1.80/h), ~20 min with setup
#   bash bench/hf_job_lora_start.sh train             # the full run on one L40S; EPOCHS=1 for the short one
#
# Then: hf jobs logs <id> (follow), hf jobs ps, hf jobs cancel <id>. After about 50 optimizer steps the log prints the
# measured rate and the projected time and cost; cancel if the projection is over budget. A cancelled or crashed run
# restarts with the same command: it pulls its checkpoints from the bucket and continues with --resume.
#
# Secrets come from your machine and are never printed: HF_TOKEN from .env (an account that accepted the Apertus
# licence) and the Kaggle token from ~/.kaggle/access_token. They reach the job as encrypted secrets.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: bash bench/hf_job_lora_start.sh setup|check|train [--dry-run]}"
DRY=()
[[ "${2:-}" == "--dry-run" ]] && DRY=(--dry-run)

if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2-)"
fi
: "${HF_TOKEN:?set HF_TOKEN in .env}"
KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat "$HOME/.kaggle/access_token" 2>/dev/null || true)}"
: "${KAGGLE_API_TOKEN:?no Kaggle token in ~/.kaggle/access_token}"
export HF_TOKEN KAGGLE_API_TOKEN

USER_NAME="$(python -c "import os; from huggingface_hub import whoami; print(whoami(token=os.environ['HF_TOKEN'])['name'])")"
BUCKET="${BUCKET:-$USER_NAME/noidea-lora}"
EPOCHS="${EPOCHS:-2}"
REF="${REF:-main}"
RUN="${RUN:-exp0004-e$EPOCHS}"

case "$MODE" in
  setup) FLAVOR=cpu-upgrade; TIMEOUT=1h;    ENV=(-e STAGE=setup) ;;
  check) FLAVOR=l40sx1;      TIMEOUT=45m;   ENV=(-e STAGE=train -e MAX_STEPS=5); RUN="$RUN-check" ;;
  train) FLAVOR=l40sx1;      TIMEOUT="${TIMEOUT:-5h30m}"; ENV=(-e STAGE=train) ;;   # 5.5 h x $1.80 = $9.90 at most
  *) echo "mode must be setup, check or train" >&2; exit 1 ;;
esac

echo "job: $MODE on $FLAVOR, timeout $TIMEOUT, run $RUN, epochs $EPOCHS, ref $REF, bucket $BUCKET (private)"
exec hf jobs run "${DRY[@]}" --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --name "noidea-lora-$MODE" \
  --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e BUCKET="$BUCKET" -e RUN="$RUN" -e EPOCHS="$EPOCHS" -e REF="$REF" -e PRICE=1.80 "${ENV[@]}" \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "$(cat bench/hf_job_lora.sh)"
