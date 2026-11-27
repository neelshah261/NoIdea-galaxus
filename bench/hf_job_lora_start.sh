#!/usr/bin/env bash
# Start the LoRA job on Hugging Face (#76). A human runs this: it starts paid compute (AGENTS.md product rule 7).
#
#   bash bench/hf_job_lora_start.sh setup --dry-run   # print the job, start nothing (free)
#   bash bench/hf_job_lora_start.sh setup             # rehearsal on cpu-upgrade ($0.03/h): all but the GPU
#   bash bench/hf_job_lora_start.sh check             # 5 optimizer steps on one L40S ($1.80/h), ~25 min with setup
#   bash bench/hf_job_lora_start.sh train             # the full run on one L40S: 1 epoch, at most $10 (Neel, #76)
#
# Then (with HF_TOKEN set, e.g. `export HF_TOKEN=$(grep ^HF_TOKEN= .env | cut -d= -f2-)`):
#   hf jobs logs -f <id>    hf jobs ps    hf jobs cancel <id>
# After about 50 optimizer steps the log prints the measured rate and the projected time and cost: cancel if the
# projection is over $10. A cancelled or crashed run restarts with the same command: it pulls its checkpoints from the
# private repo and continues with --resume.
#
# Secrets come from this machine and are never printed: HF_TOKEN from .env (an account that accepted the Apertus
# licence) and the Kaggle token from ~/.kaggle/access_token. They reach the job as encrypted secrets.
# After the job: regenerate the Kaggle token (Neel's condition).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: bash bench/hf_job_lora_start.sh setup|check|train [--dry-run]}"
DRY=()
[[ "${2:-}" == "--dry-run" ]] && DRY=(--dry-run)

if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '\r')"
fi
: "${HF_TOKEN:?set HF_TOKEN in .env}"
KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(tr -d '\r\n' < "$HOME/.kaggle/access_token" 2>/dev/null || true)}"
: "${KAGGLE_API_TOKEN:?no Kaggle token in ~/.kaggle/access_token}"
export HF_TOKEN KAGGLE_API_TOKEN

USER_NAME="$(python -c "import os; from huggingface_hub import whoami; print(whoami(token=os.environ['HF_TOKEN'])['name'])")"
CKPT_REPO="${CKPT_REPO:-$USER_NAME/noidea-lora-checkpoints}"
EPOCHS="${EPOCHS:-1}"
REF="${REF:-main}"
RUN="${RUN:-exp0004-e$EPOCHS}"

case "$MODE" in
  setup) FLAVOR=cpu-upgrade; TIMEOUT=1h;    ENV=(-e STAGE=setup) ;;
  check) FLAVOR=l40sx1;      TIMEOUT=45m;   ENV=(-e STAGE=train -e MAX_STEPS=5); RUN="$RUN-check" ;;
  train) FLAVOR=l40sx1;      TIMEOUT="${TIMEOUT:-5h30m}"; ENV=(-e STAGE=train) ;;   # 5.5 h x $1.80 = $9.90 < $10
  *) echo "mode must be setup, check or train" >&2; exit 1 ;;
esac

echo "job: $MODE on $FLAVOR, timeout $TIMEOUT, run $RUN, epochs $EPOCHS, ref $REF, checkpoints to $CKPT_REPO (private)"
# The script is sent with Unix line endings: a Windows checkout adds \r, which bash in the job rejects
# ("set: pipefail: invalid option name", the first rehearsal on 27 Nov).
exec hf jobs run "${DRY[@]}" --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --name "noidea-lora-$MODE" \
  --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e CKPT_REPO="$CKPT_REPO" -e RUN="$RUN" -e EPOCHS="$EPOCHS" -e REF="$REF" -e PRICE=1.80 -e SAVE_STEPS=100 \
  "${ENV[@]}" \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "$(tr -d '\r' < bench/hf_job_lora.sh)"
