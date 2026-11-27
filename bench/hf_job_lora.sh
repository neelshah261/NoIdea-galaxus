#!/usr/bin/env bash
# Runs INSIDE a Hugging Face Job: the LoRA on Apertus 8B (#49, #76) with checkpoints that survive the job.
#
# Started by a human with bench/hf_job_lora_start.sh (AGENTS.md product rule 7: agents never start paid GPU time).
# What it does, in order:
#   1. clones the repo at $REF and installs it (uv, extra "local");
#   2. downloads the competition data from Kaggle with the KAGGLE_API_TOKEN secret. The data exists only on this job's
#      disk and is deleted with it when the job ends: never in git, an HF repo, dataset, bucket or image;
#   3. prepares the 512 px training images, original and mirrored (galaxus.data.prep);
#   4. pulls earlier checkpoints of $RUN from the private bucket $BUCKET (so a restart continues with --resume);
#   5. trains, and every $SYNC_EVERY seconds copies the run folder (LoRA checkpoints, adapter, run.json: no data,
#      no images) to the bucket, and once more at the end.
# STAGE=setup stops after steps 1-4 plus a bucket write test, on a cheap CPU job, to rehearse everything but the GPU.
set -euo pipefail

: "${HF_TOKEN:?the HF_TOKEN secret is missing}"
: "${KAGGLE_API_TOKEN:?the KAGGLE_API_TOKEN secret is missing}"
: "${BUCKET:?BUCKET (namespace/name of the private bucket) is missing}"
STAGE="${STAGE:-train}"            # setup | train
REF="${REF:-main}"
RUN="${RUN:-exp0004}"
EPOCHS="${EPOCHS:-2}"
PRICE="${PRICE:-1.80}"             # l40sx1, `hf jobs hardware`
MAX_STEPS="${MAX_STEPS:-}"         # e.g. 5 for the GPU check
SYNC_EVERY="${SYNC_EVERY:-300}"
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-8B}"
COMPETITION="hackapertus-galaxus"

WORK=/tmp/noidea
REPO="$WORK/repo"
export GALAXUS_DATA="$WORK/data"
OUT="$GALAXUS_DATA/out/lora/$RUN"
REMOTE="hf://buckets/$BUCKET/$RUN"
log() { echo "[job $(date -u +%H:%M:%S)] $*"; }
hfcli() { uvx --quiet --from "huggingface_hub>=1.0" hf "$@"; }

mkdir -p "$WORK" "$GALAXUS_DATA"
log "stage=$STAGE ref=$REF run=$RUN epochs=$EPOCHS model=$MODEL bucket=$BUCKET (private)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "no GPU (fine for STAGE=setup)"

log "1/5 clone and install"
git clone --quiet --depth 1 --branch "$REF" https://github.com/neelshah261/NoIdea-galaxus "$REPO"
cd "$REPO"
log "commit $(git rev-parse --short HEAD)"
uv sync --quiet --extra local --python 3.12

log "2/5 competition data from Kaggle (this disk only)"
uvx --quiet kaggle competitions download -c "$COMPETITION" -p "$GALAXUS_DATA" --quiet
uv run --quiet python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$GALAXUS_DATA/$COMPETITION.zip" "$GALAXUS_DATA"
rm -f "$GALAXUS_DATA/$COMPETITION.zip"
log "data: $(ls "$GALAXUS_DATA" | tr '\n' ' ')"

log "3/5 prepare 512 px training images, original and mirrored"
uv run --quiet python -m galaxus.data.prep --part train --size 512 --workers 8
uv run --quiet python -m galaxus.data.prep --part train --size 512 --workers 8 --mirror

log "4/5 checkpoints of $RUN in the bucket"
hfcli buckets create "$BUCKET" --private --exist-ok >/dev/null
mkdir -p "$OUT"
RESUME=()
if hfcli buckets list "$BUCKET" 2>/dev/null | grep -q "$RUN/checkpoint-"; then
  hfcli buckets sync "$REMOTE" "$OUT" >/dev/null
  RESUME=(--resume)
  log "found earlier checkpoints, continuing: $(ls -d "$OUT"/checkpoint-* | xargs -n1 basename | tr '\n' ' ')"
else
  log "no earlier checkpoints: a fresh run"
fi
uv run --quiet python train/lora_apertus.py --plan --dollars-per-hour "$PRICE" | tail -3

if [[ "$STAGE" == "setup" ]]; then
  echo "setup rehearsal $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD)" > "$WORK/rehearsal.txt"
  hfcli buckets cp "$WORK/rehearsal.txt" "hf://buckets/$BUCKET/rehearsal.txt" >/dev/null
  hfcli buckets list "$BUCKET" | grep -q rehearsal.txt && log "bucket write OK"
  uv run --quiet python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', \
transformers.__version__, '| peft', peft.__version__)"
  log "setup rehearsal done: nothing was trained"
  exit 0
fi

log "5/5 train (checkpoints every 200 steps, copied to the bucket every ${SYNC_EVERY}s)"
sync_out() { hfcli buckets sync "$OUT" "$REMOTE" --exclude "merged/*" >/dev/null 2>&1 || log "bucket sync failed, retrying later"; }
( while sleep "$SYNC_EVERY"; do sync_out; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true' EXIT
STEPS=()
[[ -n "$MAX_STEPS" ]] && STEPS=(--max-steps "$MAX_STEPS")
uv run --with bitsandbytes python train/lora_apertus.py --model "$MODEL" --epochs "$EPOCHS" --output "$OUT" \
  --dollars-per-hour "$PRICE" "${STEPS[@]}" "${RESUME[@]}"
kill "$SYNC_PID" 2>/dev/null || true
sync_out
log "final copy in $REMOTE: $(hfcli buckets list "$BUCKET" | grep -c "$RUN/") files"
log "done. The job stops by itself; its disk (and the data) is deleted with it."
