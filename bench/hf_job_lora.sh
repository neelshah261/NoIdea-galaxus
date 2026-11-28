#!/usr/bin/env bash
# Runs INSIDE a Hugging Face Job: the LoRA on Apertus 8B (#49, #76) with checkpoints that survive the job.
#
# Started by a human with bench/hf_job_lora_start.sh (AGENTS.md product rule 7: agents never start paid GPU time).
# Neel's conditions on #76 (27 Nov, 21:21 and 21:23 UTC), each kept here:
#   - LoRA adapter checkpoints only, no data, to a PRIVATE model repo on Swarup's account ($CKPT_REPO);
#   - a checkpoint every 100 steps, uploaded as soon as it is written, so a crash loses at most one interval;
#   - the Kaggle token only as an encrypted job secret; the logs show no labels and no image ids;
#   - 1 epoch, $10 for #76: stop at the 50-step projection if it is over that.
# In order:
#   1. clone the repo at $REF and install it (uv, extra "local");
#   2. download the competition data from Kaggle. It exists only on this job's disk and is deleted with it when the job
#      ends: never in git, an HF repo, dataset or image;
#   3. prepare the 512 px training images, original and mirrored (galaxus.data.prep prints counts only);
#   4. pull earlier checkpoints of $RUN from $CKPT_REPO, so a restart continues with --resume;
#   5. train; a watcher uploads every new checkpoint-N folder (and at the end the adapter and run.json) to $CKPT_REPO.
# STAGE=setup stops after steps 1-4 plus a repo write test, on a cheap CPU job, to rehearse everything but the GPU.
set -euo pipefail
JOB_STARTED=$(date +%s)            # STAGE=parity keeps its passes inside the job's timeout from here

: "${HF_TOKEN:?the HF_TOKEN secret is missing}"
: "${KAGGLE_API_TOKEN:?the KAGGLE_API_TOKEN secret is missing}"
: "${CKPT_REPO:?CKPT_REPO (namespace/name of the private model repo) is missing}"
STAGE="${STAGE:-train}"            # setup | train | parity (score an adapter with the training code, #145)
REF="${REF:-main}"
RUN="${RUN:-exp0004}"
EPOCHS="${EPOCHS:-1}"
PRICE="${PRICE:-1.80}"             # l40sx1, `hf jobs hardware`
MAX_STEPS="${MAX_STEPS:-}"         # e.g. 5 for the GPU check
SAVE_STEPS="${SAVE_STEPS:-100}"
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-8B}"
COMPETITION="hackapertus-galaxus"
# Extra packages for training: the 8-bit optimizer, and torchvision, which Apertus' image processor
# (Apertus1p5ImageProcessor) imports; without it AutoProcessor fails (first GPU run, 27 Nov 21:44 UTC).
WITH=(--with bitsandbytes --with "torchvision==0.29.2")

WORK=/tmp/noidea
REPO="$WORK/repo"
export GALAXUS_DATA="$WORK/data"
OUT="$GALAXUS_DATA/out/lora/$RUN"
log() { echo "[job $(date -u +%H:%M:%S)] $*"; }
hfcli() { uvx --quiet --from "huggingface_hub>=1.0" hf "$@"; }
upload() {  # upload <local dir> <path in repo> <message>: private repo, created on first use
  hfcli upload "$CKPT_REPO" "$1" "$2" --repo-type model --private --commit-message "$3" >/dev/null
}

mkdir -p "$WORK" "$GALAXUS_DATA"
log "stage=$STAGE ref=$REF run=$RUN epochs=$EPOCHS save_steps=$SAVE_STEPS model=$MODEL repo=$CKPT_REPO (private)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "no GPU (fine for STAGE=setup)"

log "1/5 clone and install"
git clone --quiet --depth 1 --branch "$REF" https://github.com/neelshah261/NoIdea-galaxus "$REPO"
cd "$REPO"
log "commit $(git rev-parse --short HEAD)"
uv sync --quiet --extra local --python 3.12

log "2/5 competition data from Kaggle (this disk only)"
uvx --quiet --from "kaggle==2.2.5" --with six kaggle competitions download -c "$COMPETITION" -p "$GALAXUS_DATA" --quiet
uv run --quiet python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$GALAXUS_DATA/$COMPETITION.zip" "$GALAXUS_DATA"
rm -f "$GALAXUS_DATA/$COMPETITION.zip"
log "data unpacked: $(find "$GALAXUS_DATA" -maxdepth 1 -name '*.csv' | wc -l) csv files"

if [[ "$STAGE" == "parity" ]]; then
  # Is EXP-0004's 0.343 the adapter or the vLLM serving? Score the adapter through the training code's own path
  # (bench/lora_parity.py) on a fixed holdout sample. Only the adapter files are fetched; only aggregates are printed.
  ADAPTER_PATH="${ADAPTER_PATH:-$RUN/adapter}"
  # The passes stop by this time and score what they have, so a slow GPU costs images, not the whole result. 45 of
  # the 50 min timeout: the clock starts with this script, after the image pull, and the last image plus the report
  # need well under a minute.
  DEADLINE=$(( JOB_STARTED + ${PARITY_BUDGET_MIN:-45} * 60 ))
  log "parity: $ADAPTER_PATH on ${LIMIT:-200} holdout images, transformers + PEFT, with then without product context," \
      "passes stop by $(date -u -d "@$DEADLINE" +%H:%M:%S)"
  # The two files as FILENAMES: `hf download` takes one pattern per --include, so a second pattern after one
  # --include is read as a filename, and a filename makes it ignore --include (it fetched only the weights).
  hfcli download "$CKPT_REPO" "$ADAPTER_PATH/adapter_config.json" "$ADAPTER_PATH/adapter_model.safetensors" \
    --repo-type model --local-dir "$WORK/pulled" >/dev/null
  for f in adapter_config.json adapter_model.safetensors; do
    [[ -s "$WORK/pulled/$ADAPTER_PATH/$f" ]] || { log "missing $ADAPTER_PATH/$f in $CKPT_REPO"; exit 1; }
  done
  uv run --quiet "${WITH[@]}" python bench/lora_parity.py --adapter "$WORK/pulled/$ADAPTER_PATH" --model "$MODEL" \
    --limit "${LIMIT:-200}" --both --deadline "$DEADLINE"
  log "parity done: nothing was trained or uploaded"
  exit 0
fi

log "3/5 prepare 512 px training images, original and mirrored"
uv run --quiet python -m galaxus.data.prep --part train --size 512 --workers 8
uv run --quiet python -m galaxus.data.prep --part train --size 512 --workers 8 --mirror

log "4/5 earlier checkpoints of $RUN in $CKPT_REPO"
mkdir -p "$OUT"
RESUME=()
if hfcli download "$CKPT_REPO" --repo-type model --include "$RUN/checkpoint-*" --local-dir "$WORK/pulled" \
     >/dev/null 2>&1 && ls -d "$WORK/pulled/$RUN"/checkpoint-* >/dev/null 2>&1; then
  cp -r "$WORK/pulled/$RUN"/checkpoint-* "$OUT"/
  RESUME=(--resume)
  log "continuing from: $(ls -d "$OUT"/checkpoint-* | xargs -n1 basename | tr '\n' ' ')"
else
  log "no earlier checkpoints: a fresh run"
fi
uv run --quiet python train/lora_apertus.py --plan --epochs "$EPOCHS" --dollars-per-hour "$PRICE" | tail -2

if [[ "$STAGE" == "setup" ]]; then
  mkdir -p "$WORK/probe" && echo "setup rehearsal $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD)" \
    > "$WORK/probe/rehearsal.txt"
  upload "$WORK/probe" "rehearsal" "setup rehearsal" && log "private repo write OK"
  log "processor check: Apertus' processor on one real training example, the training script's own encoder (CPU)"
  uv run --quiet "${WITH[@]}" python - "$MODEL" <<'PY'
import sys
sys.path.insert(0, "train")
import torch, torchvision, transformers, peft
import lora_apertus as la
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, flush=True)
processor = transformers.AutoProcessor.from_pretrained(sys.argv[1])
print("processor", type(processor).__name__, "| image processor", type(processor.image_processor).__name__, flush=True)
_, data, classes = la.training_set()
inputs = la.apertus_encoder(processor, classes)(data.examples[0])
in_loss = sum(1 for x in inputs["labels"] if x != -100)
extra = {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, "shape") and k not in ("input_ids", "labels")}
print(f"first example: {len(inputs['input_ids'])} tokens, {in_loss} in the loss, other inputs {extra}", flush=True)
PY
  log "setup rehearsal done: nothing was trained"
  exit 0
fi

log "5/5 train: a checkpoint every $SAVE_STEPS steps, each uploaded as soon as it is written"
watch_checkpoints() {
  local done_list="$WORK/uploaded.txt"; touch "$done_list"
  while true; do
    for dir in "$OUT"/checkpoint-*; do
      [[ -f "$dir/trainer_state.json" ]] || continue           # written last by the Trainer: the save is complete
      name="$(basename "$dir")"
      grep -qx "$name" "$done_list" && continue
      sleep 5                                                   # let the optimizer files settle
      if upload "$dir" "$RUN/$name" "$RUN $name"; then echo "$name" >> "$done_list"; log "uploaded $name"
      else log "upload of $name failed, retrying"; fi
    done
    sleep 20
  done
}
watch_checkpoints &
WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT
STEPS=()
[[ -n "$MAX_STEPS" ]] && STEPS=(--max-steps "$MAX_STEPS")
uv run "${WITH[@]}" python train/lora_apertus.py --model "$MODEL" --epochs "$EPOCHS" --output "$OUT" \
  --save-steps "$SAVE_STEPS" --dollars-per-hour "$PRICE" "${STEPS[@]}" "${RESUME[@]}"
sleep 30   # the watcher takes the last checkpoint
kill "$WATCH_PID" 2>/dev/null || true
upload "$OUT/adapter" "$RUN/adapter" "$RUN final adapter" && log "uploaded the final adapter"
mkdir -p "$WORK/meta" && cp "$OUT/run.json" "$WORK/meta/" && upload "$WORK/meta" "$RUN" "$RUN run.json"
log "done. The job stops by itself; its disk (and the data) is deleted with it."
