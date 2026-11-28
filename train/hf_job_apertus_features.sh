#!/usr/bin/env bash
# Runs INSIDE a Hugging Face Job: Apertus v1.5 8B hidden states for the apertus-core heads (#218, umbrella #214), sent back through
# the job's own exposed port. Started by a human with train/hf_job_apertus_features_start.sh (AGENTS.md rule 7).
#
#   STAGE=features  train (the official holdout included) and dev: for the EXP rows and the dev file
#   STAGE=rehearse  the first $CHECK dev images end to end, and the projected time of the final pass
#   STAGE=final     the final images, decrypted on this disk with the PASSPHRASE job secret
#   STAGE=encoders  SigLIP 2 so400m embeddings only, for train and dev (galaxus.model.encoders), no Apertus
# Any stage also embeds its parts with SigLIP 2 so400m when SO400M=1 (pinned in galaxus.model.encoders), and serves
# them as embeddings.tar next to features.tar. Use it for train, dev AND final, so every so400m vector comes from the
# same device and dtype (cuda, bf16).
#
# In order:
#   1. clone $REF and install it (uv, extra "local", plus torchvision for Apertus' image processor, as the LoRA job);
#   2. the competition data from Kaggle, on this disk only. It is deleted with the job: never in a repo or a dataset;
#   3. the 512 px image cache of the parts (galaxus.data.images, the prepared pixels every member reads);
#   4. a check on the first $CHECK images of the first part: the measured speed, and that every feature is finite;
#   5. galaxus.model.heads extract for every part. It resumes, so the check's images are not computed twice;
#   6. one tar of the feature folder, served on port 8000. The HF jobs proxy only lets through an HF token with
#      access to this namespace. The job serves until its timeout: cancel it once the laptop has the file.
# The logs show counts and timings only: no image id, no label, and never the passphrase or the GitHub token
# (GH_TOKEN, a read-only job secret while the repo is private).
set -euo pipefail

: "${HF_TOKEN:?the HF_TOKEN secret is missing}"
: "${KAGGLE_API_TOKEN:?the KAGGLE_API_TOKEN secret is missing}"
STAGE="${STAGE:-features}"
REF="${REF:-main}"
MODEL="${MODEL:-swiss-ai/Apertus-v1.5-8B}"
LAYERS="${LAYERS:--1,-9}"          # always passed as --layers=..., because argparse reads "-1,-9" as an option
POOL="${POOL:-last}"
CHECK="${CHECK:-300}"              # more than one shard of 256, so heads.extract logs the measured img/s
FINAL_IMAGES="${FINAL_IMAGES:-5292}"
COMPETITION="hackapertus-galaxus"
WITH=(--with "torchvision==0.29.2")   # Apertus1p5ImageProcessor imports it (the LoRA job's first GPU run, #76)

case "$STAGE" in
  features) PARTS=(train dev) ;;
  rehearse) PARTS=(dev) ;;
  final)    PARTS=(final); : "${PASSPHRASE:?STAGE=final needs the PASSPHRASE job secret}" ;;
  encoders) PARTS=(train dev); SO400M=1 ;;
  *) echo "STAGE must be features, rehearse, final or encoders" >&2; exit 1 ;;
esac
SO400M="${SO400M:-}"; [[ "$SO400M" == "0" ]] && SO400M=""   # empty: no so400m step

WORK=/tmp/noidea
REPO="$WORK/repo"
export GALAXUS_DATA="$WORK/data"
SERVE="$WORK/serve"
log() { echo "[job $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$WORK" "$GALAXUS_DATA" "$SERVE"
log "stage=$STAGE ref=$REF model=$MODEL layers=$LAYERS pool=$POOL parts=${PARTS[*]}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "no GPU"

log "1/6 clone and install"
export GIT_TERMINAL_PROMPT=0   # a refused clone fails at once instead of waiting for a password
if [[ -n "${GH_TOKEN:-}" ]]; then
  # The repo is private until the hand-in: the read-only token goes in as an HTTP header, never in a URL,
  # so neither git's messages nor this log can show it.
  git -c "http.https://github.com/.extraHeader=Authorization: Basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')" \
    clone --quiet --depth 1 --branch "$REF" https://github.com/neelshah261/NoIdea-galaxus "$REPO"
else
  git clone --quiet --depth 1 --branch "$REF" https://github.com/neelshah261/NoIdea-galaxus "$REPO"
fi
cd "$REPO"
log "commit $(git rev-parse --short HEAD)"
uv sync --quiet --extra local --python 3.12

log "2/6 competition data from Kaggle (this disk only)"
kaggle_download() { uvx --quiet --from "kaggle==2.2.5" --with six kaggle competitions download -c "$COMPETITION" "$@"; }
if [[ "$STAGE" == "final" ]]; then
  kaggle_download -f final_encrypted.bin -p "$GALAXUS_DATA" --quiet
else
  kaggle_download -p "$GALAXUS_DATA" --quiet
fi
for archive in "$GALAXUS_DATA"/*.zip; do   # the whole set comes zipped, and a single large file may be too
  [[ -e "$archive" ]] || continue
  uv run --quiet python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$archive" "$GALAXUS_DATA"
  rm -f "$archive"
done
if [[ "$STAGE" == "final" ]]; then
  log "decrypting the final images (the passphrase is read from the job secret, never printed)"
  uv run --quiet --with py7zr python - "$GALAXUS_DATA" <<'PY'
import os
import sys
from pathlib import Path

import py7zr

data = Path(sys.argv[1])
archive = next(data.rglob("final_encrypted.bin"))
with py7zr.SevenZipFile(archive, password=os.environ["PASSPHRASE"]) as handle:
    handle.extractall(data / "final")
print(f"decrypted {sum(1 for path in (data / 'final').rglob('*') if path.is_file())} files", flush=True)
PY
fi

log "3/6 the 512 px image cache"
for part in "${PARTS[@]}"; do
  # --skip-missing: a missing (and, with #217, a corrupt) image is counted instead of stopping the job; extract then
  # records it as unreadable, and the laptop's run keeps its constant row (#216, #219).
  uv run --quiet python -m galaxus.data.images --part "$part" --workers 8 --skip-missing
done

extract() {  # extract <part> [limit]: Apertus hidden states into the heads' feature store, resumable
  local options=(--part "$1" --model "$MODEL" "--layers=$LAYERS" --pool "$POOL")
  [[ -n "${2:-}" ]] && options+=(--limit "$2")
  uv run --quiet "${WITH[@]}" python -m galaxus.model.heads extract "${options[@]}"
}
store_root() {
  uv run --quiet python -c "import sys; from galaxus.model import heads
print(heads.store_for(sys.argv[1], tuple(int(x) for x in sys.argv[2].split(',')), sys.argv[3]).root)" \
    "$MODEL" "$LAYERS" "$POOL"
}

if [[ "$STAGE" != "encoders" ]]; then   # Apertus: the check, then every part (not in STAGE=encoders)
log "4/6 check: the first $CHECK images of ${PARTS[0]}"
check_started=$SECONDS
extract "${PARTS[0]}" "$CHECK" 2>&1 | tee "$WORK/check.log"
check_seconds=$((SECONDS - check_started))
rate="$(grep -o '[0-9.]* img/s' "$WORK/check.log" | tail -1 | cut -d' ' -f1 || true)"
uv run --quiet python - "$(store_root)" "${PARTS[0]}" <<'PY'
import sys
from pathlib import Path

import numpy as np

from galaxus.model.heads import FeatureStore

features = FeatureStore(Path(sys.argv[1])).load(sys.argv[2])
bad = sum(1 for vector in features.values() if not np.isfinite(vector.astype(np.float32)).all())
print(f"[check] {len(features)} feature vectors, {bad} not finite, width {len(next(iter(features.values())))}",
      flush=True)
if not features or bad:
    sys.exit("[check] features are missing or not finite: stopping before the full run")
PY
log "check: $CHECK images in ${check_seconds}s including the model load; steady rate ${rate:-unknown} img/s"
if [[ "$STAGE" == "rehearse" ]]; then
  if [[ -n "$rate" ]]; then
    minutes="$(python3 -c "import sys; print(round(int(sys.argv[1]) / float(sys.argv[2]) / 60))" "$FINAL_IMAGES" "$rate")"
  else
    minutes="$((FINAL_IMAGES * check_seconds / CHECK / 60)) (upper bound, model load included)"
  fi
  log "projection for $FINAL_IMAGES final images: about $minutes min of extraction, after a setup of" \
      "$((check_started / 60)) min (clone, install, Kaggle data, image cache) and the model load"
else
  log "5/6 extract every part"
  for part in "${PARTS[@]}"; do
    extract "$part"
  done
fi
fi   # Apertus

if [[ -n "$SO400M" ]]; then
  log "5b/6 SigLIP 2 so400m embeddings (${HELPER_SO400M:-google/siglip2-so400m-patch16-512}, pinned)"
  so400m_started=$SECONDS
  for part in "${PARTS[@]}"; do
    if [[ "$STAGE" == "rehearse" ]]; then
      uv run --quiet python -m galaxus.model.encoders embed --part "$part" --limit "$CHECK"
    else
      uv run --quiet python -m galaxus.model.encoders embed --part "$part"
    fi
  done
  log "so400m: $((SECONDS - so400m_started))s for ${PARTS[*]}, the model load included"
fi

log "6/6 serving on port 8000 (HF-token protected): download the tar files, then cancel this job"
if [[ "$STAGE" != "encoders" ]]; then
  root="$(store_root)"
  tar -C "$(dirname "$root")" -cf "$SERVE/features.tar" "$(basename "$root")"
  log "features.tar: $(du -h "$SERVE/features.tar" | cut -f1)"
fi
if [[ -n "$SO400M" ]]; then
  folder="$(uv run --quiet python -c "from galaxus.model import encoders; from galaxus.data import neighbours
print(neighbours.embedding_dir(encoders.so400m_name()))")"
  tar -C "$(dirname "$folder")" -cf "$SERVE/embeddings.tar" "$(basename "$folder")"
  log "embeddings.tar: $(du -h "$SERVE/embeddings.tar" | cut -f1)"
fi
log "total time so far $((SECONDS / 60)) min"
exec python3 -m http.server 8000 --bind 0.0.0.0 --directory "$SERVE"
