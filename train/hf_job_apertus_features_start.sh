#!/usr/bin/env bash
# Start the Apertus feature job of #214 on Hugging Face. A human runs this: it starts paid GPU time (AGENTS.md rule 7).
#
#   bash train/hf_job_apertus_features_start.sh features --dry-run   # print the job, start nothing (free)
#   bash train/hf_job_apertus_features_start.sh features             # train (holdout included) + dev, ~1-1.5 h on l40sx1
#   bash train/hf_job_apertus_features_start.sh rehearse             # 300 dev images: the projected final-pass time
#   read -rs PASSPHRASE && export PASSPHRASE && bash train/hf_job_apertus_features_start.sh final   # at 13:00 only
#   HELPER_SO400M_REVISION=<sha> bash train/hf_job_apertus_features_start.sh encoders   # SigLIP 2 so400m only, ~15 min
#   bash train/hf_job_apertus_features_start.sh url <job id>         # once it serves: download, untar, cancel
#
# With HELPER_SO400M_REVISION set (from `python -m galaxus.model.encoders pin`), every mode also embeds its parts with
# SigLIP 2 so400m (HELPER_SO400M picks 512 or 384 px) and serves them as embeddings.tar.
#
# REF is the branch the job clones (default main; model/apertus-core until #214's PR is merged). FLAVOR=a100-large is
# faster than the default l40sx1 ($1.80/h) if `hf jobs hardware` lists it. The job's disk, with the competition data,
# is deleted when the job ends; the features reach the laptop only through the job's HF-token-protected port.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: bash train/hf_job_apertus_features_start.sh features|rehearse|final|encoders|url [--dry-run | <job id>]}"
HF=(uvx --quiet --from "huggingface_hub>=1.0" hf)

if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '\r')"
fi
: "${HF_TOKEN:?set HF_TOKEN in .env}"
export HF_TOKEN

if [[ "$MODE" == "url" ]]; then
  JOB="${2:?usage: bash train/hf_job_apertus_features_start.sh url <job id>}"
  uv run --quiet --no-project --with "huggingface_hub>=1.0" python - "$JOB" <<'PY'
import os
import sys

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
job_id = sys.argv[1].split("/")[-1]
job = api.inspect_job(job_id=job_id)
urls = getattr(job.status, "expose_urls", None) or []
print(f"# job {job_id}: {job.status.stage}")
if not urls:
    print("# no exposed URL yet: wait until the job log says 'serving on port 8000'")
for url in urls:
    base = url.rstrip("/")
    print("# Apertus features (not in the encoders mode):")
    print(f'curl -fL -H "Authorization: Bearer $HF_TOKEN" "{base}/features.tar" -o "$GALAXUS_DATA/out/features.tar"')
    print('mkdir -p "$GALAXUS_DATA/cache/features" && tar -C "$GALAXUS_DATA/cache/features" '
          '-xf "$GALAXUS_DATA/out/features.tar"')
    print("# SigLIP 2 so400m embeddings (only if the job had HELPER_SO400M_REVISION):")
    print(f'curl -fL -H "Authorization: Bearer $HF_TOKEN" "{base}/embeddings.tar" -o "$GALAXUS_DATA/out/embeddings.tar"')
    print('mkdir -p "$GALAXUS_DATA/cache/embeddings" && tar -C "$GALAXUS_DATA/cache/embeddings" '
          '-xf "$GALAXUS_DATA/out/embeddings.tar"')
print(f"hf jobs cancel {job_id}      # an idle GPU costs as much as a busy one")
PY
  exit 0
fi

DRY=()
[[ "${2:-}" == "--dry-run" ]] && DRY=(--dry-run)
KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(tr -d '\r\n' < "$HOME/.kaggle/access_token" 2>/dev/null || true)}"
: "${KAGGLE_API_TOKEN:?no Kaggle token in ~/.kaggle/access_token}"
export KAGGLE_API_TOKEN
REF="${REF:-main}"
FLAVOR="${FLAVOR:-l40sx1}"
SECRETS=(--secrets HF_TOKEN --secrets KAGGLE_API_TOKEN)
ENV=(-e STAGE="$MODE" -e REF="$REF")
# The so400m pin is a public commit sha, not a secret: it goes in as a plain variable.
[[ -n "${HELPER_SO400M_REVISION:-}" ]] && ENV+=(-e HELPER_SO400M_REVISION="$HELPER_SO400M_REVISION")
[[ -n "${HELPER_SO400M:-}" ]] && ENV+=(-e HELPER_SO400M="$HELPER_SO400M")

case "$MODE" in
  features) TIMEOUT="${TIMEOUT:-150m}" ;;   # 2.5 h x $1.80 = $4.50 at most on l40sx1; cancel after the download
  rehearse) TIMEOUT="${TIMEOUT:-45m}" ;;
  encoders)
    TIMEOUT="${TIMEOUT:-45m}"
    : "${HELPER_SO400M_REVISION:?set HELPER_SO400M_REVISION first: uv run --extra local python -m galaxus.model.encoders pin}"
    ;;
  final)
    TIMEOUT="${TIMEOUT:-75m}"
    if [[ ${#DRY[@]} -eq 0 ]]; then
      : "${PASSPHRASE:?type it first, never into a file: read -rs PASSPHRASE && export PASSPHRASE}"
      export PASSPHRASE
    fi
    SECRETS+=(--secrets PASSPHRASE)
    ;;
  *) echo "mode must be features, rehearse, final, encoders or url" >&2; exit 1 ;;
esac

echo "job: $MODE on $FLAVOR, timeout $TIMEOUT, ref $REF; the features are served on port 8000 behind the HF jobs proxy"
# The script goes with Unix line endings: a Windows checkout adds \r, which bash in the job rejects (#76).
exec "${HF[@]}" jobs run "${DRY[@]}" --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --name "noidea-apertus-$MODE" \
  --expose 8000 "${SECRETS[@]}" "${ENV[@]}" \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "$(tr -d '\r' < train/hf_job_apertus_features.sh)"
