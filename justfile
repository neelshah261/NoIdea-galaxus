# One command per step (B0, #11). Install `just` (https://github.com/casey/just), then `just` lists the recipes.
# Keys and $GALAXUS_DATA come from .env (git ignores it); `just` loads it when it exists. Nothing here uploads to
# Kaggle, posts anywhere or starts a paid GPU job by itself: the recipes marked HUMAN are run by a person
# (AGENTS.md product rule 7). Full walk-through: docs/how_to_run.md.

set dotenv-load := true

default:
    @just --list

# --- setup and checks (what ci runs) ------------------------------------------------------------------------------

# install the locked dependencies (add --extra local / figures when you need them)
setup:
    uv sync --locked

# lint, tests and the secret scan, as ci does
check: lint test secrets

lint:
    uv run ruff check .

test:
    uv run pytest -q

secrets:
    python3 .github/secret_scan.py --tree

# --- data --------------------------------------------------------------------------------------------------------

# prepared-image cache for one part: train, dev or final (512 px, the size every method uses)
prep part="dev" size="512":
    uv run python -m galaxus.data.prep --part {{part}} --size {{size}} --workers 8

# --- experiments: every number goes through the eval script into results/metrics.json ----------------------------

# score a registered predictor on the official holdout and append its EXP row
eval predictor exp:
    uv run python -m galaxus.model.eval {{predictor}} --exp-id {{exp}}

# Apertus smoke test against the gateway (Q-0001, Q-0002): needs the keys in .env
smoke backend="api":
    uv run python -m galaxus.model.apertus.smoke --backend {{backend}}

# per-dimension ensemble over saved runs (BUILD_PLAN.md phase 2 step 3, #87)
ensemble members="zeroshot=EXP-0002 zeroshot_noctx=EXP-0009":
    uv run python -m galaxus.model.ensemble choose --members {{members}}

# --- the Saturday path ------------------------------------------------------------------------------------------

# a validated dev file for one method, and the Kaggle command printed (HUMAN uploads)
submit exp predictor workers="2":
    uv run python -m galaxus.serve.submit --exp {{exp}} --run {{predictor}} --workers {{workers}}

# timed rehearsal on the dev images through the final-run code (no passphrase)
dry-run exp predictor best workers="2":
    uv run python -m galaxus.serve.final_run --dry-run --predictor {{predictor}} --exp {{exp}} --best-file {{best}} --workers {{workers}}

# dress rehearsal: decrypt a PRACTICE archive, kill and resume, merge, validate
drill exp predictor best archive workers="2":
    uv run python -m galaxus.serve.final_run --drill --predictor {{predictor}} --exp {{exp}} --best-file {{best}} --archive {{archive}} --workers {{workers}}

# HUMAN, Saturday after 13:00 CET: decrypt (passphrase asked, never stored), predict, merge, validate
final exp predictor best workers="2":
    uv run python -m galaxus.serve.final_run --predictor {{predictor}} --exp {{exp}} --best-file {{best}} --workers {{workers}}

# --- speed, insights, demo --------------------------------------------------------------------------------------

# speed matrix: the CPU cells on synthetic images, then the commands a HUMAN runs for the GPU cells
bench:
    uv run python -m bench.speed_matrix fake
    uv run python -m bench.speed_matrix plan

# perspective confusion tables and the slide 8 figure for one run
perspective exp:
    uv run --extra figures python bench/perspective_report.py --exp {{exp}}

# offline hero-shot gallery for one run's dev answers
gallery exp:
    uv run python -m galaxus.serve.demo --exp {{exp}} --part dev

# --- training (HUMAN: GPU money needs Neel's OK) ----------------------------------------------------------------

# LoRA: 5 steps to prove the model, processor and LoRA load (a minute or two on the GPU)
lora-check:
    uv run --extra local python train/lora_apertus.py --model swiss-ai/Apertus-v1.5-8B --max-steps 5

# --- report ------------------------------------------------------------------------------------------------------

# technical report: section 7 regenerated from results/metrics.json, then the PDF in the repo root
report exp="":
    uv run --extra report python scripts/build_report.py {{ if exp == "" { "" } else { "--exp " + exp } }}
