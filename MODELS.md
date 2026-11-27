# MODELS.md: every model we use, development or runtime

AGENTS.md product rule 4: at runtime only Apertus v1.5 and open-weights helpers; every model, dev or runtime, is
listed here with its version and role. The rules themselves are in
[challenge_description.md, "Models"](challenge_description.md#models). A model not in this table is not used.

## Runtime (what classifies the images)

| Model | Version / id | Role | Where it runs | Status | Source in the repo |
| --- | --- | --- | --- | --- | --- |
| Apertus v1.5 8B | `apertus-v1-5-8b` on the Hack Apertus (Keera / CSCS) gateway | zero-shot predictor, EXP-0002 and EXP-0009 | the gateway, OpenAI-compatible API | measured | `galaxus.model.zero_shot`, `galaxus.model.apertus.api`, PR #63, #70 |
| Apertus v1.5 8B | `swiss-ai/Apertus-v1.5-8B` (Hugging Face weights, gated) | base model of the LoRA (EXP-0004) and of the local backend | our GPU (training, vLLM) or this machine (transformers) | script merged, run pending (#76) | `train/lora_apertus.py`, `galaxus.model.lora`, `galaxus.model.apertus.local`, PR #71 |
| LoRA adapter on Apertus v1.5 8B | TODO(owner): Hugging Face repo and revision, once trained and published | main candidate method (BUILD_PLAN.md phase 2) | merged bf16 weights served by vLLM | not trained yet (#76) | PR #71, #86 |
| Apertus v1.5 8B heads | linear heads on Apertus hidden states | BUILD_PLAN.md "Bet", **not a runtime method** until the hosts answer Q-0007 | CPU on saved features | parked, PR #51 | `decisions.md` Q-0007 |

No other model runs at runtime: no image encoder, no OCR and no closed model. The majority baseline (EXP-0001) is
a count, not a model.

## Serving and tooling around the runtime model

| Component | Version | Role | Source |
| --- | --- | --- | --- |
| vLLM, Swiss AI's release image | `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-amd64` (TODO(owner): pin the digest at freeze) | serves Apertus / the merged LoRA on one GPU | `bench/vllm_apertus.sh` |
| transformers | `>=5.24` (the `local` extra in `pyproject.toml`) | Apertus in-process: smoke test, LoRA training | `uv.lock` |

## Development only (never at runtime)

| Model | Role | Where |
| --- | --- | --- |
| tiny random Llama, generated in the test | CPU smoke run of the LoRA training path (masking, LoRA, Trainer), no download | `train/lora_apertus.py --smoke`, `tests/model/test_model_lora.py` |
| Claude Code (Anthropic), the coding assistant of every lane's agent | writes code, tests and docs; reviews PRs. Never classifies an image and never sees competition data in a prompt | TODO(owner): the model version of each team member's sessions |

Allowed in development by challenge_description.md "Models" (any tools and models, including closed ones and
coding assistants). Not used so far: closed-model pseudo-labels on dev (BUILD_PLAN.md "Later"; the hosts' remark in
#17 is still an open question).
