# MODELS.md: every model we use, development or runtime

AGENTS.md product rule 4: at runtime only Apertus v1.5 and open-weights helpers; every model, dev or runtime, is
listed here with its version and role. The rules themselves are in
[challenge_description.md, "Models"](challenge_description.md#models). A model not in this table is not used.

## Runtime (what classifies the images)

| Model | Version / id | Role | Where it runs | Status | Source in the repo |
| --- | --- | --- | --- | --- | --- |
| Apertus v1.5 8B | `apertus-v1-5-8b` on the Hack Apertus (Keera / CSCS) gateway | zero-shot predictor: EXP-0002, EXP-0009 and the #106 sprint's variants (EXP-0006, EXP-0014 to EXP-0025, EXP-0038 to EXP-0044) | the gateway, OpenAI-compatible API | measured | `galaxus.model.zero_shot`, `galaxus.model.apertus.api`, PR #63, #70 |
| Apertus v1.5 8B | `swiss-ai/Apertus-v1.5-8B` (Hugging Face weights, gated); the Kaggle notebooks pin revision `1519de0ae7f3d114879291810a58259966f64c99`; the LoRA run's revision: TODO(swarupgaikwad1801-ux) | base model of the LoRA (EXP-0004), of the local backend and of the frozen features in `train/one_look_pilot.ipynb` and `train/self_training_heads.ipynb`; bf16, or FP8 quantized by vLLM at load with `QUANT=fp8` (#43) | our GPU (training, vLLM), Kaggle, or this machine (transformers) | LoRA run 3 on an HF Job (#76) | `train/lora_apertus.py`, `galaxus.model.lora`, `galaxus.model.apertus.local`, PR #71 |
| LoRA adapter on Apertus v1.5 8B | in a private Hugging Face repo, not published; TODO(swarupgaikwad1801-ux): repo and revision if it enters the final method | candidate method (BUILD_PLAN.md phase 2); not in EXP-0056 | merged bf16 weights served by vLLM | trained (run 3, #76); its holdout score, 0.343, is not yet in `results/metrics.json` (docs/method.md) | PR #71, #86 |
| Apertus v1.5 70B | `apertus-v1-5-70b` on the gateway (`swiss-ai/Apertus-v1.5-70B` on Hugging Face) | zero-shot EXP-0015; observe first, no product text, EXP-0049; with the product-type prior EXP-0051; **a member of EXP-0056** (with the SigLIP heads, per dimension); the teacher of #132's distillation code, which no training run has used | the gateway | measured; at runtime in EXP-0051 and EXP-0056 | `galaxus.model.zero_shot`, `galaxus.model.type_prior`, `galaxus.model.ensemble` |
| SigLIP base (helper image model) | `google/siglip-base-patch16-224`, revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` (apache-2.0, 0.2B; #170), pinned in code by #188 | its pooled embeddings feed our class-balanced heads, which pick the class (Q-0008, `decisions.md`, #178): EXP-0055, and **a member of EXP-0056** | CPU (Neel's laptop) | measured; at runtime in EXP-0055 and EXP-0056 | `galaxus.model.helper`, `galaxus.data.neighbours`, PR #174 |
| DINOv2 base (helper image model) | `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` (apache-2.0 on the Hugging Face repo; Meta's upstream `facebookresearch/dinov2` artifacts carry CC-BY-NC, #176), pinned in code by #196 | the second of `helper-fused`'s three encoders: its pooled embedding at unit length, concatenated with the other two (EXP-0059, a member of EXP-0061); alone, EXP-0062 (scratch row) | CPU (Neel's laptop) | measured; at runtime if EXP-0061 is the final method | `galaxus.model.helper` (`FUSED_ENCODERS`), PR #196 |
| SigLIP 2 base (helper image model) | `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` (apache-2.0), pinned in code by #196 | the third of `helper-fused`'s encoders (EXP-0059, a member of EXP-0061); alone, EXP-0063 (scratch row) | CPU (Neel's laptop) | measured; at runtime if EXP-0061 is the final method | `galaxus.model.helper` (`FUSED_ENCODERS`), PR #196 |
| Apertus v1.5 8B heads | linear heads on Apertus hidden states, trained by us | may be a runtime method: Q-0007(a) is answered yes (`decisions.md`, #171); trained in the Kaggle notebooks (#180) | CPU on saved features | merged (#51); no holdout row yet, not in EXP-0056 | `galaxus.model.heads`, `train/self_training_heads.ipynb` |

Besides Apertus, the runtime helpers are image encoders only (Q-0008): SigLIP base in EXP-0055/0056, and SigLIP
base, DINOv2 base and SigLIP 2 base in `helper-fused` (EXP-0059/0061). No OCR and no closed model runs at runtime.
EasyOCR and Lingua were measured as a development experiment (#108) and are not wired into any predictor. The
majority baseline (EXP-0001) is a count, and the product-type prior and the per-dimension routing are rules learned
from train, not models.

## The models in the final submission

The four models the submitted file ran through, and nothing else (AGENTS.md product rule 4; the rows above give
each one's revision and role). TODO(design): the final EXP id (EXP-0101 or EXP-0106, #243) once it is named.

| Model | Id | How it runs in the final |
| --- | --- | --- |
| Apertus v1.5 70B | `apertus-v1-5-70b` | the Hack Apertus gateway, or self-hosted with vLLM in FP8 on one H200 |
| SigLIP base | `google/siglip-base-patch16-224` | CPU, pooled embeddings into our class-balanced heads |
| DINOv2 base | `facebook/dinov2-base` | CPU, second of the three fused encoders |
| SigLIP 2 base | `google/siglip2-base-patch16-224` | CPU, third of the three fused encoders |

Apertus v1.5 performs the classification; the three encoders are open-weights image helpers, which is what the
model rules allow at runtime (challenge_description.md "Models"). No closed model and no OCR runs at runtime.

## Serving and tooling around the runtime model

| Component | Version | Role | Source |
| --- | --- | --- | --- |
| vLLM, Swiss AI's release image | `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-amd64` (TODO(owner): pin the digest at freeze) | serves Apertus / the merged LoRA on one GPU | `bench/vllm_apertus.sh` |
| transformers | `>=5.24` (the `local` extra in `pyproject.toml`) | Apertus in-process: smoke test, LoRA training | `uv.lock` |

## Development only (never at runtime)

| Model | Role | Where |
| --- | --- | --- |
| tiny random Llama, generated in the test | CPU smoke run of the LoRA training path (masking, LoRA, Trainer), no download | `train/lora_apertus.py --smoke`, `tests/model/test_model_lora.py` |
| Claude Code (Anthropic), design session (Neel) | coding assistant: opens issues, reviews PRs, writes the shared files; never classifies an image | model as declared on #116; TODO(neelshah261): copy it here |
| Claude Code (Anthropic), data lane session (Vernika) | coding assistant of `lane:data`; never classifies an image | model as declared on #116; TODO(neelshah261): copy it here |
| Claude Code (Anthropic), model lane sessions (Parth) | coding assistant of `lane:model`; never classifies an image | Claude Opus 5.5 (`claude-opus-5-5`) in the session that wrote #118, #182, #188 and #189; TODO(parthdeshmukh488-ops): the model of each other session, on #116 |
| Claude Code (Anthropic), serve lane session (Swarup) | coding assistant of `lane:serve`; never classifies an image | TODO(swarupgaikwad1801-ux): post the model on #116 |
| SigLIP base, `google/siglip-base-patch16-224` (apache-2.0, 0.2B), revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | besides its runtime role above: the similar-image index that finds suspected label errors (#112 part A), all 11,876 train rows embedded (#180) | `galaxus.data.neighbours`, PR #114, #170 |
| `gpt-6.2-astra` (closed) | a reference row and label-error candidates where it and our 70B agree against the human label (#180, #182); **never at runtime**; TODO(neelshah261): its EXP row, if any | Neel's laptop |
| EasyOCR (Latin model) and Lingua (`lingua-language-detector`), versions TODO(swarupgaikwad1801-ux) | OCR plus language detection for `text_language`, measured on the holdout (#108); not in any predictor | `bench/ocr_language.py`, their own environment (not in `uv.lock`) |

Allowed in development by challenge_description.md "Models" (any tools and models, including closed ones and
coding assistants). Not used so far: closed-model pseudo-labels on dev (BUILD_PLAN.md "Later"; the hosts' remark in
#17 is still an open question).
