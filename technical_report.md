# Technical report: Apertus product-image classifier, NoIdea

A deeper write-up than the README: what we built, how it works, and what the numbers say. Every number in this
report comes from `results/metrics.json` (written only by the eval script) or names its source; anything not
measured yet is marked `TODO`. Rebuild the PDF with `just report` (`scripts/build_report.py`).

- **Challenge partner:** Digitec Galaxus
- **Challenge:** Hack Apertus x Galaxus: product image classification
- **Event:** On-site Zurich
- **Team:** NoIdea (also the Kaggle team name, issue #25). Members: Neel Shah (`neelshah261`), Parth Deshmukh (`parthdeshmukh488-ops`), Swarup (`swarupgaikwad1801-ux`), Vernika (`vernikas-ctrl`)
- **Demo:** TODO(owner): link to the 2-minute video (shot list in `presentation/demo_script.md`)

## 1. Summary

**Final submission: EXP-0106** (holdout macro F1 0.784, cross-fitted on the official 2,444-image holdout; Kaggle
public: 0.77826). Four members, combined per dimension by `galaxus.model.ensemble` (halves-agree rule, chosen on
the holdout):

- Apertus v1.5 70B zero-shot with similar-training-image label hints (EXP-0098, the base member);
- a SigLIP base linear-head helper (EXP-0055);
- a fused SigLIP base + DINOv2 base + SigLIP 2 base head helper (EXP-0059);
- plain Apertus 70B (EXP-0049).

The routing file says which member answers each dimension. The Apertus-only version is EXP-0101: 0.639 holdout,
0.66376 public.

**How we got here** (holdout macro F1, `results/metrics.json`): zero-shot Apertus 8B 0.303 (EXP-0002) against 0.187
for the majority baseline (EXP-0001); the 70B with the observe-first prompt and no product text 0.461 (EXP-0049); a
LoRA on Apertus 8B 0.343 (EXP-0004: one epoch, product text in every example and no class weights, so it learned the
catalog text and ignored the image, #145); heads on Apertus 8B hidden states with the 70B 0.540 to 0.550 (EXP-0095,
EXP-0100); the 70B with retrieval hints from similar training images, Apertus deciding, 0.639 (EXP-0101).

## 2. Architecture

One contract (`galaxus.types`: `ImageItem` → `Predictor.predict` → `Prediction`) and one runner for the holdout,
dev and final images. A new method changes one box and nothing else (diagram: `docs/architecture.svg`;
BASE_ARCHITECTURE.md). The pipeline:

1. **Load** the competition tables from `$GALAXUS_DATA`, outside git.
2. **Prepare** each image once into a cache: white composite, 512 px, JPEG.
3. **The runner** batches items through any predictor (majority, zero-shot, LoRA, ensemble). It resumes after a
   crash without duplicates and times every run.
4. **Predictions** are one JSON line per image.
5. **Scoring:** the eval script scores the official holdout through Kaggle's `metric.py` and appends an EXP row to
   `results/metrics.json`. Our scored-cell mask matches the written rules with 0 differences on the real data (#79).
6. **Submission:** the writer and validator turn dev and final predictions into a Kaggle file, which a person
   uploads.

The Saturday script decrypts the final images (the passphrase is asked for, never stored), predicts, merges into
the best dev file and validates. It has been rehearsed on the real dev data with a practice archive, including a
kill and resume halfway (#79).

### Target architecture (mandatory)

- **a) On-premise and b) air-gapped: the production path.**
    - The LoRA's merged bf16 weights are served by vLLM from Swiss AI's release image on one GPU
    (`bench/vllm_apertus.sh`).
    - At runtime there is no external call: no closed model, no API, no helper model (MODELS.md).
    - The weights, the image and `uv.lock` are copied in once.
- **c) Sovereign Swiss cloud: our development path.** Zero-shot runs through the Hack Apertus gateway, operated
  for the hackathon by CSCS (Keera gateway), over HTTPS.

| Dependency | Build time | Runtime (production path) |
| --- | --- | --- |
| Apertus v1.5 8B weights (Hugging Face, gated) | download once | local files |
| vLLM image `ghcr.io/swiss-ai/vllm_apertus_1.5_release` | pull once | local container |
| Python packages (`uv.lock`, pinned) | `uv sync --locked` | installed |
| Competition data (Kaggle) | download once into `$GALAXUS_DATA` | local files |
| Hack Apertus gateway | development only | none |

## 3. Use of Apertus

- **Models:** Apertus v1.5 70B (`apertus-v1-5-70b` on the gateway; `swiss-ai/Apertus-v1.5-70B` self-hosted) and
  Apertus v1.5 8B (`swiss-ai/Apertus-v1.5-8B`). All models are listed in MODELS.md.

**Every model we used (rule 6), in development and at runtime; MODELS.md holds the same list:**

| Model | Version | Role | Used in |
| --- | --- | --- | --- |
| Apertus v1.5 70B | `apertus-v1-5-70b` on the Keera gateway, and `swiss-ai/Apertus-v1.5-70B` with vLLM online FP8 on one H200 (HF Jobs, `bench/hf_job_serve.sh` with `SERVE_MODE=base QUANT=fp8`, served as `apertus-70b-fp8`) | classifies the images: zero-shot with the observe-first prompt (EXP-0049) and with similar-image hints (EXP-0098); the base member of EXP-0106 and the whole of EXP-0101 | runtime |
| Apertus v1.5 8B | `apertus-v1-5-8b` (gateway); `swiss-ai/Apertus-v1.5-8B`, revision `1519de0a` in the Kaggle notebooks | zero-shot (EXP-0002), base of the LoRA (EXP-0004), hidden states for the heads (EXP-0092, EXP-0097) | development; runtime in EXP-0095 and EXP-0100 |
| SigLIP base | `google/siglip-base-patch16-224`, revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | the similar-image hint index (training images only, k=5) and the SigLIP helper (EXP-0055) | runtime |
| DINOv2 base | `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | in the fused helper (EXP-0059) | runtime |
| SigLIP 2 base | `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | in the fused helper (EXP-0059) | runtime |
| LoRA adapter on Apertus 8B | private Hugging Face repo, not published (EXP-0004, 0.343) | a measured candidate; not in the final method | development |
| Linear heads on Apertus 8B hidden states (ours) | fitted by `galaxus.model.heads` / `apertus_core` | EXP-0092 (last token, 0.466) and EXP-0097 (fields, 0.463); members of EXP-0095 and EXP-0100 | development; runtime in EXP-0100 |
| `gpt-6.2-astra` (closed) | via its API, development only | reference row (0.858 on 1,000 holdout images) and label-error candidates; never at runtime | development |
| EasyOCR, Lingua | TODO(swarupgaikwad1801-ux): versions | OCR and language detection experiment for `text_language` (#108) | development |
| Tiny random Llama, generated in a test | random weights | CPU smoke test of the LoRA training path | development |
| Claude Code (Anthropic), one session per lane | Swarup's serve-lane session: Claude Opus 5.5 (`claude-opus-5-5`); the others: TODO(owner), as declared on #116 | coding assistant: code, tests, docs, reviews; never classifies an image | development |

- **Which dimensions the helpers decide** (`design/final_submission/routing-EXP-0106.json`): the fused helper answers perspective and the SigLIP helper answers individuals; subject, background, text_classification and text_language are a weighted vote of the four members (the hinted 70B, the plain 70B and both helpers).
- **How it is used:** inference (zero-shot with and without hints), heads on hidden states, a LoRA fine-tune
  (EXP-0004), evaluation.
- **Where it runs:** the Hack Apertus gateway and vLLM FP8 on one H200 for the 70B; the encoders and heads on a
  laptop CPU.

**Prompting (zero-shot):**

- **Prompt:** a short instruction plus `labeling_guideline.md`, read from `$GALAXUS_DATA` at runtime (never
  committed; only its hash is recorded), then the product context as `name: value` lines.
- **Answer:** held to a JSON schema with one enum of allowed classes per dimension (the gateway enforces
  `response_format`, with 0 defaulted cells in EXP-0002).
- **Settings:** temperature 0, at most 200 output tokens.

**LoRA (`train/lora_apertus.py`):**

- **Adapter:** rank 32 and alpha 32 on the language model's linear layers; the vision tokenizer is frozen in fp32.
- **Training:** learning rate 2e-4, cosine schedule, 5% warmup, effective batch 16, 2 epochs.
- **Loss:** on human labels only; model prelabels are masked.
- **Data:** rare classes repeated, the mirrored training rows (left and right swapped for text-free images), and the
  official holdout kept out.
- **Serving:** the merged weights are served by vLLM; the answers are checked against the classes.

## 4. Data

- **What:** the competition files from Kaggle (train with labels, dev, encrypted final), under the competition rules
  and for this hackathon only.
- **Where:** they live only in `$GALAXUS_DATA`. No image, csv, label or the passphrase is in the repository or in
  published weights, and ci blocks data files and secrets.
- **Personal data:** product photos; some show people (the `individuals` dimension). No other personal data is used.
- **Labels:** about a fifth of the train cells are unchecked model prelabels (FAQ.md); training uses human labels
  only.
- **Holdout:** our fixed holdout is the split in challenge_description.md "Scoring locally" (seed 0, grouped by
  product): 2,444 images, kept out of every training run.

## 5. Limitations

- **The helpers choose the class on some dimensions.** The organizer's ruling was that Apertus makes the final call;
  EXP-0101 is the Apertus-only version (0.639 holdout, 0.66376 public).
- **Perspective is the weakest dimension.**
- **Two serving paths for one member.** The final rows of EXP-0049 came from FP8 on an H200, while its holdout rows
  came from the gateway. The two agree on 88% of 96 holdout images.
- **Zero-shot collapses onto frequent classes** in subject and perspective without help (PR #70): weighted F1 looks
  reasonable, macro F1 stays low.
- **Our Apertus hidden-state heads read the last token only** (EXP-0092, 0.466). A public model card of another team
  reports 0.77 for frozen Apertus 8B heads on the last token plus the mean over the image tokens with an MLP head
  (#237); we did not get to extract that pooling.
- **The LoRA (EXP-0004) underfit:** one epoch, product text in every example and no class weights (#145).

## 6. Reproducibility

- **Setup:** `uv sync --locked` (Python 3.10+, packages pinned in `uv.lock`). Optional extras: `local` (transformers 5.24 or newer for Apertus in-process, Python 3.11+) and `figures`. Every step is one `just` recipe; the full
  walk-through is in `docs/how_to_run.md`.
- **Data:** `just prep train`, `just prep dev`.
- **Scores:** `just eval <predictor> <EXP-id>` appends a row with the commit, `dirty` flag, seed, config and the
  holdout's hash. Rows from a dirty tree are not shown to judges.
- **The commands that produced the final rows:**

```bash
python -m galaxus.model.hints write --part final
ZERO_SHOT_PRESET=zero-shot-hints ZERO_SHOT_HINTS=$GALAXUS_DATA/out/hints/final.jsonl LLM_NAME=apertus-v1-5-70b \
  python -m galaxus.serve.submit --exp EXP-0098 --part final --run galaxus.model.zero_shot:from_preset --batch-size 256 --workers 2
ZERO_SHOT_PRESET=zero-shot-observe-first-no-context ZERO_SHOT_BACKEND=vllm ZERO_SHOT_SERVED_MODEL=apertus-70b-fp8 \
  python -m galaxus.serve.submit --exp EXP-0049 --part final --run galaxus.model.zero_shot:from_preset --batch-size 256 --workers 8
python -m galaxus.serve.submit --exp EXP-0055 --part final --run galaxus.model.helper:from_env --batch-size 256
python -m galaxus.serve.submit --exp EXP-0059 --part final --run galaxus.model.helper:fused_from_env --batch-size 256
python -m galaxus.model.ensemble choose --members hinted=EXP-0098 siglip=EXP-0055 fused=EXP-0059 70b=EXP-0049 --rule agree --out routing-EXP-0106.json
python -m galaxus.model.ensemble apply --routing routing-EXP-0106.json --part dev --exp EXP-0106   # then --part final
python -m galaxus.serve.submit --exp EXP-0106 --part dev --part final
```

- **Training:** `just lora-check`, then the full run in `train/lora_apertus.py` (a GPU, with the owner's OK).
- **Saturday:** `just dry-run`, `just drill`, `just final`, run by a person. The final submission's commit and
  weights revision are in section 7.
- **Hardware:**
    - development on Windows and macOS laptops (CPU);
    - zero-shot on the gateway, and the 70B's final rows with vLLM FP8 on one H200 (HF Jobs);
    - the LoRA (EXP-0004) on one Nvidia L40S (HF Jobs, #76).

<!-- section7:start (generated by scripts/build_report.py from galaxus.serve.report; do not edit by hand) -->
## 7. Galaxus challenge: results and resources

- **Kaggle:** team NoIdea; public score (dev) **0.77826**. The private score is Kaggle's.

- **Per dimension** (EXP-0106, official holdout of 2444 images, metric.py; `results/metrics.json`):

| Dimension | Macro F1 | Weighted F1 |
| --- | --- | --- |
| subject | 0.741 | 0.860 |
| background | 0.880 | 0.955 |
| perspective | 0.647 | 0.714 |
| individuals | 0.973 | 0.993 |
| text_classification | 0.716 | 0.922 |
| text_language | 0.749 | 0.836 |
| **mean** | **0.784** | **0.891** |

  The bar to beat is the hosts' baselines: [challenge_description.md, Baselines](challenge_description.md#baselines).

- **What mattered** (every EXP row, holdout macro F1; source `results/metrics.json`):

| EXP | Predictor | Mean | subject | background | perspective | individuals | text_classification | text_language | img/s | $ / 100k | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0001 | majority | 0.187 | 0.056 | 0.304 | 0.049 | 0.484 | 0.152 | 0.076 | 1805822.4 | TODO(model): not measured | baseline, no model: images/s is not a product figure |
| EXP-0002 | zero-shot | 0.303 | 0.056 | 0.458 | 0.097 | 0.529 | 0.253 | 0.427 | 6.4 | TODO(model): not measured | zero-shot Apertus 8B via Keera, 512 px, with product context |
| EXP-0009 | zero-shot-no-context | 0.277 | 0.168 | 0.400 | 0.101 | 0.521 | 0.226 | 0.247 | 6.6 | TODO(model): not measured | EXP-0002 A/B: zero-shot Apertus 8B via Keera, 512 px, no product context |
| EXP-0015 | zero-shot | 0.327 | 0.059 | 0.447 | 0.111 | 0.734 | 0.290 | 0.322 | 2.0 | TODO(model): not measured | zero-shot Apertus 70B via Keera, 512 px, with product context |
| EXP-0014 | zero-shot-image-first | 0.295 | 0.062 | 0.445 | 0.095 | 0.514 | 0.249 | 0.407 | 5.7 | TODO(model): not measured |  |
| EXP-0006 | mirror-tta:zero-shot | 0.306 | 0.058 | 0.460 | 0.112 | 0.529 | 0.251 | 0.425 | 2.8 | TODO(model): not measured |  |
| EXP-0017 | zero-shot-768 | 0.312 | 0.068 | 0.473 | 0.099 | 0.522 | 0.228 | 0.483 | 3.5 | TODO(model): not measured |  |
| EXP-0018 | zero-shot-no-title | 0.296 | 0.060 | 0.453 | 0.097 | 0.522 | 0.254 | 0.389 | 6.5 | TODO(model): not measured |  |
| EXP-0019 | zero-shot-macro-aware | 0.306 | 0.059 | 0.471 | 0.100 | 0.529 | 0.246 | 0.430 | 4.7 | TODO(model): not measured |  |
| EXP-0021 | zero-shot-observe-first | 0.343 | 0.164 | 0.422 | 0.219 | 0.514 | 0.330 | 0.411 | 4.6 | TODO(model): not measured |  |
| EXP-0027 | type-prior | 0.388 | 0.193 | 0.429 | 0.277 | 0.642 | 0.370 | 0.416 | 4.6 | TODO(model): not measured |  |
| EXP-0024 | zero-shot-observe-first-768 | 0.341 | 0.117 | 0.398 | 0.168 | 0.522 | 0.321 | 0.519 | 2.1 | TODO(model): not measured |  |
| EXP-0025 | zero-shot-observe-first-no-context | 0.360 | 0.233 | 0.393 | 0.218 | 0.514 | 0.302 | 0.500 | 4.6 | TODO(model): not measured |  |
| EXP-0029 | type-prior | 0.393 | 0.235 | 0.394 | 0.278 | 0.642 | 0.336 | 0.471 | 4.6 | TODO(model): not measured |  |
| EXP-0043 | ensemble-agree | 0.365 | 0.233 | 0.414 | 0.218 | 0.506 | 0.318 | 0.500 | 2.3 | TODO(model): not measured |  |
| EXP-0038 | zero-shot-per-dimension | 0.270 | 0.059 | 0.451 | 0.109 | 0.522 | 0.223 | 0.259 | 2.9 | TODO(model): not measured |  |
| EXP-0039 | zero-shot-per-dimension-observe-first | 0.334 | 0.169 | 0.392 | 0.221 | 0.550 | 0.293 | 0.378 | 1.5 | TODO(model): not measured |  |
| EXP-0041 | zero-shot-per-dimension-observe-first-no-context | 0.380 | 0.248 | 0.422 | 0.224 | 0.529 | 0.298 | 0.559 | 1.6 | TODO(model): not measured |  |
| EXP-0044 | ensemble-agree | 0.368 | 0.233 | 0.422 | 0.207 | 0.514 | 0.330 | 0.500 | 2.3 | TODO(model): not measured |  |
| EXP-0045 | type-prior | 0.403 | 0.235 | 0.429 | 0.268 | 0.642 | 0.370 | 0.471 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0046 | type-prior | 0.416 | 0.255 | 0.419 | 0.294 | 0.632 | 0.333 | 0.564 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0047 | recalibrated | 0.383 | 0.254 | 0.429 | 0.253 | 0.529 | 0.321 | 0.511 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0048 | type-prior | 0.416 | 0.251 | 0.423 | 0.312 | 0.632 | 0.361 | 0.517 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0049 | zero-shot-observe-first-no-context | 0.461 | 0.253 | 0.483 | 0.268 | 0.721 | 0.389 | 0.655 | 2.4 | TODO(model): not measured |  |
| EXP-0050 | type-prior | 0.431 | 0.270 | 0.444 | 0.324 | 0.648 | 0.334 | 0.564 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0051 | type-prior | 0.489 | 0.303 | 0.494 | 0.324 | 0.762 | 0.415 | 0.637 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0052 | type-prior | 0.400 | 0.235 | 0.411 | 0.306 | 0.642 | 0.334 | 0.471 | 4.6 | TODO(model): not measured |  |
| EXP-0055 | helper-siglip | 0.705 | 0.702 | 0.878 | 0.520 | 0.977 | 0.667 | 0.486 | 30.3 | TODO(model): not measured |  |
| EXP-0056 | ensemble-agree | 0.733 | 0.702 | 0.878 | 0.520 | 0.977 | 0.666 | 0.656 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0057 | type-prior | 0.729 | 0.703 | 0.878 | 0.521 | 0.977 | 0.627 | 0.668 | TODO(model): not measured | TODO(model): not measured |  |
| EXP-0004 | lora | 0.343 | 0.197 | 0.319 | 0.324 | 0.506 | 0.271 | 0.441 | 3.4 | 14.74 | #76 LoRA on Apertus 8B, 1 epoch (exp0004-e1/adapter), merged at the file level, vLLM on 1x L40S via HF Jobs, 512 px, class-enum schema |
| EXP-0092 | apertus-core | 0.466 | 0.414 | 0.510 | 0.340 | 0.756 | 0.412 | 0.365 | 38308.4 | TODO(model): not measured | #219: heads on Apertus 8B hidden states, layers -1,-9, last token, no base |
| EXP-0095 | ensemble-agree | 0.540 | 0.433 | 0.529 | 0.384 | 0.779 | 0.463 | 0.652 | 2.4 | TODO(model): not measured | #214: Apertus-only: heads on Apertus hidden states + 70B, agree |
| EXP-0097 | apertus-core | 0.463 | 0.399 | 0.566 | 0.342 | 0.687 | 0.392 | 0.389 | 1721.1 | TODO(model): not measured | #233: heads on Apertus 8B fields features (a cue per dimension), layers -1,-17, no base |
| EXP-0098 | zero-shot-hints | 0.630 | 0.571 | 0.739 | 0.456 | 0.856 | 0.611 | 0.544 | 0.8 | TODO(model): not measured | #240: EXP-0049's prompt + neighbour-label hints (k=5, SigLIP base index of training images only) |
| EXP-0099 | ensemble-agree | 0.620 | 0.572 | 0.734 | 0.456 | 0.797 | 0.611 | 0.551 | TODO(model): not measured | TODO(model): not measured | #237: ensemble-agree core=EXP-0092 70b=EXP-0098 (hinted 70B) |
| EXP-0100 | ensemble-agree | 0.550 | 0.464 | 0.628 | 0.378 | 0.827 | 0.501 | 0.499 | TODO(model): not measured | TODO(model): not measured | #237: ensemble-agree core=EXP-0092 fields=EXP-0097 70b=EXP-0049, three Apertus members (vote) |
| EXP-0101 | ensemble-agree | 0.639 | 0.571 | 0.732 | 0.457 | 0.856 | 0.611 | 0.607 | TODO(model): not measured | TODO(model): not measured | #237: ensemble-agree hinted=EXP-0098 70b=EXP-0049, the hinted and the plain 70B |
| EXP-0102 | ensemble-agree | 0.631 | 0.571 | 0.739 | 0.456 | 0.856 | 0.617 | 0.544 | TODO(model): not measured | TODO(model): not measured | #237: ensemble-agree hinted=EXP-0098 70b=EXP-0049 core=EXP-0092 fields=EXP-0097, four Apertus members |
| EXP-0103 | apertus-core | 0.450 | 0.379 | 0.634 | 0.302 | 0.711 | 0.382 | 0.290 | 24711.9 | TODO(model): not measured | #237: heads on Apertus 8B mean-pooled features (Swarup's note), layers -1,-9, no base |
| EXP-0104 | ensemble-agree | 0.630 | 0.571 | 0.739 | 0.456 | 0.856 | 0.611 | 0.544 | TODO(model): not measured | TODO(model): not measured | #237: ensemble-agree hinted=EXP-0098 + 70b=EXP-0049 + mean=EXP-0103 (mean-pooled 8B heads) |
| EXP-0105 | ensemble-agree | 0.726 | 0.702 | 0.878 | 0.498 | 0.977 | 0.645 | 0.659 | 0.6 | TODO(model): not measured | Neel 14:05: Apertus 70B hinted as the base, SigLIP/DINOv2 helpers per dimension: hinted=EXP-0098 siglip=EXP-0055 70b=EXP-0049 |
| EXP-0106 | ensemble-agree | 0.784 | 0.741 | 0.880 | 0.647 | 0.973 | 0.716 | 0.749 | TODO(model): not measured | TODO(model): not measured | Neel 14:05: Apertus 70B hinted as the base, SigLIP/DINOv2 helpers per dimension: hinted=EXP-0098 siglip=EXP-0055 fused=EXP-0059 70b=EXP-0049 |
| EXP-0070 | helper-siglip-text | 0.710 | 0.723 | 0.875 | 0.503 | 0.977 | 0.693 | 0.492 | 73.4 | TODO(model): not measured |  |
| EXP-0072 | helper-siglip | 0.705 | 0.702 | 0.878 | 0.522 | 0.977 | 0.667 | 0.486 | 2074.7 | TODO(model): not measured |  |

  What did not work: TODO(model): one line per EXP that did not help, and why.

- **Models** (the final method EXP-0106 and the rows it is built from: EXP-0098, EXP-0055, EXP-0049):

  - Runtime: apertus-v1-5-70b, apertus-v1-5-70b, google/siglip-base-patch16-224, google/siglip-base-patch16-224; helpers: google/siglip-base-patch16-224
  - Development only, never at runtime (source `MODELS.md`): tiny random Llama, generated in the test; Claude Code (Anthropic), design session (Neel); Claude Code (Anthropic), data lane session (Vernika); Claude Code (Anthropic), model lane sessions (Parth); Claude Code (Anthropic), serve lane session (Swarup); SigLIP base, `google/siglip-base-patch16-224` (apache-2.0, 0.2B), revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`; `gpt-6.2-astra` (closed); EasyOCR (Latin model) and Lingua (`lingua-language-detector`), versions TODO(swarupgaikwad1801-ux)

- **Resources** (source `$GALAXUS_DATA/out/bench/`, `galaxus.serve.bench`):

  - Hardware: CPU (arm); peak memory TODO(model): not measured GB (no GPU memory: CPU run, or torch not installed)
  - Batch size 16, workers 1: **3.09 images/s**
  - 100,000 images a day: 9.00 GPU hours, **$TODO(model): not measured** at $TODO(model): not measured/h

- **Commit and weights:** commit `7823aaff353a`; weights: TODO(owner): Hugging Face repository and revision, published by 14:30.

- **Optional:** label errors: TODO(data): not generated yet (`python -m galaxus.model.heads label-errors`).
<!-- section7:end -->

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References

- Swiss AI, Apertus v1.5 model card: huggingface.co/swiss-ai/Apertus-v1.5-8B
- Digitec Galaxus, challenge description and reference notebook: challenge_description.md, reference/finetune_qwen35_reference.ipynb
- The team's plan and architecture: BUILD_PLAN.md, BASE_ARCHITECTURE.md, decisions.md
