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

Galaxus labels every product image along six dimensions with a finetuned Qwen 3.5 9B. We build an Apertus v1.5
replacement that runs on hardware Galaxus can own. Apertus v1.5 8B reads the image, prepared exactly as the shop
shows it (transparency on white, 512 × 512 px of area), and answers all six dimensions in one JSON object held to
the allowed classes.

The plan (BUILD_PLAN.md) has three steps:

1. **Zero-shot Apertus**, measured: EXP-0002.
2. **A LoRA finetune of Apertus 8B** as the main candidate: EXP-0004, training script merged, GPU run pending.
3. **One innovation:** mirror test-time augmentation for perspective, the hosts' business priority.

A per-dimension ensemble picks the best method for each dimension, with the choice cross-fitted on the holdout.
**Headline so far:** zero-shot Apertus 8B reaches a mean macro F1 of 0.303 on our official holdout (EXP-0002),
against 0.187 for the majority baseline (EXP-0001). The final number is TODO until EXP-0004 is scored.

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

- **Model:** `swiss-ai/Apertus-v1.5-8B`; on the gateway, `apertus-v1-5-8b`. All models are listed in MODELS.md.

**Every model we used (rule 6), in development and at runtime; MODELS.md holds the same list:**

| Model | Version | Role | Used in |
| --- | --- | --- | --- |
| Apertus v1.5 8B | `apertus-v1-5-8b` (gateway); `swiss-ai/Apertus-v1.5-8B`, revision `1519de0a` in the Kaggle notebooks, the LoRA run's: TODO(swarupgaikwad1801-ux) | classifies the images: zero-shot, base of the LoRA, frozen features for the heads experiments | runtime and development |
| LoRA adapter on Apertus 8B | TODO(swarupgaikwad1801-ux): Hugging Face repo and revision | main candidate method | runtime, if chosen |
| Apertus v1.5 70B | `apertus-v1-5-70b` (gateway) | zero-shot comparison (EXP-0015) | development; runtime only if chosen |
| Linear heads on Apertus hidden states (ours) | trained in the notebooks | experiment, parked until Q-0007 is answered in writing | development |
| SigLIP base | `google/siglip-base-patch16-224`, revision TODO(neelshah261) | similar-image index for suspected label errors (#112); not run yet | development |
| EasyOCR, Lingua | TODO(swarupgaikwad1801-ux) | OCR and language detection experiment for `text_language` (#108) | development |
| Tiny random Llama, generated in a test | random weights | CPU smoke test of the LoRA training path | development |
| Claude Code (Anthropic), one session per lane | TODO(owner): each session's model, as declared on #116 | coding assistant: code, tests, docs, reviews; never classifies an image | development |
- **How it is used:** inference (zero-shot, EXP-0002 and EXP-0009), fine-tuning (LoRA, EXP-0004), evaluation.
- **Where it runs:** the Hack Apertus gateway for zero-shot; local weights for training, and vLLM for serving the
  LoRA.

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

- **Zero-shot collapses onto frequent classes.** In subject and perspective it predicts the most frequent class for
  almost every image (PR #70). Weighted F1 looks reasonable, but macro F1 stays low. This is why the LoRA is the main
  candidate.
- **The LoRA is untested on a GPU.** Its cost and time are estimates until the first 50 steps print the measured
  rate (#76).
- **Patch size is an assumption.** The vision patch size (16 px per token) is assumed from Swiss AI's serving
  script, not measured (Q-0002).
- **The demo's hero rules are mostly assumptions.** Only one rule is known from the hosts, and the others are marked
  as assumptions on the page (#31, #66).
- **The gateway's rate limit is unknown.** Concurrency is capped at the level our load test proved (#58, #59).
- **Heads are not used.** Heads on Apertus features are parked until the hosts confirm they count as Apertus
  classifying (Q-0007).

## 6. Reproducibility

- **Setup:** `uv sync --locked` (Python 3.10+, packages pinned in `uv.lock`). Optional extras: `local` (transformers 5.24 or newer for Apertus in-process, Python 3.11+) and `figures`. Every step is one `just` recipe; the full
  walk-through is in `docs/how_to_run.md`.
- **Data:** `just prep train`, `just prep dev`.
- **Scores:** `just eval <predictor> <EXP-id>` appends a row with the commit, `dirty` flag, seed, config and the
  holdout's hash. Rows from a dirty tree are not shown to judges.
- **Training:** `just lora-check`, then the full run in `train/lora_apertus.py` (a GPU, with the owner's OK).
- **Saturday:** `just dry-run`, `just drill`, `just final`, run by a person. The final submission's commit and
  weights revision are in section 7.
- **Hardware:**
    - development on Windows and macOS laptops (CPU);
    - zero-shot on the gateway;
    - GPU for the LoRA: TODO(owner), the GPU type of the run in #76.

<!-- section7:start (generated by scripts/build_report.py from galaxus.serve.report; do not edit by hand) -->
## 7. Galaxus challenge: results and resources

- **Kaggle:** team NoIdea; public score: TODO(owner) after the upload. The private score is Kaggle's.

- **Per dimension** (EXP-0101, official holdout of 2444 images, metric.py; `results/metrics.json`):

| Dimension | Macro F1 | Weighted F1 |
| --- | --- | --- |
| subject | 0.571 | 0.778 |
| background | 0.732 | 0.893 |
| perspective | 0.457 | 0.520 |
| individuals | 0.856 | 0.969 |
| text_classification | 0.611 | 0.885 |
| text_language | 0.607 | 0.765 |
| **mean** | **0.639** | **0.817** |

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

  What did not work: TODO(model): one line per EXP that did not help, and why.

- **Models** (the final method EXP-0101 and the rows it is built from: EXP-0098, EXP-0049):

  - Runtime: apertus-v1-5-70b; no helper models
  - Development only, never at runtime (source `MODELS.md`): tiny random Llama, generated in the test; Claude Code (Anthropic), design session (Neel); Claude Code (Anthropic), data lane session (Vernika); Claude Code (Anthropic), model lane sessions (Parth); Claude Code (Anthropic), serve lane session (Swarup); SigLIP base, `google/siglip-base-patch16-224` (apache-2.0, 0.2B), revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`; `gpt-6.2-astra` (closed); EasyOCR (Latin model) and Lingua (`lingua-language-detector`), versions TODO(swarupgaikwad1801-ux)

- **Resources** (source `$GALAXUS_DATA/out/bench/`, `galaxus.serve.bench`):

  - Hardware: CPU (arm); peak memory TODO(model): not measured GB (no GPU memory: CPU run, or torch not installed)
  - Batch size 16, workers 1: **3.09 images/s**
  - 100,000 images a day: 9.00 GPU hours, **$TODO(model): not measured** at $TODO(model): not measured/h

- **Commit and weights:** commit `3f340bf2e397`; weights: TODO(owner): Hugging Face repository and revision, published by 14:30.

- **Optional:** label errors: TODO(data): not generated yet (`python -m galaxus.model.heads label-errors`).
<!-- section7:end -->

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References

- Swiss AI, Apertus v1.5 model card: huggingface.co/swiss-ai/Apertus-v1.5-8B
- Digitec Galaxus, challenge description and reference notebook: challenge_description.md, reference/finetune_qwen35_reference.ipynb
- The team's plan and architecture: BUILD_PLAN.md, BASE_ARCHITECTURE.md, decisions.md
