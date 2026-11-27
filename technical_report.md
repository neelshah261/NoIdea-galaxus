# Technical report: Apertus product-image classifier, NoIdead

A deeper write-up than the README: what we built, how it works, and what the numbers say. Every number in this
report comes from `results/metrics.json` (written only by the eval script) or names its source; anything not
measured yet is marked `TODO`. Rebuild the PDF with `just report` (`scripts/build_report.py`).

- **Challenge partner:** Digitec Galaxus
- **Challenge:** Hack Apertus x Galaxus: product image classification
- **Event:** On-site Zurich
- **Team:** NoIdead (also the Kaggle team name, issue #25). Members: Neel Shah (`neelshah261`), Parth Deshmukh (`parthdeshmukh488-ops`), Swarup (`swarupgaikwad1801-ux`), Vernika (`vernikas-ctrl`)
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

- **Kaggle:** team NoIdead; public score: TODO(owner) after the upload. The private score is Kaggle's.

- **Per dimension** (EXP-0002, official holdout of 2444 images, metric.py; `results/metrics.json`):

| Dimension | Macro F1 | Weighted F1 |
| --- | --- | --- |
| subject | 0.056 | 0.422 |
| background | 0.458 | 0.767 |
| perspective | 0.097 | 0.198 |
| individuals | 0.529 | 0.912 |
| text_classification | 0.253 | 0.791 |
| text_language | 0.427 | 0.645 |
| **mean** | **0.303** | **0.635** |

  The bar to beat is the hosts' baselines: [challenge_description.md, Baselines](challenge_description.md#baselines).

- **What mattered** (every EXP row, holdout macro F1; source `results/metrics.json`):

| EXP | Predictor | Mean | subject | background | perspective | individuals | text_classification | text_language | img/s | $ / 100k | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0001 | majority | 0.187 | 0.056 | 0.304 | 0.049 | 0.484 | 0.152 | 0.076 | 1805822.4 | TODO(model): not measured | baseline, no model: images/s is not a product figure |
| EXP-0002 | zero-shot | 0.303 | 0.056 | 0.458 | 0.097 | 0.529 | 0.253 | 0.427 | 6.4 | TODO(model): not measured | zero-shot Apertus 8B via Keera, 512 px, with product context |
| EXP-0009 | zero-shot-no-context | 0.277 | 0.168 | 0.400 | 0.101 | 0.521 | 0.226 | 0.247 | 6.6 | TODO(model): not measured | EXP-0002 A/B: zero-shot Apertus 8B via Keera, 512 px, no product context |

  What did not work: TODO(model): one line per EXP that did not help, and why.

- **Models:**

  - Runtime: apertus-v1-5-8b; no helper models
  - Development only: TODO(owner): every tool and model used while building (coding assistants, any model used for targets), with version and role.

- **Resources** (source `$GALAXUS_DATA/out/bench/`, `galaxus.serve.bench`):

  TODO(serve): no benchmark on the GPU yet (`python -m galaxus.serve.bench`, #16, #33).

- **Commit and weights:** commit `8bf0efe6287b`; weights: TODO(owner): Hugging Face repository and revision, published by 14:30.

- **Optional:** label errors: TODO(data): not generated yet (`python -m galaxus.model.heads label-errors`).
<!-- section7:end -->

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References

- Swiss AI, Apertus v1.5 model card: huggingface.co/swiss-ai/Apertus-v1.5-8B
- Digitec Galaxus, challenge description and reference notebook: challenge_description.md, reference/finetune_qwen35_reference.ipynb
- The team's plan and architecture: BUILD_PLAN.md, BASE_ARCHITECTURE.md, decisions.md
