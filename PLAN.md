# PLAN.md: method, speed, insights, timeline

Team NoIdea, drafted 2026-11-27 by design. Facts come from [challenge_description.md](challenge_description.md) and
[FAQ.md](FAQ.md); this file only plans. Changes go through a `design/*` branch.

## Method ladder: every rung is an EXP with a holdout score, and a valid file is always on Kaggle

| EXP | Method | Expected gain / cost / risk | Apertus classifies? |
| --- | --- | --- | --- |
| EXP-0001 | majority class per dimension (human labels, train minus holdout): proves loader → metric → writer → validator → Kaggle | low score, ~0 cost | no, pipeline check only |
| EXP-0002 | zero-shot Apertus v1.5 8B: labeling guideline in the system prompt, one call → JSON with all 6 answers, held to the class enums, thinking off, greedy, image ~512² area composited on white. A/B with and without product context; 70B through the gateway if Q-0001 = yes | unknown score; minutes of GPU per holdout run; risk: tokens per image, JSON support | yes |
| EXP-0003 | retrieval hints: an open image encoder finds the k nearest train images (holdout products excluded while scoring); their labels go into the prompt as text; Apertus decides | big gain on frequent classes (weighted F1); embeddings on a Kaggle GPU; risk: leakage, rules | ask first (Q-0003) |
| EXP-0004 | LoRA on Apertus 8B adapted from the reference notebook: loss on human labels only, rare-class repeats, mirror augmentation, bf16 LoRA → merge → vLLM; trained without the holdout. **GPU job only after the owner's OK** | biggest expected jump; estimate 2.5–4 h on one L40S ≈ $5–7; risk: no documented Unsloth path (Q-0006) | yes |
| EXP-0005 (opt) | per-class logit bias on Apertus class logprobs, fitted on holdout folds for macro F1 | small to medium gain, ~free; risk: overfitting the holdout | yes (Q-0005) |
| EXP-0006 (opt) | mirror TTA for perspective (text-free images, left/right swapped) | perspective bonus; ~+30% calls on that subset | yes |
| EXP-0007 (opt) | open OCR + language ID feed text into the Apertus prompt | text_language; one more helper | yes, only if Apertus decides |
| EXP-0008 (opt) | closed-model pseudo-labels on dev as extra LoRA data (allowed in development) | more data; API cost; dev score stops being clean | yes, runtime stays Apertus-only |

Rules that only touch unscored cells gain nothing; rules that change scored cells ship only with holdout evidence.

## Speed (20%)

- One request per image, short constrained JSON, thinking off, prefix caching on the shared prompt, offline batched
  vLLM (`ghcr.io/swiss-ai/vllm_apertus_1.5_release`).
- Image area is an ablation (384² / 512² / 768²); tokens per image are measured (Q-0002).
- Main line: self-hosted 8B on one GPU. 70B only through the gateway (the model card serves it with tensor
  parallelism 4).
- Every EXP records img/s, peak GPU memory, GPU type and cost/100k = $/h ÷ (img/s × 3600) × 100,000.
- **Hard checkpoint Sat 10:00:** timed full run on all dev images with the final script. Below 2 img/s: smaller
  image area, shorter prompt, drop hints or TTA, or the previous rung.

## Insights (15%)

- Ablation table built only from `results/metrics.json`: model, context on/off, image area, hints, LoRA,
  calibration/TTA, macro F1 for each of the 6 dimensions, their mean, weighted F1, img/s, peak GB, $/100k.
- Weakest classes per dimension (`metric.evaluate`), human vs model label counts, context on/off per dimension.
- Label-error finder: out-of-fold predictions, high-confidence disagreements with human labels. The CSV
  (`image_id, dimension, label, suggested_label`) stays in `$GALAXUS_DATA/out/`; a human sends it on Discord.

## Lanes and contracts

Lanes as in [AGENTS.md](AGENTS.md). Contracts that design adds under `contracts/`:

- `classes.json`: the 41 classes per dimension, copied verbatim from challenge_description.md "Classes".
- `prediction.md`: `ImageItem{image_id, path, context | None}` → `Predictor.predict(batch) -> list[Prediction]`,
  `Prediction{image_id, labels: {dimension: class}, confidence: {dimension: float} | None}`; never an empty or
  unknown class.
- `experiment.md`: EXP record `{id, commit, config, seed, model + revision, docker digest, helpers, holdout macro F1
  per dimension, mean, weighted F1, img/s, GPU, peak memory, cost/100k}`.

## First issues (p0 = by Fri 13:00)

| Lane | Issue | Prio |
| --- | --- | --- |
| data | loader + white compositing + resize + official holdout split + scored-cell mask + metric wrapper | p0 |
| data | submission writer + validator (ids and order as sample_submission, classes, lowercase `none`, no empties, final rows kept before Sat 13:00) | p0 |
| data | label-error finder (out-of-fold, CSV outside the repo) | p1 |
| model | stub + majority predictor + eval script → EXP-0001 row in `results/metrics.json` | p0 |
| model | probe Q-0001 / Q-0002 with a synthetic image | p0 |
| model | EXP-0002 zero-shot 8B, context on/off | p1 |
| model | EXP-0003 retrieval hints | p1 |
| model | EXP-0004 LoRA script, smoke-tested; launch command + cost posted for the owner | p1 |
| model | EXP-0005 / 0006 / 0007 | p2 |
| serve | batched runner on the Predictor interface + CI smoke test (3 synthetic images → validated file) | p0 |
| serve | EXP-0001 Kaggle file + exact submit command for a human | p0 |
| serve | Saturday script: decrypt (passphrase via getpass) → predict → merge into best dev file → validate | p1 |
| serve | benchmark: img/s, peak memory, cost/100k | p1 |
| serve | architecture diagram, demo video script, deck | p2 |

## Timeline (CET)

| When | Checkpoint |
| --- | --- |
| Fri 13:00 | EXP-0001 scored on the holdout **and** a valid file on Kaggle |
| Fri 17:00 | EXP-0002 dev score; Q-0001 and Q-0002 answered |
| Fri 20:00 | finetune go/no-go (owner) |
| Fri 23:00 | main method chosen |
| Sat 10:00 | timed full dry run on dev images (≥ 2 img/s or fall back) |
| Sat 12:00 | method frozen, commit tagged, weights ready to publish |
| Sat 13:00 | passphrase → final inference |
| Sat 14:15 | upload **and select** the submission |
| Sat 14:30 | sprintd, label-error CSV, weights published |
| Sat 15:00 | pitch (7 min + 2 Q&A) |

Kaggle: 20 submissions per UTC day; Saturday starts 01:00 CET; keep ≥ 5 for 13:00–14:30.

## Open questions

| ID | Question | How |
| --- | --- | --- |
| Q-0001 | Does the Keera gateway accept image input for Apertus v1.5? | probe with a synthetic image; Discord if unclear |
| Q-0002 | Does the swiss-ai vLLM fork support JSON-schema output, and how many tokens does an image cost? | our own test |
| Q-0003 | Do nearest-neighbour label hints, with Apertus deciding, count as "Apertus classifying"? | Discord (owner) |
| Q-0004 | May a committed prompt paraphrase labeling_guideline.md? | Discord (owner) |
| Q-0005 | Is a per-class logit bias tuned on our train holdout allowed? | Discord (owner) |
| Q-0006 | Transformers for Apertus 1.5: the swiss-ai fork (model card) or 5.24.0 (vLLM guide)? | our own test |

## Risks → fallback

| Risk | Fallback |
| --- | --- |
| gateway takes no images (Q-0001) | self-hosted 8B on an HF GPU |
| no JSON constraints in the vLLM fork (Q-0002) | lenient parser + one retry + valid default per dimension, or logprob ranking |
| LoRA tooling for Apertus breaks (Q-0006) | ship EXP-0003 + calibration |
| GPU credits run out (4 × $10 + $20 team pool) | every spend approved by the owner with an estimate; one GPU; embeddings on Kaggle |
| inference too slow | smaller image area, shorter prompt, drop hints or TTA |
| Kaggle outage at the deadline | owner DMs the validated file to the Galaxus hosts before 14:30 |
| not reproducible | uv.lock, docker image by digest, seeds in configs, weights on HF with revision, tag at freeze |
| data leak | .gitignore + CI data-leak check; prompts read the guideline from `$GALAXUS_DATA` at runtime |
