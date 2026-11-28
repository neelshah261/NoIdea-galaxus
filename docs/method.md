# Method: how NoIdea gets its score

Status on Sat 28 Nov 2026, 06:30 CET. Every number is macro F1 on the **official holdout** (2,444 images of 1,452
products, `split_holdout`), scored with the hosts' `metric.py`, unless marked *public* (the Kaggle leaderboard, the
dev images). The rows are in `results/metrics.json`; the generated table is [results.md](results.md). The plan and
the owners are on #176.

## The method now: EXP-0056, 0.733

Two members, combined per dimension:

| Member | What it is | Holdout |
| --- | --- | --- |
| **SigLIP heads** (EXP-0055) | The open SigLIP base image encoder (`google/siglip-base-patch16-224`, revision `7fd15f0`, Apache-2.0) turns each image into one pooled embedding. Per dimension, one class-balanced softmax head (#51's `fit_heads`) is trained on the human-scored cells of train minus the holdout, calibrated out of fold on product-grouped folds. `helper-siglip`, #174. | 0.705 |
| **Apertus v1.5 70B + type prior** (EXP-0051) | Apertus 70B, unchanged, through the gateway: one call per image, the labeling guideline as system prompt, one sentence about the image before the labels ("observe first"), no product text, a JSON schema of the allowed classes (EXP-0049, 0.461). Then the product-type prior (#133, #177): per product type, the top human class in train minus the holdout replaces the answer where the cross-fit learns that it helps. | 0.489 (public 0.47664) |

**The combination** is `ensemble-agree` (#106, #165). The SigLIP heads are the base. A dimension switches to another
option, the 70B or a vote of both members, only when that option beats the base on both product halves of the
holdout. The routing: perspective, individuals and text_classification take the SigLIP heads; subject, background
and text_language take the vote.

| Dimension | 70B + prior (0051) | SigLIP heads (0055) | **EXP-0056** |
| --- | --- | --- | --- |
| subject | 0.303 | 0.702 | **0.702** |
| background | 0.494 | 0.878 | **0.878** |
| perspective | 0.324 | 0.520 | **0.520** |
| individuals | 0.762 | 0.977 | **0.977** |
| text_classification | 0.415 | 0.667 | **0.666** |
| text_language | 0.637 | 0.486 | **0.656** |
| **mean** | 0.489 | 0.705 | **0.733** |

**Speed:** the SigLIP heads embed about 30 images/s on the laptop's CPU (about 3 min for the 5,292 final images).
The 70B runs at 2.38–2.42 images/s with 8 workers (about 37 min for the final images). The two run side by side.

## How we got here

| EXP | What changed | Holdout |
| --- | --- | --- |
| 0002 | Apertus 8B zero-shot, guideline prompt, product text | 0.303 |
| 0021 | observe first | 0.343 |
| 0025 | no product text | 0.360 |
| 0041 | one question per dimension (#156) | 0.380 |
| 0046 | + product-type prior (#133) | 0.416 |
| 0050 | + type prior v2: share gates from 0.3, defaults count as least sure (#177) | 0.431 |
| 0049 | Apertus **70B**, observe first, no product text | 0.461 |
| 0051 | + type prior v2 | 0.489 (public 0.47664) |
| 0055 | **SigLIP heads** | 0.705 |
| 0056 | SigLIP heads + 70B, per dimension | 0.733 |

## What did not work, and what that taught us

- **Product text in the prompt hurt every time:** EXP-0021 0.343 vs EXP-0025 0.360 without it, and EXP-0039 0.334 vs
  EXP-0041 0.380.
- **The generative LoRA (EXP-0004, 0.343) learned the text, not the image.** Its training loss flattened at 0.08–0.09
  per answer token. That is what a model that sees only the product text reaches out of fold (0.081). The causes are
  on #145: one epoch, product text in every example, no class weights, and a loss dominated by JSON tokens.
- **Recalibration on top of the type prior** added nothing (EXP-0048).
- **The catalog text alone** (TF-IDF + logistic regression, not a method) reaches 0.425. The text gives away much of
  the label, which is why a model that has both falls back on the text.
- **The labels are learnable from the image.** As a development-only reference, `gpt-6.2-astra` with EXP-0049's
  prompt scores 0.858 on 1,000 random holdout images, where the 70B scores 0.472 on the same images. The bottleneck
  was Apertus's visual recognition, not label noise. That is why frozen image features with trained heads jump from
  0.489 to 0.705.

## The rules this rests on

- Apertus v1.5 70B, unchanged, answers the dimensions routed to it. The SigLIP heads pick the class for the others:
  the hosts answered **Q-0008** yes (decisions.md, #178). The type prior is Q-0005; heads are Q-0007.
- **At runtime:** Apertus v1.5 70B (gateway) and SigLIP base (local, open weights) only.
- **In development only:** `gpt-6.2-astra`, for the reference row and the label-error candidates. Never at runtime
  (FAQ.md).
- No final image is used in any training. The holdout is never pseudo-labelled.

## Reproduce

```bash
# 1. the 70B member (gateway) and its type prior
LLM_NAME=apertus-v1-5-70b uv run --env-file .env python -m galaxus.model.eval zero-shot-observe-first-no-context
TYPE_PRIOR_BASE=EXP-0049 uv run --env-file .env python -m galaxus.model.eval type-prior
# 2. the SigLIP heads (downloads the encoder once; embeddings are cached under $GALAXUS_DATA)
uv run --extra local --env-file .env python -m galaxus.model.eval helper-siglip
# 3. the combination, its routing, and a dev file
ENSEMBLE_MEMBERS="siglip=EXP-0055 70b=EXP-0051" uv run --env-file .env python -m galaxus.model.eval ensemble-agree
uv run --env-file .env python -m galaxus.model.ensemble choose --members siglip=EXP-0055 70b=EXP-0051 --rule agree \
    --out $GALAXUS_DATA/out/ensemble/routing-EXP-0056.json
uv run --env-file .env python -m galaxus.model.ensemble apply --routing $GALAXUS_DATA/out/ensemble/routing-EXP-0056.json \
    --part dev --exp EXP-0056
uv run --env-file .env python -m galaxus.serve.submit --exp EXP-0056 --part dev
```

Both members need their dev (or final) answers first. For the SigLIP heads:
`galaxus.serve.submit --exp EXP-0055 --part dev --run galaxus.model.helper:from_env`. For the 70B with the prior:
`--run galaxus.model.type_prior:from_env`, with `TYPE_PRIOR_RULES` and
`TYPE_PRIOR_INNER=galaxus.model.zero_shot:from_preset`.

## Next (#176)

1. More image features for the heads: DINO for perspective (geometry), SigLIP 2 for small text and detail, and
   Apertus hidden states (#180).
2. Mirror-equivariant perspective: train on flipped copies with left and right swapped, and average at prediction
   time.
3. Product-level decoding: dev and final rows carry `product_id`, so a product's photos can be decoded together.
4. Self-training on the dev images, where the heads and the 70B agree with high confidence.
