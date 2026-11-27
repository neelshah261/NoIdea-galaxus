# PITCH_DECK.md: slide plan, DRAFT v1

Built like the DigitalHeat deck: every number on a slide names its source, and **a number without a source is not
said**. `{EXP-####.field}` is filled from `results/metrics.json` at build time; links point to the single source.
Slide limit and pitch length: [README.md](../README.md) "Pitch deck" and
[challenge_description.md](../challenge_description.md) "Timeline". Look: [STYLE.md](STYLE.md). Assessment parts:
[challenge_description.md](../challenge_description.md) "How we assess"; coverage check: [JUDGING.md](JUDGING.md).

`TODO(serve)` marks what only exists once the experiments have run. The slide order follows the assessment parts so
each judge finds their criterion on one slide.

---

## 1 · Title
- Word mark: **NoIdea** with the key word in `accent`. Lead: "Apertus reads every Galaxus product image: six
  answers per image, on hardware Galaxus can own."
- Key-figure row: mean macro F1 on dev `{EXP-best.mean}` · perspective macro F1 `{EXP-best.perspective}` ·
  images per second `{EXP-best.img_s}`.
- *Say:* the one-sentence result and which performance group it puts us in.

## 2 · The job
- What Galaxus decides from each image (gallery order, hidden images, text overlays): challenge_description.md
  "Why this matters".
- Visual: 4 competition images in frames, the six answers under each in Geist Mono.
- *Say:* six questions per image, today answered by a finetuned Qwen; we replace it with Apertus.

## 3 · Why it is hard
- Macro F1: rare classes count as much as frequent ones (challenge_description.md "The score").
- Perspective is the business priority, and the hosts' own worst confusions are left/right pairs (#17, slides 12 and
  22).
- Product context can be missing in production (challenge_description.md "The task").
- *Say:* three traps, and the rest of the talk shows how each one is handled.

## 4 · Section · Approach

## 5 · Our method in one picture
- Visual: the pipeline from design/README.md, redrawn in STYLE.md colours: prepare → Apertus v1.5 8B → our
  predictor → calibration → mirror TTA → validated file.
- Headline: the claim of the chosen main method (heads, LoRA or both, decided by the holdout) `TODO(serve)`.
- *Say:* what Apertus does, what our small parts do, and that everything at runtime is Apertus plus open helpers.

## 6 · Beyond training longer (innovation)
- One row per idea, each with its holdout gain from an `EXP-####`: per-dimension heads (on the hosts' "ideas we never
  tried" slide, #17), mirror TTA for perspective, per-class calibration for macro F1, context dropout.
- *Say:* which idea moved which dimension, and by how much.

## 7 · Section · Evidence

## 8 · Results against the baselines
- Visual: bars of mean macro F1 for us (`accent`) and the Galaxus baselines (`blue`, `blue-2`) from
  challenge_description.md "Baselines"; below, six small multiples per dimension, plus weighted F1.
- Perspective: the hosts' confusion pairs before → after mirror TTA (`red` → `green`) `TODO(serve)`.
- *Say:* the performance group, the dimensions where we beat the baseline and the ones where we do not.

## 9 · Speed, cost, deployment
- Images per second, peak memory, GPU type, cost per 100,000 images: `{EXP-best.img_s}`, `{EXP-best.peak_gb}`,
  `{EXP-best.cost_100k}` from the benchmark.
- Deployment: which target architecture (technical_report.md section 2) and why it runs air-gapped: no closed model,
  no external call at runtime.
- *Say:* the cost of one day of Galaxus images, and that it runs on hardware they can own.

## 10 · What mattered, what did not, and close
- Ablation table straight from `results/metrics.json`: the rows that moved the score and the ones that did not.
- Label errors found in train.csv: the count only (the list is never shown or committed).
- Close: reproducible from commit `TODO(serve)` with weights on Hugging Face at revision `TODO(serve)`.
- *Say:* the one insight we want Galaxus to take home, then thanks.

---

**Backup for Q&A (not counted as slides):** per-class F1 table, failure cases with images, the zero-shot number,
the full ablation table.
