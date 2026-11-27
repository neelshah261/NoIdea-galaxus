# design

The team's design kit, taken over from our previous hackathon repo
([sprintd Basel GreenTech, DigitalHeat](https://github.com/Mulaydm10/sprintd-basel-greentech-digitalheat), MIT): the
same visual system and the same habits that got that project's pitch through without a number nobody could trace.

| File | What | Who uses it |
| --- | --- | --- |
| [STYLE.md](STYLE.md) | colours, type, slide anatomy, chart rules, figure settings | serve (deck, diagram, video), model (report figures) |
| [PITCH_DECK.md](PITCH_DECK.md) | the slide plan: what each slide shows, where every number comes from, speaker notes | serve builds it in `presentation/` |
| [JUDGING.md](JUDGING.md) | each assessment criterion → the artifact that proves it → issue → status | design checks it before every hand-in |

Architecture in one view (details in [BASE_ARCHITECTURE.md](../BASE_ARCHITECTURE.md)):

```
Kaggle files ($GALAXUS_DATA)
        |
   data: load → white composite + resize (cached) → official holdout + grouped folds
        |
   serve.runner  ── same code for holdout, dev and final
        |
   Predictor:  majority │ zero-shot │ heads on Apertus features │ LoRA │ ensemble + calibration + mirror TTA
        |                         \__________ model.apertus: api · local · vllm __________/
        |
   predictions jsonl ──► model.eval + metric.py ──► results/metrics.json ──► report, deck, JUDGING.md
        └──────────────► data.submission (writer + validator) ──► a human uploads to Kaggle
```

Rules carried over:

- **Never say a number that is not in `results/metrics.json`** (AGENTS.md product rule 1) or linked to its single
  source. The deck and the report cite the `EXP-####` row; a number without one is cut.
- Decisions and open questions go to decisions.md as `ADR-####` / `Q-####`, append-only.
- A criterion in JUDGING.md with no artifact is a gap to close before the hand-in.
