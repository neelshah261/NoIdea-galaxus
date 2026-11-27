# STATE.md: what is true now

Overwritten on every update; it wins over worklog.md (AGENTS.md "State files"). **As of Fri 27 Nov 2026, 19:15 CET,
`main` at `58e08a1` (#89).** Plan: [BUILD_PLAN.md](BUILD_PLAN.md). Board: issue #50.

## Results (`results/metrics.json`, official holdout of 2,444 images)

| EXP | Method | Mean macro F1 | Weighted F1 | Images/s |
| --- | --- | --- | --- | --- |
| EXP-0001 | majority | 0.187 | 0.579 | not a product figure |
| EXP-0002 | zero-shot Apertus 8B, gateway, with product context | 0.303 | 0.635 | 6.4 |
| EXP-0009 | zero-shot Apertus 8B, gateway, no context | 0.277 | 0.623 | 6.6 |

The Galaxus baselines are in challenge_description.md "Baselines". No Kaggle submission yet.

## Done (on `main`)
- **Base pipeline:** loading, the prepared-image cache, the official split, the metric (parity with the written rules
  checked on the real data, 0 differences, #79), the writer and validator, the runner, the Saturday drill and dry run.
- **Code:** shared types, the class contract and the ci secret scan (#80).
- **Methods:** zero-shot (#63), LoRA script and predictor (#71, #86), ensemble (#61).
- **Tools and docs:** perspective report, gallery, speed matrix, diagram and video script, results page (#85).
- **No typed names:** the data and serve lanes read dimensions and classes from the contract (#84, #85, #89).

## In flight

| What | Who | Issue / PR |
| --- | --- | --- |
| LoRA GPU check, full run, EXP-0004 | Neel | #76 |
| Timed zero-shot dry run on dev, first Kaggle file | Neel (Swarup: run sheet) | #59 |
| Kaggle team, then the first upload | Neel | #25, #15 |
| Discord: Q-0003, Q-0005, Q-0007, dev pseudo-labels, hero rules | Neel | #40 |
| Mirror TTA for perspective (the innovation) | Parth | #48 |
| Model literals, delete the contract copy | Parth | #83 |
| B0 rest and the technical report | Parth | #11, #54 (this PR) |
| Pitch deck (last) | Swarup | #52 |

## Blocked
- **#51 (heads + calibration):** parked until the hosts answer Q-0007 and Q-0005.

## Not done yet
- **Hand-ins:** the demo video, the final Kaggle submission, the weights published with a revision, the label-error
  CSV.
- **Freeze:** the method freezes Sat 28 Nov 12:00 CET (AGENTS.md product rule 6).
