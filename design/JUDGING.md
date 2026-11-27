# JUDGING.md: assessment part → artifact → status

Living file, like `notes/judging_alignment.md` in the DigitalHeat repo. The parts and their weights are defined in
[challenge_description.md](../challenge_description.md) "How we assess" and the hand-in in
[README.md](../README.md) "Submission Requirements"; they are not restated here. **A row without an artifact is a gap
to close before the hand-in.** Status: `gap` | `partial` | `proven`.

| Part | What the judges look for | Artifact that proves it | Issues | Status |
| --- | --- | --- | --- | --- |
| Performance on the final images | performance group against the Galaxus baselines; also weighted F1 and per-dimension scores | selected Kaggle submission; best `EXP-####` in `results/metrics.json`; deck slide 8 | #12, #15, main phase | gap |
| Innovation | how far the approach goes beyond training longer | one `EXP-####` per idea with its holdout gain; deck slide 6; report section 2 | main phase | gap |
| Speed and resource usage | images per second, memory, cost per 100,000 images | `bench/` output in `results/metrics.json`; deck slide 9; report section 7 | #16 | gap |
| Insights | clear evidence of what mattered and what did not | ablation table generated from `results/metrics.json`; failures written down too (#17, point 7); deck slide 10 | main phase | gap |
| Code quality and practicality | a clean repo that runs end to end and reproduces | `ci` green; one command per step (justfile); README / report section 6 with pinned deps, configs, seeds | #11, all lanes | partial: `justfile`, `uv.lock` with extras, report section 6, ci with secret scan |
| Bonus: perspective | very good perspective scores | perspective macro F1 per `EXP-####`; mirror TTA row | main phase | gap |
| Bonus: label errors | list of suspected errors in train.csv | CSV in `$GALAXUS_DATA/out/`, sent by a human on Discord, never committed | #7 | gap |
| Hand-in: Kaggle | final file uploaded **and selected** in time | `final_run.py` dry run; owner uploads and selects | #15, #16 | gap |
| Hand-in: repo | public, template layout kept | repo settings; layout check | #11 | gap |
| Hand-in: technical report | PDF in the repo root, section 7 filled, commit + weights revision named | `technical_report.md` → `NoIdea_Report.pdf` (`just report`) | #54 | partial: draft PDF with section 7 generated; final EXP, Kaggle score, demo link and weights revision still TODO |
| Hand-in: demo video | link in the report's Demo field | video, STYLE.md title cards | serve | gap |
| Hand-in: pitch deck | PDF within the slide limit | PITCH_DECK.md built in `presentation/`, printed to PDF | serve | gap |
| Hand-in: weights | published with a revision | Hugging Face repo, revision in report section 7 | owner | gap |
| Reproducibility re-check | the hosts re-run winners from the commit (#17, point 6) | tagged commit at freeze, `uv.lock`, seeds, weights revision | design | gap |
