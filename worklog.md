# worklog.md (append-only, CET)

AGENTS.md "State files": dated and timestamped, never rewritten. STATE.md wins where they differ. The entries for
Fri 27 Nov before 19:00 were written at once from the merge history of `main` (merge time, PR, what it brought).

## Fri 27 Nov 2026

- 10:59 #1: agent rules (AGENTS.md, CLAUDE.md).
- 11:02 #2: agents merge their own PRs on green ci (later replaced, see 15:00).
- 14:26 #20: data lane: loader, white compositing, official split and metric, writer and validator, label-error finder.
- 14:26 #19: serve lane: runner, first Kaggle file command, benchmark, Saturday final run.
- 15:00 #23: plan, lane owners, Python setup, BASE_ARCHITECTURE.md, design kit; the merge freeze (#9) lifted.
- 15:04 #30: API key rotation (`galaxus.keys`).
- 15:05 #35: design PRs no longer need the owner's approval comment.
- 15:17 #39: `.env.example` removed (owner request).
- 15:21 #22: majority baseline, eval harness, Apertus layer and smoke test.
- 15:21 #24: grouped folds, out-of-fold format, image cache.
- 15:36 #44: gateway client through the key ring.
- 15:40 #45: BUILD_PLAN.md (base, main method, one innovation).
- 16:00 #37: parallel batches; gallery, drill, speed matrix, perspective report (Vernika); vLLM script; Saturday
  preflight.
- 16:22 #56: `prep` command; mirrored training rows.
- 16:36 #62: architecture diagram, demo video script, gateway load test.
- 16:38 #60: results tables, report section 7 generator, `docs/how_to_run.md`.
- 17:09 #63: zero-shot Apertus predictor.
- 17:10 #65: patch size as a parameter; real metric parity tests.
- 17:13 #67: EXP-0001 row (majority, 0.187).
- 17:22 #66: Vernika's deep-dive write-up (`docs/deep_dive_vernika.md`).
- 17:29 #70: EXP-0002 and EXP-0009 rows (zero-shot, 0.303 and 0.277).
- 17:29 #71: LoRA training script, CPU smoke run, launch command.
- 17:40 #77: Windows `--help` fix, serve tests isolated from real data, drill timing flake fixed.
- 17:44 #79: debugging pass on the real data; metric parity 0 differences.
- 17:44 #78: validator linear again; metric.py cached per file.
- 17:50 #80: `galaxus.types`, `contracts/classes.json`, ci secret scan.
- 18:49 #84: overlay roles read from the contract; metric cache fix.
- 18:49 #86: LoRA predictor registered for EXP-0004.
- 18:49 #61: ensemble picks the main method per dimension, cross-fitted, as an EXP row.
- 19:00 B0 rest (#11) and the technical report (#54) in one design PR: justfile, extras, MODELS.md, decisions.md,
  STATE.md, this file, the report and its PDF.
