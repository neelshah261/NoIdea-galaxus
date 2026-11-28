# decisions.md: ADR-#### and Q-#### (append-only)

AGENTS.md "State files": append-only; supersede an entry with a new one, never edit an accepted one. IDs are never
reused. Each entry names its source, so `grep -rn '<ID>'` recovers the trace. Entries ADR-0001 to ADR-0007 and the
Q table record what was decided on Fri 27 Nov 2026 before this file existed; their sources are the issues and PRs
named.

## Decisions

### ADR-0001: base first, then the main method, then one innovation
- **Decided:** Parth, 2026-11-27 (BUILD_PLAN.md, PR #45).
- **Main method:** zero-shot Apertus 8B (EXP-0002), then a LoRA on Apertus 8B (EXP-0004).
- **Innovation:** mirror TTA for perspective (#48). Heads on Apertus features (#40, #51) stay a bet behind Q-0007.

### ADR-0002: how PRs merge
- **Decided:** owner, 2026-11-27 (issue #9, PRs #23 and #35, AGENTS.md "Merging").
- **History:** the merge freeze (#9) was lifted. Anyone merges a PR after three checks (green `ci`, no git conflict,
  no clash with open PRs), with a `reviewed at <sha>` comment and a merge pinned to that sha.
- **Design PRs:** need no owner approval comment since #35.

### ADR-0003: no `.env.example`
- **Decided:** owner, 2026-11-27 (PR #39).
- `src/galaxus/keys.py` names the variables (`LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_API_KEY_2`, ...)
  instead. This supersedes the `.env.example` item of #11.

### ADR-0004: one contract for types and class names
- **Decided:** PR #80.
- **Types:** `galaxus.types` (`ImageItem`, `Prediction`, `Predictor`).
- **Classes:** `contracts/classes.json`, generated from challenge_description.md "Classes" and pinned to it by
  `tests/test_contracts.py`. Lane code never types a dimension or class name.
- **Tests:** the contract is hidden from lane tests unless `@pytest.mark.real_contract`, so they keep proving that
  nothing is typed.

### ADR-0005: ci scans for secrets
- **Decided:** PR #80.
- `.github/secret_scan.py` runs on every PR diff and on `main`. A hit prints its place, never the value. A fake key
  in a test carries `secret-scan: allow`.

### ADR-0006: the ensemble picks the main method per dimension
- **Decided:** Parth, issue #87, PR #61.
- **Role:** BUILD_PLAN.md phase 2 step 3. Scored through metric.py; the routing is chosen on one product-grouped half
  of the holdout and scored on the other; an EXP row through the eval script.
- **Members:** zero-shot with and without context now, the LoRA once scored.

### ADR-0007: merge only work assigned on the board
- **Decided:** Parth, 2026-11-27 (board #50).
- Work outside a person's queue stays parked on its branch until the plan's work is done, then is reviewed. #51 is
  the parked case today.

## Open questions

| ID | Question | Status | Source |
| --- | --- | --- | --- |
| Q-0001 | Does the gateway accept image input for Apertus v1.5? | **Yes in practice.** EXP-0002 and EXP-0009 classified 2,444 holdout images through it. The smoke test's own output was never posted on #46. | PR #63, #70 |
| Q-0002 | Is a JSON schema enforced, and how many tokens does an image cost? | **Schema: yes**, `response_format` is enforced, with 0 defaulted cells in EXP-0002. **Tokens:** about 3,900 prompt tokens per image, mostly the labeling guideline. Pixels per vision token: `bench/vllm_apertus.sh` assumes 16 px (1,024 tokens at 512²), **not yet measured**. | PR #63, #70 |
| Q-0003 | Do nearest-neighbour label hints, with Apertus deciding, count as "Apertus classifying"? | **answered: yes.** The hosts, one-to-one with Neel; recorded by Neel on 2026-11-28. | PLAN.md, #40 |
| Q-0004 | May a committed prompt paraphrase labeling_guideline.md? | avoided: the guideline is read from `$GALAXUS_DATA` at runtime and only its hash is recorded | PLAN.md, PR #63 |
| Q-0005 | Is a per-class logit bias, fitted out-of-fold on train, allowed? | **answered: yes.** The hosts, one-to-one with Neel; recorded by Neel on 2026-11-28. Covers the calibration in #51, `recalibrated` and the type prior (#133). | PLAN.md, #40, #88 |
| Q-0006 | Which transformers loads Apertus 1.5? | **assumed `>=5.24`** (the vLLM guide; the `local` extra); confirmed by the GPU check in #76 | PLAN.md, PR #71 |
| Q-0007 | Do small heads on Apertus v1.5 hidden states count as "Apertus classifying"? Are they fine as a development-time label-error tool? | **(a) answered: yes.** The hosts, in a one-to-one talk with Neel; recorded by Neel on 2026-11-28. Heads on Apertus hidden states count as "Apertus classifying" and may be the final method (#51, #129 step 5). (b) development-time use: yes (#121). | BASE_ARCHITECTURE.md, #40, #88, #121 |
| Q-0008 | May an open-weights image model other than Apertus (SigLIP base with class-balanced heads, `helper-siglip`, #174) pick the class at runtime, for the dimensions the ensemble routes to it? (challenge_description.md:47: Apertus "performs the classification".) | **answered: yes.** The hosts, one-to-one with Neel; recorded by Neel on 2026-11-28. The helper and its revision go in `MODELS.md` and the report (rule 6). | #172, #174, #175 |
