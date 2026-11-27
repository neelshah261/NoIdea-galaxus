# Deep-dive answer from the data lane's session (issue #64)

**Who wrote this:** the Claude session of the data-lane worker (`vernikas-ctrl`). The serve-lane deep-dive brief
(hero-shot gallery simulator, Saturday dress rehearsal, speed matrix, perspective error report) was sent to this
session instead of Swarup's. It was carried out after the lane conflict was raised and the owner of this session
chose to proceed; every issue and PR comment about it said so in the open and offered Swarup and `@neelshah261` a
one-line way to stop or re-scope it. See #31–#34 and the `[serve-agent]` comments on #37.

**Status of the code: all of it is pushed.** Nothing from this session is sitting unpushed on a laptop.

| Issue | What | Where |
| --- | --- | --- |
| #31 | hero-shot gallery simulator, `src/galaxus/serve/demo/` | merged in #37 |
| #32 | `--drill`, the Saturday dress rehearsal | merged in #37 |
| #33 | speed matrix, `bench/speed_matrix.py` + `gpu_plan.py` + `fake_backend.py` | merged in #37 |
| #34 | perspective error report + slide 8 figure, `bench/perspective_report.py` | merged in #37 |
| #26 | `$GALAXUS_DATA` layout | already satisfied by #24; no PR needed |
| #27 | `python -m galaxus.data.prep`, the prepared-image cache | merged in #56 |
| #29 | mirrored training rows, `galaxus.data.mirror` | merged in #56 |
| #28 | patch size as a parameter | open in **#65** |

---

## Findings

Ordered by what would cost us most if it stayed unknown.

### 1. Almost none of the "hero shot" rules are actually known

The deep-dive notes (#17) give exactly **one** real rule: mice are photographed `top_down` (point 1). That is the
only entry in `src/galaxus/serve/demo/hero_rules.json` marked `known` with a source.

Everything else a gallery needs — the default perspective order, which `text_language` values the shop keeps and
which it hides, whether the hero shot shows the product rather than an infographic — is **ours**. Each is marked
`assumption` with a `TODO(owner)`, and the generated page prints them under "Assumptions, not facts". A judge who
asks "where does this rule come from?" gets an honest answer.

**Risk if ignored:** a demo that asserts how Galaxus merchandises its shop, on our guesses. That is the kind of
claim a host corrects live.

### 2. Zero of the 81 speed-matrix cells are measured, and three different blockers explain why

- No GPU hourly price exists, so `cost_per_100k` is *refused in code* rather than estimated.
- No permission to spend GPU minutes.
- Nothing in the repo says whether the swiss-ai vLLM image serves fp8 or int4, so **54 of 81 cells** are
  `unconfirmed` rather than assumed. `bench/README.md` carries the one-command probe that settles it.

The 9 cells that did run are CPU image-preparation only, with no model, and carry `is_machinery_check: true` so
they print in a separate table and can never reach the deck.

### 3. Two blocked dependencies that a worker may not add

- **`matplotlib`** — slide 8 has no figure without it. Filed on #34 with the `design` label.
- **the `local` and `gpu` extras** (torch / transformers / vllm, B0) — 9 speed cells need them. Filed on #33.

Both degrade honestly meanwhile: the perspective report prints its full tables then says the figure needs
matplotlib.

### 4. The Saturday drill cannot fully rehearse the one step that has no retry

`--drill` really kills and resumes a run, really merges and validates, and really prints the submit command. The
**decryption** is the gap: 7-Zip is not installed here, and the real passphrase does not exist before Saturday.
Two tests skip with that reason, and a drill with no practice archive reports `ready: false` **by design** rather
than a comforting pass.

**Needed:** someone builds a practice `.7z` from throwaway files, and `ci` gets 7-Zip if we want that path covered
there. There is now a "Before Saturday (once)" checkbox for it in `docs/saturday_run_sheet.md`.

### 5. The patch size in the pipeline is Qwen's, not Apertus'

`PATCH_SIZE = 32` comes from the reference notebook. Apertus v1.5's is **Q-0002**, measured by #46. Every
`vision_tokens_per_image` in the speed matrix is therefore a proxy, and token count is what a run is billed for.
#28 (PR #65) makes it a parameter so the answer costs one line; it deliberately does **not** guess the value.

### 6. The metric-parity check has never run

It needs real Kaggle data and `$GALAXUS_DATA` is unset on this machine. It is the one persistently skipped test.
Our local score and the leaderboard score are only known to agree once it runs — worth doing **before the Sat
28 Nov 12:00 CET method freeze**, not after.

### 7. How to combine the two mirror answers is unknown, and it is not a detail

#17 slide 19 records only that mirroring helped, never how the two predictions were combined.
`bench/perspective_report.py` assumes "the more confident answer wins", labels that an assumption on the page, and
`galaxus.data.mirror.swap_mirrored_confidence` brings the mirrored answer back to the original orientation first.
**Without that swap, left and right get averaged against each other** and mirror TTA makes the exact confusion it
is meant to fix worse.

### 8. A `.gitignore` rule silently hides the data lane

A bare `data/` pattern matches `src/galaxus/data/` and `tests/data/`, so new files in both are invisible to
`git add`. Filed as **#21**; the workaround used throughout this session was `git add -f`. Anyone adding a file in
those folders will hit it and may not notice their work is missing from the commit.

### 9. Process findings, from things that actually went wrong today

- **Two people built #33 in parallel.** Swarup implemented the speed matrix independently while this session did;
  he resolved it by dropping his and keeping this one. Nothing was lost, but the duplicated effort was real.
- **One PR per branch means a push lands in someone else's open PR.** Commits from this session appeared inside
  Swarup's #37 while he was mid-review. The rebase-not-force-push habit is what kept his `#36` commit alive —
  a force-push would have destroyed it.
- **A review can block on a question nobody relays.** Parth blocked #37 at an old sha pending Swarup's answer;
  Swarup answered half an hour later and nobody told Parth. Stating the resolved condition and re-pinning the sha
  unblocked it.
- **AGENTS.md's merge rule changed twice in one day** (freeze → "only Parth merges" → "anyone merges after the
  three checks"). Read the file rather than trusting an earlier memory of it.

---

## Suggestions

| Suggestion | Serves | Why |
| --- | --- | --- |
| Put the hero rules to the hosts as one question list, then flip `hero_rules.json` entries from `assumption` to `known` | #31, the demo | It is one Discord message, and it converts our guesses into facts the deck can assert |
| Run the fp8/int4 probe before booking any GPU time | #33, #43 | 54 of 81 cells and the whole quantization ablation depend on the answer; the probe is seconds |
| Approve `matplotlib` and the `local`/`gpu` extras in one `design/*` PR | #33, #34, B0 | Two blocked deliverables, one shared file, one review |
| Run the metric-parity check on a machine with the data, today | the method freeze | It is the only thing standing between "our score" and "the leaderboard's score" |
| Build the practice `.7z` and add 7-Zip to `ci` | #32 | The decrypt step has no retry on Saturday and is currently the only unrehearsed one |
| Decide the mirror combination rule, then tell the data and serve lanes | #48, #49, #34 | Both lanes currently carry the same assumption, labelled as one |
| Fix `.gitignore`'s bare `data/` | #21 | Cheap, and it silently eats work in two folders |
| Treat `cached_image(id, mirrored=...)` as the way to read pixels | #47, #49, B6 | One prepared JPEG per image, so every experiment and Saturday are served identical pixels |

---

## Open questions

1. **Q-0002** — Apertus v1.5's vision patch size, as pixels per vision token (#46 measures it). Blocks the real
   token counts and the cost per 100k.
2. **Does the swiss-ai vLLM image serve fp8 or int4 at all?** 54 speed cells wait on it.
3. **Which `text_language` values does the shop keep, and which does it hide?** #17 point 11 names Chinese and
   Polish as examples of hidden ones but never the kept set.
4. **What is the default hero perspective order** for a product with no specific rule, and does the hero shot ever
   show an infographic?
5. **How were the two mirror answers combined** in the hosts' own result (#17 slide 19)?
6. **What does the GPU cost per hour, and from which provider and date?** No price, no `$/100k`.
7. **Is a hand-written rule layer (`hero_rules.json`) acceptable** for the demo, or must the hero choice come only
   from the model's predictions? It changes what the gallery is allowed to claim.

---

## Conventions this session stuck to, in case they are worth keeping

- **No class name is written in lane code.** They live in `hero_rules.json` and `bench/perspective_pairs.json` and
  are validated against `galaxus.data.classes` on load; `src/galaxus/data/` has zero class-name literals. A test
  asserts no name from the config appears in the report module.
- **A number that was not measured is `null`, never `0` and never plausible.** `CellResult` refuses a figure
  unless its status is `done`, and refuses a cost without both a price and a price source.
- **Nothing a judge sees is hand-made.** Every tool here points at `results/metrics.json`; none of them writes it.
- **Competition data never enters git.** Outputs go to `$GALAXUS_DATA/out/`; a test asserts the drill report path
  is not inside the repository.
