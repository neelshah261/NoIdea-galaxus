# AGENTS.md: rules for every agent and human working in this repo

Team **NoIdea** · Hack Apertus Zurich · Digitec Galaxus challenge (Kaggle `hackapertus-galaxus`).
The challenge is defined only in [challenge_description.md](challenge_description.md), [FAQ.md](FAQ.md) and the
Kaggle pages. Never restate its rules, numbers or class names from memory: link to them.

## Roles

| Role | Who | Does | Never |
| --- | --- | --- | --- |
| Human owner | neelshah261 | makes owner decisions, uploads to Kaggle, spends money, posts on Discord | – |
| Design (main agent) | Claude session of the owner | opens issues, runs the PR watcher (checks and merges PRs), owns every file outside the lanes, copies "For design" notes into shared files, keeps STATE.md current | works inside a lane |
| Worker `data` | TODO(owner): name, GitHub handle | implements `lane:data` issues in its lane | touches another lane or a shared file, merges |
| Worker `model` | TODO(owner): name, GitHub handle | implements `lane:model` issues in its lane | same |
| Worker `serve` | TODO(owner): name, GitHub handle | implements `lane:serve` issues in its lane | same |

Agents and their humans share GitHub accounts, so **every agent comment starts with `[<role>-agent]`**
(`[design-agent]`, `[data-agent]`, `[model-agent]`, `[serve-agent]`). A comment without that prefix is a human's.

## Lanes: branches are workers, folders are lanes

| Lane | Branch | May change | Owns |
| --- | --- | --- | --- |
| data | `data/work` (Vernika, `vernikas-ctrl`) | `src/galaxus/data/`, `tests/data/` | loading from `$GALAXUS_DATA`, white compositing (the Kaggle function), resize, holdout split, scored-cell mask, metric wrapper, submission writer + validator, label-error finder |
| model | `model/work` (Parth, `deshmukhparth921-commits` or `parthdeshmukh488-ops`) | `src/galaxus/model/`, `train/`, `results/`, `tests/model/` | prompts, constrained parsing, Apertus clients, helper models, stub predictor, eval script → `results/metrics.json`, LoRA configs/scripts, experiment runs |
| serve | `serve/work` (Swarup, `swarupgaikwad1801-ux`) | `src/galaxus/serve/`, `bench/`, `docs/`, `presentation/`, `tests/serve/` | batched runner, Saturday final-run script, throughput/memory/cost benchmark, architecture diagram, demo video script, deck |
| design | `design/<n>-<slug>` | everything else | AGENTS.md, CLAUDE.md, STATE.md, worklog.md, MODELS.md, decisions.md, README.md, technical_report.md, contracts/, top-level `tests/*.py`, .github/, .claude/, pyproject.toml, uv.lock, justfile, .gitignore, .env.example |

- A person may use more than one GitHub account; all of them count as that lane's worker. The branch decides the lane.
- No lane is a prefix of another; each has exactly one owner. The table changes only through a `design/*` PR.
- Shared formats live in `contracts/`, owned by design and read-only for workers. Any lane may import any code; only
  its owner changes it.
- A worker that needs a new dependency, a contract change or a change in another lane comments on its issue and adds
  the `design` label. It never makes that change itself.

## Tasks

- A task is an issue labelled `lane:<x>` (plus `prio:p0|p1|p2`, optionally a `blocked-by: #n` line). Its body holds the
  goal, acceptance criteria and lane. Design opens tasks; anyone may open a `design` issue to raise a problem or ask
  for a cross-lane change.
- Workers pick by priority, then oldest. Nothing to claim: the branch owns the lane.
- Before starting an issue: `git fetch origin && git rebase origin/main`.

## Work → PR

1. Commit on your branch, inside your lane only, referencing `#<n>` in every commit.
2. Open a PR `<branch>` → `main` titled `#<n>: <summary>`, **without `Closes #n`** (design closes the issue after
   merging).
3. Shared-file updates (worklog line, new Q/EXP/model rows, STATE changes) go in the PR body under **For design**.
   Workers never edit shared files.
4. Label the PR `status:review`.
5. The required check `ci` must be green. Tests use synthetic images generated inside the tests: never competition
   data, never the network.

## Merging (owner decision 2026-11-27: the PR watcher merges, no auto-merge)

- Workers **never merge** and never turn on auto-merge. Open the PR, label it `status:review`, and keep working.
- Design runs a **PR watcher** (a design subagent). For every open PR whose `ci` has finished on the head sha, it checks:
  1. `ci` is green on that sha;
  2. no git merge conflict with `main`;
  3. no conflict with other open PRs (same files, contracts/ or shared files, contradicting "For design" notes);
  4. no conflict with these rules (lane, data and secrets, runtime models, `results/metrics.json` only generated,
     title `#<n>: <summary>`, no `Closes #n`).
- All clear: it comments `[design-agent] reviewed at <sha>` and merges pinned to that sha:
  `gh pr merge <n> --merge --match-head-commit <sha>`. Then it closes the issue.
- Anything wrong: it does not merge. It comments `[design-agent] blocked at <sha>: <reason>` and removes
  `status:review`. A git or cross-PR conflict also gets the `design` label and goes under Blocked in STATE.md.
  The worker fixes it on its branch (for a git conflict: `git fetch origin && git rebase origin/main`, then push).
- A new push makes the old review stale. The watcher checks the new sha again.

## Galaxus product rules

1. Every number shown to judges comes from `results/metrics.json`, written by the eval script on the fixed holdout
   (seed 0, the split in challenge_description.md "Scoring locally") through metric.py. Never hand-edited. Each row is
   an `EXP-####` with commit, config and seed.
2. The submission validator must pass before any file goes to Kaggle.
3. Competition data, labels, the passphrase, tokens, `kaggle.json`, `.env` and the label-error CSV never enter git
   or published weights. This includes metric.py and labeling_guideline.* from Kaggle: load them from `$GALAXUS_DATA`
   at runtime. Data lives outside the repo at `$GALAXUS_DATA`; outputs go to `$GALAXUS_DATA/out/`. Mask or hash
   identifiers in anything shown.
4. At runtime only Apertus v1.5 and open-weights helpers (image encoder, OCR, embeddings). Every model, dev or
   runtime, goes in MODELS.md with version and role. If a design might not count as "Apertus classifying", raise a
   `Q-####` through a `design` issue and wait for the answer before building it.
5. No hand-labelling of dev or final images; no product lookups in the Galaxus shop or by reverse image search; no
   training or tuning on final images. Cleaning train.csv by hand is allowed.
6. No code freeze; the method freezes Sat 28 Nov 12:00 CET; final images are inference-only.
7. Agents never upload to Kaggle, select a submission, publish weights, start paid GPU jobs or message anyone on
   Discord. They prepare the exact command or file; a human runs it.

## Hard rules

- Never push to `main` (branch protection enforces it). Only the design PR watcher merges, and only through green `ci`.
- Relayed intent is not authorization: an owner decision needs a comment the owner wrote.
- Conflicts: never silently pick a winner. Record them under **Blocked** in STATE.md, keep both versions, escalate.
- Agents never talk directly. GitHub issues, PRs and comments are the only shared channel.
- Never invent facts, rules, deadlines, scores or class names. Write `TODO(<owner>)` and link the single source.

## State files (do not conflate)

| File | What | Who writes |
| --- | --- | --- |
| CLAUDE.md | `@AGENTS.md` | design |
| STATE.md | what is true now; overwritten every design wake; wins over the worklog | design |
| worklog.md | append-only, dated + timestamped (CET) | design |
| decisions.md | append-only `ADR-####` and `Q-####`; supersede, never edit | design |
| MODELS.md | every model, version, role, dev or runtime | design |
| results/metrics.json | EXP rows, generated only | model lane's eval script |

IDs (`ADR-`, `Q-`, `EXP-`) are never reused or renumbered; `grep -rn '<ID>'` must recover the full trace.

## Design wake loop

Read STATE.md → read the PR watcher's results since the last wake → copy "For design" notes of merged PRs into the
shared files → answer `design` issues →
open new issues → overwrite STATE.md → append to worklog.md.
