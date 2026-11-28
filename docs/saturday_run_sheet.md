# Saturday run sheet (serve lane)

Who runs what, in order. Times, limits and file rules are in
[challenge_description.md](../challenge_description.md) ("Timeline", "Submission file") and PLAN.md "Timeline";
this sheet only lists the commands. Agents prepare; **a human uploads and selects** (AGENTS.md product rule 7).

## Before Saturday (once)

- [ ] 7-Zip installed on the machine that runs the final: `7zz` (macOS `brew install sevenzip`), `7z` (Ubuntu
      `apt install 7zip`) or Windows `winget install 7zip.7zip`. `python -m galaxus.serve.final_run` says so if not.
- [ ] `$GALAXUS_DATA` holds the Kaggle files, including the encrypted final archive (SHA-256 checked, issue B1).
- [ ] The frozen method's predictor name (`package.module:name`) and its EXP id are written in STATE.md.
- [ ] The best dev submission file is in `$GALAXUS_DATA/out/submissions/` and has a public score.
- [ ] A **practice** archive exists, so the drill below can rehearse the decryption itself. Build one from two
      throwaway files (never competition data, never the real archive, whose passphrase nobody has yet):

      7z a -p'practice only' -mhe=on $GALAXUS_DATA/out/practice.7z <a small folder>

## Dress rehearsal on dev (`--drill`, issue #32)

The one run that cannot be retried is the final one, so walk all of it first, on the dev images, with the frozen
predictor. Same code, same steps, one command:

```bash
python -m galaxus.serve.final_run --drill --predictor <package.module:name> --exp <EXP-id> \
    --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv \
    --archive $GALAXUS_DATA/out/practice.7z
```

It asks for the practice archive's passphrase (not echoed), then prints one line per step:

| Step | What it proves |
| --- | --- |
| `decrypt` | 7-Zip is there and really extracts a passphrase-protected archive. `skipped` without `--archive`. |
| `items` | the dev images load from `$GALAXUS_DATA` and how many batches they make |
| `kill` | the run is stopped halfway on purpose; the predictions written so far are on disk |
| `resume` | started again, it skips what is done, finishes the rest, and leaves no duplicate |
| `merge` | the predictions replace those rows in the best dev file, as the final rows will on Saturday |
| `validate` | the merged file is re-read from disk and would be accepted by Kaggle |
| `submit-command` | the `kaggle competitions submit ...` line comes out, for a human |

Then it reports the wall time, the images per second, and the margin over the **2 img/s floor** (the Sat 28 Nov
13:00 row of [challenge_description.md](../challenge_description.md) "Timeline"; `--floor` changes it, `--floor 0`
switches the check off when rehearsing on a slower machine than Saturday's).

Read the last three lines:

- `ready for Saturday: yes` — every step ran, nothing was skipped, the floor is cleared.
- `not rehearsed here: <step>` — a skip, not a failure: fix it (install 7-Zip, pass `--archive`, use at least two
  batches) and drill again.
- `is BELOW the ... floor` — exit code 1. Fall back as PLAN.md "Hard checkpoint Sat 10:00" says, or use a faster
  machine; do not go into Saturday on this number.

The full report, with every step and every number, is written to
`$GALAXUS_DATA/out/drill/<drill name>/report.json` — outside the repository, like all our output, so it can never
reach git. Each drill gets its own name (`--drill-part`, default a UTC timestamp) and its own predictions file, so
drilling again never collides with an earlier one. Other options: `--batch-size` (the kill lands between batches,
so at least two are needed), `--limit N` for a shorter rehearsal, `--part holdout` to rehearse on the holdout.

Nothing here uploads anything: a drill prints the command, a human runs it (AGENTS.md product rule 7).

## Dry run on dev (PLAN.md "Hard checkpoint")

```bash
python -m galaxus.serve.final_run --dry-run --predictor <package.module:name> --exp <EXP-id>
python -m galaxus.serve.bench --predictor <package.module:name> --part dev --limit 256 --dollars-per-hour <GPU price>
```

The dry run is the short version of the drill above: it predicts and writes a file, but does not rehearse the
decryption or the resume. It prints the total time and the images per second; the benchmark appends img/s, peak
memory and the cost per 100,000 images to `$GALAXUS_DATA/out/bench/<label>.jsonl`, for section 7 of the technical
report.

## Saturday 28 Nov: the frozen method EXP-0056

EXP-0056 is the SigLIP heads and Apertus v1.5 70B, combined per dimension: 0.733 on the holdout, **0.73546 public**
(#121, `docs/method.md`). Two members, so the final run uses the live ensemble predictor (#188). The steps are Swarup's
runbook (#181), checked against `main`. Everything runs on Neel's laptop: it is the only machine with the SigLIP
embedding cache, the routing file and the gateway key.

### Before 13:00 (once)

- [ ] These exist under `$GALAXUS_DATA/out/`: `ensemble/routing-EXP-0056.json`, `type_prior/rules-EXP-0049.json`,
      `submissions/EXP-0056_dev.csv` (validated, public 0.73546).
- [ ] `shasum -a 256 $GALAXUS_DATA/final_encrypted.bin` matches the SHA-256 in the hosts' README.
- [ ] The 70B is the model the gateway client will use. This must print `apertus-v1-5-70b`:
      `LLM_NAME=apertus-v1-5-70b uv run --env-file .env python -c "from galaxus.model.apertus.api import gateway_settings as g; print(g()['LLM_NAME'])"`
- **Gateway credit is not a limit:** the Keera key has unlimited credits, and it stays as it is (Neel, 28 Nov). The
  final run needs about 21M prompt tokens (5,292 images × about 3,900). A 429 (rate limit) is still possible:
  rerun the same command.
- [ ] **A fresh `ENSEMBLE_CACHE` folder for the final part.** The cache is keyed by member name only (#188 review),
      so a folder from a drill or from another member setup would serve stale answers.

### At 13:00: one command

```bash
ENSEMBLE_ROUTING=$GALAXUS_DATA/out/ensemble/routing-EXP-0056.json \
ENSEMBLE_PREDICTORS="siglip=galaxus.model.helper:from_env 70b=galaxus.model.type_prior:from_env" \
ENSEMBLE_CACHE=$GALAXUS_DATA/out/ensemble_cache/final \
TYPE_PRIOR_RULES=$GALAXUS_DATA/out/type_prior/rules-EXP-0049.json \
TYPE_PRIOR_INNER=galaxus.model.zero_shot:from_preset \
ZERO_SHOT_PRESET=zero-shot-observe-first-no-context \
LLM_NAME=apertus-v1-5-70b \
uv run --extra local --env-file .env python -m galaxus.serve.final_run --predictor galaxus.model.ensemble:from_env \
    --exp EXP-0056 --best-file $GALAXUS_DATA/out/submissions/EXP-0056_dev.csv
```

- It asks for the passphrase (not echoed), decrypts, runs both members on each batch, combines them with the frozen
  routing, merges into the dev file, validates, and prints the `kaggle competitions submit …` command.
- **About 45 min:** the members run in turn per batch, so the time is the 70B's (2.38–2.42 img/s, EXP-0049) plus
  the SigLIP heads' (about 18–30 img/s). **Drill on Neel's laptop, 06:53:** this exact command passed decrypt,
  kill and resume, and validation. It ran 300 dev images at **1.978 img/s**, just under the 2.0 floor, with two
  other jobs on the gateway and the CPU. That is about 45 min for the 5,292 final images. Starting by 13:10 leaves
  about 20 min before 14:15.
- **Leave `--workers` at 1.** The zero-shot predictor already keeps 8 requests in flight per batch, which is what
  EXP-0049 measured on the 70B. More is unmeasured on the 70B; the `--workers 2` advice below was measured on the 8B.
- **If it stops** (429, a crash): run the same command again. Decrypted files are reused, and the cache keeps every
  member answer already received, so nothing is asked twice.
- **The Kaggle upload is a human's** (product rule 7), and **select** the submission afterwards.

### If the one command fails: the manual steps (#181)

`ensemble apply` reads `predictions/<member EXP>/final.jsonl`, so the member runs must use exactly these EXP ids.

1. Decrypt: `7zz x -o"$GALAXUS_DATA/final" "$GALAXUS_DATA/final_encrypted.bin"`. It asks for the passphrase.
2. Both members, in two terminals at the same time (each resumes if rerun):

   ```bash
   # A: SigLIP heads, about 3 min
   uv run --extra local --env-file .env python -m galaxus.serve.submit --exp EXP-0055 --part final \
       --run galaxus.model.helper:from_env
   # B: the 70B + type prior, about 37 min
   LLM_NAME=apertus-v1-5-70b TYPE_PRIOR_RULES=$GALAXUS_DATA/out/type_prior/rules-EXP-0049.json \
   TYPE_PRIOR_INNER=galaxus.model.zero_shot:from_preset ZERO_SHOT_PRESET=zero-shot-observe-first-no-context \
   uv run --env-file .env python -m galaxus.serve.submit --exp EXP-0051 --part final \
       --run galaxus.model.type_prior:from_env --batch-size 256
   ```

3. Combine: `uv run --env-file .env python -m galaxus.model.ensemble apply --routing $GALAXUS_DATA/out/ensemble/routing-EXP-0056.json --part final --exp EXP-0056`
4. The file: `uv run --env-file .env python -m galaxus.serve.submit --exp EXP-0056 --part dev --part final`

### Fallback B: the 70B has not finished by about 13:55

The SigLIP heads alone for the final rows (holdout 0.705 instead of 0.733); the dev rows stay EXP-0056's:

```bash
uv run --extra local --env-file .env python -m galaxus.serve.submit --exp EXP-0055 --part final \
    --run galaxus.model.helper:from_env \
    --base $GALAXUS_DATA/out/submissions/EXP-0056_dev.csv --name EXP-0055-final_on_EXP-0056-dev.csv
```

## After the passphrase is published

```bash
python -m galaxus.serve.final_run --predictor <package.module:name> --exp <EXP-id> \
    --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv
```

1. If the method calls Apertus over HTTP (gateway, HF endpoint or a vLLM server), add `--workers N` with the N that
   gave the best img/s in the dry run (`bench --workers N`). Keep the default of 1 for a model loaded in this process.
   **`--workers` multiplies the predictor's own concurrency.** The zero-shot predictor already sends
   `zero_shot.WORKERS = 8` requests per batch, so `--workers 2` means 16 requests in flight and `--workers 8` means 64.
   **Over the Hack Apertus gateway: use `--workers 2`.** It is the only value measured on the real gateway: the timed
   dry run on all 5,261 dev images (#59, Neel, main `b67da0c`) ran at **4.95 img/s** with no stop, about 2.5x the
   2 img/s floor, so the 5,292 final images need about 18 min of prediction. `--workers 4` and `8` (32 and 64 in
   flight) were **not** measured; don't use them on Saturday without a timed run first. Build the predictor's
   `GatewayBackend` with **`retries=4`** (the load test, #58: with the default of 2, a 429 burst longer than the
   retries stops the run).
   **For the LoRA on our own vLLM server** (`galaxus.model.lora:from_env`, one request per image): `--workers` is the
   number of requests in flight, so it needs its own timed run (`bench --workers 4, 8, 16`) on the GPU.
2. Paste the passphrase when asked (not echoed, never stored or logged).
3. If anything stops it, run the same command again: the decrypted files are reused and the runner resumes where it
   stopped, without duplicates.
   **If the gateway answers HTTP 429 (too many requests) and the run stops:** run the same command again with
   `--workers` halved (2 → 1). Nothing is lost: finished images are kept, only the rest is sent again. Check the
   time with `python -m galaxus.serve.preflight --images-per-second <new speed>` after the restart.
   **If it stops on HTTP 402:** every key in `.env` is out of credits; add another `LLM_API_KEY_n` and run again.
4. It ends with `valid: <file>` and the exact `kaggle competitions submit ...` command.
5. A human runs that command, waits for the score, then **selects** that submission on Kaggle.
6. If Kaggle is down: the fallback in challenge_description.md "Submission file" (the owner sends the validated file).

## Where things land (never in git)

| What | Where |
| --- | --- |
| predictions, one line per image | `$GALAXUS_DATA/out/predictions/<exp>/<part>.jsonl` |
| run timings | `$GALAXUS_DATA/out/predictions/<exp>/<part>.runs.jsonl` |
| validated submission files | `$GALAXUS_DATA/out/submissions/` |
| benchmark results | `$GALAXUS_DATA/out/bench/` |
| drill reports, and the files a drill extracted | `$GALAXUS_DATA/out/drill/<drill name>/` |
| decrypted final images | `$GALAXUS_DATA/final/` |
