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

## After the passphrase is published

```bash
python -m galaxus.serve.final_run --predictor <package.module:name> --exp <EXP-id> \
    --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv
```

1. If the method calls Apertus over HTTP (gateway, HF endpoint or a vLLM server), add `--workers N` with the N that
   gave the best img/s in the dry run (`bench --workers N`). Keep the default of 1 for a model loaded in this process.
2. Paste the passphrase when asked (not echoed, never stored or logged).
3. If anything stops it, run the same command again: the decrypted files are reused and the runner resumes where it
   stopped, without duplicates.
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
