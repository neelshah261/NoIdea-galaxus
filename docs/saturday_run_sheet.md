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

## Dry run on dev (PLAN.md "Hard checkpoint")

```bash
python -m galaxus.serve.final_run --dry-run --predictor <package.module:name> --exp <EXP-id>
python -m galaxus.serve.bench --predictor <package.module:name> --part dev --limit 256 --dollars-per-hour <GPU price>
```

The dry run prints the total time and the images per second; the benchmark appends img/s, peak memory and the cost
per 100,000 images to `$GALAXUS_DATA/out/bench/<label>.jsonl`, for section 7 of the technical report.

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
| decrypted final images | `$GALAXUS_DATA/final/` |
