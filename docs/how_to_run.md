# How to run it: one command per step

For judges re-running our numbers and for the team on Saturday. The rules, times and file formats are in
[challenge_description.md](../challenge_description.md); this page only lists commands. Every command below was run
on 27 Nov with the real data unless it is marked **(GPU)**, **(needs PR #51)**, **(needs the key)** or
**(not run yet)**.

## 0. Setup (once)

```bash
git clone https://github.com/neelshah261/NoIdea-galaxus && cd NoIdea-galaxus
uv sync                      # without uv: pip install -e . pytest ruff
uv run pytest -q             # synthetic data only, no network
```

The competition data never lives in the repository. Download it into a folder of your own and point
`GALAXUS_DATA` at it:

```bash
kaggle competitions download -c hackapertus-galaxus -p ~/galaxus-data   # Kaggle API token in ~/.kaggle/
cd ~/galaxus-data && unzip -q hackapertus-galaxus.zip && cd -
export GALAXUS_DATA=~/galaxus-data          # PowerShell: $env:GALAXUS_DATA = "$HOME\galaxus-data"
sha256sum ~/galaxus-data/final_encrypted.bin    # must equal the SHA-256 on the Kaggle data page
```

Without uv, drop `uv run` from the commands below.

## 1. Baseline, end to end (CPU, minutes)

```bash
uv run python -m galaxus.model.eval majority                     # EXP row on the official holdout
uv run python -m galaxus.serve.submit --exp EXP-0001 --run galaxus.model.baselines:MajorityPredictor
```

The second command writes a validated file to `$GALAXUS_DATA/out/submissions/` and **prints** the
`kaggle competitions submit` command. A human runs it (AGENTS.md product rule 7).

## 2. Apertus

| Step | Command |
| --- | --- |
| Is the gateway answering, does it take images? **(needs the key)** | `uv run --env-file .env python -m galaxus.model.apertus.smoke --backend api` |
| Serve Apertus 8B ourselves at 512 px **(GPU)** | `bash bench/vllm_apertus.sh` (see [serving_apertus.md](serving_apertus.md)) |
| Main method: heads on Apertus, whole GPU session in one go **(GPU, needs PR #51)** | `uv run python bench/gpu_session.py --dollars-per-hour <GPU price> --exp-id <EXP>` |
| …or step by step **(needs PR #51)** | `python -m galaxus.model.heads extract --part train` (GPU), then `… fit` (CPU) |
| Bonus csv of suspected train label errors **(needs PR #51)** | `uv run python -m galaxus.model.heads label-errors` |

`bench/gpu_session.py --dry-run` lists the steps and what is still missing before any paid minute is spent.
**Stop the GPU right after the session:** idle time is billed too.

## 3. Any predictor over dev or the holdout

```bash
uv run python -m galaxus.serve.runner --predictor <package.module:name> --exp <EXP> --part dev --workers 2
uv run python -m galaxus.serve.submit --exp <EXP>                    # validated file + the submit command
```

`--workers N` keeps N batches in flight; use it for HTTP backends only. It multiplies the predictor's own concurrency
(the zero-shot predictor sends 8 requests per batch, so `--workers 2` is 16 in flight): `2` is the value measured on
the gateway (#59, 4.95 img/s); see [the Saturday run sheet](saturday_run_sheet.md). A run that stops resumes where it
left off.

## 4. Evidence for the judges

| What | Command | Lands in |
| --- | --- | --- |
| Speed, memory, cost per 100k | `uv run python -m galaxus.serve.bench --predictor <…> --limit 256 --dollars-per-hour <…>` | `$GALAXUS_DATA/out/bench/` |
| Speed matrix | see [bench/README.md](../bench/README.md) | `$GALAXUS_DATA/out/bench/` |
| Perspective errors and the slide 8 figure | `uv run python bench/perspective_report.py --exp <EXP>` | `$GALAXUS_DATA/out/` |
| Hero-shot gallery demo | `uv run python -m galaxus.serve.demo --exp <EXP> --part dev` | `$GALAXUS_DATA/out/demo/` |
| Technical report section 7 | `uv run python -m galaxus.serve.report --exp <EXP> --kaggle --out docs/report_section7.md` | that file |

Every number in the report comes from `results/metrics.json` or these generated files; unmeasured values print as
`TODO(...)`, never as a guess.

## 5. Saturday

Full run sheet: [saturday_run_sheet.md](saturday_run_sheet.md). In short:

```bash
uv run python -m galaxus.serve.final_run --drill --predictor <…> --exp <EXP>        # rehearsal on dev (not run yet on real data)
uv run python -m galaxus.serve.preflight --best-file $GALAXUS_DATA/out/submissions/<best>.csv --images-per-second <…>
uv run python -m galaxus.serve.final_run --predictor <…> --exp <EXP> --best-file $GALAXUS_DATA/out/submissions/<best>.csv   # Saturday (not run yet)
```

`preflight` prints GO / NO-GO: submissions left today, whether the run ends before the upload-by time, the archive's
SHA-256, 7-Zip, the best dev file and disk space. `final_run` asks for the passphrase (never stored), decrypts,
predicts, merges into the best dev file, validates and prints the submit command; a human uploads **and selects** it.

## Reproducing a submitted score

Check out the commit named in technical_report.md section 7, download the weights at the named revision, set
`GALAXUS_DATA`, and run step 5's `final_run` with the same predictor and EXP. The holdout numbers come back with
`python -m galaxus.model.eval` from the same commit.
