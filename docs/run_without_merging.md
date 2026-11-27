# Running the pipeline while nothing is merged

Team decision, 27 Nov: no merges into `main` for now (#9). Every lane keeps working on its own branch. To run the
whole pipeline (data → model → serve) anyway, build a **local combined checkout** that merges the lane branches
on your machine only. It is never pushed: the branch name `local/run` matches no lane, so `ci` would reject it.

## 1. Build the combined checkout (once)

From your clone of this repository:

```bash
git fetch origin
git worktree add ../noidea-run origin/design/4-data-lane-vernika   # pyproject.toml + uv.lock from PR #8
cd ../noidea-run
git switch -c local/run
git merge --no-edit origin/data/work origin/model/work origin/serve/work
uv sync                                    # without uv: pip install -e . pytest
uv run pytest -q                           # without uv: python -m pytest -q   (all lanes, synthetic data only)
```

## 2. Refresh it when a lane pushes

```bash
cd ../noidea-run
git fetch origin
git merge --no-edit origin/data/work origin/model/work origin/serve/work
```

If a merge conflicts, do not resolve it by editing another lane's file: run `git merge --abort` and raise it on
the lanes' PRs (AGENTS.md "Hard rules": conflicts are escalated, never silently resolved).

## 3. Point it at the competition data

The data lives outside every repository (AGENTS.md product rule 3). The person who downloads it joins the competition
with the link in the [README](../README.md), accepts the rules, and checks the encrypted final archive as
[challenge_description.md, "Timeline"](../challenge_description.md#timeline) and [FAQ.md](../FAQ.md) describe.
With the Kaggle CLI (an API token from the Kaggle account settings in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c hackapertus-galaxus -p ~/galaxus-data
cd ~/galaxus-data && unzip -q hackapertus-galaxus.zip
export GALAXUS_DATA=~/galaxus-data            # PowerShell: $env:GALAXUS_DATA = "$HOME\galaxus-data"
```

## 4. The first Kaggle file (#15) and the Saturday rehearsal (#16)

```bash
uv run python -m galaxus.serve.submit --exp EXP-0001 --run galaxus.model.baselines:MajorityPredictor
uv run python -m galaxus.serve.final_run --dry-run --predictor galaxus.model.baselines:MajorityPredictor \
    --exp EXP-0001 --best-file $GALAXUS_DATA/out/submissions/EXP-0001_dev.csv
```

Without uv, drop `uv run` and use `python`. The first command ends with the exact `kaggle competitions submit ...` line: **a human runs it** (AGENTS.md product
rule 7). The commit in its message is the combined checkout's merge commit; write down the three lane commits
it contains (`git log -1 --format=%P`) next to the score, so the file can be traced.

Checked on 27 Nov on synthetic data: data/work + model/work + serve/work merged on top of PR #8 gave
`pytest` 273 passed, 1 skipped, and both commands above ended in a validated file.
