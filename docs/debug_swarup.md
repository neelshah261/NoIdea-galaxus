# Debugging pass on the real data (#68)

Swarup's machine, the only one with the Kaggle files in `$GALAXUS_DATA` (SHA-256 of `final_encrypted.bin` checked
against the Kaggle data page). Windows 11, Python 3.12.10, no `uv` (pip + `PYTHONPATH=src`), pandas 3.0.6.
Code: `main` at `fdecf94`, plus the serve fixes listed below. Run on Fri 27 Nov 2026, 17:00–18:00 CET.
No image ids, labels or competition files appear in this report.

## 1. Metric parity

`pytest tests/data -k parity` selects **nothing**: no test has "parity" in its name. The test that was always
skipped is `tests/data/test_metric.py::test_the_real_metric_exposes_the_entry_points_we_use`; with the real data it
**passes** (8/8 in that file). It only checks that metric.py loads and has `score` and `evaluate`.

So a real parity check was added: [`bench/metric_parity.py`](../bench/metric_parity.py). `metric.py` scores exactly
the cells where the solution holds a label, and the hosts build that solution from the rules in
challenge_description.md "Which cells are scored". The script rebuilds the scored-cell mask **independently from the
written rules** (human labels only) and compares it with ours (`galaxus.data.split.scored_mask` /
`solution_frame`), cell by cell and score by score:

| Set | Images | Cells only in our mask | Cells only in the rules' mask | Scores (3 submissions) |
| --- | --- | --- | --- | --- |
| Official holdout | 2,444 | 0 in all 6 dimensions | 0 in all 6 dimensions | identical (difference 0.0) |
| All of train | 11,876 | 0 in all 6 dimensions | 0 in all 6 dimensions | identical (difference 0.0) |

The three submissions were the majority answer, the labels with 20% noise, and random classes. **Parity OK:** the
local holdout score follows Kaggle's rules exactly. Dev and final are other images, so their scores still differ
from the holdout's; only an upload compares with the leaderboard itself.

## 2. Whole suite with the real data

| | Before | After the fixes |
| --- | --- | --- |
| `GALAXUS_DATA` = real Kaggle files | **12 failed, 2 errors**, 538 passed | **552 passed, 0 skipped** (twice) |
| `GALAXUS_DATA` unset (as in ci) | 550 passed, 1 skipped | 551 passed, 1 skipped |

**Cause of the 12 + 2:**
- `galaxus.data.metric.load_metric` caches metric.py once per process, whatever `$GALAXUS_DATA` says.
- Three serve test files (`test_demo`, `test_perspective_report`, `test_speed_matrix`) resolve the classes while the
  real data is set, so the **real** metric.py is cached. Later tests then check their invented class names against
  the real ones.
- Two perspective-report tests also read the real classes directly.
- Found by pairing every serve test file with `test_pipeline.py`. (A first guess, that a data test caused it, was
  wrong.)

**Fix (serve lane):** an autouse fixture in `tests/serve/conftest.py` removes the machine's `$GALAXUS_DATA` and
empties the metric cache before each serve test, so they run as in ci. Tests that need a data folder build their own.

The one skip left without data is the real-metric test above; with the data it runs and passes.

## 3. EXP-0001 (PR #67)

Re-run on a clean checkout of `main` (`fdecf94`) with
`--note "baseline, no model: images/s is not a product figure"`. The row says `dirty: false`, and the mean macro F1
is unchanged at 0.187. Pushed to `model/exp-0001` (`bb8d992`), replacing the earlier row in #67.

## 4. The Saturday path on the real dev data (majority predictor)

| Step | Command | Result | Time |
| --- | --- | --- | --- |
| Image cache | `python -m galaxus.data.prep --part dev --size 512 --workers 8` | 5,261 written, 0 missing | 139 s (39 img/s) |
| Drill | `final_run --drill … --archive <practice 7z> --best-file …EXP-0001_dev.csv` | **ready for Saturday: yes** | 193 s |
| Dry run | `final_run --dry-run … --best-file …EXP-0001_dev.csv` | valid file, 10,553 rows | 117 s |
| Gallery (#31) | `python -m galaxus.serve.demo --exp EXP-0001 --part dev` | written | 146 s |
| Perspective report (#34) | `python bench/perspective_report.py --exp EXP-0001` | tables and the slide 8 figure | 2 s |

**The drill's steps, all OK:**
- **decrypt:** 7-Zip extracted a **practice** archive built for this, with a practice passphrase. The real
  passphrase does not exist yet.
- **items and kill:** killed after 164 of 329 batches.
- **resume:** skipped the 2,624 already written and predicted 2,637 more, with no duplicates.
- **merge, validate, submit command:** all OK.

The first drill run failed at "merge" because I passed a Git Bash path (`/c/Users/...`) to Windows Python. That is
my invocation, not a code fault; with a Windows path it passes.

**Where the time goes:** predicting took 0.6 s (majority), but **merge and validate took about 96 s each**. See
problem 1.

## 5. Fixes in the serve lane (on `serve/work`)

| Commit | Problem | Fix |
| --- | --- | --- |
| `1829109` | `python -m galaxus.serve.final_run --help` and `galaxus.serve.submit --help` **crashed** on a Windows console (`UnicodeEncodeError`: a "→" in the description). Every other command's `--help` was checked and is fine | ASCII arrows; both commands switch stdout/stderr to `errors="replace"`; a regression test checks every serve command's description against cp1252 |
| `b022187` | 12 failures + 2 errors with the real data (section 2) | the serve tests are isolated from the real `$GALAXUS_DATA` and the metric cache |

## 6. Open problems (other lanes)

1. **`validate_submission` is O(n²): 96–125 s per real file** (data lane, reported on #6 with the profile).
   - `validate.py:77` rebuilds `set(ids)` for each of the 10,553 ids.
   - On Saturday the file is validated in `final_run`, the drill, `preflight` and possibly by hand, which costs
     minutes of the 90.
   - The one-line fix is described on #6.
2. **`load_metric`'s cache ignores `$GALAXUS_DATA`** (data lane). The serve tests now work around it; a cache keyed by
   the file path would remove the trap for every lane.
3. **The image cache path changed** from `cache/images/512/` to `cache/images/512-p32/` (#28). A cache filled before
   that is not reused: here, `prep` rebuilt all 5,261 dev images (139 s). No harm, but the GPU session should run
   `prep` with the same code it predicts with.
