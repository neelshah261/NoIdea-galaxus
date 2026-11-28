# After training: score EXP-0004 cheaply, raise it for free, run Saturday

For Swarup's laptop, after the LoRA job (#76) has uploaded `exp0004-e1/adapter` to the private repo
`<you>/noidea-lora-checkpoints`. **Budget rule (Swarup, 27 Nov): total HF spend at most ~$7.50.** That leaves
**one packed serving session tonight (~45 min, ~$1.35) and one on Saturday (~30 min, ~$0.90)**. Everything else runs
on the laptop for free, from the saved holdout answers. An L40S costs $1.80/h ($0.03/min) whether it is busy or idle,
so **cancel the job the minute you are done**.

The code lives on local branches (not pushed): the model side on `model/after-eval-sample`
(worktree `C:\Swarup\Claude\HackApertus\noidea-after-mirror`, which contains #122 confidences and recalibration, mirror
TTA for the LoRA, consistency rules, 8 requests in flight, the holdout sample), and the serving scripts on
`serve/after-checkpoints` (worktree `noidea-after-ckpt`). Run the model commands from the model worktree:
```bash
cd /c/Swarup/Claude/HackApertus/noidea-after-mirror
export PYTHONPATH=src GALAXUS_DATA=$HOME/galaxus-data
```

## 0. Before paying anything (free, ~10 min)

The holdout is part of train. Its 512 px images, and their mirrors, must be in the laptop's cache, or the first
minutes of paid serving go to image preparation:
```bash
python -m galaxus.data.prep --part train --size 512 --workers 4
python -m galaxus.data.prep --part train --size 512 --workers 4 --mirror
```

## 1. Tonight: one serving session (~45 min, ~$1.35)

```bash
cd /c/Swarup/Claude/HackApertus/noidea-after-ckpt
TIMEOUT=60m bash bench/hf_job_serve_start.sh            # prints the job id; merge mode, the final adapter
```
The job downloads Apertus and the adapter, merges them and starts vLLM, in ~10–15 min (unmeasured). Follow it with
`hf jobs logs -f <id>` (with `HF_TOKEN` exported from `.env` and `PYTHONUTF8=1`). Once the log says vLLM is listening:
```bash
eval "$(bash bench/hf_job_serve_start.sh url <id>)"     # LORA_BASE_URL, LORA_API_KEY (from .env), LORA_SERVED_MODEL
curl -s -H "Authorization: Bearer $LORA_API_KEY" "$LORA_BASE_URL/models"
export LORA_WORKERS=16                                     # requests in flight; vLLM batches them on the GPU
```
Then, in the model worktree, back to back:

| Step | Command | What it gives |
| --- | --- | --- |
| a | `python -m galaxus.model.eval lora --exp-id EXP-0004 --usd-per-hour 1.80 --note "LoRA 1 epoch"` | **the** holdout score. Check `fallbacks` and `no_confidence_cells` in the row: both should be ~0 |
| b | `MIRROR_SCORED_ONLY=1 python -m galaxus.model.eval mirror-tta:lora --exp-id EXP-0020 --usd-per-hour 1.80` | mirror TTA on perspective. Only scored, text-free holdout images get the second request, so its img/s understates production |
| c | `python -m galaxus.serve.bench --predictor galaxus.model.lora:from_env --part dev --limit 256 --workers 1` and again with `--workers 2` | our own img/s (20% of the grade) |
| d | `hf jobs cancel <id>` | **stop paying** |

Sanity check before step a (30 s): send one request and read `usage.prompt_tokens` in the reply. Training's prompt
was ~1,380 tokens (1,423 with the answer). If it is far lower, the image never reached the model: stop, and switch
`--chat-template-content-format` from `string` to `openai` in `bench/hf_job_serve.sh`.

No LoRA dev file for Kaggle tonight: the zero-shot upload already proved the format, and 5,261 dev images would add
~10 min of paid serving.

## 2. Tonight, free on the laptop: raise the score from the saved answers

None of these calls the model again; each reads `$GALAXUS_DATA/out/predictions/<EXP>/holdout.jsonl` and writes an EXP
row through metric.py, cross-fitted on product halves (an honest number):

| Step | Command |
| --- | --- |
| recalibration (#42) | `RECAL_BASE=EXP-0004 python -m galaxus.model.eval recalibrated --exp-id EXP-0021` then `python -m galaxus.model.recalibrate learn --base EXP-0004` (saves the whole-holdout rules for Saturday) |
| recalibration on the mirror run | `RECAL_BASE=EXP-0020 python -m galaxus.model.eval recalibrated --exp-id EXP-0022` |
| consistency rules (#101 item 10) | `RULES_BASE=EXP-0021 python -m galaxus.model.eval consistent --exp-id EXP-0023` then `python -m galaxus.model.consistency select --base EXP-0021` |
| per-dimension routing (#87) | `ENSEMBLE_MEMBERS="lora=EXP-0004 recal=EXP-0021 mirror=EXP-0022" python -m galaxus.model.eval ensemble --exp-id EXP-0024` |
| compare | `python -m galaxus.model.compare EXP-0002 EXP-0004 EXP-0020 EXP-0021 EXP-0022 EXP-0023 EXP-0024` |

Choose the **best holdout mean macro F1**, then check its speed: mirror TTA costs a second request on text-free
images; recalibration and consistency cost nothing. The ensemble row's img/s assumes every member is its own model
pass, which **understates** routing between variants of the same LoRA. Zero-shot can join the routing only with its
saved holdout answers (on Neel's laptop) or a gateway run with the team key.

EXP ids 0020–0024 are suggestions, chosen to avoid the ones the plan reserves (0004, 0006, 0010). Use any free ones.

## 3. Earlier checkpoints (only if the budget allows)

Checkpoints are saved every 100 steps (`exp0004-e1/checkpoint-100` … `-800`). The research says heavy fine-tuning can
cost rare classes, so an earlier save may score higher. To compare cheaply:
- **On a sample:** `EVAL_SAMPLE=600` scores a seeded, product-grouped ~600-image subset (the row says `sample_of`).
  Compare checkpoints only against each other on the same sample (same `EVAL_SAMPLE_SEED`), never against a full row.
- **One checkpoint per session (merge, safe):** `ADAPTER_PATH=exp0004-e1/checkpoint-500 bash bench/hf_job_serve_start.sh`,
  about 15 min + 2 min for 600 images, ~$0.50.
- **Several in one session (no merge, untested):** `SERVE_MODE=lora
  ADAPTERS="final=exp0004-e1/adapter,ck500=exp0004-e1/checkpoint-500,ck300=exp0004-e1/checkpoint-300" bash
  bench/hf_job_serve_start.sh`, then per adapter `LORA_SERVED_MODEL=ck500 EVAL_SAMPLE=600 python -m galaxus.model.eval lora`.
  This only works if Swiss AI's vLLM build supports LoRA for Apertus 1.5. If vLLM fails to start, cancel at once.

## 4. Saturday

- **10:00 drill, free:** `python -m galaxus.serve.final_run --drill --predictor galaxus.model.baselines:MajorityPredictor
  --exp EXP-0001`. It rehearses decryption, resume and the file path without a GPU.
- **12:40:** `TIMEOUT=45m bash bench/hf_job_serve_start.sh` (the adapter chosen in step 2 or 3), then `url`, the exports and the curl check.
- **13:00, passphrase on Discord:** run the chosen chain. For example, the LoRA with recalibration and consistency
  (the rules files from step 2):
  ```bash
  export RECAL_RULES=$GALAXUS_DATA/out/recalibrate/rules-EXP-0004.json RECAL_INNER=galaxus.model.lora:from_env
  export CONSISTENCY_RULES=$GALAXUS_DATA/out/consistency/rules-EXP-0021.json CONSISTENCY_INNER=galaxus.model.recalibrate:from_env
  python -m galaxus.serve.final_run --predictor galaxus.model.consistency:from_env --exp EXP-00NN --workers 2 \
      --best-file <best dev file>
  ```
  Use `RECAL_INNER=galaxus.model.lora:mirror_tta_from_env` if mirror TTA won in step 2. 5,292 images at ~7 img/s is
  ~13 min (unmeasured; step 1c gives the real rate).
- **Right after the valid file is written: `hf jobs cancel <id>`.** Neel uploads and selects it.
- **Fallback** if serving fails: zero-shot through the gateway, `--workers 2` (#59: 4.95 img/s, ~18 min), run by Neel
  (the team key is his).

## Costs at a glance

| Item | Time | Cost |
| --- | --- | --- |
| Tonight's serving session | ~45 min | ~$1.35 |
| Saturday's serving session | ~30 min | ~$0.90 |
| One extra checkpoint (optional) | ~17 min | ~$0.50 |
| Everything in section 2 and the drill | laptop | $0 |
