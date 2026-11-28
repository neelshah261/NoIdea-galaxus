# Fine-tuned SigLIP 2 base (#202): code, eval rows and weights

**Status: an experiment, not the method.** SigLIP 2 picks every class here and Apertus is not used, so it is not
covered by Q-0008 (`decisions.md`). The rules question (Q-0009) is open. #214's `apertus-core` can take this model's
scores as an optional input block (EXP-0094) once that is answered. Until then EXP-0061 is the method.

| EXP | What | Holdout macro F1 |
| --- | --- | --- |
| EXP-0067 | last 4 blocks, 1 probe + 3 epochs | 0.805 |
| **EXP-0090** | last 4 blocks, 1 probe + 5 epochs (**the published weights**) | **0.816** |
| EXP-0091 | all 12 blocks, 1 probe + 3 epochs | 0.805 |
| EXP-0068 | ensemble-agree ft=EXP-0067 + 70b=EXP-0051 | 0.817 cross-fitted; its dev file is identical to EXP-0067's (#202) |
| EXP-0069 | ensemble-agree ft=EXP-0067 + fused=EXP-0059 + 70b=EXP-0051 | 0.805 |

The full eval rows are in `rows.json`. All were scored by `python -m galaxus.model.eval` on the official holdout
(2444 images) from main `604d0b2`, on an M5 Pro (MPS).

## Files

- `ft_siglip2.py`: the predictor and its training. It registers `helper-siglip2-ft` with the eval script. This file
  differs from the one that produced the rows in one line: the learning-rate lambda binds its loop variables as
  default arguments (ruff B023). Their values don't change after epoch 1, so training is identical.
- `rows.json`: the five EXP rows.
- **The weights** are not in git (372 MB, over GitHub's 100 MB file limit). They are a release asset:
  [`ft-siglip2-exp0090`](../../../../releases/tag/ft-siglip2-exp0090), file `siglip2-ft-b4-e5-exp0090.pt`,
  sha256 `6ae5941d707bd4bc87b9230beff296a54c169e4371575abfb75cfb6ddc8ca03d`.

## The weights file

The file is a `torch.save` dict that loads with `torch.load(path, weights_only=True)`, so no pickled code is run.
- `state`: the vision tower (`google/siglip2-base-patch16-224` at `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`) and
  the six linear heads, fp32, 92.9 M parameters.
- `classes`, `dims`: each dimension's class order. These are the class names that are already in git.
- `bias`: the per-class bias fitted on the validation fold.
- `meta`: the hyperparameters, the training row count (9432: train minus the official holdout) and the validation scores.

There are no images, image ids or per-image labels in it.

## Reuse the weights

Put the file at `$GALAXUS_DATA/out/finetune/siglip2-b4-e5/model.pt` and run with the same tag. The script reuses
saved weights when the row count matches:

    FT_TAG=siglip2-b4-e5 FT_EPOCHS=5 GALAXUS_DATA=... PYTHONPATH=design/ft_siglip2 \
        python design/ft_siglip2/ft_siglip2.py --exp-id EXP-#### --results rows.json

To train it again, give no saved file (about 5 min on an M-series Mac).
