"""#202 (S3) as a scratch experiment, no repo change (as #176's rows were): SigLIP 2 base fine-tuned on this Mac's
GPU (MPS) on train minus the official holdout, scored by the eval script on the holdout.

* one linear-probe epoch, then the last FT_BLOCKS blocks + the pooling head unfrozen, one linear head per dimension;
* #51's tempered class weights ((n / (k * count)) ** heads.CLASS_WEIGHT_POWER), the loss summed over the dimensions,
  unscored cells skipped;
* the data lane's mirrored rows (text-free images, left/right perspective labels swapped) as extra perspective rows,
  and the mirror average at prediction for text-free answers (as ``helper-siglip-mirror``);
* product-grouped validation (one of FT_VAL_FOLDS folds) picks the epoch and fits #51's per-class bias;
* never the holdout: the eval script passes train minus the holdout.

    GALAXUS_DATA=... PYTHONPATH=<this folder> python ft_siglip2.py --exp-id EXP-0067 --results rows-ft.json

The weights stay outside git, in $GALAXUS_DATA/out/finetune/<FT_TAG>/; a later run with the same tag reuses them
(``from_env`` for ``submit --run ft_siglip2:from_env``).
"""

import io
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from galaxus.data import split
from galaxus.data.images import IMAGE_SIZE, PATCH_SIZE, cached_jpeg
from galaxus.model import eval as ev
from galaxus.model import heads, helper
from galaxus.model.calibration import apply_bias, fit_bias, macro_f1
from galaxus.types import Prediction

REPO, REVISION = "google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SMOKE = os.environ.get("FT_SMOKE", "") not in ("", "0")
BLOCKS = int(os.environ.get("FT_BLOCKS", "4"))
EPOCHS = 1 if SMOKE else int(os.environ.get("FT_EPOCHS", "3"))
LR = float(os.environ.get("FT_LR", "3e-5"))
HEAD_LR = float(os.environ.get("FT_HEAD_LR", "1e-3"))
BATCH = int(os.environ.get("FT_BATCH", "32"))
VAL_FOLDS = int(os.environ.get("FT_VAL_FOLDS", "7"))
USE_BIAS = os.environ.get("FT_BIAS", "1") not in ("", "0")
TTA = os.environ.get("FT_TTA", "1") not in ("", "0")
TAG = os.environ.get("FT_TAG", f"siglip2-b{BLOCKS}-e{EPOCHS}") + ("-smoke" if SMOKE else "")
NAME = "helper-siglip2-ft"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SIZE = 224  # the checkpoint's processor: 224 x 224, bilinear, (x / 255 - 0.5) / 0.5


def log(message: str) -> None:
    print(f"[ft {time.strftime('%H:%M:%S')}] {message}", flush=True)


def pixels(pairs: list[tuple[str, str]]) -> torch.Tensor:
    """uint8 (n, 3, 224, 224): the prepared JPEG the frozen embeddings read, resized as the processor does."""
    def one(pair: tuple[str, str]) -> np.ndarray:
        jpeg = cached_jpeg(pair[0], pair[1], image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
        image = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

    with ThreadPoolExecutor(8) as pool:
        return torch.from_numpy(np.stack(list(pool.map(one, pairs))))


class Net(torch.nn.Module):
    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        snapshot = snapshot_download(REPO, revision=REVISION, allow_patterns=helper.SNAPSHOT_FILES)
        self.vision = AutoModel.from_pretrained(snapshot).vision_model
        width = self.vision.config.hidden_size
        self.heads = torch.nn.ModuleList(torch.nn.Linear(width, k) for k in sizes)

    def forward(self, u8: torch.Tensor, frozen: bool = False) -> list[torch.Tensor]:
        x = u8.float().div(127.5).sub(1.0)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            z = self.vision(pixel_values=x).pooler_output
        return [head(z).float() for head in self.heads]

    def unfreeze(self, blocks: int) -> list[torch.nn.Parameter]:
        for p in self.vision.parameters():
            p.requires_grad = False
        modules = [*self.vision.encoder.layers[-blocks:], self.vision.post_layernorm, self.vision.head] if blocks else []
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True
        return [p for p in self.vision.parameters() if p.requires_grad]


def class_weights(targets: np.ndarray, k: int) -> np.ndarray:
    """#51's tempered weights over the labelled cells of one dimension (``heads.fit_head``)."""
    counts = np.bincount(targets[targets >= 0], minlength=k).astype(np.float64)
    seen = counts > 0
    weights = np.zeros(k)
    weights[seen] = (counts[seen].sum() / (seen.sum() * counts[seen])) ** heads.CLASS_WEIGHT_POWER
    return weights.astype(np.float32)


@torch.no_grad()
def log_probs(net: Net, images: torch.Tensor, flip: bool = False, step: int = 128) -> list[np.ndarray]:
    """Per dimension, (n, k) log-probabilities of ``images`` (uint8, any device)."""
    net.eval()
    out: list[list[np.ndarray]] = [[] for _ in net.heads]
    for start in range(0, len(images), step):
        x = images[start : start + step].to(DEVICE)
        if flip:
            x = torch.flip(x, dims=[3])
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = net(x)
        for j, value in enumerate(logits):
            out[j].append(torch.log_softmax(value.float(), dim=-1).cpu().numpy())
    net.train()
    return [np.concatenate(parts) for parts in out]


def swap_back(classes: list[str], swap) -> list[int]:
    column = {name: k for k, name in enumerate(classes)}
    return [column.get(swap(name), k) for k, name in enumerate(classes)]


def mirror_average(original: np.ndarray, mirrored: np.ndarray, back: list[int]) -> np.ndarray:
    """``helper.mirror_average`` on log-probabilities: the mirror's left and right swapped back."""
    return np.log(np.clip(0.5 * (np.exp(original) + np.exp(mirrored)[:, back]), 1e-12, None))


def train(rows, data, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dims = list(data.dimensions())
    setup = helper.MirrorSetup.from_classes(data.allowed_classes())
    p, t = dims.index(setup.dimension), dims.index(setup.text_dimension)
    ids = rows["image_id"].astype(str).tolist()

    classes: dict[str, list[str]] = {}
    target = np.full((len(rows), len(dims)), -1, dtype=np.int64)
    for j, d in enumerate(dims):
        mask = data.scored_mask(rows, d).to_numpy()
        labels = rows[d].astype(str).to_numpy()
        classes[d] = sorted(set(labels[mask]))
        index = {c: k for k, c in enumerate(classes[d])}
        target[mask, j] = [index[c] for c in labels[mask]]
    is_val = split.product_folds(rows, n_folds=VAL_FOLDS, random_state=seed).to_numpy() == 0

    fit_rows = rows[~is_val]
    extra = setup.extra_rows(fit_rows[data.scored_mask(fit_rows, setup.dimension)])
    position = {image_id: k for k, image_id in enumerate(ids)}
    p_index = {c: k for k, c in enumerate(classes[setup.dimension])}
    extra_target = np.full((len(extra), len(dims)), -1, dtype=np.int64)
    extra_target[:, p] = [p_index.get(c, -1) for c in extra[setup.dimension].astype(str)]
    entry_row = np.concatenate([np.flatnonzero(~is_val), [position[str(i)] for i in extra["image_id"]]]).astype(int)
    entry_flip = np.concatenate([np.zeros(int((~is_val).sum()), bool), np.ones(len(extra), bool)])
    entry_target = np.concatenate([target[~is_val], extra_target])
    keep = (entry_target >= 0).any(axis=1)
    entry_row, entry_flip, entry_target = entry_row[keep], entry_flip[keep], entry_target[keep]
    log(f"{len(rows)} rows: {int((~is_val).sum())} fit + {len(extra)} mirrored, {int(is_val.sum())} validate; "
        f"device {DEVICE}, blocks {BLOCKS}, epochs 1 probe + {EPOCHS}")

    started = time.time()
    images = pixels(list(zip(ids, rows["image_file"].astype(str), strict=True))).to(DEVICE)
    log(f"images ready in {time.time() - started:.0f} s")
    net = Net([len(classes[d]) for d in dims]).to(DEVICE)
    weights = [torch.tensor(class_weights(entry_target[:, j], len(classes[d])), device=DEVICE)
               for j, d in enumerate(dims)]
    val_rows = np.flatnonzero(is_val)

    def val_score() -> tuple[float, dict[str, float]]:
        lp = log_probs(net, images[torch.as_tensor(val_rows, device=DEVICE)])
        scores = {}
        for j, d in enumerate(dims):
            true = target[val_rows, j]
            ok = true >= 0
            scores[d] = round(macro_f1(true[ok], lp[j][ok].argmax(1)), 4)
        return float(np.mean(list(scores.values()))), scores

    best = (-1.0, None, None, -1)
    steps_per_epoch = math.ceil(len(entry_row) / BATCH)
    for epoch in range(1 + EPOCHS):
        probe = epoch == 0
        if epoch <= 1:
            backbone = net.unfreeze(0 if probe else BLOCKS)
            groups = [{"params": list(net.heads.parameters()), "lr": HEAD_LR if probe else LR * 10}]
            if not probe:
                groups.append({"params": backbone, "lr": LR})
            optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
            total = steps_per_epoch * (1 if probe else EPOCHS)
            warm = 0 if probe else max(1, total // 20)
            schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s, probe=probe, warm=warm, total=total: 1.0 if probe else (
                (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
        order = rng.permutation(len(entry_row))
        losses, tick = [], time.time()
        for start in range(0, len(order), BATCH):
            pick = order[start : start + BATCH]
            x = images[torch.as_tensor(entry_row[pick], device=DEVICE)]
            flip = torch.as_tensor(entry_flip[pick], device=DEVICE)
            if bool(flip.any()):
                x[flip] = torch.flip(x[flip], dims=[3])
            y = torch.as_tensor(entry_target[pick], device=DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                logits = net(x, frozen=probe)
            loss = sum(torch.nn.functional.cross_entropy(logits[j], y[:, j], weight=weights[j], ignore_index=-1)
                       for j in range(len(dims)) if bool((y[:, j] >= 0).any()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(loss.detach().item())
        score, scores = val_score()
        log(f"epoch {epoch} ({'probe' if probe else 'fine-tune'}): loss {np.mean(losses):.3f}, "
            f"{len(order) / (time.time() - tick):.0f} img/s, val macro F1 {score:.4f} {scores}")
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, scores, epoch)

    net.load_state_dict(best[1])
    val_images = images[torch.as_tensor(val_rows, device=DEVICE)]
    lp = log_probs(net, val_images)
    free = target[val_rows, t] == classes[setup.text_dimension].index(setup.no_text) \
        if setup.no_text in classes[setup.text_dimension] else np.zeros(len(val_rows), bool)
    back = swap_back(classes[setup.dimension], setup.swap)
    if TTA and free.any():
        mirrored = log_probs(net, val_images[torch.as_tensor(np.flatnonzero(free), device=DEVICE)], flip=True)
        lp[p][free] = mirror_average(lp[p][free], mirrored[p], back)
    bias = {}
    for j, d in enumerate(dims):
        true = target[val_rows, j]
        ok = true >= 0
        bias[d] = fit_bias(lp[j][ok], true[ok]) if USE_BIAS and ok.sum() > 1 else np.zeros(len(classes[d]))
    meta = {"tag": TAG, "base": REPO, "revision": REVISION, "rows": len(rows), "blocks": BLOCKS, "epochs": EPOCHS,
            "probe_epochs": 1, "lr": LR, "head_lr": HEAD_LR, "batch": BATCH, "val_folds": VAL_FOLDS,
            "best_epoch": best[3], "val_macro_f1": round(best[0], 4), "val_per_dimension": best[2],
            "bias": USE_BIAS, "mirror_tta": TTA, "mirrored_rows": len(extra), "device": DEVICE, "smoke": SMOKE,
            "minutes": round((time.time() - started) / 60, 1)}
    log(f"best epoch {best[3]}, val {best[0]:.4f}")
    return {"state": best[1], "classes": classes, "dims": dims, "bias": {d: np.asarray(b).tolist() for d, b in bias.items()},
            "meta": meta}


class FineTunedPredictor:
    name = NAME

    def __init__(self, saved: dict, setup: helper.MirrorSetup) -> None:
        self.dims, self.classes, self.meta, self.setup = saved["dims"], saved["classes"], saved["meta"], setup
        self.bias = {d: np.asarray(b) for d, b in saved["bias"].items()}
        self.net = Net([len(self.classes[d]) for d in self.dims]).to(DEVICE)
        self.net.load_state_dict(saved["state"])
        self.net.eval()
        self.back = swap_back(self.classes[setup.dimension], setup.swap)
        self.model, self.model_revision, self.helpers = REPO, REVISION, (f"{REPO} fine-tuned ({TAG})",)

    def predict(self, batch) -> list[Prediction]:
        images = pixels([(str(item.image_id), str(item.path)) for item in batch])
        lp = log_probs(self.net, images)
        p, t = self.dims.index(self.setup.dimension), self.dims.index(self.setup.text_dimension)
        text = [self.classes[self.setup.text_dimension][k] for k in apply_bias(lp[t], self.bias[self.setup.text_dimension])]
        free = np.array([label == self.setup.no_text for label in text])
        if TTA and free.any():
            mirrored = log_probs(self.net, images[torch.as_tensor(np.flatnonzero(free))], flip=True)
            lp[p][free] = mirror_average(lp[p][free], mirrored[p], self.back)
        chosen = {d: apply_bias(lp[j], self.bias[d]) for j, d in enumerate(self.dims)}
        return [Prediction(image_id=str(item.image_id),
                           labels={d: self.classes[d][chosen[d][n]] for d in self.dims},
                           confidence={d: float(np.exp(lp[j][n, chosen[d][n]])) for j, d in enumerate(self.dims)})
                for n, item in enumerate(batch)]

    def config(self) -> dict[str, object]:
        return {"fine_tune": self.meta}


def build(rows, data, seed: int) -> FineTunedPredictor:
    if SMOKE:
        rows = rows.sample(n=min(len(rows), 300), random_state=seed)
    rows = rows.reset_index(drop=True)
    path = Path(os.environ["GALAXUS_DATA"]) / "out" / "finetune" / TAG / "model.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["meta"]["rows"] != len(rows):
            raise ValueError(f"{path} was trained on {saved['meta']['rows']} rows, not these {len(rows)}: use a new FT_TAG")
        log(f"reusing {path}")
    else:
        saved = train(rows, data, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, path)
        log(f"saved {path}")
    return FineTunedPredictor(saved, helper.MirrorSetup.from_classes(data.allowed_classes()))


ev.PREDICTORS[NAME] = build


def from_env() -> FineTunedPredictor:
    """For ``submit --run ft_siglip2:from_env``: the saved weights of FT_TAG, trained on train minus the holdout."""
    data = ev.DataLane.real()
    return build(data.split_holdout(data.read_train())[0], data, 0)


if __name__ == "__main__":
    sys.exit(ev.main([NAME, *sys.argv[1:]]))
