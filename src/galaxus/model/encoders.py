"""SigLIP 2 so400m: a larger, sharper frozen image encoder for the heads (#214, the research comment there).

BYMO's frozen linear probe scored SigLIP 2 so400m at 512 px 0.759 against SigLIP 2 base at 224 px 0.682, with
text_language 0.696 against 0.511 (#214). Our fine-tune (EXP-0090) and EXP-0059's encoders all read 224 px.

* **On the best device:** the data lane's ``neighbours.load_encoder`` runs on the CPU, which suits SigLIP base at 224 px
  (about 35 GFLOPs per image) but not so400m at 512 px (about 1 TFLOP per image). This loader puts the model on
  cuda (bf16, or fp16 without bf16), else Apple's mps (fp16), else the CPU (fp32), and keeps the data lane's contract
  (``encode(images) -> one row per image``) and cache (``neighbours.embedding_dir``), so ``helper.embeddings`` and every
  heads predictor read its vectors unchanged.
* **The pin:** :data:`PINS`, read from the Hub with ``python -m galaxus.model.encoders pin`` on Neel's laptop (#215
  review at b3ef2e2), as every other encoder is pinned (#182). ``$HELPER_SO400M_REVISION`` may override it with another
  full commit sha. ``$HELPER_SO400M`` picks the checkpoint: ``google/siglip2-so400m-patch16-512`` (default) or ``-384``
  (about twice as fast, less text detail). The EXP row records both.
* **One device and dtype for every part:** the job embeds in bf16 on cuda, a Mac in fp16 on mps. Make train, dev and
  final in the same place (all in the GPU job, ``SO400M=1``), or check their parity first (#215 review).
* **The cache, per part (resumable, counts and speed only in the log):**
      uv run --extra local python -m galaxus.model.encoders embed --part train
  or inside the GPU feature job (``train/hf_job_apertus_features_start.sh``, ``encoders`` mode or any mode with the
  revision set), which sends it back as ``embeddings.tar``.
* **Rows:** ``helper-so400m`` (heads on so400m alone), ``helper-fused-so400m`` (EXP-0059's three encoders + so400m, each at
  unit length), and ``apertus-core`` with ``APERTUS_CORE_BASE=fused-so400m``. Q-0008 was asked with SigLIP base heads as
  its example; if a so400m row ships, it needs its own MODELS.md row with this pin and the same rules check as #197.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from galaxus.model import helper

SO400M_512 = "google/siglip2-so400m-patch16-512"
SO400M_384 = "google/siglip2-so400m-patch16-384"
CHOICES = (SO400M_512, SO400M_384)
NAME_ENV = "HELPER_SO400M"
REVISION_ENV = "HELPER_SO400M_REVISION"
#: The configs and the weights; ``*.safetensors`` also covers a checkpoint that is split into shards.
SNAPSHOT_PATTERNS = ["config.json", "preprocessor_config.json", "*.safetensors", "*.safetensors.index.json"]
SHA = re.compile(r"^[0-9a-f]{40}$")
#: The Hub revisions, read with ``encoders pin`` on Neel's laptop (#215 review at b3ef2e2). Both Apache-2.0; the 512 px
#: checkpoint is one 4.55 GB ``model.safetensors`` that includes the text tower.
PINS = {
    SO400M_512: "ceea1cba8130d8271436da4828633198c176a775",
    SO400M_384: "dd658faac399427308559e2c3ac1e99cbe43845d",
}


def so400m_name(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    name = environ.get(NAME_ENV, "") or SO400M_512
    if name not in CHOICES:
        raise ValueError(f"${NAME_ENV} must be one of {', '.join(CHOICES)}")
    return name


def so400m_revision(environ: Mapping[str, str] | None = None) -> str:
    """The checkpoint's pin (:data:`PINS`), or ``$HELPER_SO400M_REVISION`` when set, which must be a full commit sha."""
    environ = os.environ if environ is None else environ
    revision = environ.get(REVISION_ENV, "").strip()
    if not revision:
        return PINS[so400m_name(environ)]
    if not SHA.match(revision):
        raise ValueError(f"${REVISION_ENV} must be a full commit sha of {so400m_name(environ)}: "
                         "`uv run --extra local python -m galaxus.model.encoders pin` prints the current one")
    return revision


def revision_of(encoder: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Any encoder's pinned revision: ``helper.REVISIONS``, or the environment's pin for so400m."""
    return so400m_revision(environ) if encoder in CHOICES else helper.REVISIONS.get(encoder)


def device_and_dtype(torch: Any) -> tuple[str, Any]:
    """cuda in bf16 (fp16 where bf16 is missing, e.g. a T4), else mps in fp16, else the CPU in fp32."""
    from galaxus.model.apertus.local import pick_device

    device = pick_device(torch)
    if device == "cuda":
        bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return device, torch.bfloat16 if bf16 else torch.float16
    return device, torch.float16 if device == "mps" else torch.float32


def load_on_device(name: str, revision: str) -> Any:
    """The pinned snapshot of ``name`` on the best device, behind the data lane's ``Encoder`` contract."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModel

    from galaxus.data.neighbours import pooled_features

    snapshot = snapshot_download(repo_id=name, revision=revision, allow_patterns=SNAPSHOT_PATTERNS)
    processor = AutoImageProcessor.from_pretrained(snapshot)
    device, dtype = device_and_dtype(torch)
    model = AutoModel.from_pretrained(snapshot, dtype=dtype)
    model.to(device)
    model.eval()

    class OnDevice:
        def encode(self, images: Sequence[Any]) -> np.ndarray:
            batch = processor(images=list(images), return_tensors="pt")
            pixels = batch["pixel_values"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                raw = model.get_image_features(pixel_values=pixels)
            return pooled_features(raw).float().cpu().numpy()

    return OnDevice()


_LOADED: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def shared(name: str, revision: str, loader: Callable[[str, str], Any] = load_on_device) -> Any:
    """One loaded so400m per process and revision (the #181 lesson: never one load per runner batch)."""
    with _LOCK:
        if (name, revision) not in _LOADED:
            _LOADED[(name, revision)] = loader(name, revision)
        return _LOADED[(name, revision)]


def so400m_embeddings(
    pairs: Iterable[tuple[str, str]],
    _name: str | None = None,
    environ: Mapping[str, str] | None = None,
    loader: Callable[[str, str], Any] = load_on_device,
) -> dict[str, np.ndarray]:
    """``{image_id: so400m embedding}`` from the cache; missing ones are embedded once, on the best device."""
    name, revision = so400m_name(environ), so400m_revision(environ)
    return helper.embeddings(pairs, name, load_encoder=lambda encoder: shared(encoder, revision, loader))


def fused_so400m(
    environ: Mapping[str, str] | None = None, loader: Callable[[str, str], Any] = load_on_device
) -> tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]:
    """EXP-0059's three encoders plus so400m, each at unit length, concatenated in that order (``helper.fused``)."""
    encoders = (*helper.FUSED_ENCODERS, so400m_name(environ))

    def per_encoder(pairs: Iterable[tuple[str, str]], name: str) -> dict[str, np.ndarray]:
        if name in CHOICES:
            return so400m_embeddings(pairs, environ=environ, loader=loader)
        return helper.embeddings(pairs, name)

    return helper.fused(per_encoder, encoders), encoders


def factory(fused: bool = False, environ: Mapping[str, str] | None = None,
            loader: Callable[[str, str], Any] = load_on_device):
    """``helper-so400m`` (``fused=False``) or ``helper-fused-so400m``: #51's heads through ``helper.factory``."""

    def build(rows: Any, data: Any, seed: int) -> helper.HelperPredictor:
        env = os.environ if environ is None else environ
        name, revision = so400m_name(env), so400m_revision(env)
        if fused:
            embed, encoders = fused_so400m(env, loader)
            label = "fused:" + "+".join(encoders)
        else:
            encoders, label = (name,), name

            def embed(pairs: Iterable[tuple[str, str]], _name: str | None = None) -> dict[str, np.ndarray]:
                return so400m_embeddings(pairs, environ=env, loader=loader)

        predictor = helper.factory(encoder_name=label, embed=embed)(rows, data, seed)
        # What the EXP row records: the helper that picks the class (Q-0008), its encoders and their pins.
        predictor.model = label
        predictor.helpers = encoders
        revisions = [revision if encoder == name else helper.REVISIONS.get(encoder) for encoder in encoders]
        predictor.model_revision = ", ".join(str(r) for r in revisions) if len(revisions) > 1 else revisions[0]
        predictor.heads_model.meta.update({"so400m": name, "so400m_revision": revision})
        return predictor

    return build


def from_env(environ: Mapping[str, str] | None = None) -> helper.HelperPredictor:
    """``so400m=galaxus.model.encoders:from_env`` for the runner and the live ensemble (``HELPER_FIT_ALL=1`` works)."""
    environ = os.environ if environ is None else environ
    return helper._live(factory(environ=environ), environ)


def fused_from_env(environ: Mapping[str, str] | None = None) -> helper.HelperPredictor:
    """``fused400=galaxus.model.encoders:fused_from_env``: the four encoders' heads, live."""
    environ = os.environ if environ is None else environ
    return helper._live(factory(fused=True, environ=environ), environ)


# ---------------------------------------------------------------- filling the cache


def embed_part(
    pairs: Sequence[tuple[str, str]],
    part: str,
    chunk: int = 256,
    environ: Mapping[str, str] | None = None,
    loader: Callable[[str, str], Any] = load_on_device,
    log: Callable[[str], None] = print,
) -> int:
    """Embed ``pairs`` in chunks, resumably, logging counts and speed only (never an image id). Returns how many."""
    started = time.perf_counter()
    unreadable = 0
    for start in range(0, len(pairs), chunk):
        batch = pairs[start : start + chunk]
        try:
            so400m_embeddings(batch, environ=environ, loader=loader)
        except (OSError, ValueError):  # one unreadable image (PIL errors are OSErrors): find it, keep the rest
            for pair in batch:
                try:
                    so400m_embeddings([pair], environ=environ, loader=loader)
                except (OSError, ValueError):
                    unreadable += 1
        done = min(start + chunk, len(pairs))
        log(f"[so400m] {part}: {done}/{len(pairs)} ({done / (time.perf_counter() - started):.1f} img/s, cache "
            "hits included)")
    if unreadable:
        log(f"[so400m] {part}: {unreadable} image(s) could not be read and were skipped; their rows keep the "
            "constant answer")
    return len(pairs) - unreadable


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.encoders", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pin = sub.add_parser("pin", help="print the export lines with the Hub's current revision (needs the Hub)")
    pin.add_argument("--name", choices=CHOICES, default=None)
    fill = sub.add_parser("embed", help="fill the so400m embedding cache for one part (resumable)")
    source = fill.add_mutually_exclusive_group(required=True)
    source.add_argument("--part", choices=("train", "dev", "final"))
    source.add_argument("--csv", help="any new batch: a table with image_id and image_path (relative to it)")
    fill.add_argument("--limit", type=int, default=None, help="only the first N rows, e.g. to time a sample")
    fill.add_argument("--chunk", type=int, default=256)
    arguments = parser.parse_args(argv)
    if arguments.command == "pin":
        from huggingface_hub import model_info

        name = arguments.name or so400m_name()
        print(f"export {NAME_ENV}={name}")
        print(f"export {REVISION_ENV}={model_info(name).sha}")
        return 0
    so400m_revision()  # refuse a malformed override before reading or downloading anything
    from galaxus.model.heads import extract_source

    frame, part = extract_source(arguments.part, arguments.csv)
    pairs = list(zip(frame[helper.ID_COLUMN].astype(str), frame[helper.FILE_COLUMN].astype(str), strict=True))
    embed_part(pairs[: arguments.limit] if arguments.limit else pairs, part, arguments.chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
