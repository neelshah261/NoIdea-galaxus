"""``python -m galaxus.data.prep``: fill the prepared-image cache (issue #27, BASE_ARCHITECTURE.md B1 "just prep").

Every experiment, the heads pass, LoRA training and the Saturday run must be served the *same* pixels. Preparing
them once into ``$GALAXUS_DATA/cache/images/<size>-p<patch>/`` is what makes that true, and what stops every pass
paying for decoding and compositing again.

    python -m galaxus.data.prep --part train --size 512 --workers 8
    python -m galaxus.data.prep --part train --mirror      # also the mirrored copies, for perspective

Safe to re-run and safe to interrupt: an image already cached is skipped, and each file is written to a ``.part``
name and renamed, so a half-written JPEG is never mistaken for a cached one. Reads only files, writes only inside
``$GALAXUS_DATA``, and never touches the network.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from galaxus.data.images import IMAGE_SIZE, PATCH_SIZE, fill_image_cache, image_cache_dir

PARTS = ("train", "dev", "final")


def frame_for_part(part: str) -> pd.DataFrame:
    """The table behind one part, through the loader, so the cache follows the same layout rules as everything else."""
    from galaxus.data import load

    if part == "train":
        return load.read_train()
    if part == "dev":
        return load.read_dev()
    if part == "final":
        return load.read_final()
    raise ValueError(f"part must be one of {PARTS}, got {part!r}")


def default_workers() -> int:
    """A sensible thread count: the work is read, decode, write, so a few per core helps, but cap it."""
    return min(8, (os.cpu_count() or 2))


def _report(label: str, counts: dict[str, object]) -> None:
    rate = counts.get("images_per_second")
    rate_text = f"{rate:.1f} images/s" if isinstance(rate, float) else "no new images"
    print(
        f"[prep] {label}: {counts['written']} written, {counts['skipped']} already cached, "
        f"{counts['missing']} missing  ({rate_text}, {float(counts['seconds']):.1f}s, "
        f"{counts['workers']} workers)"
    )
    print(f"[prep] {label} -> {counts['directory']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.data.prep", description=__doc__)
    parser.add_argument("--part", required=True, choices=PARTS)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help=f"target area per side (default {IMAGE_SIZE})")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="also write <image_id>.mirror.jpg. TODO(#29): once mirrored_rows() lands, narrow this to the rows "
        "whose perspective label actually changes and whose image carries no text",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"patch size the pixels are prepared at (default {PATCH_SIZE}; Apertus' own is Q-0002). Each patch "
        "size fills its own cache folder, so two never mix",
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--limit", type=int, help="only the first N rows, for a quick check")
    parser.add_argument("--refresh", action="store_true", help="re-prepare images that are already cached")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="count an image file that is not on disk instead of stopping",
    )
    arguments = parser.parse_args(argv)

    if arguments.workers < 1:
        parser.error("--workers must be at least 1")

    frame = frame_for_part(arguments.part)
    if arguments.limit:
        frame = frame.head(arguments.limit)

    folder = image_cache_dir(arguments.size, arguments.patch_size)
    print(
        f"[prep] {arguments.part}: {len(frame)} rows at {arguments.size} px, patch {arguments.patch_size} -> {folder}"
    )
    shared = {
        "image_size": arguments.size,
        "patch_size": arguments.patch_size,
        "refresh": arguments.refresh,
        "skip_missing": arguments.skip_missing,
        "workers": arguments.workers,
    }
    _report(arguments.part, fill_image_cache(frame, **shared))
    if arguments.mirror:
        _report(f"{arguments.part} mirrored", fill_image_cache(frame, mirrored=True, **shared))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
