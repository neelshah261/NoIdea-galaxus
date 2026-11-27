"""``python -m galaxus.serve.demo``: build the hero-shot gallery from a predictions file (issue #31).

    python -m galaxus.serve.demo --exp EXP-0001 --part dev
    python -m galaxus.serve.demo --predictions <file.jsonl> --part holdout --mirror <mirrored.jsonl>

Reads only files, writes one HTML page under ``$GALAXUS_DATA/out/demo/``, and never reaches the network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from galaxus.serve.demo.gallery import build_page, products_from_frame, write_page
from galaxus.serve.demo.rules import load_rules

DEMO_DIR = "demo"


def frame_for_part(part: str) -> pd.DataFrame:
    """The table behind one part, through the data lane.

    ``runner.items_for_part`` returns items, which drop ``product_id`` on purpose; the gallery groups by product, so
    it needs the table itself.
    """
    from galaxus.data import load, split

    if part == "holdout":
        return split.split_holdout(load.read_train())[1]
    if part == "dev":
        return load.read_dev()
    if part == "final":
        return load.read_final()
    raise ValueError(f"part must be holdout, dev or final, got {part!r}")


def last_run(runs_file: Path) -> dict[str, Any] | None:
    """The newest timing record the runner appended, or ``None`` if there is no run file yet."""
    if not runs_file.is_file():
        return None
    records = [json.loads(line) for line in runs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[-1] if records else None


def meter_from_run(record: dict[str, Any] | None, usd_per_hour: float | None) -> dict[str, Any] | None:
    """The live meter: images/s from the runner, and cost per 100,000 images when a price is given.

    The cost formula is the one in ``galaxus.serve.bench``, which implements PLAN.md "Speed"; no number is invented
    here, and without ``--usd-per-hour`` the cost cell is simply left out.
    """
    if not record:
        return None
    images_per_second = record.get("images_per_second")
    meter: dict[str, Any] = {
        "images_per_second": images_per_second,
        "images": record.get("predicted"),
        "seconds_note": f"{record.get('seconds', 0):.1f} s in the last run, batch {record.get('batch_size')}",
        "hardware": "measured by the runner, not a quoted figure",
    }
    if usd_per_hour is not None and images_per_second:
        from galaxus.serve.bench import cost_per_100k

        meter["cost_per_100k"] = cost_per_100k(usd_per_hour, float(images_per_second))
        meter["cost_note"] = f"at ${usd_per_hour:.2f}/hour"
    return meter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.demo", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--exp", help="experiment id, to find the predictions the runner wrote")
    source.add_argument("--predictions", help="a predictions jsonl to read instead")
    parser.add_argument("--part", default="dev", choices=("holdout", "dev", "final"))
    parser.add_argument("--mirror", help="predictions for the mirrored images, for the mirror panel")
    parser.add_argument("--rules", help="a hero-rule file instead of the one shipped with the package")
    parser.add_argument("--usd-per-hour", type=float, help="price of the hardware, for cost per 100k images")
    parser.add_argument("--limit", type=int, help="only the first N rows of the part")
    parser.add_argument("--out", help="where to write the page; default $GALAXUS_DATA/out/demo/")
    parser.add_argument("--title", default="Hero-shot gallery")
    arguments = parser.parse_args(argv)

    from galaxus.serve.runner import predictions_path, read_predictions, runs_path

    if arguments.predictions:
        predictions_file = Path(arguments.predictions)
    else:
        predictions_file = predictions_path(arguments.exp, arguments.part)
    predictions = read_predictions(predictions_file, repair=False)
    if not predictions:
        parser.error(f"no predictions in {predictions_file}")

    frame = frame_for_part(arguments.part)
    if arguments.limit:
        frame = frame.head(arguments.limit)

    rules = load_rules(arguments.rules)
    mirror = read_predictions(Path(arguments.mirror), repair=False) if arguments.mirror else None
    page = build_page(
        products_from_frame(frame, predictions),
        rules=rules,
        meter=meter_from_run(last_run(runs_path(predictions_file)), arguments.usd_per_hour),
        mirror=mirror,
        predictions=predictions,
        title=arguments.title,
    )

    if arguments.out:
        target = Path(arguments.out)
    else:
        from galaxus.data.paths import out_dir

        name = arguments.exp or predictions_file.stem
        target = Path(out_dir()) / DEMO_DIR / f"{name}-{arguments.part}.html"
    written = write_page(page, target)

    print(f"[demo] {len(predictions)} predictions -> {written}")
    for problem in rules.problems:
        print(f"[demo] unverified: {problem}")
    for assumption in rules.assumptions():
        print(f"[demo] assumption: {assumption['rule']} - {assumption['todo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
