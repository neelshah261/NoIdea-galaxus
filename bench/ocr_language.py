"""#101 item 9: can an open OCR model plus a language detector answer the language dimension better than Apertus?

A CPU/GPU experiment on the official holdout, no Apertus and no gateway call, so it costs nothing:

1. **OCR** every holdout image whose language cell is scored, at the same 512 px the models see
   (``galaxus.data.images.cached_jpeg``), with EasyOCR's Latin model. The words and their confidences are cached
   under ``$GALAXUS_DATA/out/ocr/``, so step 2 can be re-run without the images.
2. **Decide** per image: fewer than ``min_letters`` letters above ``min_confidence`` -> the no-text class; else the
   language Lingua detects, or the "other" class when that language is not a class of the dimension.
3. **Score** through metric.py (``galaxus.data.metric.evaluate`` on ``split.solution_frame``). The two thresholds are
   chosen on one product half of the holdout and scored on the other, in both directions (as ``ensemble``, #87),
   so the number is not optimistic.

    python bench/ocr_language.py             # OCR (cached) + the cross-fitted score and the comparison rows

The roles (which dimension, which class means no text, which means another language) are in bench/ocr_language.json,
checked against the classes when it loads. Runtime tools: EasyOCR and Lingua are open-weights supporting tools, which
FAQ.md allows at runtime ("image encoder, embeddings or OCR"). Install them in their own environment, not the
project's: ``python -m venv --system-site-packages ocrenv && ocrenv/Scripts/python -m pip install easyocr
lingua-language-detector``.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROLES_FILE = Path(__file__).with_name("ocr_language.json")
OCR_CACHE = "ocr_easyocr_latin_512.jsonl"
MIN_LETTERS = (4, 8, 12, 20, 30)
MIN_CONFIDENCE = (0.1, 0.3, 0.5)


def load_roles(path: Path = ROLES_FILE, allowed: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """The roles file, every name checked against ``allowed`` (the classes) when given."""
    roles = json.loads(path.read_text(encoding="utf-8"))
    if allowed is not None:
        dimension = roles["language_dimension"]
        if dimension not in allowed:
            raise ValueError(f"{path.name}: {dimension} is not a dimension")
        for key in ("no_text_class", "other_class"):
            if roles[key] not in allowed[dimension]:
                raise ValueError(f"{path.name}: {key} {roles[key]!r} is not a class of {dimension}")
    return roles


def letters(words: Sequence[Mapping[str, Any]], min_confidence: float) -> str:
    """The recognised words above ``min_confidence``, joined with spaces."""
    return " ".join(str(w["text"]) for w in words if float(w["confidence"]) >= min_confidence)


def decide(
    words: Sequence[Mapping[str, Any]],
    detect: Any,
    classes: Sequence[str],
    roles: Mapping[str, Any],
    min_letters: int,
    min_confidence: float,
) -> str:
    """One image's language class from its OCR words: no text, a class of the dimension, or "other".

    ``detect(text)`` returns a language name such as ``"GERMAN"`` or ``None``; it is matched to a class by name, so no
    language is typed here.
    """
    text = letters(words, min_confidence)
    if sum(ch.isalpha() for ch in text) < min_letters:
        return roles["no_text_class"]
    name = detect(text)
    if not name:
        return roles["other_class"]
    by_name = {c.lower(): c for c in classes}
    return by_name.get(str(name).lower(), roles["other_class"])


def _lingua_detector() -> Any:
    from lingua import LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    def detect(text: str) -> str | None:
        language = detector.detect_language_of(text)
        return None if language is None else language.name

    return detect


def run_ocr(rows: pd.DataFrame, cache: Path, ocr_languages: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """OCR words per image, read from ``cache`` where they are already there; new ones are appended."""
    import io

    import easyocr
    import numpy as np
    from PIL import Image

    from galaxus.data.images import cached_jpeg

    done: dict[str, list[dict[str, Any]]] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["image_id"]] = record["words"]
    todo = rows[~rows["image_id"].astype(str).isin(done)]
    if todo.empty:
        return done
    reader = easyocr.Reader(list(ocr_languages), gpu=True, verbose=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with cache.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(todo.itertuples(), 1):
            image = Image.open(io.BytesIO(cached_jpeg(row.image_id, row.image_file))).convert("RGB")
            found = reader.readtext(np.asarray(image))
            words = [{"text": text, "confidence": round(float(conf), 4)} for _, text, conf in found]
            done[str(row.image_id)] = words
            handle.write(json.dumps({"image_id": str(row.image_id), "words": words}, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                rate = index / (time.perf_counter() - started)
                print(f"[ocr] {index}/{len(todo)} images, {rate:.1f} img/s", flush=True)
    seconds = time.perf_counter() - started
    print(f"[ocr] {len(todo)} images in {seconds:.0f} s = {len(todo) / seconds:.2f} img/s (GPU, this laptop)")
    return done


def language_f1(
    answers: Mapping[str, str], part: pd.DataFrame, dimension: str, fill: Mapping[str, str],
    solution_frame: Any, evaluate: Any, dimensions: Sequence[str],
) -> float:
    """metric.py's macro F1 of ``dimension`` on ``part``; the other dimensions get ``fill`` (they are not read)."""
    ids = part["image_id"].astype(str).tolist()
    submission = pd.DataFrame({"image_id": ids, **{d: [fill[d]] * len(ids) for d in dimensions}})
    submission[dimension] = [answers.get(i, fill[dimension]) for i in ids]
    evaluation = evaluate(solution_frame(part, tuple(dimensions)), submission)
    return float(evaluation[dimension]["f1_macro"])


def main() -> int:
    from galaxus.data import classes, load, metric, split
    from galaxus.data.paths import out_dir
    from galaxus.model.baselines import MajorityPredictor
    from galaxus.model.ensemble import holdout_halves

    allowed = classes.allowed_classes()
    dims = tuple(classes.dimensions())
    roles = load_roles(allowed=allowed)
    dimension = roles["language_dimension"]
    class_names = sorted(allowed[dimension])
    holdout = split.split_holdout(load.read_train())[1].reset_index(drop=True)
    scored = holdout[split.scored_mask(holdout, dimension).to_numpy()]
    print(f"holdout {len(holdout)} images; {dimension} scored on {len(scored)}")

    words = run_ocr(scored, out_dir() / "ocr" / OCR_CACHE, roles["ocr_languages"])
    detect = _lingua_detector()
    fill = MajorityPredictor().answer
    halves = holdout_halves(holdout)
    parts = [holdout[halves.to_numpy() == h] for h in (0, 1)]

    def answers_for(min_letters: int, min_confidence: float) -> dict[str, str]:
        return {i: decide(w, detect, class_names, roles, min_letters, min_confidence) for i, w in words.items()}

    grid = {setting: answers_for(*setting) for setting in product(MIN_LETTERS, MIN_CONFIDENCE)}

    def f1(answers: Mapping[str, str], part: pd.DataFrame) -> float:
        return language_f1(answers, part, dimension, fill, split.solution_frame, metric.evaluate, dims)

    chosen = []
    crossfit: dict[str, str] = {}
    for h in (0, 1):
        best = max(grid, key=lambda s: (f1(grid[s], parts[h]), -s[0]))
        chosen.append(best)
        other = parts[1 - h]["image_id"].astype(str)
        crossfit.update({i: grid[best][i] for i in other if i in grid[best]})
    honest = f1(crossfit, holdout)
    whole = max(grid, key=lambda s: f1(grid[s], holdout))
    print(f"\nchosen on half 0 / half 1 (min_letters, min_confidence): {chosen[0]} / {chosen[1]}")
    print(f"OCR + language ID, {dimension} macro F1 (cross-fitted, metric.py): {honest:.3f}")
    print(f"  (optimistic, chosen and scored on the whole holdout: {f1(grid[whole], holdout):.3f} at {whole})")

    rows = json.loads((Path(__file__).resolve().parents[1] / "results" / "metrics.json").read_text(encoding="utf-8"))
    for row in rows:
        value = (row.get("macro_f1") or {}).get(dimension)
        if value is not None:
            print(f"  {row['id']} {row['predictor']:24} {dimension} {value:.3f}")
    best_answers = grid[whole]
    truth = scored.set_index(scored["image_id"].astype(str))[dimension]
    confusion = pd.crosstab(truth.rename("human"), pd.Series(best_answers).reindex(truth.index).rename("ocr"))
    print("\nconfusion at the whole-holdout setting (rows: human label, columns: OCR answer):")
    print(confusion.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
