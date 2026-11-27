"""Synthetic tables and a fake data lane for the model-lane tests. No competition data, no network.

The class names are invented: the real ones are read at runtime and never written down (AGENTS.md "Hard rules").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from galaxus.model.eval import DataLane

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
FAKE_CLASSES = {dimension: (f"{dimension}_a", f"{dimension}_b", f"{dimension}_c") for dimension in DIMENSIONS}


def train_table(products: int = 10, images_per_product: int = 3) -> pd.DataFrame:
    """A train table as the CSV reader returns it: strings, with ``_source`` and ``_relevant`` per dimension.

    Most labels are class ``_a`` and every seventh is ``_b``; every fourth image is a model prelabel and every fifth
    cell is not relevant.
    """
    records = []
    for index in range(products * images_per_product):
        image_id = f"img_{index:05d}"
        product = index // images_per_product
        record = {
            "image_id": image_id,
            "image_path": f"train_images/{image_id}.jpg",
            "image_file": f"/synthetic/train_images/{image_id}.jpg",
            "product_id": f"p{product:03d}",
            "product_title": f"Product {product}",
            "brand": "Brand",
            "product_type": "Type",
            "category": "Category",
            "subcategory": "Subcategory",
        }
        for dimension, (first, second, _) in FAKE_CLASSES.items():
            record[dimension] = second if index % 7 == 0 else first
            record[f"{dimension}_source"] = "model" if index % 4 == 3 else "human"
            record[f"{dimension}_relevant"] = "false" if index % 5 == 4 else "true"
        records.append(record)
    return pd.DataFrame(records)


def fake_scored_mask(rows: pd.DataFrame, dimension: str) -> pd.Series:
    """A human label in a relevant cell, the rule of challenge_description.md "Scoring locally"."""
    return (rows[f"{dimension}_source"] != "model") & (rows[f"{dimension}_relevant"] == "true")


def fake_solution_frame(holdout: pd.DataFrame, dimensions: tuple[str, ...]) -> pd.DataFrame:
    solution = holdout[["image_id", *dimensions]].astype(object)
    for dimension in dimensions:
        solution.loc[~fake_scored_mask(holdout, dimension), dimension] = None
    return solution


def first_two_products_held_out(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    held_out = table["product_id"].isin(sorted(table["product_id"].unique())[:2])
    return table[~held_out], table[held_out]


class FakeMetric:
    """Stands in for metric.py: records what it is given and answers in ``metric.evaluate``'s shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def score(self, solution: pd.DataFrame, submission: pd.DataFrame) -> float:
        self.calls.append((solution, submission))
        return 0.25

    def evaluate(self, solution: pd.DataFrame, submission: pd.DataFrame) -> dict:
        per_dimension = {
            dimension: {"f1_macro": 0.25, "f1_weighted": 0.5, "support": 3, "per_class": {}} for dimension in DIMENSIONS
        }
        return {**per_dimension, "overall": {"f1_macro": 0.25, "f1_weighted": 0.5, "support": 18}}


def fake_data_lane(train: pd.DataFrame, out: Path, metric: FakeMetric | None = None) -> DataLane:
    metric = metric or FakeMetric()
    return DataLane(
        read_train=lambda: train,
        split_holdout=first_two_products_held_out,
        dimensions=lambda: DIMENSIONS,
        allowed_classes=lambda: {dimension: frozenset(names) for dimension, names in FAKE_CLASSES.items()},
        scored_mask=fake_scored_mask,
        solution_frame=fake_solution_frame,
        score=metric.score,
        evaluate=metric.evaluate,
        out_dir=lambda: out,
    )
