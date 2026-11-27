"""EXP-0001, the majority baseline: one constant answer, the most frequent human label of each dimension.

It proves the path load → predict → score → EXP row end to end. Its score is a floor, not a method (PLAN.md
"Method ladder", issue #12).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from galaxus.model.contract import ImageItem, Prediction

#: ``scored_mask(rows, dimension)``: which cells count. The data lane's ``galaxus.data.split.scored_mask`` (#5).
ScoredMask = Callable[[pd.DataFrame, str], pd.Series]


class MajorityPredictor:
    """Answers every image with the most frequent human label of each dimension.

    Only the cells that are scored are counted: a human label where the dimension is meaningful, through the data
    lane's scored-cell mask. So a model prelabel never decides the answer. A tie goes to the class name that sorts
    first, so every run gives the same answer.
    """

    name = "majority"

    def __init__(self, answer: Mapping[str, str]) -> None:
        if not answer:
            raise ValueError("a majority answer needs at least one dimension")
        self.answer = dict(answer)

    @classmethod
    def fit(cls, rows: pd.DataFrame, dimensions: Sequence[str], scored_mask: ScoredMask) -> MajorityPredictor:
        """The majority answer of ``rows``, the training rows: train minus the official holdout."""
        answer = {}
        for dimension in dimensions:
            labels = rows.loc[scored_mask(rows, dimension), dimension]
            labels = labels[labels.notna()].astype(str)
            labels = labels[labels.str.strip() != ""]
            if labels.empty:
                raise ValueError(f"no scored human label for {dimension} in the {len(rows)} training rows")
            counts = labels.value_counts()
            answer[dimension] = min(counts.index, key=lambda name: (-counts[name], name))
        return cls(answer)

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        """The same answer for every image of ``batch``, in order."""
        return [Prediction(image_id=item.image_id, labels=dict(self.answer)) for item in batch]

    def config(self) -> dict[str, object]:
        """What the EXP row records as this predictor's config."""
        return {"answer": dict(self.answer)}
