"""Kept only as a pointer: the contract lives in ``galaxus.types`` (B0, #11, PR #80). Import from there.

Until #80 this module carried a stand-in copy of ``ImageItem``, ``Prediction`` and ``Predictor``. The copy is gone
(#83): these names are ``galaxus.types``' own classes, re-exported so nothing that still imports this module breaks.
"""

from galaxus.types import ImageItem, Prediction, Predictor

__all__ = ["ImageItem", "Prediction", "Predictor"]
