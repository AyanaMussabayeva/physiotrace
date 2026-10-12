"""Fixed-marginal association stress (Stress).

For fixed heads evaluated on the same held-out records,

    D(f) = AUROC_aligned(f) - AUROC_reversed(f),
    G    = D(exposed) - D(control),

where the exposed head was fitted under the provenance--target association and
the control head without it. ``G`` is the excess vulnerability (paper Eq. 4).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .metrics import weighted_auroc
from .weights import cell_weights


def regime_aurocs(y, u, scores, tables: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Weighted AUROC of one fixed set of scores under each regime."""

    return {name: weighted_auroc(y, scores, cell_weights(y, u, table)) for name, table in tables.items()}


def association_shift(aurocs: Mapping[str, float]) -> float:
    """``D = AUROC_aligned - AUROC_reversed``."""

    return float(aurocs["aligned"] - aurocs["reversed"])


def excess_vulnerability(y, u, exposed_scores, control_scores, tables: Mapping[str, np.ndarray]) -> float:
    """``G = D(exposed) - D(control)`` for one evaluation set."""

    return association_shift(regime_aurocs(y, u, exposed_scores, tables)) - association_shift(
        regime_aurocs(y, u, control_scores, tables)
    )


def flip_rate(view_scores: np.ndarray, y, *, threshold: float = 0.5) -> float:
    """Class-balanced share of records whose hard decision differs across views.

    ``view_scores`` has shape ``(records, views)`` with one positive-class
    probability per view of the same record.
    """

    scores = np.asarray(view_scores, dtype=np.float64)
    decisions = scores >= threshold
    flips = (decisions.min(axis=1) != decisions.max(axis=1)).astype(np.float64)
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    return float(np.average(flips, weights=len(labels) / (2.0 * counts[labels])))


__all__ = ["association_shift", "excess_vulnerability", "flip_rate", "regime_aurocs"]
