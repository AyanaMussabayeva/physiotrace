"""Weighted AUROC for point estimates and many bootstrap draws at once."""

from __future__ import annotations

import numpy as np

from ._validation import as_codes


def weighted_auroc(y, scores, weights) -> float | np.ndarray:
    """Tie-correct weighted AUROC.

    ``weights`` may be a vector (one estimate) or a ``(draws, n)`` matrix, in
    which case one AUROC per row is returned. Tied positive--negative pairs
    receive half credit, which matches ``sklearn.metrics.roc_auc_score`` with
    ``sample_weight``.
    """

    labels = as_codes(y, name="y", n_levels=2)
    values = np.asarray(scores, dtype=np.float64)
    matrix = np.asarray(weights, dtype=np.float64)
    single = matrix.ndim == 1
    if single:
        matrix = matrix[None, :]
    if values.ndim != 1 or len(values) != len(labels) or matrix.shape[1] != len(labels):
        raise ValueError("y, scores, and weights must be aligned.")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("scores and weights must be finite, and weights non-negative.")

    positive = labels == 1
    order = np.argsort(values, kind="mergesort")
    ordered_scores = values[order]
    ordered_positive = positive[order]
    ordered_weights = matrix[:, order]
    denominator = ordered_weights[:, ordered_positive].sum(axis=1) * ordered_weights[:, ~ordered_positive].sum(axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("Every weighting must give positive mass to both classes.")

    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores) != 0.0) + 1]
    stops = np.r_[starts[1:], len(order)]
    group_ids = np.repeat(np.arange(len(starts)), stops - starts)
    negative_cumulative = np.cumsum(ordered_weights * (~ordered_positive), axis=1)
    negative_before = np.zeros((len(matrix), len(starts)))
    later = starts > 0
    negative_before[:, later] = negative_cumulative[:, starts[later] - 1]
    negative_within = negative_cumulative[:, stops - 1] - negative_before
    numerator = np.sum(
        ordered_weights * ordered_positive * (negative_before[:, group_ids] + 0.5 * negative_within[:, group_ids]),
        axis=1,
    )
    result = numerator / denominator
    return float(result[0]) if single else result


__all__ = ["weighted_auroc"]
