"""Linear task heads fitted once and then held fixed throughout the audit."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._validation import as_codes, as_matrix, same_length


@dataclass(frozen=True)
class LinearHead:
    """Optional standardization followed by binary logistic regression."""

    classifier: LogisticRegression
    scaler: StandardScaler | None = None
    converged: bool = True
    meta: dict = field(default_factory=dict)

    def logits(self, embeddings) -> np.ndarray:
        values = as_matrix(embeddings, name="embeddings")
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return np.asarray(self.classifier.decision_function(values), dtype=np.float64)

    def scores(self, embeddings) -> np.ndarray:
        """Positive-class probability."""

        return 1.0 / (1.0 + np.exp(-self.logits(embeddings)))


def fit_linear_head(
    embeddings,
    y,
    sample_weight=None,
    *,
    C: float = 1.0,
    standardize: bool = False,
    max_iter: int = 2_000,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> LinearHead:
    """Fit a binary logistic head; ``sample_weight`` also weights the scaler when ``standardize``."""

    h = as_matrix(embeddings, name="embeddings")
    y = as_codes(y, name="y", n_levels=2)
    same_length(embeddings=h, y=y)
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(h, sample_weight=sample_weight)
        h = scaler.transform(h)
    classifier = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol, random_state=random_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(h, y, sample_weight=sample_weight)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return LinearHead(classifier=classifier, scaler=scaler, converged=converged, meta={"C": float(C)})


__all__ = ["LinearHead", "fit_linear_head"]
