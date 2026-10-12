"""Linear Provenance Information (Recover).

LPI is held-out predictive V-information for a calibrated multinomial
logistic probe, in bits per native provenance-labelled unit (paper Eq. 1):

    LPI = mean_i log2( q(u_i | h_i) / q0(u_i) ),

where ``q0`` is the Jeffreys-smoothed training prevalence. The probe is fitted
on training data, its regularization is selected by validation log loss, and
its temperature is calibrated on validation logits; no test score enters
selection. Intervals resample identity clusters while keeping native-unit
weighting (paper Appendix C).

LPI measures accessibility, not use. It never yields a verdict on its own.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ._validation import as_matrix, encode, same_length
from .bootstrap import percentile_interval

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class CalibratedProbe:
    """Validation-selected, temperature-calibrated linear provenance probe."""

    pipeline: Pipeline
    selected_c: float
    temperature: float
    classes: tuple
    validation_nll_bits: dict[float, float]

    def log_proba(self, embeddings) -> np.ndarray:
        return _log_softmax(_logits(self.pipeline, embeddings, len(self.classes)), self.temperature)

    def proba(self, embeddings) -> np.ndarray:
        return np.exp(self.log_proba(embeddings))


def fit_calibrated_probe(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    *,
    classes: Sequence | None = None,
    c_grid: Sequence[float] = C_GRID,
    seed: int = 42,
    calibrate_temperature: bool = True,
    strict_convergence: bool = True,
) -> CalibratedProbe:
    """Select ``C`` by raw validation NLL (ties go to the smaller ``C``), then calibrate."""

    train_h = as_matrix(train_embeddings, name="train_embeddings")
    val_h = as_matrix(val_embeddings, name="val_embeddings")
    train_codes, classes = encode(train_labels, classes)
    val_codes, _ = encode(val_labels, classes)
    same_length(train_embeddings=train_h, train_labels=train_codes)
    same_length(val_embeddings=val_h, val_labels=val_codes)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("At least two provenance groups are required.")
    for name, codes in (("training", train_codes), ("validation", val_codes)):
        if set(np.unique(codes).tolist()) != set(range(n_classes)):
            raise ValueError(f"Every provenance group must occur in the {name} split.")

    fitted = []
    for c_value in (float(c) for c in c_grid):
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight=None, max_iter=5_000, random_state=seed, solver="lbfgs"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error" if strict_convergence else "ignore", ConvergenceWarning)
            pipeline.fit(train_h, train_codes)
        logits = _logits(pipeline, val_h, n_classes)
        fitted.append((_nll_bits(val_codes, _log_softmax(logits, 1.0)), c_value, pipeline, logits))
    _, selected_c, pipeline, logits = min(fitted, key=lambda item: (item[0], item[1]))
    temperature = _fit_temperature(logits, val_codes) if calibrate_temperature else 1.0
    return CalibratedProbe(
        pipeline=pipeline,
        selected_c=selected_c,
        temperature=temperature,
        classes=classes,
        validation_nll_bits={c: nll for nll, c, _, _ in fitted},
    )


@dataclass(frozen=True)
class LPIResult:
    bits: float
    ci_low: float
    ci_high: float
    confidence_level: float
    prior_log_loss_bits: float
    probe_log_loss_bits: float
    accuracy: float
    balanced_accuracy: float
    n_units: int
    n_clusters: int
    per_unit_gain_bits: np.ndarray
    bootstrap_bits: np.ndarray
    probe: CalibratedProbe | None = None

    @property
    def baseline_bits(self) -> float:
        """Entropy of the training prior on the test units (the most LPI can be)."""

        return self.prior_log_loss_bits


def lpi(
    train_embeddings,
    train_groups,
    val_embeddings,
    val_groups,
    test_embeddings,
    test_groups,
    *,
    test_clusters=None,
    c_grid: Sequence[float] = C_GRID,
    probe_seed: int = 42,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 20260908,
    confidence_level: float = 0.95,
    prior_alpha: float = 0.5,
) -> LPIResult:
    """Fit a calibrated probe and report test-set LPI with a cluster bootstrap interval.

    ``test_clusters`` identifies the independent unit (subject, patient,
    record); all units of a cluster are resampled together. Without it, each
    test unit is its own cluster.
    """

    probe = fit_calibrated_probe(
        train_embeddings, train_groups, val_embeddings, val_groups, c_grid=c_grid, seed=probe_seed
    )
    train_codes, _ = encode(train_groups, probe.classes)
    test_codes, _ = encode(test_groups, probe.classes)
    test_h = as_matrix(test_embeddings, name="test_embeddings")
    same_length(test_embeddings=test_h, test_groups=test_codes)
    prior = jeffreys_prior(train_codes, n_classes=len(probe.classes), alpha=prior_alpha)
    result = lpi_from_log_proba(
        test_codes,
        probe.log_proba(test_h),
        prior,
        clusters=test_clusters,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return LPIResult(**{**result.__dict__, "probe": probe})


def jeffreys_prior(train_codes, *, n_classes: int, alpha: float = 0.5) -> np.ndarray:
    counts = np.bincount(np.asarray(train_codes, dtype=np.int64), minlength=n_classes).astype(np.float64)
    return (counts + alpha) / (counts.sum() + n_classes * alpha)


def lpi_from_log_proba(
    test_codes,
    log_probabilities,
    prior,
    *,
    clusters=None,
    n_bootstrap: int = 10_000,
    seed: int = 20260908,
    confidence_level: float = 0.95,
) -> LPIResult:
    """LPI from already-computed probe log probabilities (native-unit mean)."""

    codes = np.asarray(test_codes, dtype=np.int64)
    log_p = np.asarray(log_probabilities, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if log_p.ndim != 2 or len(log_p) != len(codes) or log_p.shape[1] != len(prior):
        raise ValueError("log_probabilities must have shape (n_units, n_groups).")
    probe_loss = -log_p[np.arange(len(codes)), codes] / np.log(2.0)
    prior_loss = -np.log2(prior[codes])
    gains = prior_loss - probe_loss
    clusters = np.arange(len(codes)) if clusters is None else np.asarray(clusters)
    if len(clusters) != len(codes):
        raise ValueError("clusters must align with the test units.")
    draws = cluster_ratio_bootstrap(gains, clusters, n_bootstrap=n_bootstrap, seed=seed)
    low, high = percentile_interval(draws, confidence_level)
    predictions = np.argmax(log_p, axis=1)
    return LPIResult(
        bits=float(gains.mean()),
        ci_low=low,
        ci_high=high,
        confidence_level=confidence_level,
        prior_log_loss_bits=float(prior_loss.mean()),
        probe_log_loss_bits=float(probe_loss.mean()),
        accuracy=float(accuracy_score(codes, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(codes, predictions)),
        n_units=len(codes),
        n_clusters=len(np.unique(clusters)),
        per_unit_gain_bits=gains,
        bootstrap_bits=draws,
    )


def cluster_ratio_bootstrap(values, clusters, *, n_bootstrap: int, seed: int) -> np.ndarray:
    """Bootstrap draws of a native-unit mean, resampling whole clusters.

    Each draw is ``sum_c m_c a_c / sum_c m_c n_c`` with cluster multiplicities
    ``m_c``, cluster sums ``a_c`` and sizes ``n_c`` (paper Eq. 10, one encoder).
    """

    values = np.asarray(values, dtype=np.float64)
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    if n_clusters < 2:
        raise ValueError("The cluster bootstrap needs at least two clusters.")
    sums = np.bincount(inverse, weights=values, minlength=n_clusters)
    sizes = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 1_000):
        stop = min(start + 1_000, n_bootstrap)
        sampled = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        multiplicity = np.stack([np.bincount(row, minlength=n_clusters) for row in sampled])
        draws[start:stop] = (multiplicity @ sums) / (multiplicity @ sizes)
    return draws


def _logits(pipeline: Pipeline, embeddings, n_classes: int) -> np.ndarray:
    scores = np.asarray(pipeline.decision_function(as_matrix(embeddings, name="embeddings")), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack((np.zeros(len(scores)), scores))
    if scores.shape[1] != n_classes:
        raise RuntimeError("Probe logits do not match the declared classes.")
    return scores


def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    return scaled - logsumexp(scaled, axis=1, keepdims=True)


def _nll_bits(codes: np.ndarray, log_probabilities: np.ndarray) -> float:
    return float(-np.mean(log_probabilities[np.arange(len(codes)), codes]) / np.log(2.0))


def _fit_temperature(logits: np.ndarray, codes: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        return _nll_bits(codes, _log_softmax(logits, float(np.exp(log_temperature))))

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options={"xatol": 1e-6})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature calibration did not converge.")
    return 1.0 if result.fun > objective(0.0) + 1e-10 else float(np.exp(result.x))


__all__ = [
    "C_GRID",
    "CalibratedProbe",
    "LPIResult",
    "cluster_ratio_bootstrap",
    "fit_calibrated_probe",
    "jeffreys_prior",
    "lpi",
    "lpi_from_log_proba",
]
