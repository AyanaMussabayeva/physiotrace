"""Identity-aware bootstrap multiplicities.

All views and repeated observations of one identity (subject, patient,
record) move together. Draws can be taken independently within strata (for
example cross-fitting folds or task classes), and draws that lose a required
``(y, u)`` cell are rejected so that every regime stays identifiable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap_counts(
    clusters,
    *,
    n_bootstrap: int,
    seed: int,
    strata=None,
    support=None,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Return record multiplicities of shape ``(n_bootstrap + 1, n)``.

    Row 0 is the point estimate (all ones). In every later row, each stratum
    independently resamples its own clusters with replacement, and every
    record of a sampled cluster receives that cluster's multiplicity.

    ``support`` is an optional integer cell code per record (for example
    ``n_groups * y + u``); a stratum draw is rejected until every cell observed
    in that stratum keeps positive mass.
    """

    if n_bootstrap < 0 or max_attempts < 1:
        raise ValueError("n_bootstrap must be non-negative and max_attempts positive.")
    clusters = np.asarray(clusters)
    n = len(clusters)
    strata = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata)
    if len(strata) != n:
        raise ValueError("strata must align with clusters.")
    cells = None if support is None else np.asarray(support, dtype=np.int64)
    if cells is not None and len(cells) != n:
        raise ValueError("support must align with clusters.")
    crossing = pd.DataFrame({"cluster": clusters, "stratum": strata}).groupby("cluster", sort=False)["stratum"].nunique()
    if crossing.gt(1).any():
        raise ValueError("A cluster must not occur in more than one stratum.")

    rng = np.random.default_rng(seed)
    counts = np.zeros((n_bootstrap + 1, n), dtype=np.int32)
    counts[0] = 1
    for stratum in np.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        codes, unique_clusters = pd.factorize(clusters[positions], sort=False)
        n_clusters = len(unique_clusters)
        if n_clusters < 2:
            raise ValueError(f"Stratum {stratum!r} has fewer than two clusters.")
        if cells is not None:
            local_cells = cells[positions]
            required = np.unique(local_cells)
            n_cells = int(local_cells.max()) + 1
        for draw in range(1, n_bootstrap + 1):
            for _ in range(max_attempts):
                sampled = rng.integers(0, n_clusters, size=n_clusters)
                candidate = np.bincount(sampled, minlength=n_clusters).astype(np.int32)[codes]
                if cells is None:
                    break
                mass = np.bincount(local_cells, weights=candidate, minlength=n_cells)
                if np.all(mass[required] > 0):
                    break
            else:
                raise RuntimeError(f"No full-support draw for stratum {stratum!r} after {max_attempts} attempts.")
            counts[draw, positions] = candidate
    return counts


def percentile_interval(draws, confidence_level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap draws (excluding the point row)."""

    values = np.asarray(draws, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence_level
    return float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))


__all__ = ["cluster_bootstrap_counts", "percentile_interval"]
