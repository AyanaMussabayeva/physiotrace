"""Fixed-marginal association regimes and the weights that realize them.

A regime is a joint table ``pi[y, u]`` over target class ``y`` and provenance
group ``u``. Reweighting a fixed set of records to a regime changes only the
target--provenance association; records, labels, embeddings, and heads stay
the same (paper Eq. 2 and Appendix A).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ._validation import as_codes, same_length

#: Log-odds-ratio multipliers of the observational regimes (paper Appendix A).
REGIMES: dict[str, float] = {
    "aligned": 1.0,
    "intermediate_aligned": 0.5,
    "independent": 0.0,
    "intermediate_reversed": -0.5,
    "reversed": -1.0,
}
CORE_REGIMES = ("aligned", "independent", "reversed")


def _check_table(table, *, full_support: bool = True, atol: float = 1e-12) -> np.ndarray:
    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("A regime table must be two-dimensional with at least 2 x 2 cells.")
    if not np.isfinite(values).all() or np.any(values < -atol):
        raise ValueError("A regime table must contain finite non-negative mass.")
    if not np.isclose(values.sum(), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"A regime table must sum to one, got {values.sum():.17g}.")
    values = np.maximum(values, 0.0)
    values = values / values.sum()
    if full_support and np.any(values <= 0.0):
        raise ValueError("Every cell of the regime table must have positive mass.")
    return values


def odds_ratio(table) -> float:
    """Cross-product ratio ``p00 p11 / (p01 p10)`` of a binary ``[y, u]`` table."""

    values = _check_table(table)
    if values.shape != (2, 2):
        raise ValueError("odds_ratio requires a 2 x 2 table.")
    return float(values[0, 0] * values[1, 1] / (values[0, 1] * values[1, 0]))


def empirical_table(y, u, *, n_classes: int = 2, n_groups: int = 2) -> np.ndarray:
    """Observed joint frequencies of ``(y, u)`` as a ``[n_classes, n_groups]`` table."""

    y = as_codes(y, name="y", n_levels=n_classes)
    u = as_codes(u, name="u", n_levels=n_groups)
    same_length(y=y, u=u)
    counts = np.zeros((n_classes, n_groups), dtype=np.float64)
    np.add.at(counts, (y, u), 1.0)
    return counts / counts.sum()


def _table_from_p11(p_y1: float, p_u1: float, p11: float) -> np.ndarray:
    # The alternative expression for p00 avoids cancellation at the lower
    # Frechet boundary when p_y1 + p_u1 > 1.
    lower = p_y1 + p_u1 - 1.0
    p00 = p11 - lower if lower > 0.0 else (1.0 - p_y1 - p_u1) + p11
    return np.array([[p00, p_u1 - p11], [p_y1 - p11, p11]], dtype=np.float64)


def fixed_marginal_table(p_y: float, p_u: float, odds_ratio: float) -> np.ndarray:
    """Binary ``[y, u]`` table with ``P(y=1)=p_y``, ``P(u=1)=p_u`` and the given odds ratio.

    The free cell ``P(y=1, u=1)`` is solved inside its Frechet interval, so any
    finite positive odds ratio yields positive mass in all four cells.
    """

    margins = np.asarray([p_y, p_u], dtype=np.float64)
    if not np.isfinite(margins).all() or np.any(margins <= 0.0) or np.any(margins >= 1.0):
        raise ValueError("p_y and p_u must lie strictly between 0 and 1.")
    if not np.isfinite(odds_ratio) or odds_ratio <= 0.0:
        raise ValueError("odds_ratio must be finite and positive.")
    p_y, p_u = float(p_y), float(p_u)
    lower, upper = max(0.0, p_y + p_u - 1.0), min(p_y, p_u)
    log_target = float(np.log(odds_ratio))

    def residual(p11: float) -> float:
        table = _table_from_p11(p_y, p_u, p11)
        if table[0, 0] <= 0.0 or table[1, 1] <= 0.0:
            return -np.inf
        if table[0, 1] <= 0.0 or table[1, 0] <= 0.0:
            return np.inf
        return float(np.log(table[0, 0]) + np.log(table[1, 1]) - np.log(table[0, 1]) - np.log(table[1, 0]) - log_target)

    p11 = brentq(residual, lower, upper, xtol=np.finfo(np.float64).tiny, rtol=8.0 * np.finfo(np.float64).eps, maxiter=200)
    return _check_table(_table_from_p11(p_y, p_u, p11))


def observational_regimes(
    y,
    u,
    *,
    multipliers: dict[str, float] | None = None,
    odds_ratio_magnitude: float | None = None,
) -> dict[str, np.ndarray]:
    """Regimes for a binary target and binary provenance factor.

    Marginals and the association strength ``theta`` are estimated from the
    given (training) records. Regime ``e`` has odds ratio ``theta ** m_e``, so
    ``aligned`` reproduces the observed association, ``independent`` removes it,
    and ``reversed`` inverts it.
    """

    table = empirical_table(y, u)
    if np.any(table <= 0.0):
        raise ValueError("Every (y, u) cell must be observed to define association regimes.")
    theta = odds_ratio(table) if odds_ratio_magnitude is None else float(odds_ratio_magnitude)
    if odds_ratio_magnitude is not None and odds_ratio(table) < 1.0:
        theta = 1.0 / theta
    p_y, p_u = float(table[1].sum()), float(table[:, 1].sum())
    multipliers = REGIMES if multipliers is None else multipliers
    return {name: fixed_marginal_table(p_y, p_u, float(np.exp(m * np.log(theta)))) for name, m in multipliers.items()}


def designed_coupling_table(n_groups: int, shortcut_group: int, p_shortcut: float, regime: str) -> np.ndarray:
    """Designed binary-target coupling for ``n_groups`` provenance views (paired audits).

    Both marginals are uniform. In the ``aligned`` regime the shortcut group has
    ``P(u = g | y = 1) = p_shortcut``; ``reversed`` swaps the two class rows and
    ``independent`` is uniform. Returns the joint table ``0.5 * P(u | y)``.
    """

    if n_groups < 2:
        raise ValueError("A designed coupling needs at least two groups.")
    if not 1.0 / n_groups < p_shortcut < 2.0 / n_groups:
        raise ValueError(f"p_shortcut must lie strictly between {1.0 / n_groups:.6f} and {2.0 / n_groups:.6f}.")
    if shortcut_group not in range(n_groups):
        raise ValueError(f"shortcut_group must be in [0, {n_groups}).")
    if regime == "independent":
        conditional = np.full((2, n_groups), 1.0 / n_groups)
    elif regime in ("aligned", "reversed"):
        positive = np.full(n_groups, (1.0 - p_shortcut) / (n_groups - 1))
        positive[shortcut_group] = p_shortcut
        conditional = np.stack([2.0 / n_groups - positive, positive])
        if regime == "reversed":
            conditional = conditional[::-1].copy()
    else:
        raise ValueError(f"regime must be 'aligned', 'independent', or 'reversed', got {regime!r}.")
    return 0.5 * conditional


def cell_weights(y, u, table) -> np.ndarray:
    """Mean-one weights whose normalized ``(y, u)`` cell masses equal ``table``.

    Every cell with target mass must be observed: reweighting cannot create
    support that the data lack.
    """

    target = _check_table(table, full_support=False)
    y = as_codes(y, name="y", n_levels=target.shape[0])
    u = as_codes(u, name="u", n_levels=target.shape[1])
    n = same_length(y=y, u=u)
    counts = np.zeros(target.shape, dtype=np.float64)
    np.add.at(counts, (y, u), 1.0)
    missing = np.argwhere((counts == 0) & (target > 0))
    if missing.size:
        raise ValueError(f"Observed data lack support for (y, u) cells {[tuple(map(int, c)) for c in missing]}.")
    empirical = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(counts > 0, target / empirical, 0.0)
    weights = ratio[y, u]
    return weights / weights.mean()


def independence_weights(y, u, *, n_classes: int | None = None, n_groups: int | None = None) -> np.ndarray:
    """Weights that make ``y`` and ``u`` independent while keeping both marginals."""

    y = as_codes(y, name="y")
    u = as_codes(u, name="u")
    n_classes = int(y.max()) + 1 if n_classes is None else n_classes
    n_groups = int(u.max()) + 1 if n_groups is None else n_groups
    table = empirical_table(y, u, n_classes=n_classes, n_groups=n_groups)
    return cell_weights(y, u, np.outer(table.sum(axis=1), table.sum(axis=0)))


def restored_weights(y, u, counts, table) -> np.ndarray:
    """Calibrate bootstrap multiplicities so every draw has exactly ``table``'s cell masses.

    ``counts`` has shape ``(draws, records)``; each row is rescaled within every
    ``(y, u)`` cell. Rows sum to one.
    """

    target = _check_table(table, full_support=False)
    y = as_codes(y, name="y", n_levels=target.shape[0])
    u = as_codes(u, name="u", n_levels=target.shape[1])
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(y):
        raise ValueError("counts must have shape (draws, records).")
    weights = np.zeros_like(counts)
    for label in range(target.shape[0]):
        for group in range(target.shape[1]):
            if target[label, group] <= 0.0:
                continue
            selected = (y == label) & (u == group)
            denominator = counts[:, selected].sum(axis=1)
            if np.any(denominator <= 0.0):
                raise ValueError(f"A bootstrap draw lost every record of cell ({label}, {group}).")
            weights[:, selected] = counts[:, selected] * (target[label, group] / denominator)[:, None]
    return weights


def effective_sample_size(weights) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError("weights must be a non-empty vector of non-negative values with positive mass.")
    return float(values.sum() ** 2 / np.square(values).sum())


__all__ = [
    "CORE_REGIMES",
    "REGIMES",
    "cell_weights",
    "designed_coupling_table",
    "effective_sample_size",
    "empirical_table",
    "fixed_marginal_table",
    "independence_weights",
    "observational_regimes",
    "odds_ratio",
    "restored_weights",
]
