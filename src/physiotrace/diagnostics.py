"""Validity checks that tell the user when an audit's numbers are fragile.

Checks never change a verdict. They are collected as :class:`Finding` objects
on the report (``report.findings``), printed by ``report.summary()``, and
findings of severity ``"warning"`` are also emitted as
:class:`PhysioTraceWarning`.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .weights import cell_weights, effective_sample_size


class PhysioTraceWarning(UserWarning):
    """A validity check of a PhysioTRACE audit did not pass."""


@dataclass(frozen=True)
class Gates:
    """Thresholds of the validity checks.

    The defaults suit moderate datasets. :meth:`paper` returns the stricter
    gates declared for the PTB-XL device audit of the paper.
    """

    min_cell_records: int = 30
    min_reversed_ess: float = 200.0
    min_reversed_cluster_ess: float = 100.0
    max_weight: float = 10.0
    weak_odds_ratio: float = 1.25
    min_bootstrap: int = 1_000
    min_records_per_class: int = 30

    @classmethod
    def paper(cls) -> "Gates":
        return cls(min_cell_records=75, min_reversed_ess=500.0, min_reversed_cluster_ess=300.0, max_weight=8.0)


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # "warning" or "note"
    message: str
    where: str | None = None
    value: float | None = None
    threshold: float | None = None

    def __str__(self) -> str:
        prefix = f"[{self.where}] " if self.where else ""
        return f"{self.severity}: {prefix}{self.message}"


def emit(findings: list[Finding]) -> None:
    for finding in findings:
        if finding.severity == "warning":
            warnings.warn(str(finding), PhysioTraceWarning, stacklevel=_caller_stacklevel())


def _caller_stacklevel() -> int:
    """Stack level of the first frame outside this package, so warnings point at user code."""

    package = os.path.dirname(os.path.abspath(__file__))
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.dirname(os.path.abspath(frame.f_code.co_filename)) == package:
        frame, level = frame.f_back, level + 1
    return level


def _cell_counts(y, u, n_classes: int, n_groups: int) -> np.ndarray:
    counts = np.zeros((n_classes, n_groups), dtype=np.int64)
    np.add.at(counts, (np.asarray(y), np.asarray(u)), 1)
    return counts


def check_cells(y, u, *, where: str, role: str, gates: Gates, n_groups: int = 2) -> list[Finding]:
    counts = _cell_counts(y, u, 2, n_groups)
    smallest = int(counts.min())
    if smallest < gates.min_cell_records:
        y_cell, u_cell = np.unravel_index(int(np.argmin(counts)), counts.shape)
        return [Finding(
            "small_cell", "warning",
            f"only {smallest} {role} records with (y={y_cell}, u={u_cell}); regime weights and "
            f"{'heads' if role == 'training' else 'AUROCs'} rest on few records",
            where, smallest, gates.min_cell_records,
        )]
    return []


def check_reversed_weights(y, u, clusters, table, *, where: str, gates: Gates) -> list[Finding]:
    """Effective sample sizes and the largest weight of the reversed regime on a test set."""

    findings = []
    weights = cell_weights(y, u, table)
    ess = effective_sample_size(weights)
    if ess < gates.min_reversed_ess:
        findings.append(Finding(
            "low_ess", "warning",
            f"reversed regime keeps an effective sample of {ess:.0f} records; intervals may be unreliable",
            where, ess, gates.min_reversed_ess,
        ))
    cluster_mass = pd.Series(weights).groupby(np.asarray(clusters)).sum().to_numpy()
    cluster_ess = effective_sample_size(cluster_mass)
    if cluster_ess < gates.min_reversed_cluster_ess:
        findings.append(Finding(
            "low_cluster_ess", "warning",
            f"reversed regime keeps an effective {cluster_ess:.0f} independent clusters; intervals may be unreliable",
            where, cluster_ess, gates.min_reversed_cluster_ess,
        ))
    if weights.max() > gates.max_weight:
        findings.append(Finding(
            "large_weight", "warning",
            f"one record carries weight {weights.max():.1f} (mean 1) in the reversed regime",
            where, float(weights.max()), gates.max_weight,
        ))
    return findings


def check_association(aligned_odds_ratio: float, *, where: str, gates: Gates) -> list[Finding]:
    strength = max(aligned_odds_ratio, 1.0 / aligned_odds_ratio)
    if strength < gates.weak_odds_ratio:
        return [Finding(
            "weak_association", "note",
            f"the stressed association is weak (odds ratio {aligned_odds_ratio:.2f}); a 'no reliance' verdict mostly "
            "reflects that there is little association to exploit. Pass odds_ratio_magnitude to stress at a strength "
            "seen elsewhere",
            where, aligned_odds_ratio, gates.weak_odds_ratio,
        )]
    return []


def check_convergence(converged: bool, *, where: str) -> list[Finding]:
    if converged:
        return []
    return [Finding("not_converged", "warning", "a task head did not converge; consider a smaller C or standardized inputs", where)]


def check_bootstrap(n_bootstrap: int, gates: Gates) -> list[Finding]:
    if n_bootstrap == 0:
        return [Finding(
            "no_bootstrap", "warning",
            "n_bootstrap=0: reliance and safety are decided on point estimates without intervals",
        )]
    if n_bootstrap < gates.min_bootstrap:
        return [Finding(
            "few_bootstrap_draws", "note", f"{n_bootstrap} bootstrap draws; percentile intervals are coarse",
            None, n_bootstrap, gates.min_bootstrap,
        )]
    return []


def check_clusters_given(clusters_given: bool) -> list[Finding]:
    if clusters_given:
        return []
    return [Finding(
        "no_clusters", "note",
        "no clusters given: every record is resampled independently. If records share a patient or subject, "
        "pass their identities as clusters, otherwise intervals are too narrow",
    )]


def check_class_counts(y, *, where: str, gates: Gates) -> list[Finding]:
    counts = np.bincount(np.asarray(y), minlength=2)
    if counts.min() < gates.min_records_per_class:
        return [Finding(
            "few_records", "warning", f"only {int(counts.min())} test records of class {int(np.argmin(counts))}",
            where, int(counts.min()), gates.min_records_per_class,
        )]
    return []


def check_grid_edge(value: float, grid, *, name: str) -> list[Finding]:
    grid = sorted(float(g) for g in grid)
    if len(grid) > 1 and value in (grid[0], grid[-1]):
        return [Finding(
            "grid_edge", "note", f"selected {name}={value:g} lies on the edge of the grid {grid}; consider widening it",
        )]
    return []


__all__ = ["Finding", "Gates", "PhysioTraceWarning"]
