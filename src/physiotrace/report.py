"""The audit report returned by :func:`physiotrace.audit_observational` and :func:`physiotrace.audit_paired`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .verify import Decision, Verdict


@dataclass(frozen=True)
class AuditReport:
    mode: str
    decision: Decision
    G: float
    G_ci: tuple[float, float]
    G_after: float
    G_after_ci: tuple[float, float]
    G_random: np.ndarray
    utility_metric: str
    utility_before: float
    utility_after: float
    utility_change: float
    utility_change_ci: tuple[float, float]
    epsilon: float
    delta: float
    rank: int
    confidence_level: float
    n_bootstrap: int
    recover: dict
    aurocs: pd.DataFrame
    details: dict = field(default_factory=dict)

    @property
    def verdict(self) -> Verdict:
        return self.decision.verdict

    def to_dict(self) -> dict:
        """Flat, JSON-serializable summary."""

        d = self.decision
        return {
            "mode": self.mode,
            "verdict": self.verdict.value,
            "reliance": d.reliance,
            "specific": d.specific,
            "safe": d.safe,
            "G": self.G,
            "G_ci": list(self.G_ci),
            "G_after": self.G_after,
            "G_after_ci": list(self.G_after_ci),
            "G_reduction_fraction": self.G_reduction_fraction,
            "random_controls_beaten": d.random_controls_beaten,
            "random_controls_total": d.random_controls_total,
            "specificity_p_value": d.specificity_p_value,
            "utility_metric": self.utility_metric,
            "utility_before": self.utility_before,
            "utility_after": self.utility_after,
            "utility_change": self.utility_change,
            "utility_change_ci": list(self.utility_change_ci),
            "epsilon": self.epsilon,
            "delta": self.delta,
            "rank": self.rank,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "recover": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.recover.items()},
        }

    @property
    def G_reduction_fraction(self) -> float:
        return float((self.G - self.G_after) / self.G) if self.G != 0 else float("nan")

    def summary(self) -> str:
        d = self.decision
        level = int(round(100 * self.confidence_level))

        def ci(interval) -> str:
            low, high = interval
            return "" if not np.isfinite(low) else f" [{low:.4f}, {high:.4f}]"

        lines = [f"PhysioTRACE audit ({self.mode})", f"  verdict: {self.verdict}"]
        recover = ", ".join(
            f"{key} = {value:.4f}" for key, value in self.recover.items() if isinstance(value, float)
        )
        if recover:
            lines.append(f"  Recover   {recover}")
        lines.append(f"  Stress    G = {self.G:.4f}{ci(self.G_ci)}  (epsilon = {self.epsilon:g}; {level}% interval)")
        if not d.reliance:
            lines.append("  Intervene / Verify not reached: no reliance detected.")
            return "\n".join(lines)
        lines.append(
            f"  Intervene G after rank-{self.rank} removal = {self.G_after:.4f}{ci(self.G_after_ci)}"
            f"  ({100 * self.G_reduction_fraction:.0f}% reduction)"
        )
        lines.append(
            f"  Verify    specific: {'yes' if d.specific else 'no'}"
            f" (beats {d.random_controls_beaten}/{d.random_controls_total} random controls)"
        )
        lines.append(
            f"            safe: {'yes' if d.safe else 'no'}"
            f" ({self.utility_metric} change {self.utility_change:+.4f}{ci(self.utility_change_ci)}, delta = {self.delta:g})"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


__all__ = ["AuditReport"]
