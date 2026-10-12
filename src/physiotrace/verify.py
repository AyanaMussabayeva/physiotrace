"""Verdict rule (Verify).

* **no reliance** — ``G`` is below the declared effect size ``epsilon`` or its
  lower confidence bound is not above zero;
* **reliance, remedy certified** — reliance is detected and suppression is
  *specific* (it reduces ``G`` more than every rank- and energy-matched random
  control) and *safe* (the lower bound of the utility change is at least
  ``-delta``);
* **reliance, remedy not certified** — reliance is detected but suppression
  fails either test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Verdict(str, Enum):
    NO_RELIANCE = "no_reliance"
    REMEDY_CERTIFIED = "reliance_remedy_certified"
    REMEDY_NOT_CERTIFIED = "reliance_remedy_not_certified"

    def __str__(self) -> str:
        return {
            "no_reliance": "no reliance",
            "reliance_remedy_certified": "reliance, remedy certified",
            "reliance_remedy_not_certified": "reliance, remedy not certified",
        }[self.value]


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reliance: bool
    specific: bool | None
    safe: bool | None
    random_controls_beaten: int | None
    random_controls_total: int | None
    specificity_p_value: float | None


def detect_reliance(G: float, G_ci_low: float | None, *, epsilon: float) -> bool:
    """Reliance requires ``G >= epsilon`` and, when an interval exists, a lower bound above zero."""

    if not np.isfinite(G) or G < epsilon:
        return False
    return G_ci_low is None or not np.isfinite(G_ci_low) or G_ci_low > 0.0


def decide(
    *,
    G: float,
    G_ci_low: float | None,
    G_after: float | None,
    G_random,
    utility_change_ci_low: float | None,
    epsilon: float,
    delta: float,
) -> Decision:
    reliance = detect_reliance(G, G_ci_low, epsilon=epsilon)
    if not reliance:
        return Decision(Verdict.NO_RELIANCE, False, None, None, None, None, None)
    random_g = np.asarray(G_random, dtype=np.float64)
    reduction = G - G_after
    random_reductions = G - random_g
    beaten = int(np.sum(reduction > random_reductions))
    specific = bool(random_g.size > 0 and beaten == random_g.size)
    p_value = float((1 + np.sum(random_reductions >= reduction)) / (1 + random_g.size))
    safe = bool(utility_change_ci_low is not None and utility_change_ci_low >= -delta)
    verdict = Verdict.REMEDY_CERTIFIED if specific and safe else Verdict.REMEDY_NOT_CERTIFIED
    return Decision(verdict, True, specific, safe, beaten, int(random_g.size), p_value)


__all__ = ["Decision", "Verdict", "decide", "detect_reliance"]
