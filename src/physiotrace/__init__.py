"""PhysioTRACE: behavioral audit of provenance reliance in frozen-encoder task heads.

Recover how decodable provenance is, Stress fixed heads by reversing only the
provenance--target association, Intervene by removing a train-localized
provenance component, and Verify the remedy against matched random controls
within a declared utility margin.
"""

from . import bootstrap, intervene, metrics, recover, stress, synthetic, weights
from .audit import PairedSplit, audit_observational, audit_paired
from .intervene import Subspace, Whitening, conditional_direction, paired_subspace, random_subspaces
from .metrics import weighted_auroc
from .recover import LPIResult, fit_calibrated_probe, lpi
from .report import AuditReport
from .stress import excess_vulnerability, regime_aurocs
from .verify import Verdict
from .weights import (
    cell_weights,
    designed_coupling_table,
    fixed_marginal_table,
    independence_weights,
    observational_regimes,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "AuditReport",
    "LPIResult",
    "PairedSplit",
    "Subspace",
    "Verdict",
    "Whitening",
    "audit_observational",
    "audit_paired",
    "bootstrap",
    "cell_weights",
    "conditional_direction",
    "designed_coupling_table",
    "excess_vulnerability",
    "fit_calibrated_probe",
    "fixed_marginal_table",
    "independence_weights",
    "intervene",
    "lpi",
    "metrics",
    "observational_regimes",
    "paired_subspace",
    "random_subspaces",
    "recover",
    "regime_aurocs",
    "stress",
    "synthetic",
    "weighted_auroc",
    "weights",
]
