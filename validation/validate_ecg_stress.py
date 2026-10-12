"""Reproduce the paper's PTB-XL device-audit Stress results with physiotrace.

Inputs are the saved bundle ``ptbxl_real_device_association_shift`` of the
paper repository (out-of-fold head predictions and per-fold association
targets). Embeddings are not needed: heads are already fitted, so this checks
regime construction, weighting, weighted AUROC, the within-fold patient-cluster
bootstrap, and the excess vulnerability G against the published values.

Usage:
    python validation/validate_ecg_stress.py PATH/TO/ptbxl_real_device_association_shift
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import physiotrace as pt
from physiotrace.bootstrap import cluster_bootstrap_counts, percentile_interval
from physiotrace.weights import REGIMES, restored_weights

SEED = 2026
REPLICATES = {"primary_real_device_pair": 2000}
DEFAULT_REPLICATES = 1000


def regime_tables(targets: pd.DataFrame, analysis: str) -> dict[int, dict[str, np.ndarray]]:
    """Rebuild every fold's regime tables from its design marginals and odds ratio."""

    tables: dict[int, dict[str, np.ndarray]] = {}
    rows = targets[(targets.analysis == analysis) & (targets.outer_fold >= 0)]
    for fold, group in rows.groupby("outer_fold"):
        first = group.iloc[0]
        theta = float(first.stress_odds_ratio_magnitude)
        tables[int(fold)] = {
            regime: pt.fixed_marginal_table(first.label_one_marginal, first.device_one_marginal, theta ** multiplier)
            for regime, multiplier in REGIMES.items()
        }
        saved = group.pivot_table(index=["regime", "label"], columns="device_target", values="target_mass")
        for regime, table in tables[int(fold)].items():
            if not np.allclose(saved.loc[regime].to_numpy(), table, atol=1e-12):
                raise AssertionError(f"Regime table mismatch: {analysis} fold {fold} {regime}")
    return tables


def evaluate(predictions: pd.DataFrame, tables, n_bootstrap: int) -> dict[tuple, np.ndarray]:
    """Fold-averaged AUROC draws per (model, head, representation, regime)."""

    reference = predictions.groupby(["model", "head", "representation"], sort=False).get_group(
        tuple(predictions[["model", "head", "representation"]].iloc[0])
    )
    folds = reference.outer_fold.to_numpy()
    y = reference.y_true.to_numpy()
    u = reference.device_binary.to_numpy()
    counts = cluster_bootstrap_counts(
        reference.patient_id.to_numpy(), strata=folds, support=2 * y + u, n_bootstrap=n_bootstrap, seed=SEED
    )
    fold_levels = sorted(np.unique(folds))
    weights = {
        (fold, regime): restored_weights(y[folds == fold], u[folds == fold], counts[:, folds == fold], table)
        for fold in fold_levels
        for regime, table in tables[fold].items()
    }
    draws: dict[tuple, np.ndarray] = {}
    for key, arm in predictions.groupby(["model", "head", "representation"], sort=False):
        if not np.array_equal(arm.record_id.to_numpy(), reference.record_id.to_numpy()):
            raise AssertionError("Prediction arms are not aligned record by record.")
        scores = arm.y_score.to_numpy()
        for regime in REGIMES:
            total = 0.0
            for fold in fold_levels:
                mask = folds == fold
                total = total + pt.weighted_auroc(y[mask], scores[mask], weights[(fold, regime)]) / len(fold_levels)
            draws[(*key, regime)] = total
    return draws


def main(bundle: Path) -> int:
    predictions = pd.read_csv(bundle / "oof_predictions.csv.gz")
    targets = pd.read_csv(bundle / "association_targets.csv")
    published = pd.read_csv(bundle / "bootstrap_contrasts.csv").set_index(["analysis", "model", "representation", "contrast"])
    rows = []
    for analysis, frame in predictions.groupby("analysis", sort=False):
        tables = regime_tables(targets, analysis)
        draws = evaluate(frame, tables, REPLICATES.get(analysis, DEFAULT_REPLICATES))
        for model in frame.model.unique():
            for representation in frame[frame.model == model].representation.unique():
                def shift(head):
                    return draws[(model, head, representation, "aligned")] - draws[(model, head, representation, "reversed")]

                g = shift("natural") - shift("decorrelated")
                checks = {("excess_association_shift_gap_G", representation): g}
                if representation == "target_projected":
                    natural_full = draws[(model, "natural", "full", "aligned")] - draws[(model, "natural", "full", "reversed")]
                    control_full = draws[(model, "decorrelated", "full", "aligned")] - draws[(model, "decorrelated", "full", "reversed")]
                    checks[("target_projection_reduction_G", "target_vs_full")] = (natural_full - control_full) - g
                    checks[("target_projection_independent_auc_change", "target_vs_full")] = (
                        draws[(model, "decorrelated", "target_projected", "independent")]
                        - draws[(model, "decorrelated", "full", "independent")]
                    )
                for (contrast, rep), values in checks.items():
                    reference = published.loc[(analysis, model, rep, contrast)]
                    low, high = percentile_interval(values[1:])
                    rows.append({
                        "analysis": analysis, "model": model, "representation": rep, "contrast": contrast,
                        "estimate": values[0], "published": reference.estimate,
                        "ci_low": low, "published_ci_low": reference.ci_lower,
                        "ci_high": high, "published_ci_high": reference.ci_upper,
                    })
    table = pd.DataFrame(rows)
    table["max_abs_difference"] = np.max(np.abs(np.column_stack([
        table.estimate - table.published, table.ci_low - table.published_ci_low, table.ci_high - table.published_ci_high,
    ])), axis=1)
    pd.set_option("display.width", 200)
    print(table[["analysis", "model", "representation", "contrast", "estimate", "published", "ci_low", "ci_high", "max_abs_difference"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    worst = float(table.max_abs_difference.max())
    print(f"\n{len(table)} published quantities; largest absolute difference {worst:.2e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
