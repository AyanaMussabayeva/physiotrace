"""Reproduce the paper's PTB-XL device audit from frozen embeddings with physiotrace.

Runs :func:`physiotrace.audit_observational` on each cohort and compares with
the published bundle: G before and after removing the target-conditional
device direction, their bootstrap intervals, the utility change, conditional
device accessibility, and G under each of the 100 random controls.

Usage:
    python validation/validate_ecg_end_to_end.py COHORTS_CSV EMBEDDINGS_NPZ MODEL BUNDLE_DIR

``COHORTS_CSV`` comes from ``export_ptbxl_cohorts.py``; ``EMBEDDINGS_NPZ`` is
the row-aligned embedding cache of ``MODEL`` (``ECGFounder`` or
``CLEF-Small``) for the 20,056-record natural-device cohort.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import physiotrace as pt

REPLICATES = {"primary_real_device_pair": 2000}


def published_random_g(bundle: Path, analysis: str, model: str) -> np.ndarray:
    metrics = pd.read_csv(bundle / "random_control_metrics.csv")
    metrics = metrics[(metrics.analysis == analysis) & (metrics.model == model)]
    if metrics.empty:
        return np.array([])
    fold_mean = metrics.groupby(["control_id", "head", "regime"]).auroc.mean().unstack(["head", "regime"])
    shift = lambda head: fold_mean[(head, "aligned")] - fold_mean[(head, "reversed")]  # noqa: E731
    return (shift("natural") - shift("decorrelated")).sort_index().to_numpy()


def main(cohorts_csv: Path, embeddings_npz: Path, model: str, bundle: Path) -> int:
    cohorts = pd.read_csv(cohorts_csv)
    cache = np.load(embeddings_npz)
    embeddings = cache["embeddings"]
    if "ecg_id" in cache.files:
        order = cache["ecg_id"].astype(np.int64)
        lookup = dict(zip(cohorts.ecg_id, cohorts.embedding_index))
        if not all(order[lookup[e]] == e for e in cohorts.ecg_id):
            raise AssertionError("Embedding rows are not aligned with embedding_index.")
    contrasts = pd.read_csv(bundle / "bootstrap_contrasts.csv").set_index(["analysis", "model", "representation", "contrast"])
    probes = pd.read_csv(bundle / "probe_metrics.csv")
    targets = pd.read_csv(bundle / "association_targets.csv")
    rows = []
    for analysis, frame in cohorts.groupby("analysis", sort=False):
        weak = analysis.startswith("weak")
        magnitude = None
        if weak:  # stressed at the primary pair's per-fold strength, as in the paper
            selected = targets[(targets.analysis == analysis) & (targets.outer_fold >= 0)]
            magnitude = selected.groupby("outer_fold").stress_odds_ratio_magnitude.first().to_dict()
        report = pt.audit_observational(
            embeddings[frame.embedding_index.to_numpy()], frame.target.to_numpy(), frame.device_target.to_numpy(),
            folds=frame.outer_fold.to_numpy(), clusters=frame.patient_id.to_numpy(), design=frame.design.to_numpy(),
            odds_ratio_magnitude=magnitude, n_bootstrap=REPLICATES.get(analysis, 1000), seed=2026,
            y_levels=(0, 1), u_levels=(0, 1),
        )
        published = lambda rep, contrast: contrasts.loc[(analysis, model, rep, contrast)]  # noqa: E731
        g = published("full", "excess_association_shift_gap_G")
        values = {
            "G": (report.G, g.estimate), "G ci_low": (report.G_ci[0], g.ci_lower), "G ci_high": (report.G_ci[1], g.ci_upper),
        }
        accessibility = probes[(probes.analysis == analysis) & (probes.probe == f"{model}:embedding_conditional")
                               & (probes.label_stratum == "macro_crossfit")]
        values["conditional accessibility"] = (report.recover["conditional_accessibility_auroc"],
                                               accessibility.conditional_device_auroc.iloc[0])
        if (analysis, model, "target_projected", "excess_association_shift_gap_G") in contrasts.index:
            after = published("target_projected", "excess_association_shift_gap_G")
            utility = published("target_vs_full", "target_projection_independent_auc_change")
            values.update({
                "G after": (report.G_after, after.estimate),
                "G after ci_low": (report.G_after_ci[0], after.ci_lower),
                "G after ci_high": (report.G_after_ci[1], after.ci_upper),
                "utility change": (report.utility_change, utility.estimate),
                "utility ci_low": (report.utility_change_ci[0], utility.ci_lower),
            })
            random_published = published_random_g(bundle, analysis, model)
            values["max |G_random - published|"] = (float(np.max(np.abs(report.G_random - random_published))), 0.0)
        for name, (ours, theirs) in values.items():
            rows.append({"analysis": analysis, "quantity": name, "physiotrace": ours, "published": theirs,
                         "abs_difference": abs(ours - theirs)})
        print(f"{analysis}: {report.verdict} | G={report.G:.6f} after={report.G_after:.6f}", flush=True)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nlargest absolute difference: {table.abs_difference.max():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])))
