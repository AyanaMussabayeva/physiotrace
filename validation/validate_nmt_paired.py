"""Reproduce the paper's paired NMT audit end to end with physiotrace.

For each retained encoder (seeds 13, 23, 31, 47; task-only and
task-plus-reconstruction, d = 128), the frozen embeddings are passed to
:func:`physiotrace.audit_paired` and its point estimates are compared with the
published per-seed statistics: G before and after removing the paired
reference subspace, G under each random control, the paired logit contrast,
the selected C, and calibrated LPI.

Usage:
    python validation/validate_nmt_paired.py EMBEDDING_DIR RESULTS_DIR

``EMBEDDING_DIR`` holds ``seed{S}_{objective}_d128_*.npz``; ``RESULTS_DIR`` is
the paper repository's ``additional_experiments/results``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import physiotrace as pt

SEEDS = (13, 23, 31, 47)
OBJECTIVES = ("task_only", "task_plus_reconstruction")


def load_split(embeddings: np.ndarray, metadata: pd.DataFrame) -> pt.PairedSplit:
    if len(embeddings) != len(metadata):
        raise AssertionError("Embeddings and metadata rows differ.")
    return pt.PairedSplit(embeddings, metadata.label_id.to_numpy(), metadata.group.to_numpy(), metadata.record_id.to_numpy())


def saved_probe_lpi(model_path: Path, test_embeddings: np.ndarray, test: pd.DataFrame) -> pt.LPIResult:
    """LPI from the exported, locked probe state (no refitting)."""

    state = np.load(model_path)
    standardized = (np.asarray(test_embeddings, dtype=np.float64) - state["scaler_mean"]) / state["scaler_scale"]
    logits = standardized @ state["coef"].T + state["intercept"]
    scaled = logits / float(state["temperature"])
    log_proba = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
    return pt.recover.lpi_from_log_proba(
        test.group_id.to_numpy(), log_proba, state["prior_probabilities"],
        # String cluster ids reproduce the published cluster order and therefore its bootstrap draws.
        clusters=test.record_id.astype(str).to_numpy(), n_bootstrap=10_000, seed=20260908,
    )


def main(embedding_dir: Path, results: Path) -> int:
    lpi_dir = results / "nmt_mechanism_lpi_20260908"
    metadata = {split: pd.read_csv(lpi_dir / f"metadata_{split}.csv") for split in ("train", "val", "test")}
    statistics = pd.read_csv(results / "nmt_multiseed_d128_20260908/aggregate/per_seed_shortcut_statistics.csv")
    statistics = statistics.set_index(["seed", "objective", "statistic"]).estimate
    random_g = pd.read_csv(results / "nmt_multiseed_d128_20260908/aggregate/per_seed_random_G.csv")
    published_lpi = pd.read_csv(lpi_dir / "per_seed_lpi.csv").set_index(["seed", "arm"])

    rows = []
    for seed in SEEDS:
        heads = pd.read_csv(results / f"nmt_multiseed_d128_20260908/seed{seed}/shortcut/head_audit.csv")
        for objective in OBJECTIVES:
            (path,) = embedding_dir.glob(f"seed{seed}_{objective}_d128_*.npz")
            cache = np.load(path)
            splits = {name: load_split(cache[f"{name}_z"], metadata[name]) for name in ("train", "val", "test")}
            report = pt.audit_paired(
                splits["train"], splits["test"], val=splits["val"], p_shortcut=0.6, n_random=10,
                seed=seed + 10_000, n_bootstrap=200, view_order=("AVG", "CZREF", "LE"), whitening="svd",
            )
            published_random = (
                random_g[(random_g.seed == seed) & (random_g.objective == objective) & (random_g.representation == "random_control")]
                .sort_values("control_id").G.to_numpy()
            )
            expected_c = heads[heads.objective == objective].selected_c.iloc[0]
            lpi = report.details["lpi"]
            values = {
                "G_full": (report.G, statistics[(seed, objective, "G_full")]),
                "G_target": (report.G_after, statistics[(seed, objective, "G_target")]),
                "R_pair_full": (report.details["paired_logit_contrast"]["full"], statistics[(seed, objective, "R_pair_full")]),
                "R_pair_target": (report.details["paired_logit_contrast"]["targeted"], statistics[(seed, objective, "R_pair_target")]),
                "max |G_random - published|": (float(np.max(np.abs(report.G_random - published_random))), 0.0),
                "selected_C": (report.details["C"], float(expected_c)),
                "LPI_bits (refitted probe)": (lpi.bits, published_lpi.loc[(seed, objective), "lpi_bits"]),
            }
            saved = saved_probe_lpi(lpi_dir / f"models/seed{seed}_{objective}.npz", cache["test_z"], metadata["test"])
            published_row = published_lpi.loc[(seed, objective)]
            values["LPI_bits (saved probe)"] = (saved.bits, published_row.lpi_bits)
            values["LPI_ci_low (saved probe)"] = (saved.ci_low, published_row.ci_low_bits)
            values["LPI_ci_high (saved probe)"] = (saved.ci_high, published_row.ci_high_bits)
            for name, (ours, theirs) in values.items():
                rows.append({"seed": seed, "objective": objective, "quantity": name, "physiotrace": ours,
                             "published": theirs, "abs_difference": abs(ours - theirs)})
            rows.append({"seed": seed, "objective": objective, "quantity": "verdict", "physiotrace": str(report.verdict),
                         "published": "reliance, remedy certified", "abs_difference": 0.0 if report.verdict == pt.Verdict.REMEDY_CERTIFIED else 1.0})
            print(f"seed {seed} {objective}: G={report.G:.6f} after={report.G_after:.6f} verdict={report.verdict}", flush=True)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    numeric = table[table.quantity != "verdict"]
    print("\nlargest absolute difference per quantity:")
    print(numeric.groupby("quantity").abs_difference.max().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
