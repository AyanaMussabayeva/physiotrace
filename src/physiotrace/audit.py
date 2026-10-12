"""End-to-end PhysioTRACE audits of a frozen encoder.

Two designs are supported, matching the two modes of the paper:

* :func:`audit_observational` — collected data in which a binary provenance
  factor (device, site) is naturally associated with a binary target. The
  exposed head is fitted on the natural association, the control head with
  independence weights, and provenance is localized conditionally on the
  target. Cross-fitting over folds is supported (paper Section 4.3).
* :func:`audit_paired` — every record is observed under ``K`` provenance views
  (for example EEG references). The association is induced by a designed
  coupling, each view serves in turn as the shortcut, and provenance is
  localized from within-record view contrasts without task labels (paper
  Section 4.2).

Both return an :class:`~physiotrace.report.AuditReport` with the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ._validation import as_matrix, encode, same_length
from .bootstrap import cluster_bootstrap_counts, percentile_interval
from .heads import fit_linear_head
from .intervene import class_means, conditional_direction, paired_subspace, random_subspaces
from .metrics import weighted_auroc
from .recover import lpi as _lpi
from .report import AuditReport
from .stress import flip_rate
from .verify import decide
from .weights import (
    CORE_REGIMES,
    REGIMES,
    cell_weights,
    designed_coupling_table,
    independence_weights,
    observational_regimes,
    restored_weights,
)

EXPOSED, CONTROL = "exposed", "control"
FULL, TARGETED = "full", "targeted"


def _check_regimes(names) -> None:
    missing = set(CORE_REGIMES).difference(names)
    if missing:
        raise ValueError(f"Regimes must include {sorted(missing)}.")


# --------------------------------------------------------------------------- observational


def audit_observational(
    embeddings,
    y,
    u,
    *,
    folds,
    clusters=None,
    design=None,
    test_folds: Sequence | None = None,
    epsilon: float = 0.01,
    delta: float = 0.01,
    C: float = 1.0,
    n_random: int = 100,
    regimes: Mapping[str, float] | None = None,
    odds_ratio_magnitude: float | None = None,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 2026,
    y_levels: Sequence | None = None,
    u_levels: Sequence | None = None,
) -> AuditReport:
    """Audit reliance on a naturally associated binary provenance factor.

    Parameters
    ----------
    embeddings : array (n, d)
        Frozen embeddings, one row per record.
    y, u : array (n,)
        Binary target and binary provenance labels (any two values each; the
        second of the sorted levels is coded 1, or pass ``y_levels``/``u_levels``).
    folds : array (n,)
        Cross-fitting fold of each record. Each evaluated fold is held out
        while heads, scaler, and localization are fitted on all other folds.
        Keep every identity (patient) inside one fold. For a single train/test
        split, use two fold values and ``test_folds=[test_value]``.
    clusters : array (n,), optional
        Identity of each record (for example patient); the bootstrap resamples
        clusters within each test fold. Defaults to one cluster per record.
    design : bool array (n,), optional
        Records whose association defines the regimes; within each fold only
        its training records are used. Defaults to all training records.
    epsilon, delta : float
        Declared effect size for reliance and tolerated utility loss.
    C : float
        Inverse regularization of the heads and of the provenance probe.
    n_random : int
        Number of rank-one Haar random controls per fold.
    regimes : mapping, optional
        Regime name to log-odds-ratio multiplier; defaults to
        :data:`physiotrace.weights.REGIMES` (must include aligned,
        independent, reversed).
    """

    h = as_matrix(embeddings, name="embeddings")
    y_codes, _ = encode(y, y_levels)
    u_codes, _ = encode(u, u_levels)
    folds = np.asarray(folds)
    n = same_length(embeddings=h, y=y_codes, u=u_codes, folds=folds)
    if y_codes.max() != 1 or u_codes.max() != 1:
        raise ValueError("audit_observational needs a binary target and a binary provenance factor.")
    clusters = np.arange(n) if clusters is None else np.asarray(clusters)
    design = np.ones(n, dtype=bool) if design is None else np.asarray(design, dtype=bool)
    same_length(embeddings=h, clusters=clusters, design=design)
    multipliers = dict(REGIMES if regimes is None else regimes)
    _check_regimes(multipliers)
    fold_levels = tuple(np.unique(folds).tolist()) if test_folds is None else tuple(test_folds)

    arm_scores: dict[tuple, np.ndarray] = {}
    random_scores: dict[tuple, list[np.ndarray]] = {}
    tables_by_fold: dict = {}
    accessibility: list[float] = []
    fold_rows = []
    for index, fold in enumerate(fold_levels):
        test = folds == fold
        train = ~test
        if not test.any():
            raise ValueError(f"Fold {fold!r} has no records.")
        scaler = StandardScaler().fit(h[train])
        z_train, z_test = scaler.transform(h[train]), scaler.transform(h[test])
        y_train, u_train = y_codes[train], u_codes[train]
        heads = {
            EXPOSED: fit_linear_head(z_train, y_train, C=C, tol=1e-7),
            CONTROL: fit_linear_head(z_train, y_train, independence_weights(y_train, u_train), C=C, tol=1e-7),
        }
        tables_by_fold[fold] = observational_regimes(
            y_codes[train & design], u_codes[train & design], multipliers=multipliers,
            odds_ratio_magnitude=odds_ratio_magnitude,
        )

        subspace, probe = conditional_direction(z_train, y_train, u_train, C=C)
        residual_test = z_test - class_means(z_train, y_train)[y_codes[test]]
        device_score = probe.predict_proba(subspace.whitening.transform(residual_test))[:, 1]
        accessibility.append(_conditional_auroc(u_codes[test], y_codes[test], device_score))

        z_targeted = subspace.remove(z_test)
        controls = random_subspaces(subspace.whitening, 1, n_random, seed=seed + 10_000 * (index + 1) + h.shape[1])
        for name, head in heads.items():
            arm_scores[(fold, name, FULL)] = head.scores(z_test)
            arm_scores[(fold, name, TARGETED)] = head.scores(z_targeted)
            random_scores[(fold, name)] = [head.scores(control.remove(z_test)) for control in controls]
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "heads_converged": all(head.converged for head in heads.values()),
                "aligned_odds_ratio": _odds(tables_by_fold[fold]["aligned"]),
                "conditional_accessibility_auroc": accessibility[-1],
            }
        )

    # Point estimates and bootstrap draws share one set of multiplicities.
    test_rows = np.isin(folds, np.asarray(fold_levels, dtype=folds.dtype))
    positions = np.flatnonzero(test_rows)
    counts = cluster_bootstrap_counts(
        clusters[positions], strata=folds[positions], support=2 * y_codes[positions] + u_codes[positions],
        n_bootstrap=n_bootstrap, seed=seed,
    )
    auc = {}  # (head, representation, regime) -> draws (point at index 0), fold-averaged
    for fold in fold_levels:
        local = folds[positions] == fold
        y_f, u_f = y_codes[folds == fold], u_codes[folds == fold]
        for regime, table in tables_by_fold[fold].items():
            weights = restored_weights(y_f, u_f, counts[:, local], table)
            for head in (EXPOSED, CONTROL):
                for representation in (FULL, TARGETED):
                    draws = weighted_auroc(y_f, arm_scores[(fold, head, representation)], weights)
                    key = (head, representation, regime)
                    auc[key] = auc.get(key, 0.0) + draws / len(fold_levels)

    def g_draws(representation: str) -> np.ndarray:
        shift = {head: auc[(head, representation, "aligned")] - auc[(head, representation, "reversed")] for head in (EXPOSED, CONTROL)}
        return shift[EXPOSED] - shift[CONTROL]

    g_full, g_targeted = g_draws(FULL), g_draws(TARGETED)
    utility = auc[(CONTROL, TARGETED, "independent")] - auc[(CONTROL, FULL, "independent")]

    g_random = np.zeros(n_random)
    for fold in fold_levels:
        y_f, u_f = y_codes[folds == fold], u_codes[folds == fold]
        tables = tables_by_fold[fold]
        for j in range(n_random):
            shift = {}
            for head in (EXPOSED, CONTROL):
                scores = random_scores[(fold, head)][j]
                shift[head] = weighted_auroc(y_f, scores, cell_weights(y_f, u_f, tables["aligned"])) - weighted_auroc(
                    y_f, scores, cell_weights(y_f, u_f, tables["reversed"])
                )
            g_random[j] += (shift[EXPOSED] - shift[CONTROL]) / len(fold_levels)

    aurocs = pd.DataFrame(
        [{"head": k[0], "representation": k[1], "regime": k[2], "auroc": float(v[0])} for k, v in auc.items()]
    )
    return _assemble(
        mode="observational",
        g_full=g_full,
        g_targeted=g_targeted,
        g_random=g_random,
        utility=utility,
        utility_before=float(auc[(CONTROL, FULL, "independent")][0]),
        utility_after=float(auc[(CONTROL, TARGETED, "independent")][0]),
        utility_metric="control-head AUROC, independent regime",
        recover={"conditional_accessibility_auroc": float(np.mean(accessibility))},
        aurocs=aurocs,
        details={"folds": pd.DataFrame(fold_rows)},
        rank=1,
        epsilon=epsilon,
        delta=delta,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def _odds(table) -> float:
    return float(table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0]))


def _conditional_auroc(u, y, score) -> float:
    """Provenance AUROC within each target stratum, averaged equally over strata."""

    values = []
    for label in np.unique(y):
        mask = y == label
        if len(np.unique(u[mask])) == 2:
            values.append(weighted_auroc(u[mask], score[mask], np.ones(mask.sum())))
    return float(np.mean(values)) if values else float("nan")


# --------------------------------------------------------------------------------- paired


@dataclass(frozen=True)
class PairedSplit:
    """Rows of one split: one embedding per (record, view)."""

    embeddings: np.ndarray
    y: np.ndarray
    views: np.ndarray
    records: np.ndarray


def audit_paired(
    train: PairedSplit,
    test: PairedSplit,
    *,
    val: PairedSplit | None = None,
    p_shortcut: float = 0.6,
    C: float | None = None,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    rank: int | None = None,
    n_random: int = 10,
    epsilon: float = 0.01,
    delta: float = 0.02,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    view_order: Sequence | None = None,
    y_levels: Sequence | None = None,
    compute_lpi: bool = True,
    whitening: str = "eigh",
) -> AuditReport:
    """Audit reliance on a provenance factor observed as paired views of each record.

    Heads use a weighted standardization and logistic regression. ``C`` is
    selected on ``val`` by uniform-coupling AUROC unless given; heads are then
    refitted on train and validation together. The control head is fitted
    under the independent coupling and one exposed head per shortcut view
    under the aligned coupling with ``P(view = g | y = 1) = p_shortcut``;
    ``G`` averages over shortcut views. The rank-``K-1`` subspace and the
    random controls (seeded by ``seed``) are localized on ``train`` only.
    """

    splits = {"train": train, "test": test} | ({"val": val} if val is not None else {})
    view_levels = tuple(np.unique(np.asarray(train.views)).tolist()) if view_order is None else tuple(view_order)
    k = len(view_levels)
    rank = k - 1 if rank is None else int(rank)
    encoded = {name: _encode_paired(split, view_levels, y_levels) for name, split in splits.items()}
    if C is None:
        if val is None:
            raise ValueError("Pass val= to select C, or fix C.")
        C = _select_c(encoded["train"], encoded["val"], k, c_grid, seed)
    fit_rows = _concat(encoded["train"], encoded["val"]) if val is not None else encoded["train"]
    h_fit, y_fit, v_fit = fit_rows["h"], fit_rows["y"], fit_rows["v"]

    def head(table):
        return fit_linear_head(
            h_fit, y_fit, cell_weights(y_fit, v_fit, table), C=C, standardize=True, max_iter=5_000, random_state=seed
        )

    control = head(designed_coupling_table(k, 0, p_shortcut, "independent"))
    exposed = [head(designed_coupling_table(k, g, p_shortcut, "aligned")) for g in range(k)]

    tr = encoded["train"]
    subspace = paired_subspace(tr["h"], tr["r"], tr["v"], rank=rank, view_order=range(k), whitening=whitening)
    controls = random_subspaces(subspace.whitening, rank, n_random, seed=seed)

    te = encoded["test"]
    representations = {FULL: te["h"], TARGETED: subspace.remove(te["h"])}
    representations.update({("random", j): c.remove(te["h"]) for j, c in enumerate(controls)})
    scores = {(name, rep): model.scores(values) for rep, values in representations.items()
              for name, model in [(CONTROL, control)] + [((EXPOSED, g), e) for g, e in enumerate(exposed)]}

    record_ids, record_codes = np.unique(te["r"], return_inverse=True)
    record_labels = np.zeros(len(record_ids), dtype=np.int64)
    record_labels[record_codes] = te["y"]
    counts = cluster_bootstrap_counts(record_ids, strata=record_labels, n_bootstrap=n_bootstrap, seed=seed)[:, record_codes]

    tables = {(g, regime): designed_coupling_table(k, g, p_shortcut, regime) for g in range(k) for regime in CORE_REGIMES}
    weights = {key: restored_weights(te["y"], te["v"], counts, table) for key, table in tables.items()}

    def auc(name, rep, g, regime, point_only=False):
        if point_only:
            return weighted_auroc(te["y"], scores[(name, rep)], weights[(g, regime)][0])
        return weighted_auroc(te["y"], scores[(name, rep)], weights[(g, regime)])

    def g_value(rep, point_only=False):
        total = 0.0
        for g in range(k):
            exposed_shift = auc((EXPOSED, g), rep, g, "aligned", point_only) - auc((EXPOSED, g), rep, g, "reversed", point_only)
            control_shift = auc(CONTROL, rep, g, "aligned", point_only) - auc(CONTROL, rep, g, "reversed", point_only)
            total = total + (exposed_shift - control_shift) / k
        return total

    g_full, g_targeted = g_value(FULL), g_value(TARGETED)
    g_random = np.array([g_value(("random", j), point_only=True) for j in range(n_random)])
    utility = auc(CONTROL, TARGETED, 0, "independent") - auc(CONTROL, FULL, 0, "independent")

    layout = _layout(te, k)
    details = {
        "C": float(C),
        "flip_rate_exposed": float(np.mean([flip_rate(scores[((EXPOSED, g), FULL)][layout], record_labels) for g in range(k)])),
        "flip_rate_control": flip_rate(scores[(CONTROL, FULL)][layout], record_labels),
        "paired_logit_contrast": {rep: _paired_contrast(control, exposed, representations[rep], layout, record_labels)
                                  for rep in (FULL, TARGETED)},
        "view_order": view_levels,
    }
    rows = []
    for g in range(k):
        for regime in CORE_REGIMES:
            for name, label in [((EXPOSED, g), EXPOSED), (CONTROL, CONTROL)]:
                for rep in (FULL, TARGETED):
                    rows.append({"shortcut_view": view_levels[g], "head": label, "representation": rep, "regime": regime,
                                 "auroc": auc(name, rep, g, regime, point_only=True)})
    recover = {}
    if compute_lpi and val is not None:
        va = encoded["val"]
        result = _lpi(tr["h"], tr["v"], va["h"], va["v"], te["h"], te["v"], test_clusters=te["r"],
                      n_bootstrap=max(n_bootstrap, 1), confidence_level=confidence_level)
        recover = {"lpi_bits": result.bits, "lpi_ci": (result.ci_low, result.ci_high), "lpi_max_bits": result.baseline_bits}
        details["lpi"] = result
    return _assemble(
        mode="paired",
        g_full=g_full,
        g_targeted=g_targeted,
        g_random=g_random,
        utility=utility,
        utility_before=float(auc(CONTROL, FULL, 0, "independent", True)),
        utility_after=float(auc(CONTROL, TARGETED, 0, "independent", True)),
        utility_metric="control-head AUROC, independent coupling",
        recover=recover,
        aurocs=pd.DataFrame(rows),
        details=details,
        rank=rank,
        epsilon=epsilon,
        delta=delta,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def _encode_paired(split: PairedSplit, view_levels, y_levels) -> dict:
    as_matrix(split.embeddings, name="embeddings")
    h = np.asarray(split.embeddings)  # stored dtype sets the whitening rank threshold
    y, _ = encode(split.y, y_levels)
    v, _ = encode(split.views, view_levels)
    records = np.asarray(split.records)
    same_length(embeddings=h, y=y, views=v, records=records)
    frame = pd.DataFrame({"r": records, "y": y})
    if frame.groupby("r")["y"].nunique().gt(1).any():
        raise ValueError("Every view of a record must carry the same target.")
    return {"h": h, "y": y, "v": v, "r": records}


def _concat(a: dict, b: dict) -> dict:
    overlap = np.intersect1d(a["r"], b["r"])
    if overlap.size:
        raise ValueError("Train and validation records must be disjoint.")
    return {key: np.concatenate([a[key], b[key]]) for key in a}


def _select_c(train: dict, val: dict, k: int, c_grid, seed: int) -> float:
    table = designed_coupling_table(k, 0, (1.0 / k + 2.0 / k) / 2.0, "independent")
    train_w = cell_weights(train["y"], train["v"], table)
    val_w = cell_weights(val["y"], val["v"], table)
    aurocs = []
    for c in c_grid:
        model = fit_linear_head(train["h"], train["y"], train_w, C=c, standardize=True, max_iter=5_000, random_state=seed)
        aurocs.append(weighted_auroc(val["y"], model.scores(val["h"]), val_w))
    return float(c_grid[int(np.argmax(aurocs))])


def _layout(split: dict, k: int) -> np.ndarray:
    _, record_codes = np.unique(split["r"], return_inverse=True)
    position = np.empty((int(record_codes.max()) + 1, k), dtype=np.int64)
    position[record_codes, split["v"]] = np.arange(len(split["r"]))
    return position


def _paired_contrast(control, exposed, values, layout, record_labels) -> float:
    """Class-balanced same-record excess logit contrast (paper Eq. 9)."""

    k = layout.shape[1]
    control_logits = control.logits(values)[layout]
    counts = np.bincount(record_labels, minlength=2).astype(np.float64)
    weights = len(record_labels) / (2.0 * counts[record_labels])
    total = 0.0
    for g, model in enumerate(exposed):
        exposed_logits = model.logits(values)[layout]
        others = np.arange(k) != g
        contrast = (exposed_logits[:, g] - exposed_logits[:, others].mean(axis=1)) - (
            control_logits[:, g] - control_logits[:, others].mean(axis=1)
        )
        total += float(np.average(contrast, weights=weights)) / k
    return total


# ---------------------------------------------------------------------------------- shared


def _assemble(*, mode, g_full, g_targeted, g_random, utility, utility_before, utility_after, utility_metric,
              recover, aurocs, details, rank, epsilon, delta, confidence_level, n_bootstrap) -> AuditReport:
    g_full, g_targeted, utility = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (g_full, g_targeted, utility))
    has_draws = n_bootstrap > 0
    g_ci = percentile_interval(g_full[1:], confidence_level) if has_draws else (float("nan"),) * 2
    g_after_ci = percentile_interval(g_targeted[1:], confidence_level) if has_draws else (float("nan"),) * 2
    utility_ci = percentile_interval(utility[1:], confidence_level) if has_draws else (float("nan"),) * 2
    decision = decide(
        G=float(g_full[0]),
        G_ci_low=g_ci[0] if has_draws else None,
        G_after=float(g_targeted[0]),
        G_random=g_random,
        utility_change_ci_low=utility_ci[0] if has_draws else float(utility[0]),
        epsilon=epsilon,
        delta=delta,
    )
    return AuditReport(
        mode=mode,
        decision=decision,
        G=float(g_full[0]),
        G_ci=g_ci,
        G_after=float(g_targeted[0]),
        G_after_ci=g_after_ci,
        G_random=np.asarray(g_random, dtype=np.float64),
        utility_metric=utility_metric,
        utility_before=utility_before,
        utility_after=utility_after,
        utility_change=float(utility[0]),
        utility_change_ci=utility_ci,
        epsilon=epsilon,
        delta=delta,
        rank=rank,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        recover=recover,
        aurocs=aurocs,
        details=details,
    )


__all__ = ["PairedSplit", "audit_observational", "audit_paired"]
