import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import physiotrace as pt
from physiotrace.bootstrap import cluster_bootstrap_counts, percentile_interval


def test_weighted_auroc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.round(rng.normal(size=500) + y, 1)  # many ties
    weights = rng.random(500)
    assert pt.weighted_auroc(y, scores, weights) == pytest.approx(roc_auc_score(y, scores, sample_weight=weights), abs=1e-12)


def test_weighted_auroc_matrix_equals_row_by_row():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    scores = rng.normal(size=100)
    weights = rng.random((4, 100))
    draws = pt.weighted_auroc(y, scores, weights)
    assert np.allclose(draws, [pt.weighted_auroc(y, scores, row) for row in weights])


def test_cluster_bootstrap_moves_clusters_together_within_strata():
    clusters = np.repeat(np.arange(20), 3)
    strata = (np.arange(20) % 2)[clusters]
    counts = cluster_bootstrap_counts(clusters, strata=strata, n_bootstrap=50, seed=0)
    assert np.all(counts[0] == 1)
    for row in counts[1:]:
        per_cluster = row.reshape(20, 3)
        assert np.all(per_cluster == per_cluster[:, :1])
        for stratum in (0, 1):
            assert row[strata == stratum].sum() == (strata == stratum).sum()


def test_cluster_bootstrap_keeps_required_cells():
    clusters = np.arange(40)
    cells = np.r_[np.zeros(38, dtype=int), 1, 1]  # a rare cell
    counts = cluster_bootstrap_counts(clusters, support=cells, n_bootstrap=100, seed=1)
    assert np.all(counts[:, cells == 1].sum(axis=1) > 0)


def test_cluster_bootstrap_rejects_cluster_crossing_strata():
    with pytest.raises(ValueError, match="stratum"):
        cluster_bootstrap_counts([0, 0, 1, 1], strata=[0, 1, 0, 0], n_bootstrap=1, seed=0)


def test_percentile_interval():
    low, high = percentile_interval(np.arange(1001), 0.9)
    assert (low, high) == pytest.approx((50.0, 950.0))
