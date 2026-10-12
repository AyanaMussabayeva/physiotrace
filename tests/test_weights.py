import numpy as np
import pytest

import physiotrace as pt
from physiotrace.weights import (
    effective_sample_size,
    empirical_table,
    odds_ratio,
    restored_weights,
)


@pytest.mark.parametrize("p_y,p_u,target_or", [(0.3, 0.6, 5.0), (0.8, 0.25, 0.2), (0.5, 0.5, 1.0), (0.05, 0.9, 40.0)])
def test_fixed_marginal_table_keeps_marginals_and_odds_ratio(p_y, p_u, target_or):
    table = pt.fixed_marginal_table(p_y, p_u, target_or)
    assert np.allclose(table.sum(axis=1), [1 - p_y, p_y], atol=1e-13)
    assert np.allclose(table.sum(axis=0), [1 - p_u, p_u], atol=1e-13)
    assert odds_ratio(table) == pytest.approx(target_or, rel=1e-10)
    assert np.all(table > 0)


def test_observational_regimes_span_the_observed_association():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5_000)
    u = (rng.random(5_000) < np.where(y == 1, 0.8, 0.3)).astype(int)
    regimes = pt.observational_regimes(y, u)
    observed = odds_ratio(empirical_table(y, u))
    assert odds_ratio(regimes["aligned"]) == pytest.approx(observed)
    assert odds_ratio(regimes["independent"]) == pytest.approx(1.0)
    assert odds_ratio(regimes["reversed"]) == pytest.approx(1 / observed)
    for table in regimes.values():
        assert np.allclose(table.sum(axis=1), regimes["aligned"].sum(axis=1))


def test_cell_weights_reproduce_target_table_and_are_mean_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1_000)
    u = rng.integers(0, 3, 1_000)
    target = np.array([[0.1, 0.2, 0.2], [0.25, 0.15, 0.1]])
    w = pt.cell_weights(y, u, target)
    achieved = np.zeros_like(target)
    np.add.at(achieved, (y, u), w)
    assert w.mean() == pytest.approx(1.0)
    assert np.allclose(achieved / achieved.sum(), target, atol=1e-12)


def test_cell_weights_reject_missing_support():
    with pytest.raises(ValueError, match="support"):
        pt.cell_weights([0, 0, 1], [0, 1, 0], np.full((2, 2), 0.25))


def test_independence_weights_remove_association_and_keep_marginals():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 3_000)
    u = (rng.random(3_000) < np.where(y == 1, 0.9, 0.2)).astype(int)
    w = pt.independence_weights(y, u)
    table = np.zeros((2, 2))
    np.add.at(table, (y, u), w)
    table /= table.sum()
    assert odds_ratio(table) == pytest.approx(1.0)
    assert np.allclose(table.sum(axis=1), empirical_table(y, u).sum(axis=1))


@pytest.mark.parametrize("regime", ["aligned", "independent", "reversed"])
def test_designed_coupling_has_uniform_marginals(regime):
    table = pt.designed_coupling_table(3, 1, 0.6, regime)
    assert np.allclose(table.sum(axis=1), 0.5)
    assert np.allclose(table.sum(axis=0), 1 / 3)
    if regime == "aligned":
        assert table[1, 1] / 0.5 == pytest.approx(0.6)


def test_designed_coupling_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        pt.designed_coupling_table(3, 0, 0.7, "aligned")


def test_restored_weights_hit_target_masses_in_every_draw():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    u = rng.integers(0, 2, 200)
    counts = rng.integers(1, 4, size=(5, 200))
    table = pt.fixed_marginal_table(0.4, 0.3, 3.0)
    w = restored_weights(y, u, counts, table)
    for row in w:
        achieved = np.zeros((2, 2))
        np.add.at(achieved, (y, u), row)
        assert np.allclose(achieved, table, atol=1e-12)


def test_effective_sample_size():
    assert effective_sample_size(np.ones(10)) == pytest.approx(10)
    assert effective_sample_size(np.r_[1.0, np.zeros(9)]) == pytest.approx(1)
