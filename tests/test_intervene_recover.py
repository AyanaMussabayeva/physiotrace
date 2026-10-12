import numpy as np
import pytest

import physiotrace as pt
from physiotrace.synthetic import make_paired


def test_whitening_produces_identity_covariance_on_its_support():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
    w = pt.Whitening.fit(x)
    z = w.transform(x)
    assert w.rank == 6
    assert np.allclose(z.T @ z / len(z), np.eye(6), atol=1e-10)
    assert np.allclose(w.inverse_transform(z), x, atol=1e-8)


def test_whitening_drops_null_directions():
    rng = np.random.default_rng(1)
    x = np.c_[rng.normal(size=(300, 4)), np.zeros((300, 2))]
    assert pt.Whitening.fit(x).rank == 4


@pytest.mark.parametrize("method", ["eigh", "svd"])
def test_whitening_methods_give_the_same_projection(method):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 5))
    basis = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    reference = pt.Subspace(pt.Whitening.fit(x, method="eigh"), basis)
    other_whitening = pt.Whitening.fit(x, method=method)
    # Express the same subspace in the other whitened coordinates.
    rotation = reference.whitening.matrix.T @ np.cov(x.T, bias=True) @ other_whitening.matrix
    other = pt.Subspace(other_whitening, np.linalg.qr(rotation.T @ basis)[0])
    assert np.allclose(reference.remove(x), other.remove(x), atol=1e-8)


def test_random_subspaces_are_energy_matched():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(800, 10)) @ rng.normal(size=(10, 10))
    w = pt.Whitening.fit(x)
    for subspace in pt.random_subspaces(w, 2, 5, seed=0):
        assert subspace.energy_fraction(x) == pytest.approx(2 / 10, rel=1e-8)


def test_removal_is_idempotent_and_leaves_complement():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 8))
    subspace = pt.random_subspaces(pt.Whitening.fit(x), 3, 1, seed=1)[0]
    once = subspace.remove(x)
    assert np.allclose(subspace.remove(once), once, atol=1e-10)
    assert subspace.energy_fraction(once) == pytest.approx(0.0, abs=1e-20)


def test_paired_subspace_recovers_planted_view_signatures():
    train, _, _ = make_paired(n_records=600, dim=16, view_strength=2.0, seed=5)
    subspace = pt.paired_subspace(train.embeddings, train.records, train.views)
    assert subspace.rank == 2
    cleaned = subspace.remove(train.embeddings)
    view_means = [cleaned[train.views == v].mean(axis=0) for v in np.unique(train.views)]
    raw_means = [train.embeddings[train.views == v].mean(axis=0) for v in np.unique(train.views)]
    assert np.ptp(np.array(view_means), axis=0).max() < 0.05 * np.ptp(np.array(raw_means), axis=0).max()


def test_paired_subspace_requires_complete_records():
    x = np.random.default_rng(6).normal(size=(5, 3))
    with pytest.raises(ValueError, match="exactly once"):
        pt.paired_subspace(x, [0, 0, 1, 1, 2], ["a", "b", "a", "b", "a"])


def test_lpi_is_positive_for_decodable_groups_and_near_zero_for_noise():
    rng = np.random.default_rng(7)

    def split(n, informative):
        groups = rng.integers(0, 3, n)
        x = rng.normal(size=(n, 8))
        if informative:
            x[:, 0] += 2.0 * groups
        return x, groups

    for informative in (True, False):
        (xtr, gtr), (xva, gva), (xte, gte) = split(1500, informative), split(500, informative), split(500, informative)
        result = pt.lpi(xtr, gtr, xva, gva, xte, gte, n_bootstrap=500)
        if informative:
            assert result.bits > 0.5 and result.ci_low > 0
        else:
            assert abs(result.bits) < 0.05 and result.ci_low < 0.02
        assert result.baseline_bits == pytest.approx(np.log2(3), abs=0.02)


def test_conditional_direction_higher_rank_is_orthonormal_and_extends_rank_one():
    from physiotrace.synthetic import make_observational

    data = make_observational(n=1_500, dim=12, seed=8)
    one, _ = pt.conditional_direction(data.embeddings, data.y, data.u, rank=1)
    three, _ = pt.conditional_direction(data.embeddings, data.y, data.u, rank=3)
    assert three.rank == 3
    assert np.allclose(three.basis.T @ three.basis, np.eye(3), atol=1e-8)
    assert np.allclose(np.abs(three.basis[:, 0]), np.abs(one.basis[:, 0]))
