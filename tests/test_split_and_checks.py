import warnings

import numpy as np
import pytest

import physiotrace as pt
from physiotrace.synthetic import make_observational, make_paired

QUICK = dict(n_bootstrap=100, n_random=10)


@pytest.fixture(scope="module")
def data():
    return make_observational(n=3_000, dim=12, seed=21)


def test_train_test_api_matches_crossfit_with_one_test_fold(data):
    train, test = data.train_test(test_fold=0)
    split = pt.audit_observational(train, test, **QUICK)
    folds = (data.folds == 0).astype(int)
    crossfit = pt.audit_observational_crossfit(
        data.embeddings, data.y, data.u, folds=folds, clusters=data.clusters, test_folds=[1], **QUICK
    )
    assert split.G == pytest.approx(crossfit.G, abs=1e-12)
    assert split.G_after == pytest.approx(crossfit.G_after, abs=1e-12)
    assert split.verdict == crossfit.verdict == pt.Verdict.REMEDY_CERTIFIED


def test_train_test_api_accepts_string_labels(data):
    train, test = data.train_test()
    relabel = lambda s: pt.Split(s.embeddings, np.where(s.y == 1, "abnormal", "normal"), np.where(s.u == 1, "B", "A"), s.clusters)  # noqa: E731
    report = pt.audit_observational(relabel(train), relabel(test), **QUICK)
    assert report.details["levels"]["u"] == ("A", "B")
    assert report.verdict == pt.Verdict.REMEDY_CERTIFIED


def test_shared_cluster_between_train_and_test_is_rejected(data):
    train, test = data.train_test()
    leaky = pt.Split(test.embeddings, test.y, test.u, np.r_[train.clusters[:1], test.clusters[1:]])
    with pytest.raises(ValueError, match="both train and test"):
        pt.audit_observational(train, leaky, **QUICK)


def test_fold_arguments_are_redirected_to_crossfit(data):
    train, test = data.train_test()
    with pytest.raises(TypeError, match="audit_observational_crossfit"):
        pt.audit_observational(train, test, folds=data.folds)


def test_cluster_in_two_folds_is_rejected(data):
    folds = data.folds.copy()
    folds[0] = (folds[1] + 1) % 5  # records 0 and 1 share a cluster
    with pytest.raises(ValueError, match="more than one fold"):
        pt.audit_observational_crossfit(data.embeddings, data.y, data.u, folds=folds, clusters=data.clusters, **QUICK)


def test_multiclass_target_and_multigroup_provenance_have_clear_errors(data):
    y3 = data.y + (data.embeddings[:, 0] > 2)
    with pytest.raises(ValueError, match="target must be binary; found 3"):
        pt.audit_observational_crossfit(data.embeddings, y3, data.u, folds=data.folds)
    u3 = data.u + (data.embeddings[:, 1] > 2)
    with pytest.raises(ValueError, match="two groups; found 3.*audit_paired"):
        pt.audit_observational_crossfit(data.embeddings, data.y, u3, folds=data.folds)


def test_missing_cell_names_the_split_and_cell(data):
    train, test = data.train_test()
    keep = ~((test.y == 1) & (test.u == 0))
    thin = pt.Split(test.embeddings[keep], test.y[keep], test.u[keep], test.clusters[keep])
    with pytest.raises(ValueError, match=r"\[test\] no test record has y=1 and u=0"):
        pt.audit_observational(train, thin, **QUICK)


def test_findings_flag_weak_association_and_missing_clusters():
    weak = make_observational(n=3_000, dim=12, odds_ratio=1.05, seed=22)
    train, test = weak.train_test()
    report = pt.audit_observational(
        pt.Split(train.embeddings, train.y, train.u), pt.Split(test.embeddings, test.y, test.u), **QUICK
    )
    codes = {f.code for f in report.findings}
    assert {"weak_association", "no_clusters", "few_bootstrap_draws"} <= codes
    assert "Checks" in report.summary()
    assert any(f["code"] == "weak_association" for f in report.to_dict()["findings"])


def test_small_cells_and_no_bootstrap_emit_warnings():
    tiny = make_observational(n=400, dim=6, seed=23)
    train, test = tiny.train_test()
    with pytest.warns(pt.PhysioTraceWarning) as caught:
        report = pt.audit_observational(train, test, n_bootstrap=0, n_random=5)
    messages = " ".join(str(w.message) for w in caught)
    assert "point estimates" in messages and "records with" in messages
    assert {"no_bootstrap", "small_cell"} <= {f.code for f in report.findings}


def test_gates_control_thresholds(data):
    train, test = data.train_test()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pt.PhysioTraceWarning)
        strict = pt.audit_observational(train, test, gates=pt.Gates(min_reversed_ess=1e9), **QUICK)
        lenient = pt.audit_observational(train, test, gates=pt.Gates(min_reversed_ess=0, min_reversed_cluster_ess=0), **QUICK)
    assert "low_ess" in {f.code for f in strict.findings}
    assert "low_ess" not in {f.code for f in lenient.findings}


def test_paired_reports_c_on_grid_edge():
    train, val, test = make_paired(n_records=600, dim=16, seed=4)
    report = pt.audit_paired(train, test, val=val, c_grid=(0.1,), n_bootstrap=100, compute_lpi=False)
    assert report.details["C"] == 0.1
    report = pt.audit_paired(train, test, val=val, c_grid=(0.1, 1.0), n_bootstrap=100, compute_lpi=False)
    assert ("grid_edge" in {f.code for f in report.findings}) == (report.details["C"] in (0.1, 1.0))
