import json

import numpy as np
import pytest

import physiotrace as pt
from physiotrace.synthetic import make_observational, make_paired


def observational(**kwargs):
    data = make_observational(n=2_000, dim=16, seed=11, **{k: v for k, v in kwargs.items()})
    return pt.audit_observational(
        data.embeddings, data.y, data.u, folds=data.folds, clusters=data.clusters, n_bootstrap=200, n_random=30
    )


def test_observational_shortcut_is_detected_and_remedy_certified():
    report = observational()
    assert report.verdict == pt.Verdict.REMEDY_CERTIFIED
    assert report.G > 0.05 and report.G_ci[0] > 0
    assert abs(report.G_after) < 0.01
    assert report.decision.random_controls_beaten == 30


def test_observational_without_association_returns_no_reliance():
    report = observational(odds_ratio=1.0)
    assert report.verdict == pt.Verdict.NO_RELIANCE
    assert report.recover["conditional_accessibility_auroc"] > 0.9  # decodable, yet not relied on


def test_observational_entangled_component_is_refused():
    report = observational(entangled=True)
    assert report.verdict == pt.Verdict.REMEDY_NOT_CERTIFIED
    assert report.decision.specific and not report.decision.safe
    assert report.utility_change < -report.delta


def test_single_train_test_split_via_test_folds():
    data = make_observational(n=1_500, dim=12, seed=3)
    folds = (data.folds == 0).astype(int)  # 1 = test
    report = pt.audit_observational(
        data.embeddings, data.y, data.u, folds=folds, clusters=data.clusters, test_folds=[1], n_bootstrap=100, n_random=10
    )
    assert len(report.details["folds"]) == 1
    assert report.verdict == pt.Verdict.REMEDY_CERTIFIED


def test_paired_audit_certifies_safe_view_removal():
    train, val, test = make_paired(n_records=600, dim=16, seed=4)
    report = pt.audit_paired(train, test, val=val, n_bootstrap=200)
    assert report.verdict == pt.Verdict.REMEDY_CERTIFIED
    assert report.details["flip_rate_exposed"] > report.details["flip_rate_control"]
    assert report.recover["lpi_bits"] > 0


def test_paired_audit_without_view_signal_returns_no_reliance():
    train, val, test = make_paired(n_records=600, dim=16, view_strength=0.0, seed=4)
    report = pt.audit_paired(train, test, val=val, n_bootstrap=200, compute_lpi=False)
    assert report.verdict == pt.Verdict.NO_RELIANCE


def test_report_serializes_and_summarizes():
    report = observational()
    json.dumps(report.to_dict())
    text = report.summary()
    assert "verdict: reliance, remedy certified" in text and "Verify" in text


def test_paired_audit_requires_val_or_fixed_c():
    train, _, test = make_paired(n_records=300, dim=8, seed=1)
    with pytest.raises(ValueError, match="val"):
        pt.audit_paired(train, test, n_bootstrap=0)
    report = pt.audit_paired(train, test, C=1.0, n_bootstrap=0, compute_lpi=False)
    assert np.isnan(report.G_ci[0])
