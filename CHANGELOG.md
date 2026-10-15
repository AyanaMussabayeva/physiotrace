# Changelog

## 0.1.0

First release: `pip install physiotrace`. Same code as 0.1.0a1.

## 0.1.0a1

First public pre-release.

- `audit_observational(train, test)` and `audit_observational_crossfit(...)`
  for a binary provenance factor naturally associated with a binary target;
  `audit_paired(train, test, val=...)` for records observed under several
  provenance views.
- Recover (calibrated Linear Provenance Information, conditional
  accessibility), Stress (fixed-marginal regimes, excess vulnerability `G`
  with identity-aware bootstrap intervals), Intervene (paired or
  target-conditional localization of any rank), Verify (rank- and
  energy-matched random controls, declared utility margin) and the verdict.
- Validity checks reported as `report.findings` and `PhysioTraceWarning`,
  with thresholds in `pt.Gates`.
- Synthetic data with known shortcuts, a tutorial notebook, and scripts that
  reproduce the published numbers of the PhysioTRACE paper.
