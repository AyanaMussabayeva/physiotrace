# physiotrace

**Does your task head rely on how the data were recorded?**

`physiotrace` audits a linear task head on a frozen encoder for reliance on
acquisition provenance — device, site, reference, montage, scanner, batch —
and certifies a remedy only when it is specific and safe. It works on
embeddings, so any frozen model (EEG, ECG, imaging, omics) can be audited
without retraining.

Two common measurements do not answer this question. Shifted-transfer
benchmarks can fail for unrelated reasons, and a provenance probe shows only
that provenance is decodable, not that a head uses it. PhysioTRACE tests the
head's behavior directly:

| Step | Question | Output |
|---|---|---|
| **Recover** | How decodable is provenance from the embedding? | calibrated Linear Provenance Information (bits per unit) or conditional accessibility AUROC |
| **Stress** | Does a head fitted under the provenance–target association degrade more than a control head when only that association is reversed on the same records? | excess vulnerability `G` with an identity-aware interval |
| **Intervene** | Does removing a train-localized provenance component remove the failure? | `G` after removal |
| **Verify** | Does the removal beat rank- and energy-matched random controls, and keep utility within a declared margin `δ`? | verdict |

Every audit ends in one of three verdicts: **no reliance**, **reliance – remedy
certified**, or **reliance – remedy not certified**.

## Install

```bash
pip install -e .            # from a checkout; numpy, scipy, scikit-learn, pandas
```

Python 3.10–3.13. Tested on every push from the oldest supported dependencies
(numpy 1.24, scipy 1.10, scikit-learn 1.3, pandas 2.0) to the latest.

## Quickstart

### Observational data (for example device or site)

```python
import physiotrace as pt

train = pt.Split(embeddings, y, u, clusters=patient_id)   # u: e.g. device A / device B
test = pt.Split(test_embeddings, test_y, test_u, clusters=test_patient_id)

report = pt.audit_observational(train, test, epsilon=0.01, delta=0.01)
print(report)
report.verdict        # pt.Verdict.REMEDY_CERTIFIED / REMEDY_NOT_CERTIFIED / NO_RELIANCE
report.to_dict()      # JSON-serializable summary
```

```text
PhysioTRACE audit (observational)
  verdict: reliance, remedy certified
  Recover   conditional_accessibility_auroc = 0.9946
  Stress    G = 0.1384 [0.1228, 0.1539]  (epsilon = 0.01; 95% interval)
  Intervene G after rank-1 removal = -0.0000 [-0.0013, 0.0012]  (100% reduction)
  Verify    specific: yes (beats 100/100 random controls)
            safe: yes (control-head AUROC, independent regime change -0.0000 [-0.0004, 0.0003], delta = 0.01)
```

(Output of `examples/quickstart_synthetic.py`, which also shows the other two
verdicts, a cross-fitted audit, and a paired audit.)

`y` and `u` can hold any two labels each (strings or numbers). `clusters`
identifies who each record belongs to (patient, subject); it is the bootstrap
unit, and a cluster on both sides of the split is rejected. `epsilon` is the
smallest `G` that counts as reliance and `delta` the tolerated utility loss;
declare both, and the localization `rank`, before looking at test results.

The exposed head is fitted on the natural association and the control head
with independence weights. Provenance is localized conditionally on the
target, so removal does not by default discard population signal that sources
differ in.

To cross-fit over folds instead, as in the paper's PTB-XL audit:

```python
report = pt.audit_observational_crossfit(embeddings, y, u, folds=folds, clusters=patient_id)
```

### Checks

Every report lists validity checks under `report.findings` and in
`print(report)`: too few records in a `(y, u)` cell, a small effective sample
or few effective clusters after reweighting, a very large weight, a stressed
association too weak to test, a head that did not converge, missing clusters,
and too few bootstrap draws. Checks never change the verdict; those of
severity `warning` are also raised as `pt.PhysioTraceWarning`. Thresholds live
in `pt.Gates`, and `pt.Gates.paper()` gives the stricter gates of the paper:

```python
report = pt.audit_observational(train, test, gates=pt.Gates.paper())
```

Common input mistakes fail early with a specific message: a target with more
than two classes, a provenance factor with more than two groups, a patient in
two folds or on both sides of the split, or a `(y, u)` combination with no
records.

### Paired views (for example EEG references of the same record)

```python
train = pt.PairedSplit(embeddings, y, views, records)   # one row per (record, view)
report = pt.audit_paired(train, test, val=val, p_shortcut=0.6, delta=0.02)
```

The association is induced by a designed coupling at fixed marginals; each
view serves in turn as the shortcut and `G` averages over them. Provenance is
localized from within-record view contrasts without task labels.
`report.details` adds decision flip rates across views and the same-record
logit contrast.

### Building blocks

Every step is also available on its own:

```python
pt.lpi(train_h, train_u, val_h, val_u, test_h, test_u, test_clusters=subject)  # Recover
tables = pt.observational_regimes(y_train, u_train)                             # Stress regimes
w = pt.cell_weights(y_test, u_test, tables["reversed"])
pt.weighted_auroc(y_test, scores, w)
subspace, probe = pt.conditional_direction(z_train, y_train, u_train)           # Intervene
subspace.remove(z_test)
pt.random_subspaces(subspace.whitening, rank=1, n=100, seed=0)                  # Verify controls
```

`physiotrace.synthetic` generates embeddings with a known shortcut that
reproduce each of the three verdicts on purpose.

## Interpreting the result

* A certified remedy supports the evaluated association and the linear
  intervention, not safety under every deployment shift.
* LPI and accessibility never yield a verdict on their own: a head can ignore
  what a probe decodes.
* Declare `epsilon`, `delta`, the rank, and the number of random controls
  before looking at test results.
* Keep every identity (patient, subject, record) inside one fold and pass it
  as `clusters`; resampling windows or trials instead overstates precision.

## Validation against the paper

`validation/` reproduces the published numbers of the PhysioTRACE paper from
its saved artifacts:

| Script | What is compared | Agreement |
|---|---|---|
| `validate_ecg_stress.py` | PTB-XL device audit: `G` before/after removal, its reduction, and the utility change for ECGFounder, CLEF-Small, metadata comparators, sensitivity cohorts, and the weak-association pair, with bootstrap intervals (18 quantities) | ≤ 1e-15 |
| `validate_ecg_end_to_end.py` | The same audit end to end from ECGFounder embeddings recomputed with the paper's pipeline: `G` before/after removal with intervals, utility change, conditional accessibility, all 100 random-control `G`, four cohorts | ≤ 2e-5 (embeddings recomputed on different hardware) |
| `validate_nmt_paired.py` | Paired NMT audit end to end from frozen embeddings (8 encoders): `G` before/after removal, all random-control `G`, paired logit contrast, selected `C`, verdict | ≤ 4e-8 |
| | Calibrated LPI from the paper's exported probe states, including bootstrap intervals | ≤ 4e-8 |
| | Calibrated LPI with probes refitted by `physiotrace` | ≤ 8e-4 bits (optimizer tolerance) |

The paired NMT run whitened with an SVD; pass `whitening="svd"` to draw the
identical random controls. Both methods give the same projections.

## Status

Version 0.1 (development). Binary targets with binary provenance
(observational, target-conditional localization of any rank; rank 1 as in the
paper) or `K` paired views. Multiclass targets, regression
endpoints, and multi-encoder pooling are planned.

## License

MIT.
