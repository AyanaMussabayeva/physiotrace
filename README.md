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

## Quickstart

### Observational data (for example device or site)

```python
import physiotrace as pt

report = pt.audit_observational(
    embeddings,          # (n, d) frozen embeddings
    y,                   # binary target
    u,                   # binary provenance factor, e.g. device A / device B
    folds=folds,         # cross-fitting folds; keep each patient in one fold
    clusters=patient_id, # bootstrap unit
    delta=0.01,          # tolerated utility loss
    epsilon=0.01,        # smallest G that counts as reliance
)
print(report)
```

```text
PhysioTRACE audit (observational)
  verdict: reliance, remedy certified
  Recover   conditional_accessibility_auroc = 0.9947
  Stress    G = 0.1363 [0.1267, 0.1465]  (epsilon = 0.01; 95% interval)
  Intervene G after rank-1 removal = 0.0002 [-0.0007, 0.0012]  (100% reduction)
  Verify    specific: yes (beats 100/100 random controls)
            safe: yes (control-head AUROC, independent regime change +0.0002 [-0.0000, 0.0005], delta = 0.01)
```

(Output of `examples/quickstart_synthetic.py`, which also shows the other two
verdicts and a paired audit.)

The exposed head is fitted on the natural association and the control head
with independence weights. Provenance is localized conditionally on the
target, so removal does not by default discard population signal that sources
differ in. For a single train/test split, pass two fold values and
`test_folds=[test_value]`.

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
