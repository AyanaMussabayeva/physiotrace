# Validation against the PhysioTRACE paper

These scripts check that `physiotrace` reproduces the published numbers from
the paper's saved artifacts. They need the paper repository's
`additional_experiments/results/` and, for NMT, the frozen d = 128 embedding
cache.

```bash
python validation/validate_ecg_stress.py PAPER/additional_experiments/results/ptbxl_real_device_association_shift
python validation/validate_nmt_paired.py NMT_EMBEDDINGS PAPER/additional_experiments/results
```

`validate_ecg_stress.py` exits non-zero if any of the 18 published quantities
differs by more than 1e-9.
