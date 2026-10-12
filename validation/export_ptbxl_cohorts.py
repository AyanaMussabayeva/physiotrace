"""Export the PTB-XL device-audit cohorts of the paper as one CSV.

Run in the paper repository's environment (it imports
``physiotrace_additional``). The output is read by
``validate_ecg_end_to_end.py``, which only needs ``physiotrace``.

Usage:
    python validation/export_ptbxl_cohorts.py PTBXL_DIR OUTPUT_CSV
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from physiotrace_additional.ptbxl_real_device_ood.data import (
    common_calendar_cohort,
    human_validated_cohort,
    load_real_device_data,
    primary_cohort,
    weak_association_cohort,
)

COLUMNS = ["ecg_id", "embedding_index", "patient_id", "outer_fold", "target", "device_target"]


def main(ptbxl_dir: Path, output: Path) -> None:
    data = load_real_device_data(ptbxl_dir)
    frames = []
    for build in (primary_cohort, common_calendar_cohort, human_validated_cohort, weak_association_cohort):
        cohort = build(data)
        frame = cohort.metadata[COLUMNS].copy()
        frame["design"] = frame.ecg_id.isin(cohort.design_metadata.ecg_id)
        frame["analysis"] = cohort.name
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
