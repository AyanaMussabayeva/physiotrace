"""Run the three PhysioTRACE verdicts on synthetic embeddings with a known shortcut."""

import warnings

import physiotrace as pt
from physiotrace.synthetic import make_observational, make_paired

warnings.simplefilter("ignore", pt.PhysioTraceWarning)  # findings are printed in each summary

scenarios = {
    "provenance used as a shortcut": {},
    "provenance decodable but not associated with the target": {"odds_ratio": 1.0},
    "provenance entangled with task signal": {"entangled": True, "odds_ratio": 20.0, "provenance_strength": 3.0},
}
for title, options in scenarios.items():
    data = make_observational(n=8_000, seed=1, **options)
    train, test = data.train_test(test_fold=0)  # pt.Split(embeddings, y, u, clusters) each
    print(f"# {title}\n{pt.audit_observational(train, test)}\n")

# Cross-fitting over all folds, as in the paper's PTB-XL audit.
data = make_observational(seed=1)
report = pt.audit_observational_crossfit(data.embeddings, data.y, data.u, folds=data.folds, clusters=data.clusters)
print(f"# cross-fitted\n{report}\n")

train, val, test = make_paired(seed=2)
print(f"# paired views\n{pt.audit_paired(train, test, val=val)}")
