"""Run the three PhysioTRACE verdicts on synthetic embeddings with a known shortcut."""

import physiotrace as pt
from physiotrace.synthetic import make_observational, make_paired

scenarios = {
    "provenance used as a shortcut": {},
    "provenance decodable but not associated with the target": {"odds_ratio": 1.0},
    "provenance entangled with task signal": {"entangled": True},
}
for title, options in scenarios.items():
    data = make_observational(seed=1, **options)
    report = pt.audit_observational(data.embeddings, data.y, data.u, folds=data.folds, clusters=data.clusters)
    print(f"# {title}\n{report}\n")

train, val, test = make_paired(seed=2)
print(f"# paired views\n{pt.audit_paired(train, test, val=val)}")
