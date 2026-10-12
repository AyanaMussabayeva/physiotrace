"""Synthetic embeddings with a known shortcut, for tutorials and tests.

Embeddings are ``h = A z + s(u) + noise`` with latent physiology ``z`` that
determines the target ``y``, and a provenance signature ``s(u)``. Whether a
naturally fitted head relies on ``u`` and whether removal is safe are set by
construction, so each of the three verdicts can be produced on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audit import PairedSplit, Split


@dataclass(frozen=True)
class ObservationalData:
    embeddings: np.ndarray
    y: np.ndarray
    u: np.ndarray
    clusters: np.ndarray
    folds: np.ndarray

    def train_test(self, test_fold: int = 0) -> tuple[Split, Split]:
        """Split by fold: ``test_fold`` becomes the test set."""

        test = self.folds == test_fold
        parts = []
        for mask in (~test, test):
            parts.append(Split(self.embeddings[mask], self.y[mask], self.u[mask], self.clusters[mask]))
        return parts[0], parts[1]


def make_observational(
    n: int = 4_000,
    dim: int = 32,
    *,
    odds_ratio: float = 10.0,
    provenance_strength: float = 2.0,
    task_strength: float = 1.0,
    entangled: bool = False,
    n_folds: int = 5,
    records_per_cluster: int = 2,
    seed: int = 0,
) -> ObservationalData:
    """Binary target and binary provenance with a chosen natural association.

    * ``odds_ratio`` far from 1 with ``provenance_strength > 0``: a naturally
      fitted head can use the provenance signature as a shortcut.
    * ``odds_ratio = 1``: provenance is decodable but carries no association,
      so there is nothing to rely on.
    * ``entangled=True``: the provenance signature lies along the task
      direction, so removing it also removes task signal.
    """

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Symmetric P(u=1 | y): odds sqrt(OR) for y=1 and 1/sqrt(OR) for y=0, so the u--y odds ratio is OR.
    root = np.sqrt(odds_ratio)
    p_u = np.where(y == 1, root / (1.0 + root), 1.0 / (1.0 + root))
    u = (rng.random(n) < p_u).astype(np.int64)

    task_direction = _unit(rng.normal(size=dim))
    provenance_direction = task_direction.copy() if entangled else _unit(_orthogonal_to(rng.normal(size=dim), task_direction))
    latent = rng.normal(size=(n, dim))
    embeddings = (
        latent
        + task_strength * (2 * y - 1)[:, None] * task_direction[None, :]
        + provenance_strength * (2 * u - 1)[:, None] * provenance_direction[None, :]
        + 0.5 * rng.normal(size=(n, dim))
    )
    clusters = np.arange(n) // records_per_cluster
    folds = rng.permutation(np.max(clusters) + 1)[clusters] % n_folds
    return ObservationalData(embeddings=embeddings, y=y, u=u, clusters=clusters, folds=folds)


def make_paired(
    n_records: int = 900,
    dim: int = 32,
    *,
    n_views: int = 3,
    view_strength: float = 1.5,
    task_strength: float = 1.0,
    seed: int = 0,
) -> tuple[PairedSplit, PairedSplit, PairedSplit]:
    """Records observed under ``n_views`` provenance views; returns (train, val, test).

    Each view adds a fixed signature; the task is carried by a separate
    direction. A head fitted under a designed coupling of view and target can
    rely on the view signature, and removing the view subspace is safe.
    """

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_records)
    task_direction = _unit(rng.normal(size=dim))
    signatures = np.stack([_unit(_orthogonal_to(rng.normal(size=dim), task_direction)) for _ in range(n_views)])
    base = rng.normal(size=(n_records, dim)) + task_strength * (2 * y - 1)[:, None] * task_direction[None, :]
    rows = []
    for view in range(n_views):
        rows.append(base + view_strength * signatures[view][None, :] + 0.3 * rng.normal(size=(n_records, dim)))
    embeddings = np.stack(rows, axis=1).reshape(n_records * n_views, dim)
    records = np.repeat(np.arange(n_records), n_views)
    views = np.tile(np.array([f"view{v}" for v in range(n_views)]), n_records)
    labels = np.repeat(y, n_views)
    split = rng.permutation(n_records) % 10
    parts = {"train": split < 6, "val": split == 6, "test": split >= 7}
    out = []
    for mask in parts.values():
        rows_mask = mask[records]
        out.append(PairedSplit(embeddings[rows_mask], labels[rows_mask], views[rows_mask], records[rows_mask]))
    return tuple(out)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _orthogonal_to(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return vector - (vector @ reference) * reference


__all__ = ["ObservationalData", "make_observational", "make_paired"]
