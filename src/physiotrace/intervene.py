"""Train-localized provenance suppression (Intervene).

A component is removed in support-aware whitened coordinates,

    T_U(h) = h - W^+ U U^T W (h - mu),

which leaves everything outside the localized subspace, including directions
outside the numerical covariance support, unchanged (paper Eq. 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ._validation import as_codes, as_matrix, same_length
from .weights import independence_weights


@dataclass(frozen=True)
class Whitening:
    """Whitening fitted on the numerical support of a localization matrix."""

    mean: np.ndarray
    matrix: np.ndarray
    inverse_matrix: np.ndarray
    rank: int

    @classmethod
    def fit(cls, values, *, method: str = "eigh") -> "Whitening":
        """Fit on a localization matrix.

        ``method="eigh"`` diagonalizes the ``d x d`` Gram matrix and scales to
        many observations; ``method="svd"`` uses a thin SVD of the centered
        data. Both give the same projections; they can differ in the signs of
        the whitened axes, which only changes which Haar random controls a
        given seed draws (the paper's paired NMT audit used ``"svd"``).
        """

        source = np.asarray(values)
        data = as_matrix(values, name="localization matrix")
        n, d = data.shape
        if n <= d:
            raise ValueError(f"Whitening needs more observations than dimensions, got shape {data.shape}.")
        mean = data.mean(axis=0)
        centered = data - mean
        if method == "eigh":
            eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
            order = np.argsort(eigenvalues)[::-1]
            singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
            vectors = eigenvectors[:, order]
        elif method == "svd":
            _, singular, right = np.linalg.svd(centered, full_matrices=False)
            vectors = right.T
        else:
            raise ValueError("method must be 'eigh' or 'svd'.")
        dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float64)
        threshold = np.finfo(dtype).eps * max(centered.shape) * singular[0]
        rank = int(np.sum(singular > threshold))
        if rank < 1:
            raise ValueError("The localization matrix has zero numerical rank.")
        kept, basis = singular[:rank], vectors[:, :rank]
        root_n = np.sqrt(float(n))
        return cls(
            mean=mean,
            matrix=basis * (root_n / kept)[None, :],
            inverse_matrix=(kept / root_n)[:, None] * basis.T,
            rank=rank,
        )

    def transform(self, values) -> np.ndarray:
        return (as_matrix(values, name="embeddings") - self.mean) @ self.matrix

    def inverse_transform(self, values) -> np.ndarray:
        return as_matrix(values, name="whitened embeddings") @ self.inverse_matrix + self.mean


@dataclass(frozen=True)
class Subspace:
    """An orthonormal basis in the whitened coordinates of ``whitening``."""

    whitening: Whitening
    basis: np.ndarray

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != self.whitening.rank:
            raise ValueError("basis must have shape (whitening.rank, p).")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8):
            raise ValueError("basis must be orthonormal.")
        object.__setattr__(self, "basis", basis)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def remove(self, values) -> np.ndarray:
        """Apply ``T_U`` and return embeddings in their original coordinates."""

        data = as_matrix(values, name="embeddings")
        forward = self.whitening.matrix @ self.basis
        inverse = self.basis.T @ self.whitening.inverse_matrix
        return data - ((data - self.whitening.mean) @ forward) @ inverse

    def energy_fraction(self, values) -> float:
        """Fraction of whitened energy of ``values`` that lies in the subspace."""

        whitened = self.whitening.transform(values)
        return float(np.square(whitened @ self.basis).sum() / np.square(whitened).sum())


def _orthonormal(directions: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(np.asarray(directions, dtype=np.float64).reshape(directions.shape[0], -1), mode="reduced")
    return basis


def paired_subspace(
    embeddings, records, views, *, rank: int | None = None, view_order=None, whitening: str = "eigh"
) -> Subspace:
    """Localize provenance from paired views without task labels.

    Every record must be observed under every view. Each view's whitened
    deviation from its record mean is averaged over records, and the top
    ``rank`` right singular vectors of the resulting ``K x r`` matrix form the
    basis (paper Eq. 6). ``rank`` defaults to ``K - 1``.
    """

    as_matrix(embeddings, name="embeddings")
    # Keep the stored dtype: the numerical-rank threshold of the whitening
    # depends on the precision the embeddings were produced in.
    h = np.asarray(embeddings)
    records = np.asarray(records)
    views = np.asarray(views)
    same_length(embeddings=h, records=records, views=views)
    layout = _paired_layout(h, records, views, view_order)
    n_records, n_views, dim = layout.shape
    rank = n_views - 1 if rank is None else int(rank)
    if not 1 <= rank <= n_views - 1:
        raise ValueError(f"rank must lie in [1, {n_views - 1}] for {n_views} views.")
    whitening = Whitening.fit(layout.reshape(-1, dim), method=whitening)
    if rank >= whitening.rank:
        raise ValueError("rank must be smaller than the effective whitening rank.")
    whitened = whitening.transform(layout.reshape(-1, dim).astype(np.float64)).reshape(n_records, n_views, whitening.rank)
    contrasts = (whitened - whitened.mean(axis=1, keepdims=True)).mean(axis=0)
    _, singular, right = np.linalg.svd(contrasts, full_matrices=False)
    if singular[rank - 1] <= 0.0:
        raise ValueError(f"Paired view contrasts do not have rank {rank}.")
    return Subspace(whitening, right[:rank].T)


def _paired_layout(h: np.ndarray, records: np.ndarray, views: np.ndarray, view_order) -> np.ndarray:
    view_levels = tuple(np.unique(views).tolist()) if view_order is None else tuple(view_order)
    record_levels, record_codes = np.unique(records, return_inverse=True)
    view_lookup = {value: code for code, value in enumerate(view_levels)}
    try:
        view_codes = np.array([view_lookup[value] for value in views.tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"View {error.args[0]!r} is not in view_order.") from None
    counts = np.zeros((len(record_levels), len(view_levels)), dtype=np.int64)
    np.add.at(counts, (record_codes, view_codes), 1)
    if np.any(counts != 1):
        raise ValueError("Every record must appear exactly once under every view.")
    position = np.empty_like(counts)
    position[record_codes, view_codes] = np.arange(len(h))
    return h[position]


def conditional_direction(
    embeddings,
    y,
    u,
    *,
    C: float = 1.0,
    max_iter: int = 2_000,
    tol: float = 1e-7,
) -> tuple[Subspace, LogisticRegression]:
    """Localize a binary provenance factor conditionally on the target.

    Training class means are subtracted, the residuals are whitened, and an
    independence-weighted logistic probe for ``u`` is fitted; the unit normal
    of its separator is the rank-one basis (paper Eq. 7). Returns the subspace
    and the fitted probe (which acts on whitened residuals).
    """

    h = as_matrix(embeddings, name="embeddings")
    y = as_codes(y, name="y")
    u = as_codes(u, name="u", n_levels=2)
    same_length(embeddings=h, y=y, u=u)
    residual = h - class_means(h, y)[y]
    whitening = Whitening.fit(residual)
    probe = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol)
    probe.fit(whitening.transform(residual), u, sample_weight=independence_weights(y, u))
    direction = probe.coef_.reshape(-1)
    return Subspace(whitening, (direction / np.linalg.norm(direction))[:, None]), probe


def class_means(embeddings, y) -> np.ndarray:
    h = as_matrix(embeddings, name="embeddings")
    y = as_codes(y, name="y")
    return np.stack([h[y == label].mean(axis=0) for label in range(int(y.max()) + 1)])


def random_subspaces(whitening: Whitening, rank: int, n: int, *, seed: int) -> list[Subspace]:
    """Haar-random rank-``rank`` subspaces in the same whitened support.

    Any orthonormal rank-``p`` basis removes exactly ``p / r`` of the
    localization set's whitened energy, so these controls are rank- and
    energy-matched to the targeted subspace by construction.
    """

    rng = np.random.default_rng(seed)
    return [Subspace(whitening, _orthonormal(rng.normal(size=(whitening.rank, rank)))) for _ in range(n)]


__all__ = [
    "Subspace",
    "Whitening",
    "class_means",
    "conditional_direction",
    "paired_subspace",
    "random_subspaces",
]
