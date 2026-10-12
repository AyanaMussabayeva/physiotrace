"""Small input checks shared across modules."""

from __future__ import annotations

import numpy as np


def as_matrix(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def as_codes(values, *, name: str, n_levels: int | None = None) -> np.ndarray:
    """Return non-negative integer codes; values must already be encoded."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.equal(array, np.floor(array))):
            raise ValueError(f"{name} must contain integer codes.")
    codes = array.astype(np.int64)
    if np.any(codes < 0):
        raise ValueError(f"{name} must contain non-negative codes.")
    if n_levels is not None and np.any(codes >= n_levels):
        raise ValueError(f"{name} contains a code outside [0, {n_levels}).")
    return codes


def encode(values, levels=None) -> tuple[np.ndarray, tuple]:
    """Encode arbitrary labels as integer codes in the order of ``levels``."""

    array = np.asarray(values)
    if levels is None:
        levels = tuple(np.unique(array).tolist())
    else:
        levels = tuple(levels)
    lookup = {level: code for code, level in enumerate(levels)}
    try:
        codes = np.fromiter((lookup[value] for value in array.tolist()), dtype=np.int64, count=len(array))
    except KeyError as error:
        raise ValueError(f"Value {error.args[0]!r} is not among the declared levels {levels}.") from None
    return codes, levels


def same_length(**arrays) -> int:
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inputs must have the same length, got {lengths}.")
    return next(iter(lengths.values()))
