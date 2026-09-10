"""Vector arithmetic, written out before it is imported."""

import math
from collections.abc import Sequence

import numpy as np

Vector = Sequence[float]


def dot(a: Vector, b: Vector) -> float:
    """Sum of element-wise products."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(a, b, strict=True))


def magnitude(a: Vector) -> float:
    """Euclidean length of a vector."""
    return math.sqrt(dot(a, a))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine of the angle between two vectors, -1 to 1."""
    denominator = magnitude(a) * magnitude(b)
    if denominator == 0.0:
        raise ValueError("cannot compare a zero vector")
    return dot(a, b) / denominator


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, once, up front."""
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("cannot normalize a zero vector")
    scaled: np.ndarray = matrix / lengths
    return scaled


def top_k(
    query: np.ndarray, matrix: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """Rank rows of a normalized matrix against a query.

    Both inputs must already be unit length, which turns
    cosine similarity into a single dot product.
    """
    if k <= 0:
        return []
    scores = matrix @ query
    k = min(k, scores.shape[0])
    best = np.argpartition(-scores, k - 1)[:k]
    ranked = best[np.argsort(-scores[best])]
    return [(int(i), float(scores[i])) for i in ranked]
