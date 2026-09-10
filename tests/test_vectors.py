import math

import numpy as np
import pytest

from llmapp.retrieval.vectors import (
    cosine_similarity,
    dot,
    magnitude,
    normalize,
    top_k,
)


def test_identical_vectors_score_one() -> None:
    a = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_vectors_score_minus_one() -> None:
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == (
        pytest.approx(-1.0)
    )


def test_length_does_not_change_direction() -> None:
    short = [1.0, 1.0]
    long = [100.0, 100.0]
    assert cosine_similarity(short, long) == pytest.approx(1.0)


def test_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=8), rng.normal(size=8)
    expected = float(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    )
    assert cosine_similarity(list(a), list(b)) == (
        pytest.approx(expected)
    )


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        dot([1.0], [1.0, 2.0])


def test_magnitude_is_pythagoras() -> None:
    assert magnitude([3.0, 4.0]) == pytest.approx(5.0)


def test_normalized_rows_have_unit_length() -> None:
    matrix = np.array([[3.0, 4.0], [1.0, 0.0], [-2.0, 2.0]])
    lengths = np.linalg.norm(normalize(matrix), axis=1)
    assert np.allclose(lengths, 1.0)


def test_top_k_ranks_by_similarity() -> None:
    matrix = normalize(
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    )
    query = np.array([1.0, 0.0])
    ranked = top_k(query, matrix, k=2)
    assert [i for i, _ in ranked] == [0, 1]
    assert ranked[0][1] > ranked[1][1]


def test_top_k_handles_k_larger_than_the_corpus() -> None:
    matrix = normalize(np.array([[1.0, 0.0]]))
    assert len(top_k(np.array([1.0, 0.0]), matrix, k=9)) == 1


def test_dot_product_equals_cosine_when_normalized() -> None:
    matrix = normalize(np.array([[3.0, 4.0]]))
    query = normalize(np.array([[1.0, 2.0]]))[0]
    by_dot = float(matrix[0] @ query)
    by_cosine = cosine_similarity([3.0, 4.0], [1.0, 2.0])
    assert by_dot == pytest.approx(by_cosine)
    assert math.isclose(by_dot, by_cosine, rel_tol=1e-12)
