"""Metrics for retrieval and generation."""

import math
from collections.abc import Sequence


def ndcg_at_k(
    retrieved: Sequence[str], relevant: Sequence[str], k: int
) -> float:
    """Rank-weighted relevance, normalized to 0..1.

    Unlike recall@k, a correct passage at rank 1 scores higher
    than the same passage at rank 5.
    """
    if not relevant:
        raise ValueError("a query needs at least one relevant id")
    wanted = set(relevant)
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in wanted
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(wanted), k) + 1)
    )
    return gain / ideal if ideal else 0.0


def context_precision(
    retrieved: Sequence[str], relevant: Sequence[str]
) -> float:
    """Share of supplied passages that were relevant.

    Low precision means the model is reading noise, which
    costs tokens and lowers quality.
    """
    if not retrieved:
        return 0.0
    wanted = set(relevant)
    hits = sum(1 for item in retrieved if item in wanted)
    return hits / len(retrieved)


def context_recall(
    retrieved: Sequence[str], relevant: Sequence[str]
) -> float:
    """Share of relevant passages that were supplied."""
    if not relevant:
        raise ValueError("a query needs at least one relevant id")
    wanted = set(relevant)
    return len(set(retrieved) & wanted) / len(wanted)


def contains_all(answer: str, required: Sequence[str]) -> bool:
    """Semantic assertion: every required fact is present."""
    lowered = answer.lower()
    return all(item.lower() in lowered for item in required)


def contains_none(answer: str, forbidden: Sequence[str]) -> bool:
    """Semantic assertion: nothing forbidden appears."""
    lowered = answer.lower()
    return not any(item.lower() in lowered for item in forbidden)


def abstention_correct(answered: bool, answerable: bool) -> bool:
    """Answering the answerable, refusing the rest."""
    return answered is answerable


def agreement(
    left: Sequence[str], right: Sequence[str]
) -> float:
    """Raw agreement between two sets of labels."""
    if len(left) != len(right):
        raise ValueError("label sets must be the same length")
    if not left:
        raise ValueError("no labels to compare")
    same = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    return same / len(left)


def cohens_kappa(
    left: Sequence[str], right: Sequence[str]
) -> float:
    """Agreement corrected for what chance would produce.

    Two raters who both say "good" 90% of the time agree 82%
    of the time by luck alone, so raw agreement flatters a
    judge on an unbalanced label set.
    """
    observed = agreement(left, right)
    labels = set(left) | set(right)
    total = len(left)
    expected = sum(
        (list(left).count(label) / total)
        * (list(right).count(label) / total)
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
