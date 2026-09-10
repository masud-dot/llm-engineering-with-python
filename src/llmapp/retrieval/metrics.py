"""Retrieval quality, measured the standard way."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScore:
    """Aggregate quality over a query set."""

    recall_at_k: float
    mrr: float
    queries: int

    def __str__(self) -> str:
        return (
            f"recall@k {self.recall_at_k:.2f}  "
            f"mrr {self.mrr:.2f}  n={self.queries}"
        )


def recall_at_k(
    retrieved: Sequence[str], relevant: Sequence[str], k: int
) -> float:
    """Fraction of relevant items found in the top k."""
    if not relevant:
        raise ValueError("a query needs at least one relevant id")
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Sequence[str]
) -> float:
    """1/rank of the first relevant item, or 0."""
    wanted = set(relevant)
    for position, item in enumerate(retrieved, start=1):
        if item in wanted:
            return 1.0 / position
    return 0.0


def evaluate(
    results: Mapping[str, Sequence[str]],
    gold: Mapping[str, Sequence[str]],
    k: int = 5,
) -> RetrievalScore:
    """Score a whole query set."""
    if not gold:
        raise ValueError("no gold queries supplied")
    recalls = []
    ranks = []
    for query, relevant in gold.items():
        retrieved = results.get(query, [])
        recalls.append(recall_at_k(retrieved, relevant, k))
        ranks.append(reciprocal_rank(retrieved, relevant))
    return RetrievalScore(
        recall_at_k=sum(recalls) / len(recalls),
        mrr=sum(ranks) / len(ranks),
        queries=len(gold),
    )
