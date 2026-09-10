"""Combine rankings that use incomparable scores."""

from collections.abc import Sequence

from llmapp.retrieval.store import Hit

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    k: int = 5,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    """Merge ranked lists by position, not by score.

    BM25 scores and cosine similarities are not on the same
    scale, so they cannot be added. Ranks can be.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Hit] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + position
            )
            seen.setdefault(chunk_id, hit)
    ordered = sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )
    return [
        Hit(chunk=seen[chunk_id].chunk, score=score)
        for chunk_id, score in ordered[:k]
    ]
