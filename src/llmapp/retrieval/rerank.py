"""Reorder a candidate set with a stronger, slower signal."""

import json
from dataclasses import dataclass
from typing import Protocol

from llmapp.llm.base import LLMClient, Message
from llmapp.retrieval.store import Hit

RERANK_INSTRUCTION = (
    "Score how well each passage answers the question, from 0 "
    "to 10. Judge relevance only; do not answer the question. "
    "Reply with JSON: a list of objects with keys id and score, "
    "one per passage, and nothing else."
)


class Reranker(Protocol):
    def rerank(
        self, query: str, hits: list[Hit], k: int = 5
    ) -> list[Hit]: ...


@dataclass
class LLMReranker:
    """Ask a model to score query-passage pairs together."""

    client: LLMClient
    max_candidates: int = 20

    def rerank(
        self, query: str, hits: list[Hit], k: int = 5
    ) -> list[Hit]:
        if not hits:
            return []
        candidates = hits[: self.max_candidates]
        listing = "\n\n".join(
            f"id: {h.chunk.chunk_id}\n{h.chunk.text}"
            for h in candidates
        )
        result = self.client.complete(
            [
                Message(role="system", content=RERANK_INSTRUCTION),
                Message(
                    role="user",
                    content=f"Question: {query}\n\n{listing}",
                ),
            ]
        )
        scores = _parse_scores(result.text)
        if not scores:
            return candidates[:k]  # fall back to input order
        ranked = sorted(
            candidates,
            key=lambda h: -scores.get(h.chunk.chunk_id, -1.0),
        )
        return [
            Hit(chunk=h.chunk, score=scores[h.chunk.chunk_id])
            for h in ranked
            if h.chunk.chunk_id in scores
        ][:k]


def _parse_scores(text: str) -> dict[str, float]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return {}
    try:
        rows = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    scores: dict[str, float] = {}
    for row in rows:
        if isinstance(row, dict) and "id" in row:
            try:
                scores[str(row["id"])] = float(row.get("score", 0))
            except (TypeError, ValueError):
                continue
    return scores
