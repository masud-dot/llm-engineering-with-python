"""Run both retrievers and fuse the results."""

from dataclasses import dataclass
from typing import Protocol

from llmapp.retrieval.fusion import reciprocal_rank_fusion
from llmapp.retrieval.store import Hit


class Retriever(Protocol):
    """Anything that can rank chunks for a query."""

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]: ...


@dataclass
class HybridRetriever:
    """Semantic recall plus lexical precision."""

    semantic: Retriever
    lexical: Retriever
    candidates: int = 20

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        rankings = [
            self.semantic.search(query, self.candidates, where),
            self.lexical.search(query, self.candidates, where),
        ]
        return reciprocal_rank_fusion(rankings, k=k)
