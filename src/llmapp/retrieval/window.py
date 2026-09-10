"""Retrieve small pieces, return the context around them."""

from collections.abc import Sequence
from dataclasses import dataclass

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.store import Hit


@dataclass
class ParentExpander:
    """Match on a precise chunk, answer from its neighbors.

    Small chunks rank well because their vectors are focused.
    Large chunks answer well because they carry context. This
    keeps both by expanding a hit into its window.
    """

    chunks: Sequence[Chunk]
    window: int = 1

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._by_source: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            self._by_source.setdefault(chunk.source, []).append(
                chunk
            )
        for group in self._by_source.values():
            group.sort(key=lambda c: c.ordinal)

    def expand(self, hits: Sequence[Hit]) -> list[Hit]:
        expanded: list[Hit] = []
        for hit in hits:
            group = self._by_source.get(hit.chunk.source, [])
            positions = [
                i
                for i, c in enumerate(group)
                if c.chunk_id == hit.chunk.chunk_id
            ]
            if not positions:
                expanded.append(hit)
                continue
            centre = positions[0]
            low = max(0, centre - self.window)
            high = min(len(group), centre + self.window + 1)
            neighbors = group[low:high]
            merged = Chunk(
                text="\n".join(c.text for c in neighbors),
                source=hit.chunk.source,
                ordinal=hit.chunk.ordinal,
                metadata=dict(hit.chunk.metadata),
            )
            expanded.append(Hit(chunk=merged, score=hit.score))
        return expanded
