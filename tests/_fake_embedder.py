"""A deterministic embedder for tests.

Bag-of-words over a fixed vocabulary. Not semantic, but real
lexical similarity, which makes retrieval predictable.
"""
from collections.abc import Sequence

import numpy as np

VOCAB = [
    "refund", "refunds", "fourteen", "days", "window", "parcel",
    "damaged", "broken", "replacement", "photograph", "claim",
    "delivery", "express", "surcharge", "standard", "working",
    "escalation", "senior", "engineer", "unresolved", "policy",
    "warehouse", "item", "price", "cost", "ticket",
]
INDEX = {word: i for i, word in enumerate(VOCAB)}


class KeywordEmbedder:
    @property
    def dimensions(self) -> int:
        return len(VOCAB) + 1

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(len(VOCAB) + 1)
            for word in text.lower().replace(".", " ").split():
                position = INDEX.get(word.strip(",;:()[]"))
                if position is not None:
                    vector[position] += 1.0
            vector[-1] = 0.01  # keeps zero vectors legal
            rows.append(vector)
        stacked: np.ndarray = np.vstack(rows)
        return stacked
