"""A deterministic embedder for local runs and CI.

This is not a semantic model. It hashes tokens into a fixed
space, so similar wording scores highly and paraphrase does
not. It exists so the service starts, the tests run, and the
container smoke-tests without a key or a network — never as
a substitute for a real embedding model.
"""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

WORD = re.compile(r"[a-z0-9]+")
DEFAULT_DIMENSIONS = 256


class HashEmbedder:
    """Bag of hashed tokens, unit length."""

    def __init__(
        self, dimensions: int = DEFAULT_DIMENSIONS
    ) -> None:
        if dimensions < 8:
            raise ValueError("too few dimensions to be useful")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode()).digest()
        return int.from_bytes(digest[:4], "big") % (
            self._dimensions - 1
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self._dimensions)
            for token in WORD.findall(text.lower()):
                vector[self._bucket(token)] += 1.0
            # Keeps an empty string from being a zero vector.
            vector[-1] = 0.01
            rows.append(vector)
        stacked: np.ndarray = np.vstack(rows)
        return stacked
