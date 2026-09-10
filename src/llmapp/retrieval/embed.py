"""Turning text into vectors, with batching and a cache."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from llmapp.llm.errors import PermanentError


class Embedder(Protocol):
    """What the retrieval layer needs from an embedding model."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class EmbeddingCache:
    """Text does not change, so neither does its vector."""

    model: str
    store: dict[str, np.ndarray] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def key(self, text: str) -> str:
        blob = f"{self.model}\x00{text}".encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        found = self.store.get(self.key(text))
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, text: str, vector: np.ndarray) -> None:
        self.store[self.key(text)] = vector


class OpenAIEmbedder:
    """Adapter for the embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        batch_size: int = 128,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self.cache = EmbeddingCache(model=model)
        self.prompt_tokens = 0

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimensions))
        pending = [t for t in texts if self.cache.get(t) is None]
        unique = list(dict.fromkeys(pending))
        for start in range(0, len(unique), self._batch_size):
            batch = unique[start : start + self._batch_size]
            self._fetch(batch)
        rows = []
        for text in texts:
            vector = self.cache.get(text)
            if vector is None:
                raise PermanentError("embedding was not returned")
            rows.append(vector)
        return np.vstack(rows)

    def _fetch(self, batch: list[str]) -> None:
        response = self._client.embeddings.create(
            model=self._model,
            input=batch,
            dimensions=self._dimensions,
        )
        # Note: embeddings report prompt_tokens, not the
        # input_tokens the Responses API uses.
        self.prompt_tokens += response.usage.prompt_tokens
        for item in response.data:
            vector = np.asarray(item.embedding, dtype=np.float64)
            if vector.shape[0] != self._dimensions:
                raise PermanentError(
                    f"expected {self._dimensions} dimensions, "
                    f"received {vector.shape[0]}"
                )
            self.cache.put(batch[item.index], vector)
