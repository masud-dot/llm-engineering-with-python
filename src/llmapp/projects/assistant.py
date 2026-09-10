"""Project 1: the AI Document Assistant, assembled."""

from dataclasses import dataclass, field
from pathlib import Path

from llmapp.llm.base import LLMClient
from llmapp.prompts.context import ContextBudget
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.access import AccessPolicy, Principal
from llmapp.rag.ingest import ingest_file
from llmapp.rag.pipeline import RagPipeline, RagResult
from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.embed import Embedder
from llmapp.retrieval.hybrid import HybridRetriever
from llmapp.retrieval.lexical import BM25Index
from llmapp.retrieval.search import SemanticIndex
from llmapp.retrieval.store import Hit

SUPPORTED = (".md", ".txt", ".html", ".htm", ".pdf")


@dataclass(frozen=True)
class DocumentMeta:
    """What ingestion must know that a file cannot tell it.

    Access labels come from the directory layout and a
    manifest, never from the document text (Chapter 21).
    """

    team: str
    effective_date: str
    topic: str = ""
    visibility: str = "internal"

    def as_metadata(self) -> dict[str, str]:
        data = {
            "team": self.team,
            "effective_date": self.effective_date,
            "visibility": self.visibility,
        }
        if self.topic:
            data["topic"] = self.topic
        return data


def meta_for(path: Path, root: Path) -> DocumentMeta:
    """Derive labels from where a file sits, not what it says."""
    relative = path.relative_to(root)
    team = relative.parts[0] if len(relative.parts) > 1 else "all"
    stem = path.stem
    year = "".join(c for c in stem if c.isdigit())[:4]
    date = f"{year}-01-01" if len(year) == 4 else "2026-01-01"
    topic = stem.split("-")[0]
    return DocumentMeta(team=team, effective_date=date, topic=topic)


def ingest_corpus(
    root: Path, *, max_words: int = 80, overlap_words: int = 20
) -> list[Chunk]:
    """Every supported file under root, labeled by location."""
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (
            SUPPORTED
        ):
            continue
        meta = meta_for(path, root)
        chunks.extend(
            ingest_file(
                path,
                max_words=max_words,
                overlap_words=overlap_words,
                metadata=meta.as_metadata(),
            )
        )
    return chunks


@dataclass
class FloorFilter:
    """A semantic index with a relevance floor of its own.

    Rank fusion discards score scales, so a floor applied to
    fused scores is meaningless. The floor therefore belongs
    on the semantic arm, before fusion. See Section 26.6.
    """

    inner: SemanticIndex | BM25Index
    min_score: float

    def __len__(self) -> int:
        return len(self.inner)

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        return [
            hit
            for hit in self.inner.search(query, k=k, where=where)
            if hit.score >= self.min_score
        ]


@dataclass
class HybridIndex:
    """Hybrid retrieval behind the pipeline's index interface."""

    semantic: SemanticIndex
    lexical: BM25Index
    candidates: int = 20
    min_semantic_score: float = 0.15
    # BM25 scores are unbounded, so this is a corpus-specific
    # floor found by measurement, not a universal constant.
    min_lexical_score: float = 1.0

    def __len__(self) -> int:
        return len(self.semantic)

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        return HybridRetriever(
            semantic=FloorFilter(
                self.semantic, self.min_semantic_score
            ),
            lexical=FloorFilter(
                self.lexical,
                self.min_lexical_score,
            ),
            candidates=self.candidates,
        ).search(query, k=k, where=where)


@dataclass
class DocumentAssistant:
    """Hybrid retrieval, grounded answers, enforced access."""

    client: LLMClient
    embedder: Embedder
    registry: PromptRegistry
    policy: AccessPolicy = field(default_factory=AccessPolicy)
    k: int = 4
    min_score: float = 0.15
    semantic: SemanticIndex = field(init=False)
    lexical: BM25Index = field(init=False)

    def __post_init__(self) -> None:
        self.semantic = SemanticIndex(self.embedder)
        self.lexical = BM25Index()
        # The pipeline retrieves through the same hybrid
        # retriever as retrieve(); using the semantic index
        # here and hybrid there produced different results
        # for the same question. See Section 26.6.
        self._retriever = HybridIndex(
            self.semantic,
            self.lexical,
            min_semantic_score=self.min_score,
        )
        self._pipeline = RagPipeline(
            self.client,
            self._retriever,
            self.registry,
            budget=ContextBudget(
                window_tokens=8000, reserved_output=800
            ),
            k=self.k,
            # The floor lives in the retriever now; fused
            # scores are not comparable to cosine scores.
            min_score=0.0,
            policy=self.policy,
        )

    def load(self, root: Path) -> int:
        chunks = ingest_corpus(root)
        self.semantic.add(chunks)
        self.lexical.add(chunks)
        return len(chunks)

    def __len__(self) -> int:
        return len(self.semantic)

    def retrieve(
        self,
        question: str,
        principal: Principal,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        """Hybrid retrieval under the caller's permissions."""
        scoped = self.policy.merge(principal, where)
        return self._retriever.search(
            question, k=self.k, where=scoped
        )

    def answer(
        self,
        question: str,
        principal: Principal,
        where: dict[str, str] | None = None,
    ) -> RagResult:
        return self._pipeline.answer(
            question, where=where, principal=principal
        )
