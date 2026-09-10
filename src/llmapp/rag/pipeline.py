"""Retrieval-augmented generation, one stage at a time."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from llmapp.llm.base import LLMClient, Message
from llmapp.prompts.context import ContextBudget, Section
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.access import AccessPolicy, Principal
from llmapp.rag.answer import GroundedAnswer
from llmapp.rag.grounding import (
    GroundingReport,
    check,
    demote,
    keep_verified,
)
from llmapp.retrieval.store import Hit
from llmapp.schemas.validate import ParseReport, parse_or_repair

MIN_SCORE = 0.20


class SearchIndex(Protocol):
    """What the pipeline needs from a retriever.

    Widened in Chapter 26 so a hybrid retriever can be used
    here as well as directly; SemanticIndex still satisfies it.
    """

    def __len__(self) -> int: ...

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]: ...


@dataclass(frozen=True)
class RagResult:
    """The answer, and everything needed to audit it."""

    answer: GroundedAnswer
    hits: list[Hit]
    prompt_id: str
    dropped: list[str]
    report: ParseReport
    truncated: list[str] = field(default_factory=list)
    grounding: GroundingReport | None = None

    @property
    def sources(self) -> list[str]:
        return sorted({h.chunk.source for h in self.hits})


def format_passages(hits: list[Hit]) -> list[Section]:
    """One budgeted section per passage, provenance included."""
    sections: list[Section] = []
    for rank, hit in enumerate(hits):
        header = f"[{hit.chunk.chunk_id}]"
        heading = hit.chunk.metadata.get("heading", "")
        if heading:
            header = f"{header} ({heading})"
        sections.append(
            Section(
                name=hit.chunk.chunk_id,
                content=f"{header}\n{hit.chunk.text}",
                priority=len(hits) - rank,
                truncatable=True,
            )
        )
    return sections


class RagPipeline:
    """Retrieve, assemble, generate, verify."""

    def __init__(
        self,
        client: LLMClient,
        index: SearchIndex,
        registry: PromptRegistry,
        *,
        budget: ContextBudget | None = None,
        k: int = 5,
        min_score: float = MIN_SCORE,
        prompt_version: int = 1,
        policy: AccessPolicy | None = None,
        min_coverage: float = 1.0,
    ) -> None:
        self._client = client
        self._index = index
        self._registry = registry
        self._policy = policy
        self._min_coverage = min_coverage
        self._budget = budget or ContextBudget(
            window_tokens=8000, reserved_output=800
        )
        self._k = k
        self._min_score = min_score
        self._version = prompt_version

    @property
    def index_size(self) -> int:
        """How many chunks are searchable, for health checks."""
        return len(self._index)

    def answer(
        self,
        question: str,
        where: dict[str, str] | None = None,
        principal: Principal | None = None,
    ) -> RagResult:
        if self._policy is not None:
            if principal is None:
                raise PermissionError(
                    "this pipeline requires a principal"
                )
            where = self._policy.merge(principal, where)
        hits = [
            hit
            for hit in self._index.search(
                question, k=self._k, where=where
            )
            if hit.score >= self._min_score
        ]
        if not hits:
            return RagResult(
                answer=GroundedAnswer(
                    answered=False,
                    missing="no relevant passages were retrieved",
                ),
                hits=[],
                prompt_id="",
                dropped=[],
                report=ParseReport(),
            )

        fitted = self._budget.fit(format_passages(hits))
        kept = {section.name for section in fitted.sections}
        template = self._registry.get("rag_answer", self._version)
        prompt = template.render(
            question=question, passages=fitted.text()
        )
        parsed, report = parse_or_repair(
            self._client, prompt.messages, GroundedAnswer
        )
        supplied = [h for h in hits if h.chunk.chunk_id in kept]
        final, grounding = self._ground(parsed, supplied)
        return RagResult(
            answer=final,
            hits=supplied,
            prompt_id=prompt.prompt_id,
            dropped=fitted.dropped,
            report=report,
            truncated=fitted.truncated,
            grounding=grounding,
        )

    def _ground(
        self, parsed: GroundedAnswer, supplied: list[Hit]
    ) -> tuple[GroundedAnswer, GroundingReport]:
        grounding = check(parsed, supplied)
        if not parsed.answered:
            return parsed, grounding
        if not grounding.spans:
            return (
                demote(
                    parsed,
                    grounding,
                    "the answer could not be traced to a passage",
                ),
                grounding,
            )
        if grounding.coverage < self._min_coverage:
            return (
                demote(
                    parsed,
                    grounding,
                    "part of the answer is not supported by the "
                    "retrieved passages",
                ),
                grounding,
            )
        return keep_verified(parsed, grounding), grounding


def build_index_from(
    root: Path, index: "SemanticIndex", **kwargs: object
) -> int:
    """Ingest a directory into an index. Returns chunk count."""
    from llmapp.rag.ingest import ingest_directory

    total = 0
    for chunks in ingest_directory(root, **kwargs).values():
        index.add(chunks)
        total += len(chunks)
    return total


if TYPE_CHECKING:  # imported for the ingestion helper only
    from llmapp.retrieval.search import SemanticIndex
