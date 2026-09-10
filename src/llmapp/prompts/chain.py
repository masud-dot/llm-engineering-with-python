"""Two prompts are often more reliable than one."""

from dataclasses import dataclass

from llmapp.llm.base import LLMClient
from llmapp.prompts.registry import PromptRegistry


@dataclass(frozen=True)
class ChainResult:
    """The answer plus the artifact that produced it."""

    category: str
    facts: str
    prompt_ids: tuple[str, str]
    total_tokens: int


class TriageChain:
    """Extract facts, then judge from the facts alone."""

    def __init__(
        self,
        client: LLMClient,
        registry: PromptRegistry,
        categories: str,
    ) -> None:
        self._client = client
        self._registry = registry
        self._categories = categories

    def run(self, ticket: str) -> ChainResult:
        extract = self._registry.get("extract", 1).render(
            ticket=ticket
        )
        step1 = self._client.complete(extract.messages)

        judge = self._registry.get("judge", 1).render(
            facts=step1.text, categories=self._categories
        )
        step2 = self._client.complete(judge.messages)

        used = (
            step1.input_tokens
            + step1.output_tokens
            + step2.input_tokens
            + step2.output_tokens
        )
        return ChainResult(
            category=step2.text.strip().lower(),
            facts=step1.text,
            prompt_ids=(extract.prompt_id, judge.prompt_id),
            total_tokens=used,
        )
