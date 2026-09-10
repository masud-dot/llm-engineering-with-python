"""Change the question before searching with it."""

from dataclasses import dataclass

from llmapp.llm.base import LLMClient, Message
from llmapp.retrieval.fusion import reciprocal_rank_fusion
from llmapp.retrieval.hybrid import Retriever
from llmapp.retrieval.store import Hit

REWRITE_INSTRUCTION = (
    "Rewrite the user's question as search queries for a "
    "document index. Produce up to $n short queries, one per "
    "line, using the vocabulary a policy document would use. "
    "Do not answer the question. Output only the queries."
)


def expand_instruction(n: int) -> str:
    return REWRITE_INSTRUCTION.replace("$n", str(n))


@dataclass
class MultiQueryRetriever:
    """Ask several ways, then fuse the answers."""

    client: LLMClient
    retriever: Retriever
    variants: int = 3
    include_original: bool = True

    def queries(self, question: str) -> list[str]:
        result = self.client.complete(
            [
                Message(
                    role="system",
                    content=expand_instruction(self.variants),
                ),
                Message(role="user", content=question),
            ]
        )
        lines = [
            line.strip(" -*\t")
            for line in result.text.splitlines()
            if line.strip()
        ][: self.variants]
        if self.include_original:
            lines = [question, *lines]
        # Preserve order, drop repeats.
        return list(dict.fromkeys(lines))

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        rankings = [
            self.retriever.search(variant, k * 2, where)
            for variant in self.queries(query)
        ]
        return reciprocal_rank_fusion(rankings, k=k)
