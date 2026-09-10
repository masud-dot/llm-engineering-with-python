"""Wire the evaluation harness to the application.

Kept at the repository root so the gate script and the CI
job agree on how the system under test is constructed.
"""

from pathlib import Path

from llmapp.eval.dataset import Dataset, load_dataset
from llmapp.eval.runner import Runner
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.pipeline import RagPipeline, build_index_from
from llmapp.retrieval.embed import Embedder
from llmapp.retrieval.search import SemanticIndex


def build_dataset() -> Dataset:
    return load_dataset(Path("evals/triage.jsonl"), "triage")


def build_runner(answers: object = None) -> Runner:
    from llmapp.config import Settings
    from llmapp.llm.factory import build_client

    settings = Settings()
    index = SemanticIndex(_embedder(settings))
    build_index_from(Path("tests/corpus"), index, max_words=60)
    client = answers if answers is not None else build_client(
        settings
    )
    pipeline = RagPipeline(
        client,  # type: ignore[arg-type]
        index,
        PromptRegistry(Path("prompts")),
    )
    return lambda case: pipeline.answer(case.question)


def _embedder(settings: object) -> Embedder:
    """The package ships a local embedder for this path."""
    from llmapp.retrieval.local import HashEmbedder

    return HashEmbedder()
