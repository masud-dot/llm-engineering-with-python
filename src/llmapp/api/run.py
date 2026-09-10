"""The process entry point the container runs."""

from pathlib import Path

import uvicorn

from llmapp.api.limits import SlidingWindow
from llmapp.api.main import Services, build_app
from llmapp.api.security import TokenIssuer
from llmapp.config import Settings
from llmapp.llm.factory import build_client
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.access import AccessPolicy
from llmapp.rag.pipeline import RagPipeline, build_index_from
from llmapp.retrieval.embed import Embedder, OpenAIEmbedder
from llmapp.retrieval.local import HashEmbedder
from llmapp.retrieval.search import SemanticIndex


def build_embedder(settings: Settings) -> Embedder:
    """Mirror build_client: the fake provider embeds locally.

    The provider check has to be here as well as in the model
    factory. Omitting it made the service crash at startup,
    because the SDK refuses to construct without credentials.
    """
    if settings.provider == "fake":
        return HashEmbedder(settings.embedding_dimensions)
    key = settings.api_key
    return OpenAIEmbedder(
        api_key=key.get_secret_value() if key else "",
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        base_url=settings.base_url,
    )


def build_services(settings: Settings) -> Services:
    """Construct once, at startup, never per request."""
    index = SemanticIndex(build_embedder(settings))
    if settings.corpus_dir.is_dir():
        loaded = build_index_from(settings.corpus_dir, index)
        print(f"ingested {loaded} chunks from "
              f"{settings.corpus_dir}")
    pipeline = RagPipeline(
        build_client(settings),
        index,
        PromptRegistry(Path("prompts")),
        policy=AccessPolicy(),
    )
    secret = settings.jwt_secret
    return Services(
        pipeline=pipeline,
        issuer=TokenIssuer(
            secret=secret.get_secret_value() if secret else ""
        ),
        limiter=SlidingWindow(limit=settings.rate_limit_per_minute),
    )


def main() -> None:
    settings = Settings()
    app = build_app(build_services(settings))
    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104 - bound inside a container
        port=settings.port,
        workers=1,  # scale with replicas, not with workers
        access_log=False,  # structlog handles request logging
    )


if __name__ == "__main__":
    main()
