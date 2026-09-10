"""Choose an adapter from configuration."""

from llmapp.config import Settings
from llmapp.llm.base import LLMClient
from llmapp.llm.fake import ScriptedClient


def build_client(settings: Settings) -> LLMClient:
    """Return the client the configuration asks for."""
    if settings.provider == "fake":
        return ScriptedClient(["fake reply"])

    key = settings.api_key
    if key is None:  # config validation should prevent this
        raise RuntimeError("no API key configured")
    secret = key.get_secret_value()

    if settings.provider == "openai":
        from llmapp.llm.openai_adapter import OpenAIClient

        return OpenAIClient(
            api_key=secret,
            model=settings.model,
            timeout_s=settings.request_timeout_s,
            base_url=settings.base_url,
        )

    from llmapp.llm.anthropic_adapter import AnthropicClient

    return AnthropicClient(
        api_key=secret,
        model=settings.model,
        timeout_s=settings.request_timeout_s,
        base_url=settings.base_url,
    )
