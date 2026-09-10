"""Adapter for the Anthropic Messages API."""

from collections.abc import Sequence

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

from llmapp.llm.base import Completion, Message


class AnthropicClient:
    """Satisfies LLMClient."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._client = Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout_s
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        system = "\n".join(
            m.content for m in messages if m.role == "system"
        )
        turns: list[MessageParam] = [
            MessageParam(role=m.role, content=m.content)
            for m in messages
            if m.role in ("user", "assistant")
        ]
        # max_tokens is required here, unlike the Responses API.
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_output_tokens,
            system=system,
            messages=turns,
        )
        text = "".join(
            block.text
            for block in response.content
            if isinstance(block, TextBlock)
        )
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            finish_reason=response.stop_reason or "",
        )
