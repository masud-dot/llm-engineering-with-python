"""Adapter for the OpenAI Responses API."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import openai
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputParam,
)

from llmapp.llm.base import Completion, Message, ToolCallRequest
from llmapp.llm.errors import PermanentError, TransientError


def split_system(
    messages: Sequence[Message],
) -> tuple[str | None, ResponseInputParam]:
    """Move system turns into the instructions field."""
    system = "\n".join(
        m.content for m in messages if m.role == "system"
    )
    turns: ResponseInputParam = [
        EasyInputMessageParam(role=m.role, content=m.content)
        for m in messages
        if m.role != "system"
    ]
    return (system or None), turns


@contextmanager
def translate_errors() -> Iterator[None]:
    """Turn SDK exceptions into the neutral taxonomy."""
    try:
        yield
    except openai.RateLimitError as exc:
        header = exc.response.headers.get("retry-after")
        wait = float(header) if header else None
        raise TransientError(str(exc), retry_after_s=wait) from exc
    except (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    ) as exc:
        raise TransientError(str(exc)) from exc
    except (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.BadRequestError,
    ) as exc:
        raise PermanentError(str(exc)) from exc


def to_completion(response: Response) -> Completion:
    """Read only what the application needs."""
    usage = response.usage
    calls = tuple(
        ToolCallRequest(
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )
        for item in response.output
        if item.type == "function_call"
    )
    return Completion(
        text=response.output_text,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        model=response.model,
        finish_reason=response.status or "",
        tool_calls=calls,
    )


class OpenAIClient:
    """Satisfies LLMClient, SupportsStreaming, SupportsAsync."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        base_url: str | None = None,
        sdk_retries: int = 0,
    ) -> None:
        self._model = model
        # The SDK retries twice by default. We set zero and own
        # the policy ourselves; see Section 6.3.
        self._sync = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=sdk_retries,
        )
        self._async = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=sdk_retries,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        system, turns = split_system(messages)
        with translate_errors():
            response = self._sync.responses.create(
                model=self._model,
                input=turns,
                instructions=system,
                max_output_tokens=max_output_tokens,
            )
        return to_completion(response)

    def complete_with_tools(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, object]],
        extra_input: Sequence[dict[str, object]] = (),
        max_output_tokens: int = 512,
    ) -> Completion:
        """One turn of a tool-using conversation."""
        system, turns = split_system(messages)
        payload = [*turns, *extra_input]
        with translate_errors():
            response = self._sync.responses.create(
                model=self._model,
                input=payload,  # type: ignore[arg-type]
                instructions=system,
                max_output_tokens=max_output_tokens,
                tools=tools,  # type: ignore[arg-type]
            )
        return to_completion(response)

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, object],
        schema_name: str,
        max_output_tokens: int = 512,
    ) -> Completion:
        """Ask the provider to enforce the schema itself."""
        system, turns = split_system(messages)
        with translate_errors():
            response = self._sync.responses.create(
                model=self._model,
                input=turns,
                instructions=system,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        return to_completion(response)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Iterator[str]:
        system, turns = split_system(messages)
        with self._sync.responses.create(
            model=self._model,
            input=turns,
            instructions=system,
            max_output_tokens=max_output_tokens,
            stream=True,
        ) as events:
            for event in events:
                if event.type == "response.output_text.delta":
                    yield event.delta

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        system, turns = split_system(messages)
        response = await self._async.responses.create(
            model=self._model,
            input=turns,
            instructions=system,
            max_output_tokens=max_output_tokens,
        )
        return to_completion(response)
