"""Parse model output into a typed object, or fail loudly."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llmapp.llm.base import Completion, LLMClient, Message
from llmapp.llm.errors import SemanticError

T = TypeVar("T", bound=BaseModel)

REPAIR_PREFIX = (
    "Your previous reply did not match the required schema. "
    "Fix these problems and reply with corrected JSON only:\n"
)

TRUNCATED = {"incomplete", "length", "max_tokens"}


@dataclass
class ParseReport:
    """How much work the parse took. Track this."""

    attempts: int = 0
    first_pass_valid: bool = False
    errors: list[str] = field(default_factory=list)


def extract_json(text: str) -> str:
    """Strip code fences and prose around a JSON object."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1]
        body = body.rsplit("```", 1)[0]
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end == -1 or end < start:
        return body.strip()
    return body[start : end + 1]


def _summarize(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "(root)"
        lines.append(f"- {where}: {item['msg']}")
    return "\n".join(lines)


def parse_or_repair(
    client: LLMClient,
    messages: Sequence[Message],
    model_cls: type[T],
    *,
    attempts: int = 2,
    max_output_tokens: int = 512,
) -> tuple[T, ParseReport]:
    """Validate the reply, feeding errors back once or twice."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    report = ParseReport()
    turns = list(messages)

    for attempt in range(attempts):
        report.attempts = attempt + 1
        result: Completion = client.complete(
            turns, max_output_tokens=max_output_tokens
        )
        if result.finish_reason in TRUNCATED:
            raise SemanticError(
                "response was truncated; raise the output cap "
                "or shrink the schema"
            )
        try:
            payload = json.loads(extract_json(result.text))
            parsed = model_cls.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            problems = (
                _summarize(exc)
                if isinstance(exc, ValidationError)
                else f"- (root): {exc}"
            )
            report.errors.append(problems)
            if attempt == attempts - 1:
                raise SemanticError(
                    f"output failed validation after "
                    f"{attempts} attempts:\n{problems}"
                ) from exc
            turns = [
                *turns,
                Message(role="assistant", content=result.text),
                Message(
                    role="user", content=REPAIR_PREFIX + problems
                ),
            ]
            continue
        report.first_pass_valid = attempt == 0
        return parsed, report

    raise AssertionError("unreachable")
