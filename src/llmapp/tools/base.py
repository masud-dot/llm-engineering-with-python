"""What a tool is, and what the model is told about it."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from llmapp.schemas.strict import strict_schema

MAX_RESULT_CHARS = 4000


class SideEffect(str, Enum):
    """What running this tool does to the world."""

    READ = "read"          # safe to repeat
    WRITE = "write"        # changes state; needs idempotency
    EXTERNAL = "external"  # leaves the building


class ToolError(Exception):
    """A failure the model should see and may recover from."""


class ApprovalRequired(Exception):
    """Execution stopped pending a human decision."""


class ToolDenied(Exception):
    """The caller may not use this tool."""


@dataclass(frozen=True)
class ToolSpec:
    """The contract for one tool."""

    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[Any], str]
    side_effect: SideEffect = SideEffect.READ
    requires_approval: bool = False
    timeout_s: float = 10.0
    max_result_chars: int = MAX_RESULT_CHARS

    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the provider.

        The Responses API takes these fields flat; Chat
        Completions nests them under a "function" key.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": strict_schema(self.arguments),
            "strict": True,
        }


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run a tool."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    """What goes back to the model."""

    call_id: str
    name: str
    output: str
    ok: bool = True
    truncated: bool = False

    def as_input(self) -> dict[str, str]:
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": self.output,
        }
