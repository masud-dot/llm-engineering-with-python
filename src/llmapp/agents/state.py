"""Everything an agent run knows about itself."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from llmapp.llm.base import ToolCallRequest
from llmapp.tools.base import ToolResult


class RunStatus(str, Enum):
    """How a run ended, or why it stopped."""

    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_APPROVAL = "needs_approval"
    STEP_LIMIT = "step_limit"
    COST_LIMIT = "cost_limit"
    STALLED = "stalled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self not in (
            RunStatus.RUNNING,
            RunStatus.NEEDS_APPROVAL,
        )


@dataclass(frozen=True)
class Step:
    """One iteration of the loop, recorded."""

    number: int
    thought: str
    calls: tuple[ToolCallRequest, ...]
    results: tuple[ToolResult, ...]
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Limits:
    """The bounds a run may not exceed."""

    max_steps: int = 8
    max_cost_usd: float = 0.50
    max_tool_calls: int = 20
    repeat_threshold: int = 2


@dataclass
class AgentState:
    """Mutable state carried across steps."""

    goal: str
    limits: Limits = field(default_factory=Limits)
    plan: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    status: RunStatus = RunStatus.RUNNING
    answer: str = ""
    note: str = ""
    approved: set[str] = field(default_factory=set)
    pending: ToolCallRequest | None = None
    # The whole turn is held, not just the awaited call, so a
    # resumed run executes what the human saw.
    pending_calls: tuple[ToolCallRequest, ...] = ()
    pending_thought: str = ""
    spent_usd: float = 0.0
    _signatures: list[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(len(step.calls) for step in self.steps)

    def record(self, step: Step) -> None:
        self.steps.append(step)
        self.spent_usd += step.cost_usd
        for call in step.calls:
            self._signatures.append(signature(call))

    def repeat_count(self) -> int:
        """How many times the most recent call has been made."""
        if not self._signatures:
            return 0
        return self._signatures.count(self._signatures[-1])

    def repeated(self) -> str | None:
        """The last call, if it has been made too often."""
        if self.repeat_count() > self.limits.repeat_threshold:
            return self._signatures[-1]
        return None

    def exhausted(self) -> RunStatus | None:
        """Which bound, if any, has been reached."""
        if self.step_count >= self.limits.max_steps:
            return RunStatus.STEP_LIMIT
        if self.spent_usd >= self.limits.max_cost_usd:
            return RunStatus.COST_LIMIT
        if self.tool_calls >= self.limits.max_tool_calls:
            return RunStatus.STEP_LIMIT
        return None

    def transcript(self) -> str:
        """What the model sees of its own history."""
        lines: list[str] = []
        for step in self.steps:
            if step.thought:
                lines.append(f"step {step.number}: {step.thought}")
            for call, result in zip(
                step.calls, step.results, strict=False
            ):
                lines.append(
                    f"  called {call.name}({call.arguments})"
                )
                lines.append(f"  -> {result.output}")
        return "\n".join(lines)


def signature(call: ToolCallRequest) -> str:
    """Identify a call by tool and arguments, not by id."""
    blob = f"{call.name}\x00{call.arguments}".encode()
    return hashlib.sha256(blob).hexdigest()[:12]
