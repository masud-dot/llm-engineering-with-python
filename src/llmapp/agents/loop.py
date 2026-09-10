"""A bounded agent loop, written by hand."""

from collections.abc import Sequence
from dataclasses import dataclass

from llmapp.llm.base import Completion, LLMClient, Message
from llmapp.prompts.registry import PromptRegistry
from llmapp.tools.base import ApprovalRequired
from llmapp.tools.loop import run_calls
from llmapp.tools.registry import ToolRegistry
from llmapp.agents.state import (
    AgentState,
    Limits,
    RunStatus,
    Step,
)

REPEAT_NUDGE = (
    "You already made that exact call and have the result. "
    "Use it, choose a different call, or answer the goal."
)


@dataclass
class Agent:
    """Perceive, decide, act, repeat — until a bound is hit."""

    client: LLMClient
    tools: ToolRegistry
    registry: PromptRegistry
    allowed: Sequence[str] | None = None
    max_output_tokens: int = 512

    def plan(self, state: AgentState) -> list[str]:
        """Write the plan before acting on it."""
        names = ", ".join(
            self.allowed
            if self.allowed is not None
            else sorted(self.tools.specs)
        )
        prompt = self.registry.get("agent_plan", 1).render(
            goal=state.goal, tools=names
        )
        result = self.client.complete(prompt.messages)
        state.plan = [
            line.strip(" -*\t")
            for line in result.text.splitlines()
            if line.strip()
        ][:4]
        return state.plan

    def run(self, goal: str, limits: Limits | None = None) -> AgentState:
        state = AgentState(goal=goal, limits=limits or Limits())
        self.plan(state)
        return self.resume(state)

    def resume(self, state: AgentState) -> AgentState:
        """Continue a run, including after an approval."""
        state.status = RunStatus.RUNNING
        if state.pending_calls and not self._flush(state):
            return state
        while True:
            reached = state.exhausted()
            if reached is not None:
                state.status = reached
                state.note = (
                    f"stopped after {state.step_count} steps "
                    f"and ${state.spent_usd:.4f}"
                )
                return state
            if not self._step(state):
                return state

    def _flush(self, state: AgentState) -> bool:
        """Execute the calls the human was shown, not new ones."""
        calls = state.pending_calls
        try:
            results = run_calls(
                self.tools,
                calls,
                allowed=self.allowed,
                approved=sorted(state.approved),
            )
        except ApprovalRequired as exc:
            state.status = RunStatus.NEEDS_APPROVAL
            state.note = str(exc)
            return False
        state.record(
            Step(
                number=state.step_count + 1,
                thought=state.pending_thought,
                calls=calls,
                results=tuple(results),
            )
        )
        state.pending = None
        state.pending_calls = ()
        state.pending_thought = ""
        return True

    def _step(self, state: AgentState) -> bool:
        """One iteration. False means the run has stopped."""
        completion = self._decide(state)
        if not completion.tool_calls:
            state.answer = completion.text
            state.status = RunStatus.COMPLETED
            self._record(state, completion, (), ())
            return False

        try:
            results = run_calls(
                self.tools,
                completion.tool_calls,
                allowed=self.allowed,
                approved=sorted(state.approved),
            )
        except ApprovalRequired as exc:
            state.pending = next(
                call
                for call in completion.tool_calls
                if call.call_id not in state.approved
            )
            state.pending_calls = completion.tool_calls
            state.pending_thought = completion.text
            state.status = RunStatus.NEEDS_APPROVAL
            state.note = str(exc)
            return False

        self._record(
            state, completion, completion.tool_calls, tuple(results)
        )
        repeated = state.repeated()
        if repeated is not None:
            state.status = RunStatus.STALLED
            state.note = "the same call was made repeatedly"
            return False
        return True

    def _decide(self, state: AgentState) -> Completion:
        remaining = state.limits.max_steps - state.step_count
        prompt = self.registry.get("agent_step", 1).render(
            goal=state.goal,
            plan="\n".join(state.plan) or "(none)",
            transcript=state.transcript() or "(nothing yet)",
            remaining=str(remaining),
        )
        messages = list(prompt.messages)
        # Nudge as soon as a call repeats once; stall only if
        # the nudge is ignored (see _step).
        if state.repeat_count() >= 2:
            messages.append(
                Message(role="user", content=REPEAT_NUDGE)
            )
        return self._call(messages)

    def _call(self, messages: list[Message]) -> Completion:
        schemas = self.tools.schemas(self.allowed)
        complete_with_tools = getattr(
            self.client, "complete_with_tools", None
        )
        if complete_with_tools is None:
            return self.client.complete(
                messages, max_output_tokens=self.max_output_tokens
            )
        result: Completion = complete_with_tools(
            messages,
            tools=schemas,
            max_output_tokens=self.max_output_tokens,
        )
        return result

    def _record(
        self,
        state: AgentState,
        completion: Completion,
        calls: tuple[object, ...],
        results: tuple[object, ...],
    ) -> None:
        state.record(
            Step(
                number=state.step_count + 1,
                thought=completion.text,
                calls=completion.tool_calls,
                results=results,  # type: ignore[arg-type]
                tokens=completion.input_tokens
                + completion.output_tokens,
            )
        )


def approve(state: AgentState, call_id: str) -> AgentState:
    """Record a human decision and clear the suspension."""
    state.approved.add(call_id)
    state.pending = None
    return state
