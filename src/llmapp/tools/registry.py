"""Validate, authorize, and run tool calls."""

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from llmapp.tools.base import (
    ApprovalRequired,
    SideEffect,
    ToolCall,
    ToolDenied,
    ToolError,
    ToolResult,
    ToolSpec,
)


@dataclass
class ToolRegistry:
    """The only place a tool call becomes a function call."""

    specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"tool {spec.name!r} already exists")
        self.specs[spec.name] = spec

    def schemas(
        self, allowed: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Only advertise tools the caller may use."""
        names = self.specs if allowed is None else allowed
        return [
            self.specs[name].schema()
            for name in names
            if name in self.specs
        ]

    def dispatch(
        self,
        call: ToolCall,
        *,
        allowed: Sequence[str] | None = None,
        approved: Sequence[str] = (),
    ) -> ToolResult:
        """Run one call, or return the failure to the model."""
        try:
            spec = self._authorize(call, allowed)
            arguments = self._parse(spec, call)
            self._check_approval(spec, call, approved)
            output = self._run(spec, arguments)
        except ApprovalRequired:
            raise
        except (ToolDenied, ToolError) as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=f"error: {exc}",
                ok=False,
            )
        return self._truncate(spec, call, output)

    def _authorize(
        self, call: ToolCall, allowed: Sequence[str] | None
    ) -> ToolSpec:
        spec = self.specs.get(call.name)
        if spec is None:
            raise ToolDenied(f"no tool named {call.name!r}")
        if allowed is not None and call.name not in allowed:
            raise ToolDenied(
                f"tool {call.name!r} is not available here"
            )
        return spec

    def _parse(self, spec: ToolSpec, call: ToolCall) -> Any:
        try:
            payload = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(f"arguments were not JSON: {exc}")
        try:
            return spec.arguments.model_validate(payload)
        except ValidationError as exc:
            fields = ", ".join(
                ".".join(str(p) for p in item["loc"])
                for item in exc.errors()
            )
            raise ToolError(
                f"invalid arguments ({fields}); "
                f"correct them and call again"
            )

    def _check_approval(
        self,
        spec: ToolSpec,
        call: ToolCall,
        approved: Sequence[str],
    ) -> None:
        if spec.requires_approval and call.call_id not in approved:
            raise ApprovalRequired(
                f"{spec.name} ({spec.side_effect.value}) needs "
                f"approval for call {call.call_id}"
            )

    def _run(self, spec: ToolSpec, arguments: Any) -> str:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(spec.handler, arguments)
            try:
                return future.result(timeout=spec.timeout_s)
            except TimeoutError:
                # The worker keeps running; see Section 15.6.
                raise ToolError(
                    f"{spec.name} timed out after "
                    f"{spec.timeout_s:g}s"
                )
            except Exception as exc:
                raise ToolError(f"{spec.name} failed: {exc}")

    def _truncate(
        self, spec: ToolSpec, call: ToolCall, output: str
    ) -> ToolResult:
        limit = spec.max_result_chars
        if len(output) <= limit:
            return ToolResult(call.call_id, call.name, output)
        head = output[: limit - 60]
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=(
                f"{head}\n[truncated: {len(output)} characters, "
                f"showing {limit - 60}]"
            ),
            truncated=True,
        )

    def writes(self) -> list[str]:
        """Tools that change state, for review."""
        return sorted(
            name
            for name, spec in self.specs.items()
            if spec.side_effect is not SideEffect.READ
        )
