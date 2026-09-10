"""Project 2: the AI Test-Case Generator, assembled."""

import json
from dataclasses import dataclass
from pathlib import Path

from llmapp.llm.base import LLMClient
from llmapp.prompts.registry import PromptRegistry
from llmapp.projects.testgen.cases import TestCase, TestCaseBatch
from llmapp.projects.testgen.emit import Module, emit
from llmapp.projects.testgen.plan import Slot, plan_endpoint
from llmapp.projects.testgen.review import Review, review
from llmapp.projects.testgen.spec import Endpoint, load_spec
from llmapp.schemas.validate import parse_or_repair


@dataclass(frozen=True)
class Suite:
    """One endpoint's generated tests and their review."""

    endpoint: Endpoint
    slots: tuple[Slot, ...]
    cases: tuple[TestCase, ...]
    module: Module
    review: Review


def describe(endpoint: Endpoint) -> str:
    return json.dumps(
        {
            "method": endpoint.method,
            "path": endpoint.path,
            "summary": endpoint.summary,
            "secured": endpoint.secured,
            "statuses": list(endpoint.statuses),
            "fields": [
                {
                    "name": f.name,
                    "type": f.type,
                    "required": f.required,
                    "minimum": f.minimum,
                    "maximum": f.maximum,
                    "min_length": f.min_length,
                    "max_length": f.max_length,
                    "enum": list(f.enum),
                    "pattern": f.pattern,
                }
                for f in endpoint.fields
            ],
        },
        indent=1,
        sort_keys=True,
    )


def describe_slots(slots: list[Slot]) -> str:
    return "\n".join(
        f"- {slot.test_name} [{slot.kind.value}] {slot.note}"
        for slot in slots
    )


@dataclass
class TestGenerator:
    """Plan in code, generate with a model, verify in code."""

    client: LLMClient
    registry: PromptRegistry
    attempts: int = 2

    def for_endpoint(self, endpoint: Endpoint) -> Suite:
        slots = plan_endpoint(endpoint)
        prompt = self.registry.get("testgen", 1).render(
            endpoint=describe(endpoint),
            slots=describe_slots(slots),
            statuses=", ".join(endpoint.statuses) or "unknown",
        )
        batch, _ = parse_or_repair(
            self.client,
            prompt.messages,
            TestCaseBatch,
            attempts=self.attempts,
            max_output_tokens=2000,
        )
        module = emit(list(batch.cases))
        return Suite(
            endpoint=endpoint,
            slots=tuple(slots),
            cases=tuple(batch.cases),
            module=module,
            review=review(slots, batch.cases, module.source),
        )

    def for_spec(self, path: Path) -> list[Suite]:
        return [
            self.for_endpoint(endpoint)
            for endpoint in load_spec(path)
        ]


def write_suites(
    suites: list[Suite], out_dir: Path
) -> list[Path]:
    """Deterministic file names, one module per endpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suite in suites:
        path = out_dir / f"test_{suite.endpoint.name}.py"
        path.write_text(suite.module.source, encoding="utf-8")
        written.append(path)
    return sorted(written)
