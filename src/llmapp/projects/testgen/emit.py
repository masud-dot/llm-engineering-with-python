"""Turn validated cases into a runnable pytest module."""

import ast
import json
import keyword
from dataclasses import dataclass

from llmapp.projects.testgen.cases import TestCase

HEADER = '''"""Generated API tests. Review before trusting.

Produced by the test-case generator from the OpenAPI spec.
Every case is a starting point for a reviewer, not a
substitute for one.
"""

import pytest

BASE_URL = "http://localhost:8000"
'''

TEMPLATE = '''

def {name}(client) -> None:
    """{description}"""
    response = client.request(
        "{method}",
        BASE_URL + {path},
        headers={headers},
        json={body},
    )
    assert response.status_code == {status}
'''


class EmitError(ValueError):
    """The generated module would not be valid Python."""


@dataclass(frozen=True)
class Module:
    """A generated file and what it contains."""

    source: str
    case_names: tuple[str, ...]

    @property
    def line_count(self) -> int:
        return len(self.source.splitlines())


def safe_name(name: str, used: set[str]) -> str:
    """A unique, legal Python identifier."""
    cleaned = "".join(
        c if c.isalnum() or c == "_" else "_" for c in name
    ).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"test_{cleaned}"
    if not cleaned.startswith("test_"):
        cleaned = f"test_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_case"
    candidate, suffix = cleaned, 2
    while candidate in used:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def emit(cases: list[TestCase]) -> Module:
    """Render, then parse. A module that will not parse is a bug."""
    used: set[str] = set()
    names: list[str] = []
    body = [HEADER]
    for case in cases:
        name = safe_name(case.name, used)
        names.append(name)
        body.append(
            TEMPLATE.format(
                name=name,
                description=case.description.replace('"', "'"),
                method=case.method.upper(),
                path=json.dumps(case.path),
                headers=json.dumps(case.headers),
                body=json.dumps(case.body),
                status=case.expected_status,
            )
        )
    source = "".join(body)
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise EmitError(f"generated module is invalid: {exc}")
    return Module(source=source, case_names=tuple(names))
