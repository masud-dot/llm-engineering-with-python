"""Read an OpenAPI document into an endpoint inventory.

A spec file is untrusted input: it may come from another
team, a vendor, or a pull request. Everything read from it
is validated and bounded before it reaches a prompt.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METHODS = ("get", "post", "put", "patch", "delete")
MAX_ENDPOINTS = 200
MAX_FIELDS = 60


class SpecError(ValueError):
    """The document is not a spec this tool can use."""


@dataclass(frozen=True)
class Field_:
    """One request field the generator can vary."""

    name: str
    type: str
    required: bool
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str = ""
    enum: tuple[str, ...] = ()

    @property
    def has_bounds(self) -> bool:
        return any(
            value is not None
            for value in (
                self.minimum,
                self.maximum,
                self.min_length,
                self.max_length,
            )
        ) or bool(self.enum or self.pattern)


@dataclass(frozen=True)
class Endpoint:
    """One operation, with what a test would need."""

    method: str
    path: str
    operation_id: str
    summary: str = ""
    secured: bool = False
    fields: tuple[Field_, ...] = ()
    statuses: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        cleaned = self.path.strip("/").replace("/", "_")
        cleaned = cleaned.replace("{", "").replace("}", "")
        return f"{self.method}_{cleaned or 'root'}"


def _resolve(schema: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    """Follow one level of $ref, which is all we support."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = doc
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(node, dict) or part not in node:
            raise SpecError(f"unresolvable reference {ref}")
        node = node[part]
    if not isinstance(node, dict):
        raise SpecError(f"reference {ref} is not an object")
    return node


def _fields(schema: dict[str, Any]) -> list[Field_]:
    required = set(schema.get("required", []))
    found: list[Field_] = []
    for name, prop in list(
        schema.get("properties", {}).items()
    )[:MAX_FIELDS]:
        if not isinstance(prop, dict):
            continue
        found.append(
            Field_(
                name=name,
                type=str(prop.get("type", "string")),
                required=name in required,
                minimum=prop.get("minimum"),
                maximum=prop.get("maximum"),
                min_length=prop.get("minLength"),
                max_length=prop.get("maxLength"),
                pattern=str(prop.get("pattern", "")),
                enum=tuple(str(v) for v in prop.get("enum", [])),
            )
        )
    return found


def load_spec(path: Path) -> list[Endpoint]:
    """Parse a spec into endpoints, bounded and validated."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"not valid JSON: {exc}") from None
    if not isinstance(doc, dict) or "paths" not in doc:
        raise SpecError("no paths section; is this OpenAPI?")

    endpoints: list[Endpoint] = []
    for path_name, operations in doc["paths"].items():
        if not isinstance(operations, dict):
            continue
        for method, operation in operations.items():
            if method.lower() not in METHODS:
                continue
            if not isinstance(operation, dict):
                continue
            endpoints.append(
                _endpoint(path_name, method, operation, doc)
            )
            if len(endpoints) >= MAX_ENDPOINTS:
                return endpoints
    if not endpoints:
        raise SpecError("the spec declares no operations")
    return endpoints


def _endpoint(
    path_name: str,
    method: str,
    operation: dict[str, Any],
    doc: dict[str, Any],
) -> Endpoint:
    body = (
        operation.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    schema = _resolve(body, doc) if body else {}
    responses = tuple(
        sorted(str(code) for code in operation.get("responses", {}))
    )
    return Endpoint(
        method=method.lower(),
        path=path_name,
        operation_id=str(
            operation.get("operationId", f"{method}{path_name}")
        ),
        summary=str(operation.get("summary", ""))[:200],
        secured=bool(operation.get("security"))
        or "Authorization" in json.dumps(
            operation.get("parameters", [])
        ),
        fields=tuple(_fields(schema)),
        statuses=responses,
    )
