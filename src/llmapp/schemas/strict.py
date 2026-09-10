"""Turn a Pydantic model into a strict JSON Schema.

Strict schema modes are narrower than JSON Schema. Every
object must forbid extra properties and list every property
as required, including the optional ones, which carry null
in their type union instead.
"""

from typing import Any

from pydantic import BaseModel


def _tighten(node: dict[str, Any]) -> None:
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        properties = node.get("properties", {})
        node["required"] = list(properties)
        for child in properties.values():
            if isinstance(child, dict):
                _tighten(child)
    for key in ("items", "not"):
        child = node.get(key)
        if isinstance(child, dict):
            _tighten(child)
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for child in node.get(key, []):
            if isinstance(child, dict):
                _tighten(child)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A JSON Schema a strict decoder will accept."""
    schema = model.model_json_schema()
    for definition in schema.get("$defs", {}).values():
        _tighten(definition)
    _tighten(schema)
    return schema
