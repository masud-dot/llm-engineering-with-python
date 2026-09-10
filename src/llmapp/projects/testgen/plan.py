"""Decide which cases an endpoint needs, before generating."""

from dataclasses import dataclass
from enum import Enum

from llmapp.projects.testgen.spec import Endpoint, Field_


class CaseKind(str, Enum):
    """The categories a reviewer expects to see covered."""

    HAPPY = "happy"
    MISSING_REQUIRED = "missing_required"
    WRONG_TYPE = "wrong_type"
    BOUNDARY = "boundary"
    UNAUTHENTICATED = "unauthenticated"
    EXTRA_FIELD = "extra_field"


@dataclass(frozen=True)
class Slot:
    """One planned case: what to write before writing it."""

    endpoint: Endpoint
    kind: CaseKind
    field_name: str = ""
    note: str = ""

    @property
    def test_name(self) -> str:
        parts = ["test", self.endpoint.name, self.kind.value]
        if self.field_name:
            parts.append(self.field_name)
        return "_".join(parts).replace("__", "_")


def plan_endpoint(endpoint: Endpoint) -> list[Slot]:
    """A deterministic coverage plan, not a model call.

    The plan is code because coverage is a decision the team
    owns. The model writes the cases; it does not choose
    which cases exist.
    """
    slots = [Slot(endpoint, CaseKind.HAPPY)]
    if endpoint.secured:
        slots.append(Slot(endpoint, CaseKind.UNAUTHENTICATED))
    for field in endpoint.fields:
        if field.required:
            slots.append(
                Slot(
                    endpoint,
                    CaseKind.MISSING_REQUIRED,
                    field.name,
                    f"omit {field.name}",
                )
            )
        if field.type in ("integer", "number"):
            slots.append(
                Slot(
                    endpoint,
                    CaseKind.WRONG_TYPE,
                    field.name,
                    f"send a string for {field.name}",
                )
            )
        if field.has_bounds:
            slots.append(
                Slot(
                    endpoint,
                    CaseKind.BOUNDARY,
                    field.name,
                    describe_bounds(field),
                )
            )
    if endpoint.fields:
        slots.append(Slot(endpoint, CaseKind.EXTRA_FIELD))
    return slots


def describe_bounds(field: Field_) -> str:
    parts: list[str] = []
    if field.minimum is not None:
        parts.append(f"below {field.minimum}")
    if field.maximum is not None:
        parts.append(f"above {field.maximum}")
    if field.min_length is not None:
        parts.append(f"shorter than {field.min_length}")
    if field.max_length is not None:
        parts.append(f"longer than {field.max_length}")
    if field.enum:
        parts.append("outside the allowed values")
    if field.pattern:
        parts.append("not matching the pattern")
    return f"{field.name}: " + ", ".join(parts)


def plan(endpoints: list[Endpoint]) -> list[Slot]:
    slots: list[Slot] = []
    for endpoint in endpoints:
        slots.extend(plan_endpoint(endpoint))
    return slots


def coverage(slots: list[Slot]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in slots:
        counts[slot.kind.value] = counts.get(slot.kind.value, 0) + 1
    return counts
