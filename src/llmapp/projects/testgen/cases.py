"""The output contract for one generated test case."""

from pydantic import BaseModel, ConfigDict, Field

from llmapp.projects.testgen.plan import CaseKind


class TestCase(BaseModel):
    """What the model must return for one planned slot."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=5, max_length=80, pattern=r"^test_[a-z0-9_]+$"
    )
    kind: CaseKind
    description: str = Field(min_length=5, max_length=200)
    method: str = Field(pattern="^(get|post|put|patch|delete)$")
    path: str = Field(min_length=1, max_length=200)
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, object] = Field(default_factory=dict)
    expected_status: int = Field(ge=100, le=599)
    requirement_id: str = Field(default="", max_length=64)


class TestCaseBatch(BaseModel):
    """One model call returns the cases for one endpoint."""

    model_config = ConfigDict(extra="forbid")

    cases: list[TestCase] = Field(min_length=1, max_length=20)
