"""The public contract. Changing these is a breaking change."""

from pydantic import BaseModel, ConfigDict, Field

from llmapp.rag.answer import Citation


class AskRequest(BaseModel):
    """What a caller may send."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """What a caller receives. Internals stay internal."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    answered: bool
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    missing: str = ""
    sources: list[str] = Field(default_factory=list)


class SqlRequest(BaseModel):
    """A natural-language question about the database."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)


class SqlResponse(BaseModel):
    """The query that ran, and what it returned."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    answered: bool
    sql: str = ""
    explanation: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    row_count: int = 0
    refused_because: str = ""


class JobAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    poll_after_seconds: float = 1.0


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    result: AskResponse | None = None
    error: str = ""


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    verdict: str = Field(
        pattern="^(good|bad|wrong_fact|missing|unsafe)$"
    )
    comment: str = Field(default="", max_length=1000)


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    checks: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """One shape for every failure, with no internals."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str = ""
    request_id: str = ""
