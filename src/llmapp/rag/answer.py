"""The output contract for a grounded answer."""

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A claim traced back to the chunk that supports it."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=300)


class GroundedAnswer(BaseModel):
    """Either a cited answer, or an explicit refusal."""

    model_config = ConfigDict(extra="forbid")

    answered: bool
    answer: str = Field(default="", max_length=2000)
    citations: list[Citation] = Field(default_factory=list)
    missing: str = Field(default="", max_length=300)
