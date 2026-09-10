"""The output contract for ticket triage."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    """The only categories a consumer will accept."""

    BILLING = "billing"
    SHIPPING = "shipping"
    TECHNICAL = "technical"
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class OrderRef(BaseModel):
    """A referenced order, when the ticket names one."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=1, max_length=32)
    amount_usd: float | None = Field(default=None, ge=0)


class TicketExtraction(BaseModel):
    """What the model must return, or the call has failed."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity
    order: OrderRef | None = None
    summary: str = Field(min_length=1, max_length=200)
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary")
    @classmethod
    def single_line(cls, value: str) -> str:
        return " ".join(value.split())
