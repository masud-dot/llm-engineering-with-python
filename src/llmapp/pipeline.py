"""Orchestration: compose the layers into one feature."""

from dataclasses import dataclass

from llmapp.llm.base import LLMClient, Message

CATEGORIES = ("billing", "shipping", "technical", "other")

TEMPLATE = (
    "Classify the support ticket into exactly one of: "
    "{categories}.\n"
    "Reply with the category name and nothing else.\n\n"
    "Ticket: {ticket}"
)


@dataclass(frozen=True)
class Triage:
    """The result the caller receives."""

    category: str
    cost_tokens: int


class TicketTriage:
    """Turn a ticket into a validated category."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def run(self, ticket: str) -> Triage:
        if not ticket.strip():
            raise ValueError("ticket text is empty")
        prompt = TEMPLATE.format(
            categories=", ".join(CATEGORIES),
            ticket=ticket.strip(),
        )
        result = self._client.complete(
            [Message(role="user", content=prompt)]
        )
        category = result.text.strip().lower()
        if category not in CATEGORIES:
            category = "other"
        return Triage(
            category=category,
            cost_tokens=result.input_tokens + result.output_tokens,
        )
