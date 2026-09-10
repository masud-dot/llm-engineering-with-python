import pytest

from llmapp.llm.fake import ScriptedClient
from llmapp.pipeline import TicketTriage


def test_valid_category_is_returned() -> None:
    client = ScriptedClient(["billing"])
    result = TicketTriage(client).run("I was charged twice")
    assert result.category == "billing"


def test_unknown_category_falls_back() -> None:
    client = ScriptedClient(["Sure! I think it is BILLING."])
    result = TicketTriage(client).run("I was charged twice")
    assert result.category == "other"


def test_prompt_contains_the_ticket() -> None:
    client = ScriptedClient(["shipping"])
    TicketTriage(client).run("  Where is my parcel?  ")
    sent = client.calls[0][0].content
    assert "Where is my parcel?" in sent


def test_empty_ticket_is_rejected() -> None:
    with pytest.raises(ValueError):
        TicketTriage(ScriptedClient([])).run("   ")
