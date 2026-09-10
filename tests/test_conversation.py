from llmapp.llm.conversation import Conversation
from llmapp.llm.fake import ScriptedClient


def test_history_is_resent_each_turn() -> None:
    client = ScriptedClient(["one", "two"])
    chat = Conversation(client=client, system="Be brief.")
    chat.ask("first")
    chat.ask("second")
    sent = client.calls[1]
    assert [m.role for m in sent] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sent[0].content == "Be brief."


def test_old_turns_are_dropped_but_system_is_kept() -> None:
    client = ScriptedClient([str(i) for i in range(5)])
    chat = Conversation(
        client=client, system="Be brief.", max_turns=2
    )
    for i in range(5):
        chat.ask(f"q{i}")
    sent = client.calls[-1]
    assert sent[0].role == "system"
    assert len(sent) == 1 + 4
    assert sent[-1].content == "q4"


def test_turn_count() -> None:
    client = ScriptedClient(["a", "b"])
    chat = Conversation(client=client)
    chat.ask("x")
    chat.ask("y")
    assert chat.turns() == 2
