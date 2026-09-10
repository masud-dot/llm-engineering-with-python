from llmapp.llm.base import Completion, Message
from llmapp.prompts.memory import (
    BufferMemory,
    HybridMemory,
    SummaryMemory,
    WindowMemory,
)


class KeepsNothing:
    """A summarizer that discards detail, on purpose."""

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        return Completion("They talked.", 0, 0, "stub", "stop")


def talk(memory: object, turns: int) -> None:
    for i in range(turns):
        memory.add(  # type: ignore[attr-defined]
            Message(role="user", content=f"point {i}")
        )
        memory.add(  # type: ignore[attr-defined]
            Message(role="assistant", content=f"noted {i}")
        )


def test_buffer_keeps_every_turn() -> None:
    memory = BufferMemory()
    talk(memory, 10)
    assert len(memory.recall()) == 20


def test_window_is_bounded() -> None:
    memory = WindowMemory(max_turns=3)
    talk(memory, 10)
    recalled = memory.recall()
    assert len(recalled) == 6
    assert recalled[-1].content == "noted 9"


def test_summary_replaces_old_turns() -> None:
    memory = SummaryMemory(
        client=KeepsNothing(), keep_turns=2, summarize_after=4
    )
    talk(memory, 10)
    recalled = memory.recall()
    assert recalled[0].role == "system"
    assert "They talked." in recalled[0].content
    # The invariant: one summary plus at most a full
    # pre-compaction window, never the whole transcript.
    assert len(recalled) <= 1 + memory.summarize_after * 2
    assert len(recalled) < 20


def test_pinned_facts_survive_a_lossy_summarizer() -> None:
    inner = SummaryMemory(
        client=KeepsNothing(), keep_turns=2, summarize_after=4
    )
    memory = HybridMemory(inner=inner)
    memory.pin("order_id", "48812")
    talk(memory, 20)
    payload = memory.recall()
    assert any("48812" in m.content for m in payload)


def test_the_same_facts_are_lost_without_pinning() -> None:
    memory = SummaryMemory(
        client=KeepsNothing(), keep_turns=2, summarize_after=4
    )
    memory.add(Message(role="user", content="order 48812 broke"))
    talk(memory, 20)
    payload = memory.recall()
    assert not any("48812" in m.content for m in payload)
