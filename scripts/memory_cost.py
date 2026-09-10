"""Run one conversation through four memory strategies."""

import sys

from llmapp.llm.base import Completion, Message
from llmapp.prompts.memory import (
    BufferMemory,
    HybridMemory,
    Memory,
    SummaryMemory,
    WindowMemory,
    transcript_tokens,
)

TURNS = 60
ORDER_ID = "48812"


class StubSummarizer:
    """Deterministic stand-in for a summarizing model.

    `bias="head"` keeps the oldest words, `bias="tail"` keeps
    the newest. Real summarizers fall somewhere between, which
    is the point of running both.
    """

    def __init__(self, bias: str = "head") -> None:
        self.bias = bias

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        body = messages[-1].content  # type: ignore[index]
        words = body.replace("\n", " ").split()
        kept = words[:60] if self.bias == "head" else words[-60:]
        return Completion(" ".join(kept), 0, 0, "stub", "stop")


def build(bias: str) -> dict[str, Memory]:
    hybrid = HybridMemory(
        inner=SummaryMemory(client=StubSummarizer(bias))
    )
    hybrid.pin("order_id", ORDER_ID)
    return {
        "buffer": BufferMemory(),
        "window(6)": WindowMemory(max_turns=6),
        "summary": SummaryMemory(client=StubSummarizer(bias)),
        "hybrid": hybrid,
    }


def main() -> int:
    bias = sys.argv[1] if len(sys.argv) > 1 else "head"
    memories = build(bias)
    totals = {name: 0 for name in memories}
    finals = {name: 0 for name in memories}
    retained = {name: False for name in memories}

    for turn in range(TURNS):
        if turn == 0:
            user = f"My order {ORDER_ID} arrived damaged."
        else:
            user = f"Follow-up question number {turn}."
        reply = f"Acknowledged point {turn}."
        for name, memory in memories.items():
            memory.add(Message(role="user", content=user))
            payload = memory.recall()
            totals[name] += transcript_tokens(payload)
            memory.add(Message(role="assistant", content=reply))

    for name, memory in memories.items():
        payload = memory.recall()
        finals[name] = transcript_tokens(payload)
        retained[name] = any(
            ORDER_ID in m.content for m in payload
        )

    print(f"summarizer bias: {bias}")
    print(f"{'strategy':>12} {'turn-60':>9} {'cumulative':>11} "
          f"{'order id kept':>14}")
    for name in memories:
        print(
            f"{name:>12} {finals[name]:>9} {totals[name]:>11} "
            f"{('yes' if retained[name] else 'no'):>14}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
