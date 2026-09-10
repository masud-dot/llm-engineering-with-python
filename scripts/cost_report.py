"""Measure what each optimization is worth on a workload."""

import random
from dataclasses import dataclass

import tiktoken

from llmapp.llm.base import Completion, Message
from llmapp.llm.pricing import Rates, cost_usd
from llmapp.obs.cache import ExactCache, cache_key
from llmapp.obs.quota import Attribution
from llmapp.obs.router import (
    ModelRouter,
    Tier,
    escalate_on_refusal,
)

ENC = tiktoken.get_encoding("o200k_base")
CHEAP = Rates(input_per_mtok=0.15, output_per_mtok=0.60)
STRONG = Rates(input_per_mtok=3.00, output_per_mtok=15.00)
REQUESTS = 500
DISTINCT = 80
REFUSAL_RATE = 0.2


@dataclass
class Stub:
    """A model that refuses a fixed share of the time."""

    text: str
    refusal: str = ""
    rate: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)
        self.calls = 0

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls += 1
        refuse = self._random.random() < self.rate
        text = self.refusal if refuse else self.text
        return Completion(
            text,
            input_tokens=4400,
            output_tokens=len(ENC.encode(text)),
            model="stub",
            finish_reason="stop",
        )


def workload() -> list[tuple[str, str]]:
    rng = random.Random(7)
    users = [f"u{n}" for n in range(10)]
    questions = [f"question number {n}" for n in range(DISTINCT)]
    return [
        (rng.choice(users), rng.choice(questions))
        for _ in range(REQUESTS)
    ]


def main() -> int:
    calls = workload()
    answer = "Refunds are issued within fourteen days."

    # 1. No cache, strong model only.
    plain = Stub(answer)
    baseline = 0.0
    ledger = Attribution()
    for tenant, question in calls:
        result = plain.complete(
            [Message(role="user", content=question)]
        )
        spend = cost_usd(result, STRONG)
        baseline += spend
        ledger.record(
            feature="rag",
            tenant=tenant,
            model="large",
            cost_usd=spend,
        )

    # 2. Exact cache, keyed per principal.
    cached_model = Stub(answer)
    cache = ExactCache()
    cached_cost = 0.0
    for tenant, question in calls:
        messages = [Message(role="user", content=question)]
        key = cache_key(
            principal=tenant,
            model="large",
            prompt_id="rag_answer.v1",
            messages=messages,
        )
        hit = cache.get(key)
        if hit is not None:
            continue
        result = cached_model.complete(messages)
        cache.put(key, result)
        cached_cost += cost_usd(result, STRONG)

    # 3. Routing: cheap first, escalate on refusal.
    small = Stub(answer, refusal="I don't know.",
                 rate=REFUSAL_RATE, seed=3)
    large = Stub(answer)
    route = ModelRouter(
        tiers=[Tier("small", small, CHEAP),
               Tier("large", large, STRONG)],
        should_escalate=escalate_on_refusal(),
    )
    for _, question in calls:
        route.complete([Message(role="user", content=question)])

    print(f"workload: {REQUESTS} requests, {DISTINCT} distinct "
          f"questions, 10 tenants\n")
    print(f"{'configuration':<28}{'calls':>7}{'cost':>10}"
          f"{'vs baseline':>13}")
    print(f"{'strong model, no cache':<28}{plain.calls:>7}"
          f"{baseline:>10.4f}{'—':>13}")
    print(f"{'strong model, exact cache':<28}"
          f"{cached_model.calls:>7}{cached_cost:>10.4f}"
          f"{cached_cost / baseline - 1:>12.0%}")
    routed = route.total_usd
    print(f"{'routed, cheap first':<28}"
          f"{small.calls + large.calls:>7}{routed:>10.4f}"
          f"{routed / baseline - 1:>12.0%}")
    print()
    # What a shared cache would hit, and must not be used.
    shared = ExactCache()
    for _, question in calls:
        messages = [Message(role="user", content=question)]
        unsafe = cache_key(
            principal="SHARED",
            model="large",
            prompt_id="rag_answer.v1",
            messages=messages,
        )
        if shared.get(unsafe) is None:
            shared.put(unsafe, Completion(answer, 4400, 8, "m", ""))

    print(f"cache hit rate    : {cache.hit_rate:.0%} "
          f"per principal, {shared.hit_rate:.0%} shared "
          f"(unsafe)")
    print(f"escalation rate   : {route.escalation_rate:.0%}")
    print(f"cost per request  : ${ledger.mean_usd:.4f} baseline, "
          f"${routed / REQUESTS:.4f} routed")
    print(f"top tenant        : {ledger.top('tenant')[0][0]} "
          f"(${ledger.top('tenant')[0][1]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
