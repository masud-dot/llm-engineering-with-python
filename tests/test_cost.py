import pytest

from llmapp.llm.base import Completion, Message
from llmapp.llm.pricing import Rates
from llmapp.obs.cache import (
    CacheKeyError,
    ExactCache,
    cache_key,
)
from llmapp.obs.quota import (
    Attribution,
    QuotaExceeded,
    ServiceDisabled,
    TenantQuota,
)
from llmapp.obs.router import (
    ModelRouter,
    Tier,
    any_of,
    escalate_on_refusal,
    escalate_on_short_answer,
)

ASK = [Message(role="user", content="what is the refund window")]
CHEAP = Rates(input_per_mtok=0.15, output_per_mtok=0.60)
STRONG = Rates(input_per_mtok=3.00, output_per_mtok=15.00)


def key(principal: str = "u1", model: str = "m") -> str:
    return cache_key(
        principal=principal,
        model=model,
        prompt_id="rag_answer.v1.abcd1234",
        messages=ASK,
    )


# --- cache keys -----------------------------------------------


def test_a_key_without_a_principal_is_refused() -> None:
    with pytest.raises(CacheKeyError, match="principal"):
        cache_key(
            principal="",
            model="m",
            prompt_id="p",
            messages=ASK,
        )


def test_two_users_do_not_share_an_entry() -> None:
    """Chapter 21: a shared cache undoes access filtering."""
    assert key("alice") != key("bob")


def test_the_model_is_part_of_the_key() -> None:
    assert key(model="small") != key(model="large")


def test_the_prompt_version_is_part_of_the_key() -> None:
    first = cache_key(
        principal="u1", model="m", prompt_id="v1", messages=ASK
    )
    second = cache_key(
        principal="u1", model="m", prompt_id="v2", messages=ASK
    )
    assert first != second


def test_extra_scope_separates_entries() -> None:
    base = cache_key(
        principal="u1", model="m", prompt_id="p", messages=ASK
    )
    scoped = cache_key(
        principal="u1",
        model="m",
        prompt_id="p",
        messages=ASK,
        scope=("corpus-2026-09",),
    )
    assert base != scoped


# --- cache behavior -------------------------------------------


def answer(text: str = "fourteen days") -> Completion:
    return Completion(text, 4400, 300, "m", "stop")


def test_a_miss_then_a_hit() -> None:
    cache = ExactCache()
    assert cache.get(key()) is None
    cache.put(key(), answer())
    assert cache.get(key()) is not None
    assert cache.hit_rate == 0.5


def test_an_expired_entry_is_a_miss() -> None:
    now = [1000.0]
    cache = ExactCache(ttl_seconds=60, clock=lambda: now[0])
    cache.put(key(), answer())
    now[0] += 61
    assert cache.get(key()) is None
    assert cache.expired == 1


def test_the_cache_is_bounded() -> None:
    now = [0.0]
    cache = ExactCache(max_entries=2, clock=lambda: now[0])
    for index in range(3):
        now[0] += 1
        cache.put(f"k{index}", answer())
    assert len(cache.entries) == 2
    assert "k0" not in cache.entries


def test_invalidation_by_prefix() -> None:
    cache = ExactCache()
    cache.put("aa1", answer())
    cache.put("aa2", answer())
    cache.put("bb1", answer())
    assert cache.invalidate("aa") == 2
    assert set(cache.entries) == {"bb1"}


# --- routing --------------------------------------------------


class Fixed:
    def __init__(self, text: str, output_tokens: int = 20) -> None:
        self.text = text
        self.output_tokens = output_tokens
        self.calls = 0

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls += 1
        return Completion(
            self.text, 4400, self.output_tokens, "m", "stop"
        )


def router(cheap_text: str) -> tuple[ModelRouter, Fixed, Fixed]:
    small = Fixed(cheap_text)
    large = Fixed("Refunds are issued within fourteen days.")
    route = ModelRouter(
        tiers=[
            Tier("small", small, CHEAP),
            Tier("large", large, STRONG),
        ],
        should_escalate=any_of(
            escalate_on_refusal(),
            escalate_on_short_answer(minimum=3),
        ),
    )
    return route, small, large


def test_a_good_cheap_answer_is_not_escalated() -> None:
    route, small, large = router("Refunds take fourteen days.")
    result = route.complete(ASK)
    assert result.tier == "small"
    assert result.escalated is False
    assert large.calls == 0


def test_a_refusal_escalates() -> None:
    route, small, large = router("I don't know.")
    result = route.complete(ASK)
    assert result.tier == "large"
    assert result.escalated is True
    assert result.attempts == ("small", "large")
    assert large.calls == 1


def test_an_escalated_call_costs_both_tiers() -> None:
    route, _, _ = router("I don't know.")
    result = route.complete(ASK)
    cheap = (4400 * 0.15 + 20 * 0.60) / 1_000_000
    strong = (4400 * 3.00 + 20 * 15.00) / 1_000_000
    assert result.cost_usd == pytest.approx(cheap + strong)


def test_the_escalation_rate_is_tracked() -> None:
    route, _, _ = router("I don't know.")
    route.complete(ASK)
    good, _, _ = router("Refunds take fourteen days.")
    good.complete(ASK)
    assert route.escalation_rate == 1.0
    assert good.escalation_rate == 0.0


def test_a_router_needs_a_tier() -> None:
    with pytest.raises(ValueError, match="at least one tier"):
        ModelRouter(tiers=[], should_escalate=lambda c: False)


# --- quotas ---------------------------------------------------


def test_a_tenant_is_refused_past_its_limit() -> None:
    quota = TenantQuota(daily_limit_usd=1.00)
    quota.record("acme", 0.99)
    with pytest.raises(QuotaExceeded, match="acme"):
        quota.check("acme", 0.05)


def test_tenants_have_separate_allowances() -> None:
    quota = TenantQuota(daily_limit_usd=1.00)
    quota.record("acme", 0.99)
    quota.check("globex", 0.50)
    assert quota.remaining("globex") == pytest.approx(1.00)


def test_the_allowance_resets_the_next_day() -> None:
    now = [1_700_000_000.0]
    quota = TenantQuota(
        daily_limit_usd=1.00, clock=lambda: now[0]
    )
    quota.record("acme", 0.99)
    now[0] += 86_400
    assert quota.spent_today("acme") == 0.0
    quota.check("acme", 0.50)


def test_the_kill_switch_stops_everything() -> None:
    quota = TenantQuota(daily_limit_usd=100.0)
    quota.disable()
    with pytest.raises(ServiceDisabled):
        quota.check("acme", 0.01)
    quota.enable()
    quota.check("acme", 0.01)


# --- attribution ----------------------------------------------


def test_cost_is_attributed_three_ways() -> None:
    ledger = Attribution()
    ledger.record(
        feature="rag", tenant="acme", model="large", cost_usd=0.02
    )
    ledger.record(
        feature="rag", tenant="globex", model="small",
        cost_usd=0.001,
    )
    ledger.record(
        feature="triage", tenant="acme", model="small",
        cost_usd=0.002,
    )
    assert ledger.total_usd == pytest.approx(0.023)
    assert ledger.top("feature")[0] == ("rag", pytest.approx(0.021))
    assert ledger.top("tenant")[0][0] == "acme"
    assert ledger.mean_usd == pytest.approx(0.023 / 3)
