import pytest

from llmapp.llm.budget import BudgetExceeded, SpendGuard


def test_allows_calls_under_the_ceiling() -> None:
    guard = SpendGuard(limit_usd=1.0)
    guard.check(0.25)
    guard.record(0.25)
    assert guard.remaining_usd == pytest.approx(0.75)


def test_blocks_a_call_that_would_breach() -> None:
    guard = SpendGuard(limit_usd=1.0, spent_usd=0.99)
    with pytest.raises(BudgetExceeded, match="0.9900"):
        guard.check(0.05)


def test_rejects_negative_projection() -> None:
    with pytest.raises(ValueError):
        SpendGuard(limit_usd=1.0).check(-0.1)
