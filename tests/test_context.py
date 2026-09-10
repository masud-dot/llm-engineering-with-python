import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llmapp.prompts.context import (
    BudgetExceeded,
    ContextBudget,
    Section,
    count_tokens,
    truncate_to,
)

BUDGET = ContextBudget(window_tokens=200, reserved_output=50)


def section(name: str, words: int, **kw: object) -> Section:
    return Section(
        name=name,
        content=" ".join([name] * words),
        **kw,  # type: ignore[arg-type]
    )


def test_everything_fits_when_there_is_room() -> None:
    result = BUDGET.fit([section("a", 5), section("b", 5)])
    assert result.dropped == []
    assert [s.name for s in result.sections] == ["a", "b"]


def test_lowest_priority_is_dropped_first() -> None:
    result = BUDGET.fit(
        [
            section("keep", 40, priority=9),
            section("drop", 200, priority=1),
        ]
    )
    assert result.dropped == ["drop"]
    assert [s.name for s in result.sections] == ["keep"]


def test_truncatable_sections_are_trimmed_not_dropped() -> None:
    result = BUDGET.fit(
        [
            section("instr", 10, priority=9, required=True),
            section("docs", 400, priority=1, truncatable=True),
        ]
    )
    assert result.truncated == ["docs"]
    assert result.dropped == []
    assert result.tokens_used <= BUDGET.available


def test_impossible_required_sections_raise() -> None:
    with pytest.raises(BudgetExceeded, match="required"):
        BUDGET.fit([section("huge", 500, required=True)])


def test_original_order_is_restored() -> None:
    result = BUDGET.fit(
        [
            section("first", 5, priority=1),
            section("second", 5, priority=9),
        ]
    )
    assert [s.name for s in result.sections] == ["first", "second"]


def test_truncate_never_exceeds_the_limit() -> None:
    text = "policy " * 300
    for limit in (0, 1, 7, 50, 5000):
        assert count_tokens(truncate_to(text, limit)) <= limit


@settings(max_examples=40, deadline=None)
@given(
    sizes=st.lists(
        st.integers(min_value=1, max_value=120),
        min_size=1,
        max_size=6,
    )
)
def test_budget_is_never_exceeded(sizes: list[int]) -> None:
    sections = [
        section(f"s{i}", n, priority=i, truncatable=i % 2 == 0)
        for i, n in enumerate(sizes)
    ]
    result = BUDGET.fit(sections)
    assert result.tokens_used <= BUDGET.available
