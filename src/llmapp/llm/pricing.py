"""Cost arithmetic. Rates come from configuration."""

from dataclasses import dataclass

from llmapp.llm.base import Completion

PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Rates:
    """Price per million tokens, read from settings."""

    input_per_mtok: float
    output_per_mtok: float


def cost_usd(completion: Completion, rates: Rates) -> float:
    """Cost of one completed call."""
    return (
        completion.input_tokens * rates.input_per_mtok
        + completion.output_tokens * rates.output_per_mtok
    ) / PER_MILLION


def estimate_usd(
    input_tokens: int, max_output_tokens: int, rates: Rates
) -> float:
    """Worst-case cost, for checking a budget before calling."""
    return (
        input_tokens * rates.input_per_mtok
        + max_output_tokens * rates.output_per_mtok
    ) / PER_MILLION
