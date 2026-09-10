"""A spend ceiling that fails before money is spent."""

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised when a call would breach the spend ceiling."""


@dataclass
class SpendGuard:
    """Track spend against a ceiling for one process."""

    limit_usd: float
    spent_usd: float = field(default=0.0)

    def check(self, projected_usd: float) -> None:
        """Raise if this call would breach the ceiling."""
        if projected_usd < 0:
            raise ValueError("projected cost cannot be negative")
        if self.spent_usd + projected_usd > self.limit_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of "
                f"${self.limit_usd:.2f}; this call adds "
                f"${projected_usd:.4f}"
            )

    def record(self, actual_usd: float) -> None:
        """Add a completed call to the running total."""
        self.spent_usd += actual_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)
