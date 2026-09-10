"""Per-tenant spend limits and a kill switch."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


class QuotaExceeded(RuntimeError):
    """This tenant has spent its allowance for the period."""


class ServiceDisabled(RuntimeError):
    """The kill switch is engaged."""


def utc_day(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")


@dataclass
class TenantQuota:
    """Spend per tenant per day, with a global kill switch.

    This is a single-process implementation. A deployment
    with several workers needs a shared counter, which is why
    the storage is behind two small methods.
    """

    daily_limit_usd: float
    clock: Callable[[], float] = time.time
    enabled: bool = True
    spend: dict[tuple[str, str], float] = field(
        default_factory=dict
    )

    def _key(self, tenant: str) -> tuple[str, str]:
        return (tenant, utc_day(self.clock()))

    def spent_today(self, tenant: str) -> float:
        return self.spend.get(self._key(tenant), 0.0)

    def remaining(self, tenant: str) -> float:
        return max(
            0.0, self.daily_limit_usd - self.spent_today(tenant)
        )

    def check(self, tenant: str, projected_usd: float) -> None:
        """Refuse before spending, not after."""
        if not self.enabled:
            raise ServiceDisabled("the kill switch is engaged")
        if projected_usd < 0:
            raise ValueError("projected cost cannot be negative")
        if self.spent_today(tenant) + projected_usd > (
            self.daily_limit_usd
        ):
            raise QuotaExceeded(
                f"{tenant} has spent "
                f"${self.spent_today(tenant):.4f} of "
                f"${self.daily_limit_usd:.2f} today"
            )

    def record(self, tenant: str, actual_usd: float) -> None:
        key = self._key(tenant)
        self.spend[key] = self.spend.get(key, 0.0) + actual_usd

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True


@dataclass
class Attribution:
    """Cost broken down the way finance will ask for it."""

    by_feature: dict[str, float] = field(default_factory=dict)
    by_tenant: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)
    requests: int = 0

    def record(
        self,
        *,
        feature: str,
        tenant: str,
        model: str,
        cost_usd: float,
    ) -> None:
        self.requests += 1
        self.by_feature[feature] = (
            self.by_feature.get(feature, 0.0) + cost_usd
        )
        self.by_tenant[tenant] = (
            self.by_tenant.get(tenant, 0.0) + cost_usd
        )
        self.by_model[model] = (
            self.by_model.get(model, 0.0) + cost_usd
        )

    @property
    def total_usd(self) -> float:
        return sum(self.by_feature.values())

    @property
    def mean_usd(self) -> float:
        return (
            self.total_usd / self.requests if self.requests else 0.0
        )

    def top(self, kind: str = "feature", n: int = 3) -> list[
        tuple[str, float]
    ]:
        source = {
            "feature": self.by_feature,
            "tenant": self.by_tenant,
            "model": self.by_model,
        }[kind]
        return sorted(
            source.items(), key=lambda kv: -kv[1]
        )[:n]
