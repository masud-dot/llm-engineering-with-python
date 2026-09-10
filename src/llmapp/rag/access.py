"""Retrieval that cannot be asked to ignore permissions."""

from dataclasses import dataclass, field

RESERVED_FILTERS = ("visibility", "team")


class AccessDenied(PermissionError):
    """The caller tried to widen their own permissions."""


@dataclass(frozen=True)
class Principal:
    """Who is asking, and what they may see."""

    user_id: str
    team: str
    clearance: str = "internal"


@dataclass(frozen=True)
class AccessPolicy:
    """Turn a principal into a filter the caller cannot undo."""

    public_only: frozenset[str] = field(
        default_factory=lambda: frozenset({"public"})
    )

    def filter_for(self, principal: Principal) -> dict[str, str]:
        """The minimum restriction every query must carry."""
        return {"team": principal.team}

    def merge(
        self,
        principal: Principal,
        requested: dict[str, str] | None,
    ) -> dict[str, str]:
        """Caller filters may narrow, never widen."""
        enforced = self.filter_for(principal)
        merged = dict(requested or {})
        for key, value in enforced.items():
            if key in merged and merged[key] != value:
                raise AccessDenied(
                    f"cannot override the {key!r} restriction"
                )
            merged[key] = value
        return merged
