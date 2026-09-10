"""A read-only SQL gate for a model-generated query.

The safe design is not to let a model write arbitrary SQL.
Where a query tool is unavoidable, this narrows what can
reach the database: one statement, read-only, on approved
tables, with a bounded row count. The database role is the
real boundary; this is the layer above it.
"""

import re
from dataclasses import dataclass

COMMENT = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)
FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "merge",
    "vacuum",
)
TABLE_REF = re.compile(
    r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
    re.IGNORECASE,
)
# Names defined by the query itself are not base tables.
CTE_NAME = re.compile(
    r"(?:\bwith\s+|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(",
    re.IGNORECASE,
)


class UnsafeQuery(ValueError):
    """The generated query will not be executed."""


@dataclass(frozen=True)
class SqlPolicy:
    """What a generated query is permitted to do."""

    allowed_tables: frozenset[str]
    max_rows: int = 100

    def check(self, sql: str) -> str:
        """Return a safe query, or raise explaining why not."""
        stripped = COMMENT.sub(" ", sql).strip().rstrip(";")
        if not stripped:
            raise UnsafeQuery("the query is empty")
        if ";" in stripped:
            raise UnsafeQuery(
                "only one statement may be executed"
            )
        lowered = stripped.lower()
        if not (
            lowered.startswith("select")
            or lowered.startswith("with")
        ):
            raise UnsafeQuery(
                "only SELECT statements are permitted"
            )
        for word in FORBIDDEN:
            if re.search(rf"\b{word}\b", lowered):
                raise UnsafeQuery(
                    f"the keyword {word!r} is not permitted"
                )
        referenced = {
            name.lower() for name in TABLE_REF.findall(stripped)
        }
        defined = {
            name.lower() for name in CTE_NAME.findall(stripped)
        }
        unknown = referenced - self.allowed_tables - defined
        if unknown:
            raise UnsafeQuery(
                f"table(s) not permitted: "
                f"{', '.join(sorted(unknown))}"
            )
        if not referenced - defined:
            raise UnsafeQuery("no permitted table was referenced")
        return self._limit(stripped)

    def _limit(self, sql: str) -> str:
        if re.search(r"\blimit\s+\d+\s*$", sql, re.IGNORECASE):
            return sql
        return f"{sql} LIMIT {self.max_rows}"
