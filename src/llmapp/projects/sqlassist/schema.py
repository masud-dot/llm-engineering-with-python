"""Read the schema the assistant is allowed to describe.

Only the tables the read-only role can see are introspected,
so the schema description handed to the model cannot mention
a table the query could never reach.
"""

from dataclasses import dataclass
from typing import Any

SCHEMA_SQL = """
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = %s
  AND table_name = ANY(%s)
ORDER BY table_name, ordinal_position
"""


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]

    def describe(self) -> str:
        body = ", ".join(
            f"{c.name} {c.type}" for c in self.columns
        )
        return f"{self.name}({body})"


@dataclass(frozen=True)
class Schema:
    """What the model is told the database contains."""

    namespace: str
    tables: tuple[Table, ...]

    @property
    def table_names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tables)

    def describe(self) -> str:
        return "\n".join(t.describe() for t in self.tables)


def load_schema(
    connection: Any, namespace: str, allowed: list[str]
) -> Schema:
    """Introspect only the tables the role may read."""
    rows = connection.execute(
        SCHEMA_SQL, (namespace, allowed)
    ).fetchall()
    grouped: dict[str, list[Column]] = {}
    for table, column, data_type in rows:
        grouped.setdefault(table, []).append(
            Column(name=column, type=data_type)
        )
    return Schema(
        namespace=namespace,
        tables=tuple(
            Table(name=name, columns=tuple(columns))
            for name, columns in sorted(grouped.items())
        ),
    )
