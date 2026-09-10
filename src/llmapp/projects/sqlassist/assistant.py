"""Project 3: the AI SQL Assistant, assembled."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmapp.llm.base import LLMClient
from llmapp.prompts.registry import PromptRegistry
from llmapp.projects.sqlassist.schema import Schema
from llmapp.schemas.validate import parse_or_repair
from llmapp.security.sql import SqlPolicy, UnsafeQuery

MAX_ROWS = 50


class GeneratedQuery(BaseModel):
    """What the model must return."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    sql: str = Field(default="", max_length=2000)
    explanation: str = Field(default="", max_length=400)
    missing: str = Field(default="", max_length=200)


@dataclass(frozen=True)
class QueryResult:
    """The answer, and everything needed to audit it."""

    answered: bool
    sql: str = ""
    explanation: str = ""
    missing: str = ""
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    plan_cost: float | None = None
    refused_because: str = ""

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class SqlAssistant:
    """Generate, gate, dry-run, then execute as a reader."""

    client: LLMClient
    registry: PromptRegistry
    schema: Schema
    connection: Any
    max_rows: int = MAX_ROWS
    max_plan_cost: float = 100_000.0
    policy: SqlPolicy = field(init=False)

    def __post_init__(self) -> None:
        qualified = {
            f"{self.schema.namespace}.{name}"
            for name in self.schema.table_names
        }
        self.policy = SqlPolicy(
            allowed_tables=frozenset(
                qualified | self.schema.table_names
            ),
            max_rows=self.max_rows,
        )

    def ask(self, question: str) -> QueryResult:
        generated = self._generate(question)
        if not generated.answerable:
            return QueryResult(
                answered=False, missing=generated.missing
            )
        try:
            sql = self.policy.check(generated.sql)
        except UnsafeQuery as exc:
            return QueryResult(
                answered=False,
                sql=generated.sql,
                refused_because=str(exc),
            )
        try:
            cost = self._dry_run(sql)
        except Exception as exc:
            return QueryResult(
                answered=False,
                sql=sql,
                refused_because=f"the query would not plan: {exc}",
            )
        if cost > self.max_plan_cost:
            return QueryResult(
                answered=False,
                sql=sql,
                plan_cost=cost,
                refused_because=(
                    f"estimated cost {cost:.0f} exceeds the "
                    f"limit of {self.max_plan_cost:.0f}"
                ),
            )
        columns, rows = self._execute(sql)
        return QueryResult(
            answered=True,
            sql=sql,
            explanation=generated.explanation,
            columns=columns,
            rows=rows,
            plan_cost=cost,
        )

    def _generate(self, question: str) -> GeneratedQuery:
        prompt = self.registry.get("sql_assist", 1).render(
            question=question,
            schema=self.schema.describe(),
            namespace=self.schema.namespace,
            limit=str(self.max_rows),
        )
        parsed, _ = parse_or_repair(
            self.client, prompt.messages, GeneratedQuery
        )
        return parsed

    def _dry_run(self, sql: str) -> float:
        """Plan the query without running it."""
        rows = self.connection.execute(
            f"EXPLAIN (FORMAT JSON) {sql}"
        ).fetchone()
        plan = rows[0][0]["Plan"]
        return float(plan["Total Cost"])

    def _execute(
        self, sql: str
    ) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
        cursor = self.connection.execute(sql)
        columns = tuple(
            description[0] for description in cursor.description
        )
        rows = tuple(tuple(row) for row in cursor.fetchall())
        return columns, rows
