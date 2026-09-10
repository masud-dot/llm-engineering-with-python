"""Projects 3 and 5, tested against a real database."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from llmapp.prompts.registry import PromptRegistry
from llmapp.projects.sqlassist.assistant import SqlAssistant
from llmapp.projects.sqlassist.schema import load_schema
from llmapp.security.sql import UnsafeQuery
from tests._sql_stub import ScriptedSql

READER_DSN = os.environ.get("LLMAPP_TEST_READER_DSN", "")
try:  # the driver is an optional extra
    import psycopg  # noqa: F401
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False
REGISTRY = PromptRegistry(Path("prompts"))

pytestmark = pytest.mark.skipif(
    not (READER_DSN and _HAS_PSYCOPG),
    reason="needs LLMAPP_TEST_READER_DSN and the postgres extra"
)


@pytest.fixture
def assistant() -> Iterator[SqlAssistant]:
    import psycopg

    with psycopg.connect(READER_DSN, autocommit=True) as conn:
        schema = load_schema(conn, "shop", ["customers", "orders"])
        yield SqlAssistant(
            ScriptedSql(), REGISTRY, schema, conn
        )


# --- schema introspection --------------------------------------


@pytest.mark.local_server
def test_only_permitted_tables_are_described(
    assistant: SqlAssistant,
) -> None:
    """The role cannot read salaries, so the model is not
    told the table exists."""
    assert assistant.schema.table_names == {
        "customers",
        "orders",
    }
    assert "salaries" not in assistant.schema.describe()


@pytest.mark.local_server
def test_column_types_reach_the_description(
    assistant: SqlAssistant,
) -> None:
    described = assistant.schema.describe()
    assert "total_usd numeric" in described
    assert "placed_on date" in described


# --- the happy path --------------------------------------------


@pytest.mark.local_server
def test_a_legitimate_question_returns_rows(
    assistant: SqlAssistant,
) -> None:
    result = assistant.ask("shipped orders")
    assert result.answered
    assert result.row_count == 3
    assert result.columns == ("id", "total_usd")
    assert result.sql.rstrip().endswith("LIMIT 50")


@pytest.mark.local_server
def test_a_join_is_permitted_within_the_schema(
    assistant: SqlAssistant,
) -> None:
    result = assistant.ask("business customers")
    assert result.answered
    assert result.row_count == 2
    assert result.plan_cost is not None


@pytest.mark.local_server
def test_a_row_limit_is_always_applied(
    assistant: SqlAssistant,
) -> None:
    result = assistant.ask("shipped orders")
    assert "LIMIT" in result.sql.upper()


# --- the gate ---------------------------------------------------


@pytest.mark.local_server
def test_an_unlisted_table_is_refused(
    assistant: SqlAssistant,
) -> None:
    result = assistant.ask("salaries")
    assert result.answered is False
    assert "not permitted" in result.refused_because


@pytest.mark.local_server
def test_a_write_is_refused(assistant: SqlAssistant) -> None:
    result = assistant.ask("delete everything")
    assert result.answered is False
    assert "SELECT" in result.refused_because


@pytest.mark.local_server
def test_a_second_statement_is_refused(
    assistant: SqlAssistant,
) -> None:
    result = assistant.ask("two statements")
    assert result.answered is False
    assert "one statement" in result.refused_because


@pytest.mark.local_server
def test_an_unanswerable_question_is_refused_by_the_model(
    assistant: SqlAssistant,
) -> None:
    result = assistant.ask("weather")
    assert result.answered is False
    assert "weather" in result.missing


@pytest.mark.local_server
def test_an_invented_column_is_caught_by_the_dry_run(
    assistant: SqlAssistant,
) -> None:
    """The policy accepts it; the planner does not."""
    result = assistant.ask("invented column")
    assert result.answered is False
    assert "would not plan" in result.refused_because
    assert "margin_pct" in result.refused_because


# --- the role is the real boundary ------------------------------


@pytest.mark.local_server
def test_the_role_refuses_writes_even_without_the_gate() -> None:
    """Defense in depth: if every application control were
    bypassed, the database still refuses."""
    import psycopg

    with psycopg.connect(READER_DSN, autocommit=True) as conn:
        for sql in (
            "DELETE FROM shop.orders",
            "UPDATE shop.orders SET total_usd = 0",
            "SELECT * FROM shop.salaries",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(sql)


# --- the assembled service --------------------------------------


@pytest.mark.local_server
def test_the_service_exposes_the_sql_endpoint(
    assistant: SqlAssistant,
) -> None:
    from fastapi.testclient import TestClient

    from llmapp.api.main import build_app
    from tests.test_api import auth, make_services

    services = make_services()
    services.sql = assistant
    with TestClient(build_app(services)) as client:
        response = client.post(
            "/v1/sql",
            json={"question": "shipped orders"},
            headers=auth(services),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answered"] is True
        assert body["row_count"] == 3
        assert body["columns"] == ["id", "total_usd"]


@pytest.mark.local_server
def test_a_refusal_reaches_the_caller_without_internals(
    assistant: SqlAssistant,
) -> None:
    from fastapi.testclient import TestClient

    from llmapp.api.main import build_app
    from tests.test_api import auth, make_services

    services = make_services()
    services.sql = assistant
    with TestClient(build_app(services)) as client:
        body = client.post(
            "/v1/sql",
            json={"question": "delete everything"},
            headers=auth(services),
        ).json()
        assert body["answered"] is False
        assert "SELECT" in body["refused_because"]
        assert "plan_cost" not in body


def test_the_endpoint_is_disabled_when_unconfigured() -> None:
    from fastapi.testclient import TestClient

    from llmapp.api.main import build_app
    from tests.test_api import auth, make_services

    services = make_services()
    with TestClient(build_app(services)) as client:
        response = client.post(
            "/v1/sql",
            json={"question": "anything"},
            headers=auth(services),
        )
        assert response.status_code == 503
