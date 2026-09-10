import pytest

from llmapp.security.output import (
    UrlPolicy,
    check_links,
    escape_for_html,
    find_links,
    strip_disallowed_links,
)
from llmapp.security.redact import Redactor
from llmapp.security.sql import SqlPolicy, UnsafeQuery

POLICY = SqlPolicy(
    allowed_tables=frozenset({"orders", "customers"}),
    max_rows=50,
)
LINKS = UrlPolicy(allowed_hosts=frozenset({"example.com"}))


# --- redaction ------------------------------------------------


def test_credential_shapes_are_removed() -> None:
    r = Redactor()
    out = r.scrub("key sk-proj-abcdefghijklmnopqrstuvwx here")
    assert "sk-proj" not in out
    assert "[redacted:openai_key]" in out
    assert r.found["openai_key"] == 1


def test_several_categories_are_counted() -> None:
    r = Redactor()
    r.scrub("a@b.com and 4111 1111 1111 1111 and AKIAABCDEFGHIJKLMNOP")
    assert set(r.found) == {"email", "card", "aws_key"}
    assert r.clean is False


def test_clean_text_is_unchanged() -> None:
    r = Redactor()
    text = "The refund window is fourteen days."
    assert r.scrub(text) == text
    assert r.clean is True


def test_redaction_is_a_floor_not_a_boundary() -> None:
    """A customer's name is not a shape, so it survives."""
    r = Redactor()
    out = r.scrub("The complaint was filed by Priya Raman.")
    assert "Priya Raman" in out
    assert r.clean is True


# --- output handling ------------------------------------------


def test_generated_markup_is_escaped() -> None:
    hostile = '<img src=x onerror="steal()">'
    assert "<img" not in escape_for_html(hostile)


def test_links_are_found_in_markdown() -> None:
    text = "See [docs](https://example.com/a) and [x](http://q.io)"
    assert find_links(text) == [
        "https://example.com/a",
        "http://q.io",
    ]


def test_a_disallowed_host_is_reported() -> None:
    text = "[exfil](https://attacker.test/?d=secret)"
    assert check_links(text, LINKS) == [
        "https://attacker.test/?d=secret"
    ]


def test_plain_http_is_refused_even_on_an_allowed_host() -> None:
    assert not LINKS.permits("http://example.com/a")


def test_a_subdomain_of_an_allowed_host_is_permitted() -> None:
    assert LINKS.permits("https://docs.example.com/a")


def test_a_lookalike_host_is_refused() -> None:
    assert not LINKS.permits("https://example.com.attacker.test/")


def test_stripping_keeps_the_label_and_drops_the_target(
) -> None:
    text = "See [report](https://attacker.test/?d=secret) now"
    cleaned = strip_disallowed_links(text, LINKS)
    assert "attacker.test" not in cleaned
    assert "report" in cleaned


# --- sql gate -------------------------------------------------


def test_a_plain_select_is_allowed_and_bounded() -> None:
    assert POLICY.check("SELECT id FROM orders") == (
        "SELECT id FROM orders LIMIT 50"
    )


def test_an_existing_limit_is_respected() -> None:
    assert POLICY.check(
        "SELECT id FROM orders LIMIT 5"
    ).endswith("LIMIT 5")


def test_a_write_is_refused() -> None:
    with pytest.raises(UnsafeQuery, match="SELECT"):
        POLICY.check("DELETE FROM orders")


def test_a_second_statement_is_refused() -> None:
    with pytest.raises(UnsafeQuery, match="one statement"):
        POLICY.check("SELECT 1 FROM orders; DROP TABLE orders")


def test_a_comment_cannot_hide_a_write() -> None:
    with pytest.raises(UnsafeQuery):
        POLICY.check(
            "SELECT id FROM orders /* x */ ; DELETE FROM orders"
        )


def test_an_unlisted_table_is_refused() -> None:
    with pytest.raises(UnsafeQuery, match="salaries"):
        POLICY.check("SELECT * FROM salaries")


def test_a_join_to_an_unlisted_table_is_refused() -> None:
    with pytest.raises(UnsafeQuery, match="salaries"):
        POLICY.check(
            "SELECT o.id FROM orders o JOIN salaries s "
            "ON s.id = o.id"
        )


def test_a_cte_is_permitted_when_its_tables_are() -> None:
    query = (
        "WITH recent AS (SELECT id FROM orders) "
        "SELECT id FROM recent"
    )
    assert POLICY.check(query).endswith("LIMIT 50")


def test_an_empty_query_is_refused() -> None:
    with pytest.raises(UnsafeQuery, match="empty"):
        POLICY.check("   ")
