"""A scripted SQL generator, keyed on the question."""
import json

from llmapp.llm.base import Completion

REPLIES = {
    "shipped orders": {
        "answerable": True,
        "sql": "SELECT id, total_usd FROM shop.orders "
               "WHERE status = 'shipped'",
        "explanation": "Selects shipped orders and their totals.",
        "missing": "",
    },
    "business customers": {
        "answerable": True,
        "sql": "SELECT c.name, count(o.id) AS orders "
               "FROM shop.customers c "
               "JOIN shop.orders o ON o.customer_id = c.id "
               "WHERE c.tier = 'business' GROUP BY c.name",
        "explanation": "Counts orders per business customer.",
        "missing": "",
    },
    "salaries": {
        "answerable": True,
        "sql": "SELECT * FROM shop.salaries",
        "explanation": "Reads the salary table.",
        "missing": "",
    },
    "delete": {
        "answerable": True,
        "sql": "DELETE FROM shop.orders",
        "explanation": "Removes every order.",
        "missing": "",
    },
    "two statements": {
        "answerable": True,
        "sql": "SELECT 1 FROM shop.orders; DROP TABLE shop.orders",
        "explanation": "Two statements.",
        "missing": "",
    },
    "weather": {
        "answerable": False,
        "sql": "",
        "explanation": "",
        "missing": "the schema has no weather data",
    },
    "invented column": {
        "answerable": True,
        "sql": "SELECT margin_pct FROM shop.orders",
        "explanation": "Reads a column that does not exist.",
        "missing": "",
    },
}


class ScriptedSql:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_output_tokens=512):
        self.calls += 1
        body = messages[-1].content.lower()
        for marker, reply in REPLIES.items():
            if marker in body:
                return Completion(
                    json.dumps(reply), 400, 60, "stub", "stop")
        return Completion(
            json.dumps({"answerable": False, "sql": "",
                        "explanation": "",
                        "missing": "no scripted reply"}),
            400, 30, "stub", "stop")
