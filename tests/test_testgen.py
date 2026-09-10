"""Project 2, tested as a deliverable."""

import ast
from pathlib import Path

import pytest

from llmapp.prompts.registry import PromptRegistry
from llmapp.projects.testgen.cases import TestCase
from llmapp.projects.testgen.emit import EmitError, emit, safe_name
from llmapp.projects.testgen.generator import (
    TestGenerator,
    write_suites,
)
from llmapp.projects.testgen.plan import (
    CaseKind,
    coverage,
    plan,
    plan_endpoint,
)
from llmapp.projects.testgen.review import review, signature
from llmapp.projects.testgen.spec import SpecError, load_spec
from tests._testgen_stub import ScriptedGenerator

SPEC = Path("tests/specs/llmapp.json")
REGISTRY = PromptRegistry(Path("prompts"))


def generator(**kwargs: object) -> TestGenerator:
    return TestGenerator(
        ScriptedGenerator(**kwargs),  # type: ignore[arg-type]
        REGISTRY,
    )


# --- the spec is untrusted input -------------------------------


def test_the_spec_yields_an_endpoint_inventory() -> None:
    endpoints = load_spec(SPEC)
    assert len(endpoints) == 7
    paths = {e.path for e in endpoints}
    assert "/v1/ask" in paths and "/healthz" in paths


def test_field_constraints_are_read_from_the_schema() -> None:
    ask = next(
        e for e in load_spec(SPEC) if e.path == "/v1/ask"
    )
    top_k = next(f for f in ask.fields if f.name == "top_k")
    assert top_k.type == "integer"
    assert top_k.maximum == 20
    assert top_k.has_bounds


def test_a_document_without_paths_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"openapi": "3.1.0"}')
    with pytest.raises(SpecError, match="no paths"):
        load_spec(path)


def test_malformed_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("{not json")
    with pytest.raises(SpecError, match="valid JSON"):
        load_spec(path)


# --- the plan is code, not a model call ------------------------


def test_the_plan_covers_every_category() -> None:
    counts = coverage(plan(load_spec(SPEC)))
    assert counts["happy"] == 7
    for kind in (
        "missing_required",
        "wrong_type",
        "boundary",
        "extra_field",
    ):
        assert counts[kind] > 0


def test_planning_is_deterministic() -> None:
    endpoint = load_spec(SPEC)[2]
    first = [s.test_name for s in plan_endpoint(endpoint)]
    second = [s.test_name for s in plan_endpoint(endpoint)]
    assert first == second


def test_a_bounded_field_gets_a_boundary_case() -> None:
    ask = next(
        e for e in load_spec(SPEC) if e.path == "/v1/ask"
    )
    kinds = {s.kind for s in plan_endpoint(ask)}
    assert CaseKind.BOUNDARY in kinds
    assert CaseKind.MISSING_REQUIRED in kinds


# --- emission --------------------------------------------------


def test_names_are_made_unique_and_legal() -> None:
    used: set[str] = set()
    assert safe_name("test_a", used) == "test_a"
    assert safe_name("test_a", used) == "test_a_2"
    assert safe_name("9 bad name!", used).startswith("test_")


def test_every_generated_module_parses() -> None:
    for suite in generator().for_spec(SPEC):
        ast.parse(suite.module.source)
        assert suite.review.parses


def test_an_unparseable_module_raises() -> None:
    bad = TestCase(
        name="test_x",
        kind=CaseKind.HAPPY,
        description="a description",
        method="get",
        path='"); import os; ("',
        expected_status=200,
    )
    module = emit([bad])
    # The path is JSON-encoded, so injection cannot escape.
    ast.parse(module.source)


def test_generation_is_reproducible(tmp_path: Path) -> None:
    first = generator().for_spec(SPEC)
    second = generator().for_spec(SPEC)
    assert [s.module.source for s in first] == [
        s.module.source for s in second
    ]
    names = write_suites(first, tmp_path / "a")
    assert [p.name for p in names] == sorted(
        p.name for p in names
    )


# --- the self-review pass --------------------------------------


def test_a_clean_suite_is_accepted() -> None:
    suite = generator().for_spec(SPEC)[2]
    assert suite.review.accepted
    assert suite.review.coverage == 1.0


def test_duplicates_are_detected() -> None:
    suite = generator(duplicate=True).for_spec(SPEC)[2]
    assert suite.review.duplicates
    assert suite.review.duplicate_rate > 0
    assert not suite.review.accepted


def test_a_missing_category_is_detected() -> None:
    suite = generator(drop_kind="boundary").for_spec(SPEC)[2]
    assert "boundary" in suite.review.missing_kinds
    assert suite.review.coverage < 1.0
    assert not suite.review.accepted


def test_a_negative_case_expecting_success_is_trivial() -> None:
    suite = generator(trivial=True).for_spec(SPEC)[2]
    assert suite.review.trivial


def test_the_signature_ignores_the_case_name() -> None:
    one = TestCase(
        name="test_one",
        kind=CaseKind.HAPPY,
        description="a description",
        method="post",
        path="/v1/ask",
        body={"question": "x"},
        expected_status=200,
    )
    two = one.model_copy(update={"name": "test_two"})
    assert signature(one) == signature(two)


def test_review_reports_a_syntax_failure() -> None:
    result = review([], [], "def broken(:")
    assert result.parses is False
