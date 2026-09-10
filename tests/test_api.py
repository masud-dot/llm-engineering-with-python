import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmapp.api.limits import SlidingWindow
from llmapp.api.main import Services, build_app
from llmapp.api.security import TokenIssuer
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.access import AccessPolicy
from llmapp.rag.pipeline import RagPipeline, build_index_from
from llmapp.retrieval.search import SemanticIndex
from tests._fake_embedder import KeywordEmbedder
from tests.test_eval import Answers

CORPUS = Path("tests/corpus")
REGISTRY = PromptRegistry(Path("prompts"))
# PyJWT warns below 32 bytes for HS256; see Section 24.6.
SECRET = "test-secret-not-a-real-one-32-bytes-min"

GOOD = {
    "refunds": {
        "answered": True,
        "answer": "Refunds are issued within fourteen days.",
        "citations": [
            {"chunk_id": "refunds.md#0", "quote": "fourteen days"}
        ],
        "missing": "",
    }
}


def make_services(limit: int = 100) -> Services:
    index = SemanticIndex(KeywordEmbedder())
    build_index_from(
        CORPUS, index, max_words=60,
        metadata={"team": "billing"},
    )
    pipeline = RagPipeline(
        Answers(GOOD), index, REGISTRY, policy=AccessPolicy()
    )
    return Services(
        pipeline=pipeline,
        issuer=TokenIssuer(secret=SECRET),
        limiter=SlidingWindow(limit=limit),
    )


@pytest.fixture
def services() -> Services:
    return make_services()


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(build_app(services)) as test_client:
        yield test_client


def auth(services: Services, team: str = "billing") -> dict[str, str]:
    token = services.issuer.issue(
        user_id="u1", team=team, tenant="acme"
    )
    return {"Authorization": f"Bearer {token}"}


# --- health ---------------------------------------------------


def test_liveness_needs_no_token(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_checks_dependencies(
    client: TestClient, services: Services
) -> None:
    assert client.get("/readyz").status_code == 200
    services.ready = False
    assert client.get("/readyz").status_code == 503


# --- authentication -------------------------------------------


def test_a_missing_token_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": "hi"})
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_a_malformed_scheme_is_rejected(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/ask",
        json={"question": "hi"},
        headers={"Authorization": "Basic abc"},
    )
    assert response.status_code == 401


def test_a_forged_token_is_rejected(client: TestClient) -> None:
    other = TokenIssuer(secret="a-different-secret")
    token = other.issue(user_id="u9", team="billing", tenant="x")
    response = client.post(
        "/v1/ask",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_an_expired_token_is_rejected(
    client: TestClient, services: Services
) -> None:
    expired = TokenIssuer(secret=SECRET, ttl_seconds=-10)
    token = expired.issue(
        user_id="u1", team="billing", tenant="acme"
    )
    response = client.post(
        "/v1/ask",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


# --- the contract ---------------------------------------------


def test_a_valid_request_is_answered(
    client: TestClient, services: Services
) -> None:
    response = client.post(
        "/v1/ask",
        json={"question": "refunds fourteen days"},
        headers=auth(services),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert "fourteen days" in body["answer"]
    assert "refunds.md" in body["sources"]
    assert len(body["request_id"]) == 12


def test_an_unknown_field_is_refused(
    client: TestClient, services: Services
) -> None:
    response = client.post(
        "/v1/ask",
        json={"question": "hi", "temperature": 0.9},
        headers=auth(services),
    )
    assert response.status_code == 422


def test_an_out_of_range_value_is_refused(
    client: TestClient, services: Services
) -> None:
    response = client.post(
        "/v1/ask",
        json={"question": "hi", "top_k": 500},
        headers=auth(services),
    )
    assert response.status_code == 422


def test_the_response_carries_no_internals(
    client: TestClient, services: Services
) -> None:
    body = client.post(
        "/v1/ask",
        json={"question": "refunds fourteen days"},
        headers=auth(services),
    ).json()
    for leaked in ("prompt_id", "hits", "dropped", "grounding"):
        assert leaked not in body


# --- authorization --------------------------------------------


def test_a_caller_cannot_widen_their_own_filter(
    client: TestClient, services: Services
) -> None:
    response = client.post(
        "/v1/ask",
        json={
            "question": "delivery",
            "filters": {"team": "shipping"},
        },
        headers=auth(services, team="billing"),
    )
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


# --- rate limiting --------------------------------------------


def test_the_rate_limit_returns_429_with_retry_after() -> None:
    services = make_services(limit=2)
    with TestClient(build_app(services)) as client:
        headers = auth(services)
        payload = {"question": "refunds fourteen days"}
        for _ in range(2):
            assert (
                client.post(
                    "/v1/ask", json=payload, headers=headers
                ).status_code
                == 200
            )
        limited = client.post(
            "/v1/ask", json=payload, headers=headers
        )
        assert limited.status_code == 429
        assert int(limited.headers["Retry-After"]) >= 1


# --- streaming ------------------------------------------------


def test_streaming_emits_events(
    client: TestClient, services: Services
) -> None:
    with client.stream(
        "POST",
        "/v1/ask/stream",
        json={"question": "refunds fourteen days"},
        headers=auth(services),
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers[
            "content-type"
        ]
        body = "".join(response.iter_text())
    assert "event: meta" in body
    assert "event: token" in body
    assert body.rstrip().endswith("event: done\ndata:")


# --- background jobs ------------------------------------------


def test_a_job_is_accepted_and_completes(
    client: TestClient, services: Services
) -> None:
    accepted = client.post(
        "/v1/ask/async",
        json={"question": "refunds fourteen days"},
        headers=auth(services),
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    status_response = client.get(
        f"/v1/jobs/{job_id}", headers=auth(services)
    )
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "done"
    assert body["result"]["answered"] is True


def test_another_users_job_is_not_found(
    client: TestClient, services: Services
) -> None:
    accepted = client.post(
        "/v1/ask/async",
        json={"question": "refunds fourteen days"},
        headers=auth(services),
    )
    job_id = accepted.json()["job_id"]
    other = services.issuer.issue(
        user_id="u2", team="billing", tenant="acme"
    )
    response = client.get(
        f"/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {other}"},
    )
    assert response.status_code == 404


# --- feedback and schema --------------------------------------


def test_feedback_is_recorded(
    client: TestClient, services: Services
) -> None:
    response = client.post(
        "/v1/feedback",
        json={"request_id": "r1", "verdict": "wrong_fact"},
        headers=auth(services),
    )
    assert response.status_code == 204
    assert len(services.feedback.negative) == 1


def test_an_invalid_verdict_is_refused(
    client: TestClient, services: Services
) -> None:
    response = client.post(
        "/v1/feedback",
        json={"request_id": "r1", "verdict": "excellent"},
        headers=auth(services),
    )
    assert response.status_code == 422


def test_the_openapi_schema_is_published(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/ask" in schema["paths"]
    assert "/v1/jobs/{job_id}" in schema["paths"]
