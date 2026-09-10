"""The HTTP surface for the reference application."""

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse

from llmapp.api.jobs import JobState, JobStore
from llmapp.api.limits import RateLimited, SlidingWindow
from llmapp.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    FeedbackRequest,
    Health,
    JobAccepted,
    JobStatus,
    SqlRequest,
    SqlResponse,
)
from llmapp.api.security import AuthError, TokenIssuer, bearer_token
from llmapp.obs.feedback import Feedback, FeedbackStore, Verdict
from llmapp.obs.metrics import Metrics
from llmapp.obs.tracing import request_span
from llmapp.rag.access import AccessDenied, Principal
from llmapp.rag.pipeline import RagPipeline


@dataclass
class Services:
    """Everything a request needs, constructed once."""

    pipeline: RagPipeline
    issuer: TokenIssuer
    limiter: SlidingWindow
    sql: object | None = None
    jobs: JobStore = field(default_factory=JobStore)
    feedback: FeedbackStore = field(default_factory=FeedbackStore)
    metrics: Metrics = field(default_factory=Metrics)
    ready: bool = True


def build_app(services: Services) -> FastAPI:
    """One app, with its dependencies supplied not imported."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Warm caches and open pools here; close them after.
        yield

    app = FastAPI(
        title="llmapp",
        version="1.0.0",
        lifespan=lifespan,
    )

    def principal_from(token: str = Depends(bearer_token)) -> Principal:
        return services.issuer.verify(token)

    Caller = Annotated[Principal, Depends(principal_from)]

    def request_id() -> str:
        return uuid.uuid4().hex[:12]

    RequestId = Annotated[str, Depends(request_id)]

    def answer_for(
        payload: AskRequest, caller: Principal, rid: str
    ) -> AskResponse:
        services.limiter.check(caller.user_id)
        services.metrics.count("requests")
        with request_span(
            feature="ask",
            principal=caller.user_id,
            correlation_id=rid,
        ):
            result = services.pipeline.answer(
                payload.question,
                where=payload.filters or None,
                principal=caller,
            )
        if not result.answer.answered:
            services.metrics.count("abstained")
        return AskResponse(
            request_id=rid,
            answered=result.answer.answered,
            answer=result.answer.answer,
            citations=list(result.answer.citations),
            missing=result.answer.missing,
            sources=result.sources,
        )

    @app.exception_handler(AccessDenied)
    async def on_access_denied(
        request: Request, exc: AccessDenied
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=ErrorResponse(
                error="forbidden", detail=str(exc)
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def on_unexpected(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # The type name is safe; the message may not be.
        services.metrics.count("errors")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                detail="the request could not be completed",
            ).model_dump(),
        )

    @app.get("/healthz", response_model=Health)
    def healthz() -> Health:
        """Liveness: is the process up?"""
        return Health(status="ok")

    @app.get("/readyz", response_model=Health)
    def readyz() -> Health:
        """Readiness: can it serve a request right now?"""
        checks = {
            "index": (
                "ok" if services.pipeline.index_size else "empty"
            ),
            "dependencies": "ok" if services.ready else "down",
        }
        healthy = all(v == "ok" for v in checks.values())
        if not healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="not ready",
            )
        return Health(status="ok", checks=checks)

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(
        payload: AskRequest, caller: Caller, rid: RequestId
    ) -> AskResponse:
        return answer_for(payload, caller, rid)

    @app.post("/v1/ask/stream")
    def ask_stream(
        payload: AskRequest, caller: Caller, rid: RequestId
    ) -> StreamingResponse:
        """Stream prose. Structured fields arrive first."""
        response = answer_for(payload, caller, rid)

        def chunks() -> Iterator[str]:
            yield f"event: meta\ndata: {response.request_id}\n\n"
            for word in response.answer.split(" "):
                yield f"event: token\ndata: {word} \n\n"
            yield "event: done\ndata: \n\n"

        return StreamingResponse(
            chunks(), media_type="text/event-stream"
        )

    @app.post(
        "/v1/ask/async",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def ask_async(
        payload: AskRequest,
        caller: Caller,
        rid: RequestId,
        background: BackgroundTasks,
    ) -> JobAccepted:
        job = services.jobs.create(caller.user_id)
        background.add_task(
            services.jobs.run,
            job.job_id,
            lambda: answer_for(payload, caller, rid),
        )
        return JobAccepted(job_id=job.job_id, status=job.state.value)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str, caller: Caller) -> JobStatus:
        job = services.jobs.get(job_id, caller.user_id)
        if job is None:
            # Not found and not yours are the same response.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job not found",
            )
        return JobStatus(
            job_id=job.job_id,
            status=job.state.value,
            result=job.result,
            error=job.error,
        )

    @app.post("/v1/sql", response_model=SqlResponse)
    def sql_query(
        payload: SqlRequest, caller: Caller, rid: RequestId
    ) -> SqlResponse:
        """Natural language to a validated read-only query."""
        if services.sql is None:
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail="the database assistant is not enabled",
            )
        services.limiter.check(caller.user_id)
        services.metrics.count("requests")
        with request_span(
            feature="sql",
            principal=caller.user_id,
            correlation_id=rid,
        ):
            result = services.sql.ask(  # type: ignore[attr-defined]
                payload.question
            )
        if not result.answered:
            services.metrics.count("abstained")
        return SqlResponse(
            request_id=rid,
            answered=result.answered,
            sql=result.sql,
            explanation=result.explanation,
            columns=list(result.columns),
            rows=[[str(v) for v in row] for row in result.rows],
            row_count=result.row_count,
            refused_because=result.refused_because
            or result.missing,
        )

    @app.post(
        "/v1/feedback", status_code=status.HTTP_204_NO_CONTENT
    )
    def feedback(
        payload: FeedbackRequest, caller: Caller
    ) -> None:
        services.feedback.add(
            Feedback(
                request_id=payload.request_id,
                question="",
                verdict=Verdict(payload.verdict),
                principal=caller.user_id,
                comment=payload.comment,
            )
        )

    return app
