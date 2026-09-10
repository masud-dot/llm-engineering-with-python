"""Work that outlives an HTTP request."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from llmapp.api.schemas import AskResponse


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    """One unit of background work and its outcome."""

    job_id: str
    principal: str
    state: JobState = JobState.QUEUED
    result: AskResponse | None = None
    error: str = ""


@dataclass
class JobStore:
    """In-process jobs. Chapter 25 replaces the storage."""

    jobs: dict[str, Job] = field(default_factory=dict)

    def create(self, principal: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], principal=principal)
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str, principal: str) -> Job | None:
        """A job belongs to the principal that created it."""
        job = self.jobs.get(job_id)
        if job is None or job.principal != principal:
            return None
        return job

    def run(self, job_id: str, work: Callable[[], AskResponse]) -> None:
        job = self.jobs[job_id]
        job.state = JobState.RUNNING
        try:
            job.result = work()
        except Exception as exc:  # reported, never raised here
            job.state = JobState.FAILED
            job.error = type(exc).__name__
            return
        job.state = JobState.DONE
