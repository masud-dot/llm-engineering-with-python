"""Structured logging with a correlation ID and redaction."""

import re
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

_correlation_id: ContextVar[str] = ContextVar(
    "correlation_id", default=""
)

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),
)


def new_correlation_id() -> str:
    """Start a new request scope and return its id."""
    value = uuid.uuid4().hex[:12]
    _correlation_id.set(value)
    return value


def correlation_id() -> str:
    return _correlation_id.get()


def redact(text: str) -> str:
    """Remove obvious secrets and personal data."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def add_correlation_id(
    _logger: Any,
    _name: str,
    event: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    value = correlation_id()
    if value:
        event["correlation_id"] = value
    return event


def configure_logging() -> None:
    """JSON logs with a correlation id on every line."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            add_correlation_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
