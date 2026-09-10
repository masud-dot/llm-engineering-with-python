"""A provider-neutral error taxonomy.

Every adapter translates its SDK's exceptions into these
three classes, so callers never import a vendor error.
"""


class LLMError(Exception):
    """Base class for every failure in the model layer."""


class TransientError(LLMError):
    """The call may succeed if repeated. Retry."""

    def __init__(
        self, message: str, *, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class PermanentError(LLMError):
    """The call will never succeed as written. Do not retry."""


class SemanticError(LLMError):
    """The call succeeded; the output is unusable."""


class CircuitOpen(TransientError):
    """The breaker is open; the call was not attempted."""
