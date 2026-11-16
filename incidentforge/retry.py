"""Retry logic with exponential backoff for IncidentForge."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from .client import LLMResponse

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryableError(Exception):
    """Base class for retryable errors."""

    def __init__(
        self, message: str, status_code: int | None = None, retry_after: float | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """Rate limit error (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429, retry_after=retry_after)


class ServerError(RetryableError):
    """Server error (HTTP 5xx)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message, status_code=status_code)


class TimeoutError(RetryableError):
    """Timeout error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=408)


class NonRetryableError(Exception):
    """Non-retryable error (e.g., 401, 403, 400)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def is_retryable_exception(exc: BaseException) -> bool:
    """Check if exception is retryable."""
    if isinstance(exc, RetryableError):
        return True
    if isinstance(exc, NonRetryableError):
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 429} or 500 <= status_code < 600
    return False


def extract_retry_after(exc: BaseException) -> float | None:
    """Extract Retry-After header from exception."""
    if isinstance(exc, RetryableError) and exc.retry_after is not None:
        return exc.retry_after
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


class _WaitWithRetryAfter(wait_base):
    """Wait using Retry-After when available, else exponential backoff with jitter."""

    def __init__(self, base_wait: float, max_wait: float, jitter: float) -> None:
        self._max_wait = max_wait
        self._fallback = wait_exponential_jitter(initial=base_wait, max=max_wait, jitter=jitter)

    def __call__(self, retry_state: RetryCallState) -> float:
        if retry_state.outcome:
            exc = retry_state.outcome.exception()
            if exc is not None:
                retry_after = extract_retry_after(exc)
                if retry_after is not None:
                    return max(0.0, min(float(retry_after), self._max_wait))
        return float(self._fallback(retry_state))


def before_sleep_log_retry(retry_state: RetryCallState) -> None:
    """Log before sleep with retry info."""
    if retry_state.outcome:
        exc = retry_state.outcome.exception()
        if exc is not None:
            retry_after = extract_retry_after(exc)
            attempt = retry_state.attempt_number
            if retry_after is not None:
                logger.warning(
                    "Retry attempt %d: %s. Waiting %.1fs (Retry-After header)",
                    attempt,
                    exc,
                    retry_after,
                )
            else:
                logger.warning(
                    "Retry attempt %d: %s. Waiting with exponential backoff",
                    attempt,
                    exc,
                )


def create_retry_policy(
    max_attempts: int = 3,
    base_wait: float = 1.0,
    max_wait: float = 60.0,
    jitter: float = 0.5,
) -> AsyncRetrying:
    """
    Create retry policy with exponential backoff, jitter, and Retry-After support.
    """
    return AsyncRetrying(
        retry=retry_if_exception(is_retryable_exception),
        stop=stop_after_attempt(max_attempts),
        wait=_WaitWithRetryAfter(base_wait=base_wait, max=max_wait, jitter=jitter)
        if False
        else _WaitWithRetryAfter(base_wait=base_wait, max_wait=max_wait, jitter=jitter),
        before_sleep=before_sleep_log_retry,
        reraise=True,
    )


async def with_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 3,
    base_wait: float = 1.0,
    max_wait: float = 60.0,
    jitter: float = 0.5,
    **kwargs: Any,
) -> T:
    """Execute async function with retry policy."""
    retryer = create_retry_policy(
        max_attempts=max_attempts,
        base_wait=base_wait,
        max_wait=max_wait,
        jitter=jitter,
    )
    return await retryer(func, *args, **kwargs)


def classify_http_error(
    status_code: int, message: str, headers: dict[str, Any] | None = None
) -> Exception:
    """
    Classify HTTP error into appropriate exception type.

    Args:
        status_code: HTTP status code.
        message: Error message.
        headers: Response headers.

    Returns:
        Appropriate exception instance.
    """
    headers = headers or {}
    retry_after = headers.get("Retry-After")

    if status_code == 429:
        retry_after_float = None
        if retry_after:
            try:
                retry_after_float = float(retry_after)
            except ValueError:
                pass
        return RateLimitError(message, retry_after=retry_after_float)

    if 500 <= status_code < 600:
        return ServerError(message, status_code)

    if status_code == 408:
        return TimeoutError(message)

    # Non-retryable errors
    return NonRetryableError(message, status_code)


async def handle_llm_response(response: LLMResponse) -> LLMResponse:
    """
    Handle LLM response, raising appropriate exceptions for errors.

    This is a placeholder for handling errors from the LLM response itself
    (e.g., if the API returns an error in the response body).
    """
    # The actual HTTP errors are handled at the client level
    # This function can be extended to handle response-level errors
    return response


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_wait: float = 1.0,
        max_wait: float = 60.0,
        jitter: float = 0.5,
        max_validation_retries: int = 2,
    ):
        self.max_attempts = max_attempts
        self.base_wait = base_wait
        self.max_wait = max_wait
        self.jitter = jitter
        self.max_validation_retries = max_validation_retries

    def create_retryer(self) -> AsyncRetrying:
        """Create retryer with this config."""
        return create_retry_policy(
            max_attempts=self.max_attempts,
            base_wait=self.base_wait,
            max_wait=self.max_wait,
            jitter=self.jitter,
        )


DEFAULT_RETRY_CONFIG = RetryConfig()
