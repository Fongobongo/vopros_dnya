from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BASE = 1.0
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_RETRY_MAX = 20.0
DEFAULT_RETRY_JITTER = 0.2


@dataclass(frozen=True)
class RetryConfig:
    attempts: int
    base_sleep: float
    backoff: float
    max_sleep: float
    jitter: float
    retryable_statuses: set[int]


def _coerce_int(raw: str | None, default: int, min_value: int = 1) -> int:
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        return default
    return max(min_value, value)


def _coerce_float(raw: str | None, default: float, min_value: float = 0.0) -> float:
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return max(min_value, value)


def load_retry_config(prefix: str = "REQUEST", fallback_prefix: str | None = None) -> RetryConfig:
    def _get(key: str) -> str | None:
        value = os.getenv(f"{prefix}_{key}")
        if value is None and fallback_prefix:
            value = os.getenv(f"{fallback_prefix}_{key}")
        return value

    attempts = _coerce_int(_get("RETRIES"), DEFAULT_RETRIES)
    base_sleep = _coerce_float(_get("RETRY_BASE"), DEFAULT_RETRY_BASE)
    backoff = _coerce_float(_get("RETRY_BACKOFF"), DEFAULT_RETRY_BACKOFF, min_value=1.0)
    max_sleep = _coerce_float(_get("RETRY_MAX"), DEFAULT_RETRY_MAX)
    jitter = _coerce_float(_get("RETRY_JITTER"), DEFAULT_RETRY_JITTER)
    return RetryConfig(
        attempts=attempts,
        base_sleep=base_sleep,
        backoff=backoff,
        max_sleep=max_sleep,
        jitter=jitter,
        retryable_statuses=set(RETRYABLE_STATUS_CODES),
    )


def should_retry(exc: Exception, retryable_statuses: Iterable[int]) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in set(retryable_statuses)
    if isinstance(exc, URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    return False


def _compute_delay(attempt_index: int, config: RetryConfig) -> float:
    delay = config.base_sleep * (config.backoff ** attempt_index)
    delay = min(delay, config.max_sleep)
    if config.jitter > 0:
        spread = delay * config.jitter
        delay = delay + random.uniform(-spread, spread)
        delay = max(0.0, delay)
    return delay


def run_with_retry(
    func: Callable[[], object],
    config: RetryConfig,
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> object:
    last_exc: Exception | None = None
    for attempt in range(config.attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= config.attempts - 1:
                raise
            if not should_retry(exc, config.retryable_statuses):
                raise
            delay = _compute_delay(attempt, config)
            if on_retry:
                on_retry(attempt + 1, config.attempts, delay, exc)
            if delay > 0:
                time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry handler failed without exception")
