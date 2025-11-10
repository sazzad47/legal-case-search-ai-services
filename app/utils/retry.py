import asyncio
import time
from typing import Callable, Any, Awaitable


async def retry_async(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 4.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Simple async retry with exponential backoff.

    - Retries the given async function up to `retries` times on exception.
    - Backoff starts at `initial_delay` and multiplies by `backoff_factor`.
    - Caps delay at `max_delay`.
    """
    attempt = 0
    delay = initial_delay
    last_exc = None
    while attempt <= retries:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
            attempt += 1


def retry_sync(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 4.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Simple sync retry with exponential backoff for non-async functions."""
    attempt = 0
    delay = initial_delay
    last_exc = None
    while attempt <= retries:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
            attempt += 1