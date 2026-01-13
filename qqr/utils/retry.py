import logging
import time
from functools import wraps
from typing import Any, Callable

from .envs import RETRY_STOP_AFTER_ATTEMPT, RETRY_WAIT_FIXED

logger = logging.getLogger(__name__)


def retry(
    stop_after_attempt: int = RETRY_STOP_AFTER_ATTEMPT,
    wait_fixed: float = RETRY_WAIT_FIXED,
    retry_if_result: Callable[[Any], bool] = None,
    retry_if_not_result: Callable[[Any], bool] = None,
):
    """
    Decorator that retries a function until a specified condition is met or until stop_after_attempt is reached.

    Args:
        stop_after_attempt: Maximum number of attempts (default: 3)
        wait_fixed: Delay in seconds between attempts (default: 1.0)
        retry_if_result: A callable that takes the function's result and retries if the condition is met
        retry_if_not_result: A callable that takes the function's result and retries if the condition is not met
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(stop_after_attempt):
                retry_msg = []
                try:
                    retry_msg.append(
                        f"Retrying attempt {attempt + 1}/{stop_after_attempt} failed for function '{func.__name__}'"
                    )

                    result = func(*args, **kwargs)

                    retry_msg.append(f"Result: {result}")
                    if retry_if_result is not None and retry_if_result(result):
                        retry_msg.append("Result met `retry_if_result` condition")
                        continue
                    elif retry_if_not_result is not None and not retry_if_not_result(
                        result
                    ):
                        retry_msg.append("Result met `retry_if_not_result` condition")
                        continue
                    else:
                        return result

                except Exception as e:
                    retry_msg += [f"[{type(e).__name__}]: {e}"]

                retry_msg = "\n".join(retry_msg)
                logger.warning(retry_msg)
                time.sleep(wait_fixed)

        return wrapper

    return decorator
