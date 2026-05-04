"""Async retry handler with exponential backoff."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def retry_handler(callable, agent_name, max_retries=3, base_delay=2):
    total_attempts = 1 + max_retries
    last_exception = None
    for attempt in range(total_attempts):
        try:
            result = await callable()
            return result
        except Exception as e:
            last_exception = e
            if attempt == total_attempts - 1:
                raise RuntimeError(
                    f"{agent_name} failed after {total_attempts} attempts: {last_exception}"
                ) from last_exception
            delay = base_delay * (2 ** attempt)
            retry_number = attempt + 1
            logger.warning(
                "Attempt %d/%d for %s — retrying in %.1fs...",
                retry_number + 1, max_retries, agent_name, delay,
            )
            await asyncio.sleep(delay)
