"""Tests for async retry handler with exponential backoff."""

import asyncio

import pytest

from blog_mas.retry import retry_handler


class TestRetryHandlerSuccess:
    @pytest.mark.asyncio
    async def test_returns_result_immediately_on_first_success(self):
        async def succeed():
            return "ok"

        result = await retry_handler(succeed, "TestAgent")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_works_with_async_callables(self):
        async def fetch_data():
            return {"status": "success", "data": [1, 2, 3]}

        result = await retry_handler(fetch_data, "DataAgent")
        assert result == {"status": "success", "data": [1, 2, 3]}


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retries_once_and_succeeds_on_second_attempt(self):
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection lost")
            return "recovered"

        result = await retry_handler(
            fail_then_succeed, "TestAgent", base_delay=0.01
        )
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_with_correct_exponential_delays(self):
        delays = []

        async def always_fail():
            raise RuntimeError("fail")

        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            delays.append(delay)

        asyncio.sleep = mock_sleep
        try:
            with pytest.raises(Exception):
                await retry_handler(
                    always_fail, "TestAgent", max_retries=3, base_delay=0.01
                )
        finally:
            asyncio.sleep = original_sleep

        # max_retries=3 means 3 retries (4 total attempts), 3 delays
        # delay = base_delay * 2^attempt (0-indexed): 0.01, 0.02, 0.04
        assert len(delays) == 3
        assert delays[0] == pytest.approx(0.01)
        assert delays[1] == pytest.approx(0.02)
        assert delays[2] == pytest.approx(0.04)

    @pytest.mark.asyncio
    async def test_raises_exception_after_all_retries_exhausted(self):
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent failure")

        with pytest.raises(Exception) as exc_info:
            await retry_handler(
                always_fail, "TestAgent", max_retries=3, base_delay=0.01
            )

        assert "TestAgent" in str(exc_info.value)
        assert "permanent failure" in str(exc_info.value)
        # 1 initial + 3 retries = 4 total attempts
        assert call_count == 4


class TestErrorTypes:
    @pytest.mark.asyncio
    async def test_handles_network_style_connection_error(self):
        call_count = 0

        async def fail_connection():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("connection refused")
            return "connected"

        result = await retry_handler(
            fail_connection, "NetAgent", max_retries=3, base_delay=0.01
        )
        assert result == "connected"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_handles_network_style_timeout_error(self):
        call_count = 0

        async def fail_timeout():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise TimeoutError("request timed out")
            return "completed"

        result = await retry_handler(
            fail_timeout, "NetAgent", base_delay=0.01
        )
        assert result == "completed"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_handles_pydantic_validation_error(self):
        from pydantic import BaseModel, ValidationError

        class StrictModel(BaseModel):
            value: int

        call_count = 0

        async def fail_validation():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                StrictModel(value="not_an_int")
            return "valid"

        result = await retry_handler(
            fail_validation, "ValidationAgent", base_delay=0.01
        )
        assert result == "valid"
        assert call_count == 2


class TestRetryOutput:
    @pytest.mark.asyncio
    async def test_logs_retry_status_on_each_attempt(self, caplog):
        import logging

        call_count = 0

        async def fail_once_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network down")
            return "ok"

        with caplog.at_level(logging.WARNING):
            await retry_handler(
                fail_once_then_succeed, "TestAgent", base_delay=0.01
            )

        assert any("Attempt 2/3 for TestAgent" in record.message for record in caplog.records)
        assert any("retrying in" in record.message for record in caplog.records)
