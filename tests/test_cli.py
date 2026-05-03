"""Tests for the CLI application (interactive loop + integration)."""

from unittest.mock import AsyncMock, patch

import pytest

from blog_mas.mcp.models import BlogDraft
from tests.conftest import make_mock_llm


class TestCliStartup:
    def test_prints_welcome_message_with_kb_topics(self, capsys):
        from blog_mas.cli import print_welcome

        print_welcome()
        output = capsys.readouterr().out
        assert "Blog" in output
        assert "Mediterranean diet" in output
        assert "Artificial intelligence" in output
        assert "Climate change" in output
        assert "Space exploration" in output
        assert "Mental health" in output


class TestInputValidation:
    def test_rejects_empty_input(self, capsys):
        from blog_mas.cli import validate_input

        assert validate_input("") is None
        assert "Please provide a blog topic" in capsys.readouterr().out

    def test_rejects_whitespace_only_input(self, capsys):
        from blog_mas.cli import validate_input

        assert validate_input("   ") is None
        assert "Please provide a blog topic" in capsys.readouterr().out

    def test_rejects_input_over_500_characters(self, capsys):
        from blog_mas.cli import validate_input

        long_input = "x" * 501
        assert validate_input(long_input) is None
        output = capsys.readouterr().out
        assert "500" in output

    def test_accepts_valid_input(self):
        from blog_mas.cli import validate_input

        assert validate_input("Write about the Mediterranean diet") == (
            "Write about the Mediterranean diet"
        )


class TestCliIntegration:
    @pytest.mark.asyncio
    async def test_displays_blog_post_with_markers(self, capsys):
        from blog_mas.cli import display_result

        draft = BlogDraft(title="Test", body="Great content.", word_count=2)
        display_result({"success": True, "draft": draft})
        output = capsys.readouterr().out
        assert "--- BLOG POST ---" in output
        assert "--- END ---" in output
        assert "Great content." in output

    @pytest.mark.asyncio
    async def test_displays_error_on_pipeline_failure(self, capsys):
        from blog_mas.cli import display_result

        display_result({"success": False, "error": "LLM timeout"})
        output = capsys.readouterr().out
        assert "LLM timeout" in output
        assert "Traceback" not in output

    @pytest.mark.asyncio
    async def test_runs_full_pipeline_on_valid_input(self):
        """Valid input → LangGraph pipeline → returns result."""
        from blog_mas.cli import process_request

        mock_llm = make_mock_llm(BlogDraft(title="AI Post", body="Content", word_count=1))

        with patch("blog_mas.cli.run_pipeline_async", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = {
                "success": True,
                "draft": BlogDraft(title="AI Post", body="Content", word_count=1),
            }

            result = await process_request("Write about AI", llm=mock_llm)
            assert result["success"] is True
            mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_pipeline_failure_gracefully(self, capsys):
        from blog_mas.cli import process_request

        mock_llm = make_mock_llm(BlogDraft(title="", body="", word_count=0))

        with patch("blog_mas.cli.run_pipeline_async", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = {
                "success": False,
                "error": "ConnectionError: timeout",
            }

            result = await process_request("Write about AI", llm=mock_llm)
            assert result["success"] is False
            assert "ConnectionError" in result["error"]


class TestLoopBehavior:
    def test_exits_on_quit(self):
        from blog_mas.cli import should_exit

        assert should_exit("quit") is True

    def test_exits_on_exit(self):
        from blog_mas.cli import should_exit

        assert should_exit("exit") is True

    def test_does_not_exit_on_normal_input(self):
        from blog_mas.cli import should_exit

        assert should_exit("Write about AI") is False
