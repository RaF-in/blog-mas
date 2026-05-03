"""Tests for MCP message protocol (factory, envelope validator) and Pydantic content models."""

from pydantic import ValidationError
import pytest

from blog_mas.mcp.protocol import create_mcp_message, validate_mcp_envelope
from blog_mas.mcp.models import (
    BlogSpec,
    ResearchRequest,
    ResearchSummary,
    WriterInput,
    BlogDraft,
    ValidationInput,
    ValidationVerdict,
)


# --- MCP Message Factory ---


class TestCreateMcpMessage:
    def test_creates_valid_envelope_with_all_four_fields(self):
        result = create_mcp_message(sender="TestAgent", content={"key": "value"})
        assert result["protocol_version"] == "1.0"
        assert result["sender"] == "TestAgent"
        assert result["content"] == {"key": "value"}
        assert result["metadata"] == {}

    def test_defaults_metadata_to_empty_dict(self):
        result = create_mcp_message(sender="TestAgent", content="hello")
        assert result["metadata"] == {}

    def test_protocol_version_always_1_0(self):
        result = create_mcp_message(sender="A", content=None)
        assert result["protocol_version"] == "1.0"

    def test_no_mutable_default_aliasing_across_calls(self):
        r1 = create_mcp_message(sender="A", content="x")
        r2 = create_mcp_message(sender="B", content="y")
        r1["metadata"]["foo"] = "bar"
        assert r2["metadata"] == {}

    def test_accepts_custom_metadata(self):
        result = create_mcp_message(
            sender="A", content="x", metadata={"task_id": "abc"}
        )
        assert result["metadata"] == {"task_id": "abc"}


# --- MCP Envelope Validator ---


class TestValidateMcpEnvelope:
    def test_returns_true_for_valid_envelope(self):
        msg = create_mcp_message(sender="Agent", content={})
        assert validate_mcp_envelope(msg) is True

    def test_returns_false_when_message_is_none(self, capsys):
        assert validate_mcp_envelope(None) is False
        assert "Message is not a dictionary" in capsys.readouterr().out

    def test_returns_false_when_message_is_string(self, capsys):
        assert validate_mcp_envelope("not a dict") is False
        assert "Message is not a dictionary" in capsys.readouterr().out

    def test_returns_false_when_protocol_version_missing(self, capsys):
        msg = {"sender": "A", "content": {}, "metadata": {}}
        assert validate_mcp_envelope(msg) is False
        assert "Missing key" in capsys.readouterr().out

    def test_returns_false_when_sender_missing(self, capsys):
        msg = {"protocol_version": "1.0", "content": {}, "metadata": {}}
        assert validate_mcp_envelope(msg) is False
        assert "Missing key" in capsys.readouterr().out

    def test_returns_false_when_content_missing(self, capsys):
        msg = {"protocol_version": "1.0", "sender": "A", "metadata": {}}
        assert validate_mcp_envelope(msg) is False
        assert "Missing key" in capsys.readouterr().out

    def test_returns_false_when_metadata_missing(self, capsys):
        msg = {"protocol_version": "1.0", "sender": "A", "content": {}}
        assert validate_mcp_envelope(msg) is False
        assert "Missing key" in capsys.readouterr().out

    def test_returns_false_when_sender_empty_string(self, capsys):
        msg = {"protocol_version": "1.0", "sender": "", "content": {}, "metadata": {}}
        assert validate_mcp_envelope(msg) is False
        assert "Empty sender field" in capsys.readouterr().out


# --- Pydantic Content Models ---


class TestBlogSpec:
    def test_creates_with_all_fields(self):
        spec = BlogSpec(
            topic="AI",
            audience="tech",
            tone="neutral",
            goal="inform",
            constraints=["no jargon"],
        )
        assert spec.topic == "AI"
        assert spec.audience == "tech"
        assert spec.tone == "neutral"
        assert spec.goal == "inform"
        assert spec.constraints == ["no jargon"]

    def test_rejects_missing_topic(self):
        with pytest.raises(ValidationError):
            BlogSpec(audience="tech", tone="neutral", goal="inform", constraints=[])


class TestResearchRequest:
    def test_creates_with_required_fields(self):
        req = ResearchRequest(topic="AI", audience="tech", goal="inform")
        assert req.topic == "AI"
        assert req.audience == "tech"
        assert req.goal == "inform"


class TestResearchSummary:
    def test_creates_with_bullet_points_and_source(self):
        summary = ResearchSummary(
            topic="AI", bullet_points=["point1", "point2"], source="kb"
        )
        assert summary.topic == "AI"
        assert summary.bullet_points == ["point1", "point2"]
        assert summary.source == "kb"


class TestWriterInput:
    def test_defaults_revision_feedback_to_none(self):
        wi = WriterInput(
            research_summary=ResearchSummary(
                topic="AI", bullet_points=["p1"], source="kb"
            ),
            blog_spec=BlogSpec(
                topic="AI",
                audience="general readers",
                tone="informative",
                goal="educate",
                constraints=[],
            ),
        )
        assert wi.revision_feedback is None

    def test_accepts_optional_revision_feedback(self):
        wi = WriterInput(
            research_summary=ResearchSummary(
                topic="AI", bullet_points=["p1"], source="kb"
            ),
            blog_spec=BlogSpec(
                topic="AI",
                audience="general readers",
                tone="informative",
                goal="educate",
                constraints=[],
            ),
            revision_feedback="fix claims",
        )
        assert wi.revision_feedback == "fix claims"


class TestBlogDraft:
    def test_creates_with_title_body_word_count(self):
        draft = BlogDraft(title="Test", body="Content here", word_count=2)
        assert draft.title == "Test"
        assert draft.body == "Content here"
        assert draft.word_count == 2


class TestValidationInput:
    def test_creates_with_research_summary_and_draft(self):
        summary = ResearchSummary(
            topic="AI", bullet_points=["p1"], source="kb"
        )
        draft = BlogDraft(title="T", body="B", word_count=1)
        vi = ValidationInput(research_summary=summary, draft=draft)
        assert vi.research_summary == summary
        assert vi.draft == draft


class TestValidationVerdict:
    def test_accepts_verdict_pass(self):
        v = ValidationVerdict(verdict="pass", reason="all good")
        assert v.verdict == "pass"

    def test_accepts_verdict_fail(self):
        v = ValidationVerdict(verdict="fail", reason="unsupported claim")
        assert v.verdict == "fail"

    def test_rejects_other_verdict_values(self):
        with pytest.raises(ValidationError):
            ValidationVerdict(verdict="maybe", reason="unsure")
