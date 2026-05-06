"""Tests for rag/blueprints.py — schema validation, injection scan, neutral default."""

import json
import logging

import pytest

from blog_mas.rag.blueprints import (
    NEUTRAL_BLUEPRINT,
    Blueprint,
    Participant,
    validate_blueprint_payload,
)


def _valid_blueprint_dict(**overrides) -> dict:
    base = {
        "id": "bp-test",
        "description": "A test blueprint.",
        "scene_goal": "Test scene goal.",
        "style_guide": "Test style guide.",
        "participants": [{"name": "Alice", "role": "editor"}],
        "instruction": "Write clearly and concisely.",
        "metadata": {"version": 1},
    }
    base.update(overrides)
    return base


def _valid_blueprint_json(**overrides) -> str:
    return json.dumps(_valid_blueprint_dict(**overrides))


# ── Schema validation ──────────────────────────────────────────────────


class TestSchemaValidation:
    def test_accepts_well_formed_json(self):
        result = validate_blueprint_payload(_valid_blueprint_json())
        assert isinstance(result, Blueprint)
        assert result.id == "bp-test"

    def test_rejects_exceeding_8kb_serialized(self):
        payload = _valid_blueprint_json(instruction="x" * 10_000)
        result = validate_blueprint_payload(payload)
        assert result is None

    @pytest.mark.parametrize("field", ["description", "scene_goal", "style_guide", "instruction"])
    def test_rejects_per_field_length_violation(self, field):
        payload = _valid_blueprint_json(**{field: "a" * 5000})
        result = validate_blueprint_payload(payload)
        assert result is None

    def test_strips_whitespace_on_string_fields(self):
        payload = _valid_blueprint_json(description="  padded  ")
        result = validate_blueprint_payload(payload)
        assert result is not None
        assert result.description == "padded"

    def test_bounds_participants_list_count(self):
        many = [{"name": f"P{i}", "role": "role"} for i in range(15)]
        payload = _valid_blueprint_json(participants=many)
        result = validate_blueprint_payload(payload)
        assert result is None

    def test_bounds_participant_item_length(self):
        payload = _valid_blueprint_json(
            participants=[{"name": "x" * 200, "role": "ok"}]
        )
        result = validate_blueprint_payload(payload)
        assert result is None

    def test_rejects_metadata_with_non_primitive_values(self):
        payload = _valid_blueprint_json(metadata={"nested": {"a": 1}})
        result = validate_blueprint_payload(payload)
        assert result is None


# ── Injection-marker scan ──────────────────────────────────────────────


class TestInjectionScan:
    @pytest.mark.parametrize("marker", ["{{", "}}", "<script", "</script", "<|", "|>"])
    def test_rejects_marker_in_instruction(self, marker):
        payload = _valid_blueprint_json(instruction=f"do stuff {marker} more")
        result = validate_blueprint_payload(payload)
        assert result is None

    def test_allows_markers_in_non_instruction_fields(self):
        payload = _valid_blueprint_json(
            description="Use <script> tags for demo",
            instruction="Write clearly.",
        )
        result = validate_blueprint_payload(payload)
        assert result is not None

    def test_scan_happens_after_stripping(self):
        payload = _valid_blueprint_json(instruction="  {{  ")
        result = validate_blueprint_payload(payload)
        assert result is None

    def test_security_event_logged(self, caplog):
        with caplog.at_level(logging.INFO):
            payload = _valid_blueprint_json(instruction="{{inject}}")
            validate_blueprint_payload(payload)
        assert any("injection marker" in r.message.lower() for r in caplog.records)


# ── JSON parse failures ────────────────────────────────────────────────


class TestJsonParseFailures:
    def test_returns_none_on_malformed_json(self, caplog):
        with caplog.at_level(logging.INFO):
            result = validate_blueprint_payload("not json {{{")
        assert result is None
        assert any("json_parse_error" in r.message for r in caplog.records)

    def test_returns_none_on_missing_required_field(self, caplog):
        data = _valid_blueprint_dict()
        del data["instruction"]
        with caplog.at_level(logging.INFO):
            result = validate_blueprint_payload(json.dumps(data))
        assert result is None
        assert any("schema_violation" in r.message for r in caplog.records)


# ── Neutral default ────────────────────────────────────────────────────


class TestNeutralDefault:
    def test_validates_against_schema(self):
        result = validate_blueprint_payload(NEUTRAL_BLUEPRINT.model_dump_json())
        assert isinstance(result, Blueprint)

    def test_has_documented_id(self):
        assert NEUTRAL_BLUEPRINT.id == "blueprint_neutral_default"

    def test_has_empty_participants(self):
        assert NEUTRAL_BLUEPRINT.participants == []
