"""Blueprint Pydantic schema, injection-marker scan, and neutral default.

Every blueprint retrieved from the vector store MUST pass through
``validate_blueprint_payload`` before being injected into any LLM prompt.
The vector store is treated as untrusted input — this module is the
security boundary.
"""

import json
import logging
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_INJECTION_MARKERS = ("{{", "}}", "<script", "</script", "<|", "|>")
_MAX_SERIALIZED_BYTES = 8 * 1024  # 8 KB


class Participant(BaseModel):
    name: str = Field(max_length=100)
    role: str = Field(max_length=200)

    @field_validator("name", "role")
    @classmethod
    def _strip_strings(cls, v: str) -> str:
        return v.strip()


class Blueprint(BaseModel):
    MAX_PARTICIPANTS: ClassVar[int] = 10

    id: str = Field(max_length=100)
    description: str = Field(max_length=2000)
    scene_goal: str = Field(max_length=1000)
    style_guide: str = Field(max_length=2000)
    participants: list[Participant] = Field(default_factory=list, max_length=10)
    instruction: str = Field(max_length=3000)
    metadata: dict[str, str | int | bool] | None = None

    @field_validator("id", "description", "scene_goal", "style_guide", "instruction")
    @classmethod
    def _strip_text_fields(cls, v: str) -> str:
        return v.strip()

    @field_validator("instruction")
    @classmethod
    def _scan_injection_markers(cls, v: str) -> str:
        stripped = v.strip()
        lower = stripped.lower()
        for marker in _INJECTION_MARKERS:
            if marker.lower() in lower:
                raise ValueError(f"injection marker detected: {marker!r}")
        return stripped

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_primitives(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        for key, val in v.items():
            if not isinstance(val, (str, int, bool)):
                raise ValueError(
                    f"metadata value for key {key!r} must be str|int|bool, "
                    f"got {type(val).__name__}"
                )
        return v

    @model_validator(mode="after")
    def _check_serialized_size(self) -> "Blueprint":
        serialized = self.model_dump_json()
        if len(serialized.encode("utf-8")) > _MAX_SERIALIZED_BYTES:
            raise ValueError(
                f"serialized blueprint exceeds {_MAX_SERIALIZED_BYTES} bytes "
                f"({len(serialized.encode('utf-8'))} bytes)"
            )
        return self


def validate_blueprint_payload(json_str: str) -> Blueprint | None:
    """Validate a blueprint JSON string retrieved from the vector store.

    Returns a validated ``Blueprint`` on success, or ``None`` on any
    failure (malformed JSON, schema violation, injection markers, size
    exceeded).  Logs the reason for every fallback.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.info("librarian.fallback reason=json_parse_error detail=%s", exc)
        return None

    try:
        bp = Blueprint.model_validate(data)
    except Exception as exc:
        logger.info(
            "librarian.fallback reason=schema_violation detail=%s", exc
        )
        return None

    return bp


NEUTRAL_BLUEPRINT = Blueprint(
    id="blueprint_neutral_default",
    description="A generic, neutral blog blueprint with no specific style requirements.",
    scene_goal="Inform the reader clearly and concisely.",
    style_guide="Neutral tone, standard formatting, clear paragraphs.",
    participants=[],
    instruction=(
        "Write a well-structured blog post using standard formatting. "
        "Use clear headings, short paragraphs, and a neutral informative tone."
    ),
)
