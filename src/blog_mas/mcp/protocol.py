"""MCP message factory and envelope validator."""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SENDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class MCPEnvelope(BaseModel):
    protocol_version: str
    sender: str
    content: Any
    metadata: dict = Field(default_factory=dict)


def create_mcp_message(
    sender: str, content: Any, metadata: dict | None = None
) -> MCPEnvelope:
    return MCPEnvelope(
        protocol_version="1.0",
        sender=sender,
        content=content,
        metadata=metadata if metadata is not None else {},
    )


def validate_mcp_envelope(message: Any) -> bool:
    if not isinstance(message, dict):
        logger.warning("MCP Validation Failed: Invalid envelope format")
        return False

    required_keys = ("protocol_version", "sender", "content", "metadata")
    for key in required_keys:
        if key not in message:
            logger.debug("MCP Validation Failed: Missing key '%s'", key)
            logger.warning("MCP Validation Failed: Invalid envelope format")
            return False

    if not message["sender"]:
        logger.warning("MCP Validation Failed: Invalid envelope format")
        return False

    if not _SENDER_PATTERN.match(message["sender"]):
        logger.debug(
            "MCP Validation Failed: Sender '%s' does not match required format",
            message["sender"],
        )
        logger.warning("MCP Validation Failed: Invalid envelope format")
        return False

    return True
