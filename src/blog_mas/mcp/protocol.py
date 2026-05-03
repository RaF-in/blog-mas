"""MCP message factory and envelope validator."""

from typing import Any


def create_mcp_message(
    sender: str, content: Any, metadata: dict | None = None
) -> dict:
    return {
        "protocol_version": "1.0",
        "sender": sender,
        "content": content,
        "metadata": metadata if metadata is not None else {},
    }


def validate_mcp_envelope(message: Any) -> bool:
    if not isinstance(message, dict):
        print(f"MCP Validation Failed: Message is not a dictionary")
        return False

    required_keys = ("protocol_version", "sender", "content", "metadata")
    for key in required_keys:
        if key not in message:
            sender = message.get("sender", "unknown")
            print(
                f"MCP Validation Failed: Missing key '{key}' in message from {sender}"
            )
            return False

    if not message["sender"]:
        print("MCP Validation Failed: Empty sender field")
        return False

    return True
