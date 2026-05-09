"""Minimal MCP-style dict envelope (chapter 2 pattern)."""


def create_mcp_message(sender: str, content: dict) -> dict:
    """Wrap content in a minimal MCP envelope."""
    return {"sender": sender, "content": content}
