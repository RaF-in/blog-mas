"""Logging configuration shim — preserved for backward compatibility.

All existing imports of the form:
  from blog_mas.logging_config import setup_logging

continue to work.  The implementation is now in logging_json.py, which adds
Ch10 §1.4 structured JSON logging with trace_id binding.
"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Backward-compatible shim — delegates to logging_json.setup_logging."""
    from blog_mas.logging_json import setup_logging as _setup
    level_name = logging.getLevelName(level) if isinstance(level, int) else str(level)
    _setup(level=level_name)
