"""Centralized logging configuration for the blog MAS pipeline."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )
