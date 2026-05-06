"""Tests for T19: blueprint seed files exist and validate."""

import json
from pathlib import Path

import pytest

from blog_mas.rag.blueprints import validate_blueprint_payload

BLUEPRINTS_DIR = Path(__file__).resolve().parent.parent / "data" / "blueprints"

EXPECTED_IDS = {
    "technical-deep-dive",
    "executive-summary",
    "casual-explainer",
    "tutorial-stepwise",
    "news-brief",
    "opinion-essay",
}


def test_all_blueprint_files_exist():
    files = {p.stem for p in BLUEPRINTS_DIR.glob("*.json")}
    assert files == EXPECTED_IDS


def test_each_file_validates_against_schema():
    for path in sorted(BLUEPRINTS_DIR.glob("*.json")):
        raw = path.read_text()
        bp = validate_blueprint_payload(raw)
        assert bp is not None, f"{path.name} failed Blueprint validation"
        assert bp.id == path.stem
