"""Blueprint ingestion: load .json files → validate → embed description → upsert."""

import hashlib
import logging
import uuid
from pathlib import Path

from blog_mas.rag.blueprints import validate_blueprint_payload

logger = logging.getLogger(__name__)


def _blueprint_id(bp_json_str: str) -> str:
    return hashlib.sha256(bp_json_str.encode("utf-8")).hexdigest()


def run_blueprint_ingestion(
    source_dir: str,
    namespace: str,
    store,
    embedder,
) -> None:
    dim = getattr(embedder, "_dim", 384)
    store.ensure_collection(namespace, dim=dim)

    json_files = sorted(Path(source_dir).glob("*.json"))
    if not json_files:
        return

    descriptions = []
    payloads = []
    bp_ids = []

    for path in json_files:
        raw = path.read_text()
        bp = validate_blueprint_payload(raw)
        if bp is None:
            logger.warning("blueprint_graph.skipped file=%s", path.name)
            continue

        bp_id = _blueprint_id(raw)
        descriptions.append(bp.description)
        bp_ids.append(bp_id)
        payloads.append({
            "blueprint_id": bp.id,
            "description": bp.description,
            "blueprint_json": raw,
        })

    if not descriptions:
        return

    vectors = embedder.embed_batch(descriptions)

    points = []
    for bp_id, vec, payload in zip(bp_ids, vectors, payloads):
        points.append({"id": str(uuid.UUID(bp_id[:32])), "vector": vec, "payload": payload})

    store.upsert_points(namespace, points)
    logger.info("blueprint_graph.upserted namespace=%s count=%d", namespace, len(points))
