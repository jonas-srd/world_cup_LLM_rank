from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path

from ..config import AnalysisConfig, sha256_file
from ..manifest import Manifest, git_commit


def _integrity_and_counts(path: Path) -> tuple[str, dict[str, int], int, str]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in tables
        }
        schema_sql = "\n".join(
            str(row[0])
            for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
            )
        )
        return (
            integrity,
            counts,
            schema_version,
            hashlib.sha256(schema_sql.encode("utf-8")).hexdigest(),
        )
    finally:
        connection.close()


def freeze_database(config: AnalysisConfig, manifest: Manifest) -> Path:
    source = config.resolve_path("source_db")
    destination = config.resolve_path("frozen_db")
    sidecar = destination.with_suffix(".freeze.json")
    if not source.exists():
        raise FileNotFoundError(f"Source SQLite database not found: {source}")
    source_hash = sha256_file(source)
    if destination.exists() and sidecar.exists():
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            existing.get("source_sha256") == source_hash
            and sha256_file(destination) == existing.get("frozen_sha256")
            and existing.get("repository_commit") == git_commit(config.root.parent)
            and existing.get("schema_sha256")
        ):
            manifest.add(
                "frozen_db", destination, "sqlite", "freeze", {"source_db": source_hash}, existing
            )
            manifest.add(
                "freeze_metadata",
                sidecar,
                "json",
                "freeze",
                {"frozen_db": existing["frozen_sha256"]},
            )
            manifest.write()
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.sqlite")
    if temporary.exists():
        temporary.unlink()
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(temporary)
    try:
        source_connection.execute("BEGIN")
        source_connection.backup(destination_connection)
        source_connection.rollback()
    finally:
        destination_connection.close()
        source_connection.close()
    if destination.exists():
        destination.unlink()
    temporary.replace(destination)
    integrity, table_counts, schema_version, schema_hash = _integrity_and_counts(destination)
    if integrity != "ok":
        raise ValueError(f"Frozen database integrity check failed: {integrity}")
    frozen_hash = sha256_file(destination)
    metadata = {
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": source_hash,
        "frozen_sha256": frozen_hash,
        "sqlite_user_version": schema_version,
        "schema_sha256": schema_hash,
        "integrity_check": integrity,
        "table_counts": table_counts,
        "repository_commit": git_commit(config.root.parent),
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add("frozen_db", destination, "sqlite", "freeze", {"source_db": source_hash}, metadata)
    manifest.add("freeze_metadata", sidecar, "json", "freeze", {"frozen_db": frozen_hash})
    manifest.write()
    return destination
