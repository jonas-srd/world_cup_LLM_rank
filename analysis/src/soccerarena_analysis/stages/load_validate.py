from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import AnalysisConfig, sha256_file
from ..manifest import Manifest

REQUIRED_TABLES = {
    "matches",
    "benchmark_predictions",
    "prediction_evaluations",
    "special_predictions",
    "special_prediction_options",
}


def validate_frozen_database(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    frozen = Path(manifest.require("frozen_db").path)
    connection = sqlite3.connect(f"file:{frozen.as_posix()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise ValueError(f"Frozen DB is missing required tables: {missing}")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"Frozen DB integrity check failed: {integrity}")
        fixture_count = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        completed_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM matches WHERE home_score_90 IS NOT NULL AND away_score_90 IS NOT NULL"
            ).fetchone()[0]
        )
        prediction_count = int(
            connection.execute("SELECT COUNT(*) FROM benchmark_predictions").fetchone()[0]
        )
        special_question_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT question_id) FROM special_predictions"
            ).fetchone()[0]
        )
        expected = int(config.section("validation")["expected_fixtures"])
        if fixture_count != expected:
            raise ValueError(f"Expected {expected} fixtures, found {fixture_count}")
        if (
            config.is_final
            and config.section("validation")["require_complete_tournament_in_final"]
            and completed_count != expected
        ):
            raise ValueError(
                f"Final mode requires {expected} completed 90-minute outcomes; found {completed_count}"
            )
        report = {
            "frozen_db_sha256": sha256_file(frozen),
            "integrity_check": integrity,
            "fixture_count": fixture_count,
            "completed_90_minute_outcomes": completed_count,
            "benchmark_prediction_count": prediction_count,
            "special_question_count": special_question_count,
            "run_mode": config.raw["run_mode"],
        }
    finally:
        connection.close()
    output = config.resolve_path("verification") / "database_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add(
        "database_validation",
        output,
        "json",
        "load_validate",
        {"frozen_db": sha256_file(frozen)},
        report,
    )
    manifest.write()
    return report
