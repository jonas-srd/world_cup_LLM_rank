from __future__ import annotations

import json
import sqlite3

import pandas as pd

from soccerarena_analysis.stages.external_baselines import _markus_audit, evaluate_gate


def test_source_database_has_registered_fixture_and_question_units(config):
    connection = sqlite3.connect(config.resolve_path("source_db"))
    try:
        fixtures = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        questions = connection.execute(
            "SELECT COUNT(DISTINCT question_id) FROM special_predictions"
        ).fetchone()[0]
    finally:
        connection.close()
    assert fixtures == config.section("validation")["expected_fixtures"]
    assert questions == config.section("special_questions")["expected_questions"]


def test_derived_stable_prediction_key_is_unique(config):
    path = config.resolve_path("derived") / "analysis_panel.parquet"
    if not path.exists():
        return
    panel = pd.read_parquet(path)
    assert not panel.duplicated(config.section("validation")["stable_key"]).any()


def test_markus_live_paths_and_runtime_gate(config):
    audit = _markus_audit(config)
    assert (
        audit["observed_repository_head"]
        == config.section("external_baselines")["markus"]["expected_repository_head"]
    )
    assert audit["bundled_prediction_log_rows"] == 72
    assert all(audit["prohibited_paths"].values())
    assert audit["gate_status"] == "excluded"
    assert audit["bundled_predictions_used_as_analysis_input"] is False


def test_gate_excludes_unverified_timing(config):
    required = config.section("external_baselines")["markus"]["required_gate_fields"]
    status, _ = evaluate_gate({"timing_verified": False}, required)
    assert status == "excluded"


def test_public_reconciliation_is_report_only(config):
    path = config.resolve_path("verification") / "public_export_reconciliation.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["policy"] == "report_only_sqlite_remains_source_of_truth"
