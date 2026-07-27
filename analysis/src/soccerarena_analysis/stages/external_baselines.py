from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..config import AnalysisConfig, sha256_file
from ..manifest import Manifest


def remove_bookmaker_overround(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"fixture_id", "bookmaker", "odds_home", "odds_draw", "odds_away"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Bookmaker odds require columns {sorted(required)}")
    work = frame.copy()
    inverse = 1.0 / work[["odds_home", "odds_draw", "odds_away"]].astype(float)
    denominator = inverse.sum(axis=1)
    if (denominator <= 0).any() or (~np.isfinite(inverse)).any().any():
        raise ValueError("Odds must be finite and strictly positive")
    probabilities = inverse.div(denominator, axis=0)
    probabilities.columns = ["prob_home", "prob_draw", "prob_away"]
    return pd.concat(
        [
            work[["fixture_id", "bookmaker"]].reset_index(drop=True),
            probabilities.reset_index(drop=True),
        ],
        axis=1,
    )


def aggregate_bookmakers(frame: pd.DataFrame) -> pd.DataFrame:
    deoverrounded = remove_bookmaker_overround(frame)
    aggregated = deoverrounded.groupby("fixture_id", as_index=False).agg(
        prob_home=("prob_home", "median"),
        prob_draw=("prob_draw", "median"),
        prob_away=("prob_away", "median"),
        n_bookmakers=("bookmaker", "nunique"),
    )
    probability_columns = ["prob_home", "prob_draw", "prob_away"]
    probability_sum = aggregated[probability_columns].sum(axis=1)
    if (probability_sum <= 0).any():
        raise ValueError("Aggregated bookmaker probabilities must have a positive sum")
    aggregated[probability_columns] = aggregated[probability_columns].div(probability_sum, axis=0)
    return aggregated


def evaluate_gate(metadata: dict[str, Any], required_fields: list[str]) -> tuple[str, list[str]]:
    missing = [field for field in required_fields if metadata.get(field) in (None, "", [], {})]
    if metadata.get("leakage_risk") is True or metadata.get("timing_verified") is False:
        return "excluded", missing
    if missing:
        return "provisional", missing
    return "passed", []


def _nested_bytes(outer: ZipFile, configured_name: str) -> tuple[str, bytes]:
    matches = [
        name
        for name in outer.namelist()
        if name.endswith(f"/{configured_name}") or name == configured_name
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {configured_name} in archive, found {matches}"
        )
    return matches[0], outer.read(matches[0])


def _markus_audit(config: AnalysisConfig) -> dict[str, Any]:
    settings = config.section("external_baselines")["markus"]
    archive = config.resolve_path("markus_archive")
    audit: dict[str, Any] = {
        "baseline": "markus",
        "outer_archive": str(archive),
        "outer_archive_sha256": sha256_file(archive),
        "expected_repository_head": settings["expected_repository_head"],
        "expected_raw_results_source_sha": settings["expected_source_sha"],
        "comparison_horizon": settings["comparison_horizon"],
        "target_semantics": "90-minute home/draw/away",
        "bundled_predictions_used_as_analysis_input": False,
        "live_refit_paths_eligible": False,
        "candidate_audits": [],
    }
    runtime_lock = config.resolve_path("markus_runtime_lock")
    audit["project_runtime_lock"] = str(runtime_lock)
    audit["project_runtime_lock_sha256"] = sha256_file(runtime_lock)
    audit["reported_package_versions"] = settings["reported_package_versions"]
    with ZipFile(archive) as outer:
        nested_name, nested_raw = _nested_bytes(outer, settings["nested_archive"])
    audit["nested_archive_member"] = nested_name
    audit["nested_archive_sha256"] = hashlib.sha256(nested_raw).hexdigest()
    with ZipFile(BytesIO(nested_raw)) as nested:
        names = set(nested.namelist())
        head_name = next((name for name in names if name.endswith("/.git/refs/heads/main")), None)
        observed_head = nested.read(head_name).decode().strip() if head_name else None
        audit["observed_repository_head"] = observed_head
        audit["repository_head_verified"] = observed_head == settings["expected_repository_head"]
        log_name = next(
            (name for name in names if name.endswith("/data/track/predictions_log.csv")), None
        )
        if log_name:
            rows = list(
                csv.DictReader(StringIO(nested.read(log_name).decode("utf-8", errors="replace")))
            )
        else:
            rows = []
        audit["bundled_prediction_log_rows"] = len(rows)
        audit["bundled_group_forecast_count_verified"] = len(rows) == int(
            settings["group_prediction_count"]
        )
        audit["bundled_kickoff_resolution"] = "date-only midnight anchors"
        audit["prohibited_paths"] = {
            path: any(name.endswith(f"/{path}") for name in names)
            for path in settings["prohibited_paths"]
        }
        audit["nested_renv_lock_present"] = any(name.endswith("/renv.lock") for name in names)
        audit["market_value_row_provenance"] = "not established"

    rscript = str(settings["rscript_path"])
    runtime_ready = rscript != "CONFIGURE_ISOLATED_RSCRIPT" and Path(rscript).is_file()
    audit["configured_r_version"] = settings["r_version"]
    audit["rscript_path"] = rscript
    audit["rscript_exists"] = runtime_ready
    audit["runtime_restored"] = False
    audit["reproduction_status"] = "not_run_runtime_or_lock_unavailable"
    audit["reproduced_prediction_hash"] = None
    audit["numerical_parity_status"] = "not_run"
    audit["retrospective_knockout_generation"] = "not_run"

    common = {
        "source_identifier": "martj42/international_results plus nested repository archive",
        "source_url": "https://github.com/martj42/international_results",
        "retrieval_timestamp": None,
        "raw_snapshot_hash": settings["expected_source_sha"],
        "oriented_fixture_key": None,
        "training_cutoff": "2026-06-09 for bundled group log; regeneration not completed",
        "feature_timing": "date-only kickoff anchors prevent exact snapshot verification",
        "target_semantics": "90-minute H/D/A claimed; reproduction pending",
        "reproduction_hash": None,
        "redistribution_terms": None,
        "timing_verified": False,
        "leakage_risk": False,
    }
    for candidate in settings["candidates"]:
        candidate_metadata = dict(common)
        candidate_metadata["candidate"] = candidate
        candidate_metadata.update(settings["candidate_details"][candidate])
        candidate_metadata["tournament_outcome_refitting"] = (
            "prohibited; supplied live-refit path excluded"
        )
        candidate_metadata["reproduction_status"] = audit["reproduction_status"]
        if candidate.endswith("market_value"):
            candidate_metadata["feature_timing"] = "market-value row provenance not established"
        status, missing = evaluate_gate(candidate_metadata, settings["required_gate_fields"])
        audit["candidate_audits"].append(
            {**candidate_metadata, "gate_status": status, "missing_gate_fields": missing}
        )
    audit["gate_status"] = "excluded"
    audit["gate_reasons"] = [
        "isolated R 4.5.1 runtime and package lock were not restored",
        "match-level predictions were not regenerated from frozen inputs",
        "date-only kickoff anchors do not establish exact feature timing",
        "live update paths refit after tournament matches and are prohibited",
        "market-value row-level provenance is insufficient",
    ]
    return audit


def _combined_file_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _closing_odds_audit(config: AnalysisConfig) -> dict[str, Any]:
    website_root = config.resolve_path("website_root")
    odds_directory = config.resolve_path("website_exports") / "worldcup2026_closing_odds"
    csv_path = odds_directory / "closing-odds.csv"
    summary_path = odds_directory / "summary.json"
    raw_directory = odds_directory / "raw"
    fetch_script = (
        website_root / "apps" / "cron" / "src" / "jobs" / "fetch-worldcup-closing-odds.ts"
    )
    required_columns = {
        "match_id",
        "kickoff_utc",
        "actual_snapshot_utc",
        "home_team",
        "away_team",
        "bookmaker_key",
        "market_last_update_utc",
        "home_odds_decimal",
        "draw_odds_decimal",
        "away_odds_decimal",
    }
    files_present = all(path.is_file() for path in (csv_path, summary_path, fetch_script))
    raw_files = sorted(raw_directory.glob("*.json")) if raw_directory.is_dir() else []
    validation_errors: list[str] = []
    frame = pd.DataFrame()
    summary: dict[str, Any] = {}
    if not files_present:
        validation_errors.append("closing-odds CSV, summary, or fetch script is missing")
    else:
        frame = pd.read_csv(csv_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            validation_errors.append(f"missing CSV columns: {missing_columns}")
        else:
            prices = frame[["home_odds_decimal", "draw_odds_decimal", "away_odds_decimal"]].apply(
                pd.to_numeric, errors="coerce"
            )
            if prices.isna().any().any() or (prices <= 1.0).any().any():
                validation_errors.append("odds are missing, non-finite, or not greater than one")
            kickoff = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
            snapshot = pd.to_datetime(frame["actual_snapshot_utc"], utc=True, errors="coerce")
            update = pd.to_datetime(frame["market_last_update_utc"], utc=True, errors="coerce")
            if kickoff.isna().any() or snapshot.isna().any() or update.isna().any():
                validation_errors.append("kickoff, snapshot, or market update timestamp is invalid")
            elif (snapshot >= kickoff).any() or (update >= kickoff).any():
                validation_errors.append(
                    "at least one snapshot or market update is not pre-kickoff"
                )
            if frame["match_id"].nunique() != 104:
                validation_errors.append("closing odds do not cover all 104 matches")
            if frame.groupby("match_id")["bookmaker_key"].nunique().min() < 10:
                validation_errors.append("at least one match has fewer than 10 bookmakers")
            if frame.duplicated(["match_id", "bookmaker_key"]).any():
                validation_errors.append("duplicate match/bookmaker rows are present")
        metadata = summary.get("metadata", {})
        if metadata.get("selected_matches") != 104 or metadata.get("unmatched_matches") != 0:
            validation_errors.append("summary does not certify complete 104-match coverage")
    if len(raw_files) != 92:
        validation_errors.append(f"expected 92 raw kickoff snapshots, found {len(raw_files)}")

    raw_snapshot_hash = _combined_file_hash(raw_files) if raw_files else None
    retrieval_timestamp = (
        datetime.fromtimestamp(min(path.stat().st_mtime for path in raw_files), tz=UTC).isoformat()
        if raw_files
        else None
    )
    reproduction_hash = None
    if files_present and raw_snapshot_hash is not None:
        payload = ":".join(
            [
                sha256_file(fetch_script),
                sha256_file(csv_path),
                sha256_file(summary_path),
                raw_snapshot_hash,
            ]
        )
        reproduction_hash = hashlib.sha256(payload.encode("ascii")).hexdigest()

    audit: dict[str, Any] = {
        "baseline": "closing_odds",
        "source_identifier": "The Odds API v4 historical FIFA World Cup odds",
        "source_url": "https://the-odds-api.com/sports/fifa-world-cup-odds.html",
        "retrieval_timestamp": retrieval_timestamp,
        "raw_snapshot_hash": raw_snapshot_hash,
        "oriented_fixture_key": "local match_id plus ordered home_team/away_team",
        "training_cutoff": "not applicable (external market forecast)",
        "feature_timing": "closest historical snapshot before kickoff; bookmaker updates verified pre-kickoff",
        "target_semantics": "90-minute home/draw/away (1X2)",
        "reproduction_hash": reproduction_hash,
        "redistribution_terms": "raw subscription odds remain local; paper artifacts contain derived consensus probabilities and scores only",
        "timing_verified": not validation_errors,
        "leakage_risk": False,
        "bookmaker_level_values_retained": True,
        "overround_protocol_compatible": True,
        "aggregation": config.section("external_baselines")["closing_odds"]["aggregation"],
        "csv_path": str(csv_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "raw_snapshot_count": len(raw_files),
        "match_count": int(frame["match_id"].nunique()) if "match_id" in frame else 0,
        "bookmaker_count": (
            int(frame["bookmaker_key"].nunique()) if "bookmaker_key" in frame else 0
        ),
        "validation_errors": validation_errors,
        "publication_constraint": "confirm vendor attribution and derived-data publication terms before submission; do not redistribute raw snapshots",
    }
    required = config.section("external_baselines")["markus"]["required_gate_fields"]
    status, missing = evaluate_gate(audit, required)
    audit["gate_status"] = "excluded" if validation_errors else status
    audit["missing_gate_fields"] = missing
    audit["reason"] = (
        "; ".join(validation_errors)
        if validation_errors
        else "complete timestamped bookmaker-level closing odds support the registered estimator"
    )
    return audit


def _closing_odds_consensus(config: AnalysisConfig, audit: dict[str, Any]) -> pd.DataFrame:
    if audit["gate_status"] != "passed":
        return pd.DataFrame()
    frame = pd.read_csv(audit["csv_path"])
    prepared = frame.rename(
        columns={
            "match_id": "fixture_id",
            "bookmaker_key": "bookmaker",
            "home_odds_decimal": "odds_home",
            "draw_odds_decimal": "odds_draw",
            "away_odds_decimal": "odds_away",
        }
    )
    consensus = aggregate_bookmakers(prepared)
    snapshots = frame.groupby("match_id", as_index=False).agg(
        snapshot=("actual_snapshot_utc", "first"),
        kickoff_utc=("kickoff_utc", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
    )
    consensus = consensus.merge(snapshots, left_on="fixture_id", right_on="match_id")
    consensus.insert(0, "baseline", "closing_odds")
    consensus.insert(1, "candidate", "median_devigged_bookmaker_consensus")
    consensus["reproduction_hash"] = audit["reproduction_hash"]
    return consensus[
        [
            "baseline",
            "candidate",
            "match_id",
            "snapshot",
            "kickoff_utc",
            "home_team",
            "away_team",
            "prob_home",
            "prob_draw",
            "prob_away",
            "n_bookmakers",
            "reproduction_hash",
        ]
    ]


def _t24_odds_audit(config: AnalysisConfig) -> dict[str, Any]:
    website_root = config.resolve_path("website_root")
    odds_directory = config.resolve_path("website_exports") / "worldcup2026_t24_odds"
    csv_path = odds_directory / "closing-odds.csv"
    summary_path = odds_directory / "summary.json"
    raw_directory = odds_directory / "raw"
    fetch_script = (
        website_root / "apps" / "cron" / "src" / "jobs" / "fetch-worldcup-closing-odds.ts"
    )
    required_columns = {
        "match_id",
        "kickoff_utc",
        "requested_snapshot_utc",
        "actual_snapshot_utc",
        "home_team",
        "away_team",
        "bookmaker_key",
        "market_last_update_utc",
        "home_odds_decimal",
        "draw_odds_decimal",
        "away_odds_decimal",
    }
    files_present = all(path.is_file() for path in (csv_path, summary_path, fetch_script))
    raw_files = sorted(raw_directory.glob("*.json")) if raw_directory.is_dir() else []
    validation_errors: list[str] = []
    frame = pd.DataFrame()
    summary: dict[str, Any] = {}
    unmatched_ids: list[str] = []
    if not files_present:
        validation_errors.append("T-24h odds CSV, summary, or fetch script is missing")
    else:
        frame = pd.read_csv(csv_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            validation_errors.append(f"missing CSV columns: {missing_columns}")
        else:
            prices = frame[["home_odds_decimal", "draw_odds_decimal", "away_odds_decimal"]].apply(
                pd.to_numeric, errors="coerce"
            )
            if prices.isna().any().any() or (prices <= 1.0).any().any():
                validation_errors.append("odds are missing, non-finite, or not greater than one")
            kickoff = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
            requested = pd.to_datetime(frame["requested_snapshot_utc"], utc=True, errors="coerce")
            snapshot = pd.to_datetime(frame["actual_snapshot_utc"], utc=True, errors="coerce")
            update = pd.to_datetime(frame["market_last_update_utc"], utc=True, errors="coerce")
            if any(values.isna().any() for values in (kickoff, requested, snapshot, update)):
                validation_errors.append(
                    "kickoff, request, snapshot, or update timestamp is invalid"
                )
            else:
                requested_lead = (kickoff - requested).dt.total_seconds()
                if not np.allclose(requested_lead, 86_400.0, atol=1.0):
                    validation_errors.append("requested snapshots are not exactly T-24h")
                if (snapshot > requested).any():
                    validation_errors.append("historical API returned a post-request snapshot")
                if (update > snapshot).any():
                    validation_errors.append("market update occurs after the historical snapshot")
            if frame.groupby("match_id")["bookmaker_key"].nunique().min() < 10:
                validation_errors.append("at least one match has fewer than 10 bookmakers")
            if frame.duplicated(["match_id", "bookmaker_key"]).any():
                validation_errors.append("duplicate match/bookmaker rows are present")
        metadata = summary.get("metadata", {})
        matches = summary.get("matches", [])
        unmatched_ids = sorted(
            str(item["matchId"])
            for item in matches
            if isinstance(item, dict) and item.get("reason") is not None
        )
        match_count = int(frame["match_id"].nunique()) if "match_id" in frame else 0
        if metadata.get("selected_matches") != 104:
            validation_errors.append("summary does not certify 104 selected matches")
        if metadata.get("matched_matches") != match_count:
            validation_errors.append("summary matched count disagrees with the CSV")
        if metadata.get("unmatched_matches") != 104 - match_count:
            validation_errors.append("summary unmatched count disagrees with complete coverage")
        if len(unmatched_ids) != 104 - match_count:
            validation_errors.append("unmatched fixture IDs are not fully documented")
    if len(raw_files) != 92:
        validation_errors.append(f"expected 92 raw kickoff snapshots, found {len(raw_files)}")

    raw_snapshot_hash = _combined_file_hash(raw_files) if raw_files else None
    retrieval_timestamp = (
        datetime.fromtimestamp(min(path.stat().st_mtime for path in raw_files), tz=UTC).isoformat()
        if raw_files
        else None
    )
    reproduction_hash = None
    if files_present and raw_snapshot_hash is not None:
        payload = ":".join(
            [
                sha256_file(fetch_script),
                sha256_file(csv_path),
                sha256_file(summary_path),
                raw_snapshot_hash,
            ]
        )
        reproduction_hash = hashlib.sha256(payload.encode("ascii")).hexdigest()

    match_count = int(frame["match_id"].nunique()) if "match_id" in frame else 0
    audit: dict[str, Any] = {
        "baseline": "t24_odds",
        "source_identifier": "The Odds API v4 historical FIFA World Cup odds",
        "source_url": "https://the-odds-api.com/sports/fifa-world-cup-odds.html",
        "retrieval_timestamp": retrieval_timestamp,
        "raw_snapshot_hash": raw_snapshot_hash,
        "oriented_fixture_key": "local match_id plus ordered home_team/away_team",
        "training_cutoff": "not applicable (external market forecast)",
        "feature_timing": "requested exactly 24 hours before kickoff; API snapshot at or before request; bookmaker updates verified no later than snapshot",
        "target_semantics": "90-minute home/draw/away (1X2)",
        "reproduction_hash": reproduction_hash,
        "redistribution_terms": "raw subscription odds remain local; paper artifacts contain derived consensus probabilities and scores only",
        "timing_verified": not validation_errors,
        "leakage_risk": False,
        "bookmaker_level_values_retained": True,
        "overround_protocol_compatible": True,
        "aggregation": config.section("external_baselines")["t24_odds"]["aggregation"],
        "csv_path": str(csv_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "raw_snapshot_count": len(raw_files),
        "match_count": match_count,
        "coverage_fraction": match_count / 104.0,
        "unmatched_match_ids": unmatched_ids,
        "coverage_policy": "mechanical complete-case comparison; no odds imputation",
        "bookmaker_count": (
            int(frame["bookmaker_key"].nunique()) if "bookmaker_key" in frame else 0
        ),
        "validation_errors": validation_errors,
        "publication_constraint": "confirm vendor attribution and derived-data publication terms before submission; do not redistribute raw snapshots",
    }
    required = config.section("external_baselines")["markus"]["required_gate_fields"]
    status, missing = evaluate_gate(audit, required)
    audit["gate_status"] = "excluded" if validation_errors else status
    audit["missing_gate_fields"] = missing
    audit["reason"] = (
        "; ".join(validation_errors)
        if validation_errors
        else f"validated T-24h bookmaker snapshots cover {match_count}/104 fixtures; "
        f"{len(unmatched_ids)} fixtures had no listed EU h2h event at the requested time"
    )
    return audit


def _t24_odds_consensus(config: AnalysisConfig, audit: dict[str, Any]) -> pd.DataFrame:
    if audit["gate_status"] != "passed":
        return pd.DataFrame()
    frame = pd.read_csv(audit["csv_path"])
    prepared = frame.rename(
        columns={
            "match_id": "fixture_id",
            "bookmaker_key": "bookmaker",
            "home_odds_decimal": "odds_home",
            "draw_odds_decimal": "odds_draw",
            "away_odds_decimal": "odds_away",
        }
    )
    consensus = aggregate_bookmakers(prepared)
    snapshots = frame.groupby("match_id", as_index=False).agg(
        snapshot=("actual_snapshot_utc", "first"),
        requested_snapshot=("requested_snapshot_utc", "first"),
        kickoff_utc=("kickoff_utc", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
    )
    consensus = consensus.merge(snapshots, left_on="fixture_id", right_on="match_id")
    consensus.insert(0, "baseline", "t24_odds")
    consensus.insert(1, "candidate", "median_devigged_bookmaker_consensus")
    consensus["reproduction_hash"] = audit["reproduction_hash"]
    return consensus[
        [
            "baseline",
            "candidate",
            "match_id",
            "snapshot",
            "requested_snapshot",
            "kickoff_utc",
            "home_team",
            "away_team",
            "prob_home",
            "prob_draw",
            "prob_away",
            "n_bookmakers",
            "reproduction_hash",
        ]
    ]


def _other_audits(config: AnalysisConfig) -> list[dict[str, Any]]:
    required = config.section("external_baselines")["markus"]["required_gate_fields"]
    odds = _closing_odds_audit(config)
    t24_odds = _t24_odds_audit(config)
    opta = {
        "baseline": "opta",
        "source_identifier": "Stats Perform public predictions feed snapshot",
        "source_url": "https://dataviz.theanalyst.com/",
        "retrieval_timestamp": None,
        "raw_snapshot_hash": None,
        "oriented_fixture_key": "feed fixture ID in raw snapshot",
        "training_cutoff": None,
        "feature_timing": "snapshot date documented; per-row forecast generation time absent",
        "target_semantics": "90-minute H/D/A",
        "reproduction_hash": None,
        "redistribution_terms": None,
        "timing_verified": False,
        "leakage_risk": False,
    }
    audits = []
    for item in (odds, t24_odds, opta):
        if "gate_status" in item:
            audits.append(item)
            continue
        status, missing = evaluate_gate(item, required)
        audits.append({**item, "gate_status": status, "missing_gate_fields": missing})
    return audits


def _write_markdown(audits: list[dict[str, Any]], path: Path) -> None:
    lines = ["# External baseline provenance and timing audit", ""]
    for audit in audits:
        lines.extend(
            [f"## {audit['baseline']}", "", f"Gate status: **{audit['gate_status']}**.", ""]
        )
        if audit["baseline"] == "markus":
            lines.extend(
                [
                    f"Nested repository head: `{audit['observed_repository_head']}`.",
                    f"Raw international-results source pin: `{audit['expected_raw_results_source_sha']}`.",
                    f"Bundled log: {audit['bundled_prediction_log_rows']} group forecasts; it is used only for a future parity diagnostic.",
                    "The supplied live-update path downloads results and refits after matches, so it is ineligible.",
                    "The market-value candidate remains ineligible until row-level source and timing provenance are established.",
                    f"Reproduction status: {audit['reproduction_status']}; reproduced hash: {audit['reproduced_prediction_hash']}.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"Reason: {audit.get('reason', '; '.join(audit.get('missing_gate_fields', [])))}.",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: AnalysisConfig, manifest: Manifest) -> list[dict[str, Any]]:
    archive = config.resolve_path("markus_archive")
    if not archive.exists():
        raise FileNotFoundError(f"External artifact archive not found: {archive}")
    markus = _markus_audit(config)
    audits = [markus, *_other_audits(config)]
    directory = config.resolve_path("external")
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "baseline_gate_audit.json"
    json_path.write_text(json.dumps(audits, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = directory / "baseline_gate_audit.md"
    _write_markdown(audits, markdown_path)
    odds_audit = next(audit for audit in audits if audit["baseline"] == "closing_odds")
    t24_odds_audit = next(audit for audit in audits if audit["baseline"] == "t24_odds")
    external_frames = [
        _closing_odds_consensus(config, odds_audit),
        _t24_odds_consensus(config, t24_odds_audit),
    ]
    external_frames = [frame for frame in external_frames if not frame.empty]
    if external_frames:
        external = pd.concat(external_frames, ignore_index=True, sort=False)
    else:
        external = pd.DataFrame(
            columns=[
                "baseline",
                "candidate",
                "match_id",
                "snapshot",
                "requested_snapshot",
                "kickoff_utc",
                "home_team",
                "away_team",
                "prob_home",
                "prob_draw",
                "prob_away",
                "n_bookmakers",
                "reproduction_hash",
            ]
        )
    output_path = config.resolve_path("derived") / "external_baselines.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external.to_parquet(output_path, index=False)
    source_hashes = {
        "markus_archive": sha256_file(archive),
        "markus_runtime_lock": sha256_file(config.resolve_path("markus_runtime_lock")),
        "closing_odds_csv": (
            sha256_file(Path(odds_audit["csv_path"]))
            if Path(odds_audit["csv_path"]).is_file()
            else "missing"
        ),
        "closing_odds_raw_snapshots": odds_audit.get("raw_snapshot_hash") or "missing",
        "t24_odds_csv": (
            sha256_file(Path(t24_odds_audit["csv_path"]))
            if Path(t24_odds_audit["csv_path"]).is_file()
            else "missing"
        ),
        "t24_odds_raw_snapshots": t24_odds_audit.get("raw_snapshot_hash") or "missing",
    }
    manifest.add(
        "external_baseline_audit_json",
        json_path,
        "json",
        "external_baselines",
        source_hashes,
        {"passed": sum(audit["gate_status"] == "passed" for audit in audits)},
    )
    manifest.add(
        "external_baseline_audit_markdown",
        markdown_path,
        "markdown",
        "external_baselines",
        source_hashes,
    )
    manifest.add(
        "derived_external_baselines",
        output_path,
        "parquet",
        "external_baselines",
        source_hashes,
        {"rows": len(external)},
    )
    manifest.write()
    return audits
