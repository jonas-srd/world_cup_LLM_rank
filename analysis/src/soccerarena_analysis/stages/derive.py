from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import AnalysisConfig
from ..manifest import Manifest, git_commit
from ..schemas import (
    EVALUATIONS_SCHEMA,
    MATCHES_SCHEMA,
    PREDICTIONS_SCHEMA,
    SPECIAL_OPTIONS_SCHEMA,
    SPECIAL_OUTCOMES_SCHEMA,
    SPECIAL_PREDICTIONS_SCHEMA,
)
from ..statistics.metrics import recompute_evaluations, result_from_scores


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _nullable_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda value: None if pd.isna(value) else bool(value)).astype("boolean")


def _write_parquet(frame: pd.DataFrame, path: Path, schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=False)
    pq.write_table(table, path, compression="zstd", version="2.6")


def _read_sql(connection: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, connection)


def _matches(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _read_sql(
        connection,
        """
        SELECT id AS match_id, utc_date AS kickoff_utc, competition, home_team, away_team,
               status, stage, group_name, matchday, is_knockout,
               home_score_90, away_score_90, home_score_full, away_score_full,
               actual_advancer, result_winner, source, source_match_id
        FROM matches ORDER BY utc_date, id
        """,
    )
    frame["kickoff_utc"] = _utc(frame["kickoff_utc"])
    frame["is_knockout"] = _nullable_bool(frame["is_knockout"])
    frame["actual_result_90"] = [
        result_from_scores(home, away)
        for home, away in zip(frame["home_score_90"], frame["away_score_90"])
    ]
    return frame[list(name for name in MATCHES_SCHEMA.names)]


def _predictions(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _read_sql(
        connection,
        """
        SELECT bp.id AS prediction_id, bp.match_id,
               COALESCE(bp.model_id, bp.predictor_id) AS model_id,
               COALESCE(bp.model_version, 'unspecified') AS model_version,
               bp.provider, bp.forecast_horizon, bp.access_condition, bp.prompt_strategy,
               bp.sample_id, bp.scheduled_prediction_time_utc, bp.actual_prediction_time_utc,
               bp.kickoff_time_utc, bp.minutes_before_kickoff, bp.timing_status,
               bp.home_win_90_prob, bp.draw_90_prob, bp.away_win_90_prob,
               bp.expected_home_goals_90, bp.expected_away_goals_90,
               bp.home_advances_prob, bp.away_advances_prob,
               bp.most_likely_score_90_home AS predicted_home_90,
               bp.most_likely_score_90_away AS predicted_away_90,
               bp.confidence, bp.reason, bp.validation_status, bp.is_valid_for_scoring,
               bp.repair_attempted, bp.normalization_applied, bp.tools_enabled,
               bp.tool_calls_observed, bp.num_tool_calls, bp.tool_trace_available,
               bp.tool_trace, bp.open_book_compliance, bp.input_tokens, bp.output_tokens,
               bp.latency_ms, bp.cost_usd, bp.raw_prompt, bp.raw_response, bp.prompt_hash
        FROM benchmark_predictions bp
        ORDER BY bp.match_id, model_id, bp.forecast_horizon, bp.access_condition,
                 bp.prompt_strategy, bp.sample_id
        """,
    )
    for column in (
        "scheduled_prediction_time_utc",
        "actual_prediction_time_utc",
        "kickoff_time_utc",
    ):
        frame[column] = _utc(frame[column])
    for column in (
        "is_valid_for_scoring",
        "repair_attempted",
        "normalization_applied",
        "tools_enabled",
        "tool_calls_observed",
        "tool_trace_available",
    ):
        frame[column] = _nullable_bool(frame[column])
    return frame


def _stored_evaluations(connection: sqlite3.Connection) -> pd.DataFrame:
    return _read_sql(
        connection,
        """
        SELECT prediction_id, brier_90, log_loss_90, advancement_brier,
               advancement_log_loss, advancement_accuracy,
               score_result_matches_prob_argmax_90
        FROM prediction_evaluations
        """,
    )


def _special_predictions(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _read_sql(
        connection,
        """
        SELECT id AS prediction_id, question_id, question_label, prediction_type,
               CAST(k AS REAL) AS k, COALESCE(model_id, predictor_id) AS model_id,
               COALESCE(model_version, 'unspecified') AS model_version,
               access_condition, prompt_strategy, forecast_horizon, sample_id,
               final_pick, final_picks, confidence, reasoning_summary,
               is_valid_for_scoring, actual_prediction_time_utc
        FROM special_predictions
        ORDER BY question_id, model_id, access_condition, prompt_strategy
        """,
    )
    frame["actual_prediction_time_utc"] = _utc(frame["actual_prediction_time_utc"])
    frame["is_valid_for_scoring"] = _nullable_bool(frame["is_valid_for_scoring"])
    return frame


def _special_options(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = _read_sql(
        connection,
        """
        SELECT id AS option_id, prediction_id, question_id, candidate_id,
               candidate_label, candidate_type, probability, rank, is_final_pick
        FROM special_prediction_options
        ORDER BY prediction_id, rank, candidate_id
        """,
    )
    frame["is_final_pick"] = _nullable_bool(frame["is_final_pick"])
    return frame


def _official_team(row: pd.Series, side_value: object) -> str | None:
    if side_value == "home":
        return str(row["home_team"])
    if side_value == "away":
        return str(row["away_team"])
    return None


def _derived_special_outcomes(
    connection: sqlite3.Connection,
    matches: pd.DataFrame,
    options: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    settings = config.section("special_questions")
    table_name = str(settings["normalized_outcome_table"])
    table_exists = connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()[0]
    explicit = pd.DataFrame(columns=SPECIAL_OUTCOMES_SCHEMA.names)
    if table_exists:
        explicit = _read_sql(
            connection,
            f"SELECT question_id, candidate_id, is_realized, outcome_status, derivation FROM {table_name}",
        )
        explicit["is_realized"] = _nullable_bool(explicit["is_realized"])

    configured_records: list[dict[str, object]] = []
    for question_id, outcome in settings.get("official_outcomes", {}).items():
        realized_candidates = {str(candidate) for candidate in outcome["realized_candidates"]}
        question_candidates = set(
            options.loc[options["question_id"].eq(str(question_id)), "candidate_id"].astype(str)
        )
        if not realized_candidates.issubset(question_candidates):
            missing = sorted(realized_candidates - question_candidates)
            raise ValueError(
                f"Configured official outcome candidates are absent from {question_id}: {missing}"
            )
        derivation = str(outcome["derivation"])
        if outcome.get("source_url"):
            derivation = f"{derivation}; source={outcome['source_url']}"
        for candidate in sorted(question_candidates):
            configured_records.append(
                {
                    "question_id": str(question_id),
                    "candidate_id": candidate,
                    "is_realized": candidate in realized_candidates,
                    "outcome_status": str(outcome["outcome_status"]),
                    "derivation": derivation,
                }
            )
    configured = pd.DataFrame(configured_records, columns=SPECIAL_OUTCOMES_SCHEMA.names)
    if not configured.empty:
        explicit = pd.concat(
            [
                explicit[~explicit["question_id"].isin(configured["question_id"])],
                configured,
            ],
            ignore_index=True,
        )

    records: list[dict[str, object]] = []
    prefix = str(settings["group_question_prefix"])
    group_matches = matches[
        (matches["stage"] == "group_stage") & matches["actual_result_90"].notna()
    ].copy()
    for group_name, fixtures in group_matches.groupby("group_name", dropna=True):
        team_rows: list[dict[str, object]] = []
        teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
        for team in teams:
            points = goals_for = goals_against = 0
            for fixture in fixtures.itertuples(index=False):
                if team not in {fixture.home_team, fixture.away_team}:
                    continue
                is_home = team == fixture.home_team
                scored = int(fixture.home_score_90 if is_home else fixture.away_score_90)
                conceded = int(fixture.away_score_90 if is_home else fixture.home_score_90)
                goals_for += scored
                goals_against += conceded
                points += 3 if scored > conceded else 1 if scored == conceded else 0
            team_rows.append(
                {
                    "team": team,
                    "points": points,
                    "goal_difference": goals_for - goals_against,
                    "goals_for": goals_for,
                }
            )
        if len(fixtures) == int(settings["expected_group_matches"]):
            standings = pd.DataFrame(team_rows).sort_values(
                ["points", "goal_difference", "goals_for"], ascending=[False, False, False]
            )
            ranking_columns = ["points", "goal_difference", "goals_for"]
            if len(standings) > 1 and standings.iloc[0][ranking_columns].equals(
                standings.iloc[1][ranking_columns]
            ):
                continue
            winner = str(standings.iloc[0]["team"])
            sqlite_prefix = str(settings["sqlite_group_name_prefix"])
            normalized_group = str(group_name).removeprefix(sqlite_prefix)
            question_id = f"{prefix}{normalized_group}"
            for candidate in options.loc[
                options["question_id"] == question_id, "candidate_id"
            ].drop_duplicates():
                records.append(
                    {
                        "question_id": question_id,
                        "candidate_id": candidate,
                        "is_realized": candidate == winner,
                        "outcome_status": "resolved",
                        "derivation": "group standings from frozen SQLite match outcomes",
                    }
                )

    semifinal_id = str(settings["semifinal_question_id"])
    semifinals = matches[matches["stage"] == "semifinal"]
    semifinalists = {
        team
        for team in pd.concat([semifinals["home_team"], semifinals["away_team"]]).dropna()
        if team != "TBD"
    }
    if len(semifinalists) == int(settings["semifinal_k"]):
        for candidate in options.loc[
            options["question_id"] == semifinal_id, "candidate_id"
        ].drop_duplicates():
            records.append(
                {
                    "question_id": semifinal_id,
                    "candidate_id": candidate,
                    "is_realized": candidate in semifinalists,
                    "outcome_status": "resolved",
                    "derivation": "semifinal fixture participants from frozen SQLite",
                }
            )

    champion_id = str(settings["champion_question_id"])
    final = matches[matches["stage"] == "final"]
    if len(final) == 1 and final.iloc[0]["result_winner"] in {"home", "away"}:
        champion = _official_team(final.iloc[0], final.iloc[0]["result_winner"])
        for candidate in options.loc[
            options["question_id"] == champion_id, "candidate_id"
        ].drop_duplicates():
            records.append(
                {
                    "question_id": champion_id,
                    "candidate_id": candidate,
                    "is_realized": candidate == champion,
                    "outcome_status": "resolved",
                    "derivation": "final result from frozen SQLite",
                }
            )

    derived = pd.DataFrame(records, columns=SPECIAL_OUTCOMES_SCHEMA.names)
    if explicit.empty:
        return derived
    combined = pd.concat(
        [derived[~derived["question_id"].isin(explicit["question_id"])], explicit],
        ignore_index=True,
    )
    return combined


def _validate_predictions(
    predictions: pd.DataFrame, matches: pd.DataFrame, config: AnalysisConfig
) -> dict[str, Any]:
    tolerance = float(config.section("validation")["probability_tolerance"])
    valid = predictions[predictions["is_valid_for_scoring"].fillna(False)].copy()
    sums = valid[["home_win_90_prob", "draw_90_prob", "away_win_90_prob"]].sum(axis=1, min_count=3)
    bad_probability_sums = int((~np.isclose(sums, 1.0, atol=tolerance, rtol=0.0)).sum())
    probabilities = valid[["home_win_90_prob", "draw_90_prob", "away_win_90_prob"]]
    bad_probability_ranges = int(
        ((probabilities < -tolerance) | (probabilities > 1.0 + tolerance)).any(axis=1).sum()
    )
    stable_key = config.section("validation")["stable_key"]
    duplicate_keys = int(predictions.duplicated(stable_key, keep=False).sum())
    unknown_matches = int((~predictions["match_id"].isin(matches["match_id"])).sum())
    if bad_probability_sums or bad_probability_ranges or duplicate_keys or unknown_matches:
        raise ValueError(
            "Prediction validation failed: "
            f"bad sums={bad_probability_sums}, bad ranges={bad_probability_ranges}, "
            f"duplicate keys={duplicate_keys}, unknown matches={unknown_matches}"
        )
    return {
        "valid_prediction_count": len(valid),
        "bad_probability_sums": bad_probability_sums,
        "bad_probability_ranges": bad_probability_ranges,
        "duplicate_stable_keys": duplicate_keys,
        "unknown_match_references": unknown_matches,
    }


def _validate_complete_panel(panel: pd.DataFrame, config: AnalysisConfig) -> dict[str, Any]:
    design = config.section("design")
    completed = panel[panel["actual_result_90"].notna()]
    primary = completed[
        (completed["forecast_horizon"] == design["primary_horizon"])
        & completed["is_valid_for_scoring"].fillna(False)
    ]
    expected_cells = (
        len(completed["match_id"].unique())
        * len(design["access_conditions"])
        * len(design["prompt_strategies"])
        * len(design["sample_ids"])
    )
    counts = primary.groupby("model_id").size().to_dict()
    missing = {
        model: expected_cells - int(counts.get(model, 0))
        for model in design["complete_panel"]
        if int(counts.get(model, 0)) != expected_cells
    }
    if config.is_final and design["require_complete_panel_in_final"] and missing:
        raise ValueError(f"Configured complete panel is incomplete in final mode: {missing}")
    return {
        "expected_primary_cells_per_model": expected_cells,
        "observed_cells": counts,
        "configured_panel_missing_cells": missing,
    }


def derive_tables(config: AnalysisConfig, manifest: Manifest) -> dict[str, Path]:
    frozen_record = manifest.require("frozen_db")
    frozen = Path(frozen_record.path)
    connection = sqlite3.connect(f"file:{frozen.as_posix()}?mode=ro", uri=True)
    try:
        matches = _matches(connection)
        predictions = _predictions(connection)
        stored = _stored_evaluations(connection)
        specials = _special_predictions(connection)
        options = _special_options(connection)
        special_outcomes = _derived_special_outcomes(connection, matches, options, config)
    finally:
        connection.close()

    prediction_validation = _validate_predictions(predictions, matches, config)
    joined = predictions.merge(matches, on="match_id", how="left", validate="many_to_one")
    joined = joined.merge(stored, on="prediction_id", how="left", validate="one_to_one")
    evaluations = recompute_evaluations(joined, config)
    tolerance = float(config.section("validation")["reconciliation_absolute_tolerance"])
    metric_pairs = [
        ("brier_90_stored", "brier_90_recomputed"),
        ("log_loss_90_stored", "log_loss_90_recomputed"),
    ]
    metric_mismatches: dict[str, int] = {}
    for stored_name, computed_name in metric_pairs:
        comparable = evaluations[[stored_name, computed_name]].dropna()
        metric_mismatches[computed_name] = int(
            (
                ~np.isclose(
                    comparable[stored_name], comparable[computed_name], atol=tolerance, rtol=0.0
                )
            ).sum()
        )
    if any(metric_mismatches.values()):
        raise ValueError(f"Stored/recomputed metric mismatch: {metric_mismatches}")

    panel = joined.drop(
        columns=[
            "brier_90",
            "log_loss_90",
            "advancement_brier",
            "advancement_log_loss",
            "advancement_accuracy",
            "score_result_matches_prob_argmax_90",
        ]
    ).merge(
        evaluations,
        on=["prediction_id", "match_id", "actual_result_90"],
        how="left",
        validate="one_to_one",
    )
    complete_panel_validation = _validate_complete_panel(panel, config)
    rationale = predictions.loc[
        predictions["reason"].notna() & predictions["reason"].str.strip().ne(""),
        [
            "prediction_id",
            "match_id",
            "model_id",
            "forecast_horizon",
            "access_condition",
            "prompt_strategy",
            "reason",
        ],
    ].rename(columns={"reason": "rationale_text"})
    tool_events = predictions[
        [
            "prediction_id",
            "match_id",
            "model_id",
            "forecast_horizon",
            "access_condition",
            "prompt_strategy",
            "tools_enabled",
            "tool_calls_observed",
            "num_tool_calls",
            "tool_trace_available",
            "tool_trace",
            "open_book_compliance",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        ]
    ].copy()

    directory = config.resolve_path("derived")
    outputs = {
        "matches": directory / "matches.parquet",
        "predictions": directory / "predictions.parquet",
        "evaluations": directory / "evaluations.parquet",
        "analysis_panel": directory / "analysis_panel.parquet",
        "tool_events": directory / "tool_events.parquet",
        "rationale_corpus": directory / "rationale_corpus.parquet",
        "special_predictions": directory / "special_predictions.parquet",
        "special_options": directory / "special_options.parquet",
        "special_outcomes": directory / "special_outcomes.parquet",
    }
    _write_parquet(matches, outputs["matches"], MATCHES_SCHEMA)
    _write_parquet(predictions, outputs["predictions"], PREDICTIONS_SCHEMA)
    _write_parquet(evaluations, outputs["evaluations"], EVALUATIONS_SCHEMA)
    _write_parquet(panel, outputs["analysis_panel"])
    _write_parquet(tool_events, outputs["tool_events"])
    _write_parquet(rationale, outputs["rationale_corpus"])
    _write_parquet(specials, outputs["special_predictions"], SPECIAL_PREDICTIONS_SCHEMA)
    _write_parquet(options, outputs["special_options"], SPECIAL_OPTIONS_SCHEMA)
    _write_parquet(special_outcomes, outputs["special_outcomes"], SPECIAL_OUTCOMES_SCHEMA)

    frames = {
        "matches": matches,
        "predictions": predictions,
        "evaluations": evaluations,
        "analysis_panel": panel,
        "tool_events": tool_events,
        "rationale_corpus": rationale,
        "special_predictions": specials,
        "special_options": options,
        "special_outcomes": special_outcomes,
    }
    source_hashes = {"frozen_db": frozen_record.sha256}
    key_columns = {
        "matches": ["match_id"],
        "predictions": ["prediction_id"],
        "evaluations": ["prediction_id"],
        "analysis_panel": config.section("validation")["stable_key"],
        "tool_events": ["prediction_id"],
        "rationale_corpus": ["prediction_id"],
        "special_predictions": ["prediction_id"],
        "special_options": ["option_id"],
        "special_outcomes": ["question_id", "candidate_id"],
    }
    for name, path in outputs.items():
        duplicate_keys = int(frames[name].duplicated(key_columns[name]).sum())
        if duplicate_keys:
            raise ValueError(f"Derived table {name} has {duplicate_keys} duplicate keys")
        manifest.add(
            f"derived_{name}",
            path,
            "parquet",
            "derive",
            source_hashes,
            {
                "rows": len(frames[name]),
                "key_columns": key_columns[name],
                "duplicate_keys": duplicate_keys,
                "repository_commit": git_commit(config.root.parent),
            },
        )
    verification = {
        "prediction_validation": prediction_validation,
        "metric_mismatches": metric_mismatches,
        "complete_panel": complete_panel_validation,
        "derived_rows": {
            name: int(pq.read_metadata(path).num_rows) for name, path in outputs.items()
        },
    }
    verification_path = config.resolve_path("verification") / "derived_validation.json"
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest.add(
        "derived_validation", verification_path, "json", "derive", source_hashes, verification
    )
    manifest.write()
    return outputs
