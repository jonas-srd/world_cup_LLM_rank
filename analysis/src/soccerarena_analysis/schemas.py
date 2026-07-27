from __future__ import annotations

import pyarrow as pa

MATCHES_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("kickoff_utc", pa.timestamp("us", tz="UTC")),
        ("competition", pa.string()),
        ("home_team", pa.string()),
        ("away_team", pa.string()),
        ("status", pa.string()),
        ("stage", pa.string()),
        ("group_name", pa.string()),
        ("matchday", pa.int64()),
        ("is_knockout", pa.bool_()),
        ("home_score_90", pa.int64()),
        ("away_score_90", pa.int64()),
        ("home_score_full", pa.int64()),
        ("away_score_full", pa.int64()),
        ("actual_result_90", pa.string()),
        ("actual_advancer", pa.string()),
        ("result_winner", pa.string()),
        ("source", pa.string()),
        ("source_match_id", pa.string()),
    ]
)


PREDICTIONS_SCHEMA = pa.schema(
    [
        ("prediction_id", pa.string()),
        ("match_id", pa.string()),
        ("model_id", pa.string()),
        ("model_version", pa.string()),
        ("provider", pa.string()),
        ("forecast_horizon", pa.string()),
        ("access_condition", pa.string()),
        ("prompt_strategy", pa.string()),
        ("sample_id", pa.int64()),
        ("scheduled_prediction_time_utc", pa.timestamp("us", tz="UTC")),
        ("actual_prediction_time_utc", pa.timestamp("us", tz="UTC")),
        ("kickoff_time_utc", pa.timestamp("us", tz="UTC")),
        ("minutes_before_kickoff", pa.float64()),
        ("timing_status", pa.string()),
        ("home_win_90_prob", pa.float64()),
        ("draw_90_prob", pa.float64()),
        ("away_win_90_prob", pa.float64()),
        ("expected_home_goals_90", pa.float64()),
        ("expected_away_goals_90", pa.float64()),
        ("home_advances_prob", pa.float64()),
        ("away_advances_prob", pa.float64()),
        ("predicted_home_90", pa.float64()),
        ("predicted_away_90", pa.float64()),
        ("confidence", pa.float64()),
        ("reason", pa.string()),
        ("validation_status", pa.string()),
        ("is_valid_for_scoring", pa.bool_()),
        ("repair_attempted", pa.bool_()),
        ("normalization_applied", pa.bool_()),
        ("tools_enabled", pa.bool_()),
        ("tool_calls_observed", pa.bool_()),
        ("num_tool_calls", pa.float64()),
        ("tool_trace_available", pa.bool_()),
        ("tool_trace", pa.string()),
        ("open_book_compliance", pa.string()),
        ("input_tokens", pa.float64()),
        ("output_tokens", pa.float64()),
        ("latency_ms", pa.float64()),
        ("cost_usd", pa.float64()),
        ("raw_prompt", pa.string()),
        ("raw_response", pa.string()),
        ("prompt_hash", pa.string()),
    ]
)


EVALUATIONS_SCHEMA = pa.schema(
    [
        ("prediction_id", pa.string()),
        ("match_id", pa.string()),
        ("actual_result_90", pa.string()),
        ("brier_90_stored", pa.float64()),
        ("brier_90_recomputed", pa.float64()),
        ("log_loss_90_stored", pa.float64()),
        ("log_loss_90_recomputed", pa.float64()),
        ("top_outcome_accuracy_90_recomputed", pa.float64()),
        ("exact_score_90_correct_recomputed", pa.bool_()),
        ("goal_difference_90_correct_recomputed", pa.bool_()),
        ("tendency_90_correct_recomputed", pa.bool_()),
        ("kicktipp_points_90_recomputed", pa.float64()),
        ("advancement_brier_recomputed", pa.float64()),
        ("advancement_log_loss_recomputed", pa.float64()),
        ("advancement_accuracy_recomputed", pa.bool_()),
        ("score_probability_consistent_recomputed", pa.bool_()),
    ]
)


SPECIAL_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("prediction_id", pa.string()),
        ("question_id", pa.string()),
        ("question_label", pa.string()),
        ("prediction_type", pa.string()),
        ("k", pa.float64()),
        ("model_id", pa.string()),
        ("model_version", pa.string()),
        ("access_condition", pa.string()),
        ("prompt_strategy", pa.string()),
        ("forecast_horizon", pa.string()),
        ("sample_id", pa.int64()),
        ("final_pick", pa.string()),
        ("final_picks", pa.string()),
        ("confidence", pa.float64()),
        ("reasoning_summary", pa.string()),
        ("is_valid_for_scoring", pa.bool_()),
        ("actual_prediction_time_utc", pa.timestamp("us", tz="UTC")),
    ]
)


SPECIAL_OPTIONS_SCHEMA = pa.schema(
    [
        ("option_id", pa.string()),
        ("prediction_id", pa.string()),
        ("question_id", pa.string()),
        ("candidate_id", pa.string()),
        ("candidate_label", pa.string()),
        ("candidate_type", pa.string()),
        ("probability", pa.float64()),
        ("rank", pa.int64()),
        ("is_final_pick", pa.bool_()),
    ]
)


SPECIAL_OUTCOMES_SCHEMA = pa.schema(
    [
        ("question_id", pa.string()),
        ("candidate_id", pa.string()),
        ("is_realized", pa.bool_()),
        ("outcome_status", pa.string()),
        ("derivation", pa.string()),
    ]
)
