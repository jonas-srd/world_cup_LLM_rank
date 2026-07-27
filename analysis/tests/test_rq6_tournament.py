from __future__ import annotations

import numpy as np
import pandas as pd

from soccerarena_analysis.analyses.rq6_tournament import (
    _analysis_prediction_panel,
    _semifinalists,
    _uniform_random_reference,
    _validate_normalized_inputs,
)


def test_normalized_tournament_tables_match_registered_design(config):
    predictions = pd.read_parquet(config.resolve_path("derived") / "special_predictions.parquet")
    options = pd.read_parquet(config.resolve_path("derived") / "special_options.parquet")
    outcomes = pd.read_parquet(config.resolve_path("derived") / "special_outcomes.parquet")
    _validate_normalized_inputs(predictions, options, outcomes, config)
    counts = predictions.groupby("question_id")["prediction_id"].nunique()
    assert counts.eq(config.section("special_questions")["expected_forecasts_per_question"]).all()
    analysis_panel = _analysis_prediction_panel(predictions, config)
    analysis_counts = analysis_panel.groupby("question_id")["prediction_id"].nunique()
    assert analysis_counts.eq(
        config.section("special_questions")["analysis_forecasts_per_question"]
    ).all()
    assert config.section("design")["partial_models"]["fable"] not in set(
        analysis_panel["model_id"]
    )


def test_semifinal_marginals_are_audited_without_normalization():
    question = pd.DataFrame(
        {
            "prediction_id": ["p1"] * 4,
            "candidate_id": ["A", "B", "C", "D"],
            "probability": [0.9, 0.8, 0.7, 0.6],
            "is_final_pick": [True, True, False, False],
        }
    )
    outcomes = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C", "D"],
            "is_realized": [True, False, True, False],
        }
    )
    result = _semifinalists(question, outcomes, expected_k=2).iloc[0]
    assert np.isclose(result["semifinal_probability_sum"], 3.0)
    assert np.isclose(result["absolute_sum_from_expected_k"], 1.0)
    assert result["correct_count"] == 1


def test_uniform_random_references_follow_question_candidate_universe():
    options = pd.DataFrame(
        {
            "question_id": ["group_a"] * 8 + ["semifinalists"] * 96 + ["winner"] * 96,
            "prediction_id": (
                ["group_1"] * 4
                + ["group_2"] * 4
                + ["semi_1"] * 48
                + ["semi_2"] * 48
                + ["winner_1"] * 48
                + ["winner_2"] * 48
            ),
            "candidate_id": (
                list("ABCD")
                + list("ABCD")
                + [f"T{i}" for i in range(48)]
                + [f"T{i}" for i in range(48)]
                + [f"T{i}" for i in range(48)]
                + [f"T{i}" for i in range(48)]
            ),
        }
    )
    assert np.isclose(_uniform_random_reference(options, "group_a", 1), 0.25)
    assert np.isclose(_uniform_random_reference(options, "semifinalists", 4), 4 / 48)
    assert np.isclose(_uniform_random_reference(options, "winner", 1), 1 / 48)
