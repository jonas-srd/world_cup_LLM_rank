from __future__ import annotations

import numpy as np
import pandas as pd

from soccerarena_analysis.analyses.rq1_access import _chronological_access_trend
from soccerarena_analysis.analyses.rq2_elicitation import (
    _derive_draw_structure,
    _format_p,
    _reverse_effect,
)
from soccerarena_analysis.analyses.rq3_calibration import _within_cell_confidence_groups
from soccerarena_analysis.analyses.rq4_diversity import _ensemble
from soccerarena_analysis.stages.external_baselines import (
    aggregate_bookmakers,
    remove_bookmaker_overround,
)
from soccerarena_analysis.statistics.bootstrap import studentized_cluster_bootstrap
from soccerarena_analysis.statistics.metrics import (
    brier_score,
    probability_modal_outcomes,
    top_outcome_fractional_accuracy,
)
from soccerarena_analysis.statistics.multiplicity import holm_adjust


def test_bootstrap_and_sign_flip_are_reproducible(fast_config):
    frame = pd.DataFrame(
        {
            "match_id": [f"m{index}" for index in range(12)],
            "stage": ["group_stage"] * 8 + ["round_of_16"] * 4,
            "difference": np.linspace(-0.03, 0.08, 12),
        }
    )
    first = studentized_cluster_bootstrap(
        frame, "difference", "stage", fast_config, "test.reproducibility"
    )
    second = studentized_cluster_bootstrap(
        frame, "difference", "stage", fast_config, "test.reproducibility"
    )
    assert first == second
    assert first.replicates == 200
    assert first.permutation_replicates == 200
    assert 0 < first.p_raw <= 1


def test_bootstrap_rejects_more_than_one_unit_per_match(fast_config):
    frame = pd.DataFrame(
        {"match_id": ["m1", "m1"], "stage": ["group_stage"] * 2, "difference": [0.1, 0.2]}
    )
    try:
        studentized_cluster_bootstrap(frame, "difference", "stage", fast_config, "test.duplicates")
    except ValueError as error:
        assert "one value per match" in str(error)
    else:
        raise AssertionError("duplicate match units were not rejected")


def test_holm_adjustment_is_monotone():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_top_outcome_accuracy_uses_fractional_credit_for_ties(fast_config):
    tolerance = float(fast_config.section("metrics")["top_outcome"]["tie_tolerance"])
    probabilities = (0.4, 0.4, 0.2)
    assert probability_modal_outcomes(probabilities, tolerance) == ("H", "D")
    assert top_outcome_fractional_accuracy(probabilities, "H", tolerance) == 0.5
    assert top_outcome_fractional_accuracy(probabilities, "D", tolerance) == 0.5
    assert top_outcome_fractional_accuracy(probabilities, "A", tolerance) == 0.0
    assert top_outcome_fractional_accuracy((0.2, 0.3, 0.5), "A", tolerance) == 1.0


def test_rq2_advantage_reverses_effect_and_interval():
    original = {
        "estimate": 0.2,
        "ci_low": -0.1,
        "ci_high": 0.4,
        "median": 0.05,
        "p_raw": 0.3,
    }
    reversed_effect = _reverse_effect(original)
    assert reversed_effect["estimate"] == -0.2
    assert reversed_effect["ci_low"] == -0.4
    assert reversed_effect["ci_high"] == 0.1
    assert reversed_effect["median"] == -0.05
    assert reversed_effect["p_raw"] == original["p_raw"]


def test_rq2_draw_structure_separates_score_and_probability_draws(fast_config):
    frame = pd.DataFrame(
        {
            "predicted_home_90": [1.0, 2.0],
            "predicted_away_90": [1.0, 1.0],
            "home_win_90_prob": [0.45, 0.35],
            "draw_90_prob": [0.30, 0.40],
            "away_win_90_prob": [0.25, 0.25],
        }
    )
    derived = _derive_draw_structure(frame, fast_config)
    assert derived["score_implied_draw"].tolist() == [1.0, 0.0]
    assert derived["exact_1_1_score"].tolist() == [1.0, 0.0]
    assert derived["score_draw_probability_nondraw"].tolist() == [1.0, 0.0]


def test_small_p_values_are_not_rendered_as_zero():
    assert _format_p(0.0001, 3) == "<0.001"
    assert _format_p(0.0126, 3) == "0.013"


def test_rq3_confidence_groups_preserve_ties_within_cell(fast_config):
    frame = pd.DataFrame(
        {
            "model_id": ["m"] * 8,
            "access_condition": ["open_book"] * 8,
            "prompt_strategy": ["direct_score"] * 8,
            "confidence": [0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.9],
        }
    )
    groups = _within_cell_confidence_groups(frame, fast_config)
    paired = frame.assign(group=groups).groupby("confidence")["group"].nunique()
    assert paired.eq(1).all()
    assert groups.min() >= 0
    assert groups.max() < int(fast_config.section("calibration")["confidence_groups"])


def test_rq4_average_member_comparison_is_algebraic_not_in_holm_family(fast_config):
    records = []
    outcomes = ["H", "D", "A"]
    models = fast_config.section("design")["complete_panel"]
    for match_index in range(12):
        actual = outcomes[match_index % len(outcomes)]
        for model_index, model_id in enumerate(models):
            home = 0.36 + 0.012 * model_index + 0.003 * (match_index % 4)
            draw = 0.28 - 0.004 * model_index
            away = 1.0 - home - draw
            probabilities = (home, draw, away)
            records.append(
                {
                    "match_id": f"m{match_index}",
                    "stage": "group_stage" if match_index < 8 else "round_of_16",
                    "forecast_horizon": "T_24H",
                    "access_condition": "closed_book",
                    "prompt_strategy": "direct_score",
                    "model_id": model_id,
                    "actual_result_90": actual,
                    "home_win_90_prob": home,
                    "draw_90_prob": draw,
                    "away_win_90_prob": away,
                    "brier_90_recomputed": brier_score(probabilities, actual),
                }
            )
    differences, results = _ensemble(pd.DataFrame(records), fast_config)
    average = differences[differences["member"] == "average_member"]
    assert np.allclose(
        average["difference"],
        -average["outcome_independent_dispersion"],
    )
    average_result = results[results["comparison"] == "average_member"].iloc[0]
    assert np.isnan(average_result["p_raw"])
    assert np.isnan(average_result["p_adjusted"])
    model_results = results[results["comparison"] != "average_member"]
    assert model_results["p_adjusted"].notna().all()


def test_chronological_access_trend_is_reproducible(fast_config):
    frame = pd.DataFrame(
        {
            "match_id": [f"m{index}" for index in range(12)],
            "stage": ["group_stage"] * 8 + ["round_of_16"] * 4,
            "kickoff_utc": pd.date_range("2026-06-10", periods=12, freq="D", tz="UTC"),
            "difference": np.linspace(-0.04, 0.08, 12),
        }
    )
    first = _chronological_access_trend(frame, fast_config)
    second = _chronological_access_trend(frame, fast_config)
    columns = ["smoothed_difference", "band_low", "band_high"]
    assert np.allclose(first[columns], second[columns])
    assert first["match_number"].tolist() == list(range(1, 13))
    assert (first["band_low"] <= first["band_high"]).all()


def test_bookmaker_overround_is_removed_before_median():
    odds = pd.DataFrame(
        {
            "fixture_id": ["f1", "f1"],
            "bookmaker": ["a", "b"],
            "odds_home": [2.0, 2.2],
            "odds_draw": [3.2, 3.0],
            "odds_away": [4.0, 3.8],
        }
    )
    individual = remove_bookmaker_overround(odds)
    assert np.allclose(individual[["prob_home", "prob_draw", "prob_away"]].sum(axis=1), 1.0)
    consensus = aggregate_bookmakers(odds)
    assert np.allclose(consensus[["prob_home", "prob_draw", "prob_away"]].sum(axis=1), 1.0)
    aggregated = aggregate_bookmakers(odds)
    assert aggregated.loc[0, "n_bookmakers"] == 2
    assert np.isclose(aggregated.loc[0, "prob_home"], individual["prob_home"].median())
