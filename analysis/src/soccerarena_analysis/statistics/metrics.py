from __future__ import annotations

import math

import pandas as pd

from ..config import AnalysisConfig

OUTCOMES = ("H", "D", "A")


def probability_modal_outcomes(
    probabilities: tuple[float, float, float], tie_tolerance: float
) -> tuple[str, ...]:
    """Return all H/D/A outcomes tied for the largest elicited probability."""
    if any(pd.isna(value) for value in probabilities):
        return ()
    maximum = max(float(value) for value in probabilities)
    return tuple(
        outcome
        for outcome, probability in zip(OUTCOMES, probabilities)
        if abs(float(probability) - maximum) <= tie_tolerance
    )


def top_outcome_fractional_accuracy(
    probabilities: tuple[float, float, float], actual: str | None, tie_tolerance: float
) -> float | None:
    """Score tied modal outcomes by expected accuracy under uniform tie-breaking."""
    modal_outcomes = probability_modal_outcomes(probabilities, tie_tolerance)
    if actual not in OUTCOMES or not modal_outcomes:
        return None
    return 1.0 / len(modal_outcomes) if actual in modal_outcomes else 0.0


def result_from_scores(home: float | int | None, away: float | int | None) -> str | None:
    if pd.isna(home) or pd.isna(away):
        return None
    if float(home) > float(away):
        return "H"
    if float(home) < float(away):
        return "A"
    return "D"


def brier_score(probabilities: tuple[float, float, float], actual: str | None) -> float | None:
    if actual not in OUTCOMES or any(pd.isna(value) for value in probabilities):
        return None
    target = [1.0 if outcome == actual else 0.0 for outcome in OUTCOMES]
    return float(
        sum((probability - truth) ** 2 for probability, truth in zip(probabilities, target))
    )


def log_loss(
    probabilities: tuple[float, float, float], actual: str | None, epsilon: float
) -> float | None:
    if actual not in OUTCOMES or any(pd.isna(value) for value in probabilities):
        return None
    probability = probabilities[OUTCOMES.index(actual)]
    return float(-math.log(min(max(float(probability), epsilon), 1.0 - epsilon)))


def advancement_metrics(
    home_probability: float | None, actual: str | None, epsilon: float
) -> tuple[float | None, float | None, bool | None]:
    if actual not in {"home", "away"} or pd.isna(home_probability):
        return None, None, None
    probability = float(home_probability)
    truth = 1.0 if actual == "home" else 0.0
    brier = (probability - truth) ** 2 + ((1.0 - probability) - (1.0 - truth)) ** 2
    actual_probability = probability if truth == 1.0 else 1.0 - probability
    loss = -math.log(min(max(actual_probability, epsilon), 1.0 - epsilon))
    predicted = "home" if probability >= 0.5 else "away"
    return float(brier), float(loss), predicted == actual


def kicktipp_points(
    pred_home: float | None,
    pred_away: float | None,
    actual_home: float | None,
    actual_away: float | None,
    config: AnalysisConfig,
) -> float | None:
    if any(pd.isna(value) for value in (pred_home, pred_away, actual_home, actual_away)):
        return None
    points = config.section("metrics")["kicktipp_points"]
    if int(pred_home) == int(actual_home) and int(pred_away) == int(actual_away):
        return float(points["exact"])
    if int(pred_home) - int(pred_away) == int(actual_home) - int(actual_away):
        return float(points["goal_difference"])
    if result_from_scores(pred_home, pred_away) == result_from_scores(actual_home, actual_away):
        return float(points["tendency"])
    return float(points["miss"])


def recompute_evaluations(panel: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    epsilon = float(config.section("statistics")["log_loss_epsilon"])
    top_outcome_config = config.section("metrics")["top_outcome"]
    if top_outcome_config["tie_rule"] != "fractional_credit":
        raise ValueError(
            "Only the prespecified fractional-credit top-outcome tie rule is supported"
        )
    tie_tolerance = float(top_outcome_config["tie_tolerance"])
    records: list[dict[str, object]] = []
    for row in panel.itertuples(index=False):
        probabilities = (row.home_win_90_prob, row.draw_90_prob, row.away_win_90_prob)
        actual = row.actual_result_90
        brier = brier_score(probabilities, actual)
        loss = log_loss(probabilities, actual, epsilon)
        probability_results = probability_modal_outcomes(probabilities, tie_tolerance)
        score_result = result_from_scores(row.predicted_home_90, row.predicted_away_90)
        actual_goal_difference = (
            None
            if pd.isna(row.home_score_90) or pd.isna(row.away_score_90)
            else int(row.home_score_90) - int(row.away_score_90)
        )
        predicted_goal_difference = (
            None
            if pd.isna(row.predicted_home_90) or pd.isna(row.predicted_away_90)
            else int(row.predicted_home_90) - int(row.predicted_away_90)
        )
        advancement_brier, advancement_loss, advancement_accuracy = advancement_metrics(
            row.home_advances_prob, row.actual_advancer, epsilon
        )
        records.append(
            {
                "prediction_id": row.prediction_id,
                "match_id": row.match_id,
                "actual_result_90": actual,
                "brier_90_stored": row.brier_90,
                "brier_90_recomputed": brier,
                "log_loss_90_stored": row.log_loss_90,
                "log_loss_90_recomputed": loss,
                "top_outcome_accuracy_90_recomputed": top_outcome_fractional_accuracy(
                    probabilities, actual, tie_tolerance
                ),
                "exact_score_90_correct_recomputed": None
                if pd.isna(row.predicted_home_90) or pd.isna(row.home_score_90)
                else int(row.predicted_home_90) == int(row.home_score_90)
                and int(row.predicted_away_90) == int(row.away_score_90),
                "goal_difference_90_correct_recomputed": None
                if actual_goal_difference is None or predicted_goal_difference is None
                else actual_goal_difference == predicted_goal_difference,
                "tendency_90_correct_recomputed": None
                if score_result is None or actual is None
                else score_result == actual,
                "kicktipp_points_90_recomputed": kicktipp_points(
                    row.predicted_home_90,
                    row.predicted_away_90,
                    row.home_score_90,
                    row.away_score_90,
                    config,
                ),
                "advancement_brier_recomputed": advancement_brier,
                "advancement_log_loss_recomputed": advancement_loss,
                "advancement_accuracy_recomputed": advancement_accuracy,
                "score_probability_consistent_recomputed": None
                if not probability_results or score_result is None
                else score_result in probability_results,
            }
        )
    return pd.DataFrame.from_records(records)
