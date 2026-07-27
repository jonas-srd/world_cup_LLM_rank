from __future__ import annotations

import pandas as pd
import pytest

from soccerarena_analysis.analyses.closing_odds import score_forecasts
from soccerarena_analysis.analyses.closing_odds_t24 import aggregate_scored_cells


def _cells() -> pd.DataFrame:
    probabilities = (
        (0.80, 0.10, 0.10),
        (0.60, 0.20, 0.20),
    )
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "stage": "group_stage",
                "kickoff_utc": "2026-06-11T00:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "actual_result_90": "H",
                "forecaster_id": "model",
                "forecaster_type": "individual_llm",
                "prob_home": home,
                "prob_draw": draw,
                "prob_away": away,
            }
            for home, draw, away in probabilities
        ]
    )


def test_t24_aggregation_averages_cell_scores_not_rescored_mean_probability() -> None:
    scored = score_forecasts(_cells())
    aggregated = aggregate_scored_cells(scored)

    assert len(aggregated) == 1
    assert aggregated.loc[0, "condition_rows"] == 2
    assert aggregated.loc[0, "brier"] == pytest.approx(scored["brier"].mean())
    mean_probability_score = score_forecasts(
        _cells()
        .iloc[[0]]
        .assign(
            prob_home=scored["prob_home"].mean(),
            prob_draw=scored["prob_draw"].mean(),
            prob_away=scored["prob_away"].mean(),
        )
    ).loc[0, "brier"]
    assert aggregated.loc[0, "brier"] != pytest.approx(mean_probability_score)


def test_t24_aggregation_requires_both_access_cells() -> None:
    with pytest.raises(ValueError, match="two probabilities-first access cells"):
        aggregate_scored_cells(score_forecasts(_cells().iloc[:1]))
