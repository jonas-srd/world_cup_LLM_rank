from __future__ import annotations

import numpy as np
import pandas as pd

from soccerarena_analysis.analyses.closing_odds import (
    ranked_probability_score,
    score_forecasts,
)


def test_ranked_probability_score_extremes():
    assert ranked_probability_score((1.0, 0.0, 0.0), "H") == 0.0
    assert ranked_probability_score((1.0, 0.0, 0.0), "A") == 1.0


def test_score_forecasts_matches_multiclass_definitions():
    frame = pd.DataFrame(
        {
            "match_id": ["m1"],
            "stage": ["group_stage"],
            "forecaster_id": ["model"],
            "actual_result_90": ["H"],
            "prob_home": [0.5],
            "prob_draw": [0.3],
            "prob_away": [0.2],
        }
    )
    scored = score_forecasts(frame).iloc[0]
    assert np.isclose(scored["brier"], 0.38)
    assert np.isclose(scored["log_loss"], -np.log(0.5))
    assert np.isclose(scored["rps"], ((0.5 - 1.0) ** 2 + (0.8 - 1.0) ** 2) / 2.0)
    assert scored["accuracy"] == 1.0


def test_score_forecasts_uses_fractional_credit_for_modal_ties():
    frame = pd.DataFrame(
        {
            "match_id": ["m1"],
            "stage": ["group_stage"],
            "forecaster_id": ["model"],
            "actual_result_90": ["H"],
            "prob_home": [0.4],
            "prob_draw": [0.4],
            "prob_away": [0.2],
        }
    )
    assert score_forecasts(frame).iloc[0]["accuracy"] == 0.5
