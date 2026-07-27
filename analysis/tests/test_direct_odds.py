from __future__ import annotations

import pandas as pd
import pytest

from soccerarena_analysis.analyses.closing_odds import MARKET_ID
from soccerarena_analysis.analyses.direct_odds import (
    T2_CLOSING,
    T24_ODDS,
    _comparison_forecasts,
)


@pytest.mark.parametrize(
    ("scenario", "horizon", "baseline"),
    [
        (T24_ODDS, "T_24H", "t24_odds"),
        (T2_CLOSING, "T_2H", "closing_odds"),
    ],
)
def test_direct_odds_uses_only_open_book_direct_score(
    config,
    scenario,
    horizon: str,
    baseline: str,
) -> None:
    rows: list[dict[str, object]] = []
    other_horizon = "T_2H" if horizon == "T_24H" else "T_24H"
    for model in config.section("design")["complete_panel"]:
        common = {
            "match_id": "m1",
            "model_id": model,
            "is_valid_for_scoring": True,
            "actual_result_90": "H",
            "stage": "group_stage",
            "kickoff_utc": "2026-06-11T19:00:00Z",
            "home_team": "A",
            "away_team": "B",
            "minutes_before_kickoff": 1_440.0 if horizon == "T_24H" else 120.0,
        }
        rows.extend(
            [
                {
                    **common,
                    "forecast_horizon": horizon,
                    "access_condition": "open_book",
                    "prompt_strategy": "direct_score",
                    "home_win_90_prob": 0.60,
                    "draw_90_prob": 0.25,
                    "away_win_90_prob": 0.15,
                },
                {
                    **common,
                    "forecast_horizon": horizon,
                    "access_condition": "open_book",
                    "prompt_strategy": "probabilistic_forecast",
                    "home_win_90_prob": 0.10,
                    "draw_90_prob": 0.20,
                    "away_win_90_prob": 0.70,
                },
                {
                    **common,
                    "forecast_horizon": horizon,
                    "access_condition": "closed_book",
                    "prompt_strategy": "direct_score",
                    "home_win_90_prob": 0.20,
                    "draw_90_prob": 0.30,
                    "away_win_90_prob": 0.50,
                },
                {
                    **common,
                    "forecast_horizon": other_horizon,
                    "access_condition": "open_book",
                    "prompt_strategy": "direct_score",
                    "home_win_90_prob": 0.30,
                    "draw_90_prob": 0.30,
                    "away_win_90_prob": 0.40,
                },
            ]
        )
    panel = pd.DataFrame(rows)
    external = pd.DataFrame(
        [
            {
                "baseline": baseline,
                "match_id": "m1",
                "snapshot": (
                    "2026-06-10T18:55:37Z" if horizon == "T_24H" else "2026-06-11T18:59:59Z"
                ),
                "prob_home": 0.55,
                "prob_draw": 0.25,
                "prob_away": 0.20,
                "n_bookmakers": 20,
            }
        ]
    )

    forecasts, metadata = _comparison_forecasts(panel, external, config, scenario)

    model_rows = forecasts[forecasts["forecaster_type"].eq("individual_llm")]
    assert len(model_rows) == len(config.section("design")["complete_panel"])
    assert model_rows["prob_home"].eq(0.60).all()
    market = forecasts[forecasts["forecaster_id"].eq(MARKET_ID)]
    assert market["prob_home"].item() == 0.55
    assert metadata["horizon"] == horizon
    assert metadata["prompt_strategy"] == "direct_score"
    assert metadata["access_condition"] == "open_book"
