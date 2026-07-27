from __future__ import annotations

import pandas as pd

from soccerarena_analysis.analyses.closing_odds import MARKET_ID
from soccerarena_analysis.analyses.t24_odds import _comparison_forecasts


def test_t24_odds_uses_only_open_book_probabilities_first(config) -> None:
    rows: list[dict[str, object]] = []
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
            "minutes_before_kickoff": 1_440.0,
        }
        rows.extend(
            [
                {
                    **common,
                    "forecast_horizon": "T_24H",
                    "access_condition": "open_book",
                    "prompt_strategy": "probabilistic_forecast",
                    "home_win_90_prob": 0.60,
                    "draw_90_prob": 0.25,
                    "away_win_90_prob": 0.15,
                },
                {
                    **common,
                    "forecast_horizon": "T_24H",
                    "access_condition": "closed_book",
                    "prompt_strategy": "probabilistic_forecast",
                    "home_win_90_prob": 0.10,
                    "draw_90_prob": 0.20,
                    "away_win_90_prob": 0.70,
                },
                {
                    **common,
                    "forecast_horizon": "T_2H",
                    "access_condition": "open_book",
                    "prompt_strategy": "probabilistic_forecast",
                    "home_win_90_prob": 0.20,
                    "draw_90_prob": 0.30,
                    "away_win_90_prob": 0.50,
                },
            ]
        )
    panel = pd.DataFrame(rows)
    external = pd.DataFrame(
        [
            {
                "baseline": "t24_odds",
                "match_id": "m1",
                "snapshot": "2026-06-10T18:55:37Z",
                "requested_snapshot": "2026-06-10T19:00:00Z",
                "prob_home": 0.55,
                "prob_draw": 0.25,
                "prob_away": 0.20,
                "n_bookmakers": 20,
            }
        ]
    )

    forecasts, metadata = _comparison_forecasts(panel, external, config)

    models = forecasts[forecasts["forecaster_type"].eq("individual_llm")]
    assert len(models) == len(config.section("design")["complete_panel"])
    assert models["prob_home"].eq(0.60).all()
    assert forecasts[forecasts["forecaster_id"].eq(MARKET_ID)]["prob_home"].item() == 0.55
    assert metadata["matches"] == 1
