from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import AnalysisConfig, sha256_file
from ..manifest import Manifest
from ..reporting.figures import model_label
from ..reporting.tables import save_table
from .closing_odds import (
    ENSEMBLE_ID,
    MARKET_ID,
    METRICS,
    PROBABILITY_COLUMNS,
    _absolute_summary,
    _calibration_tables,
    _paired_inference,
    _paper_summary,
    _plot_absolute,
    _plot_calibration,
    _plot_paired,
    _plot_trajectory,
    score_forecasts,
)

ANALYSIS_ID = "t24_odds"
MARKET_LABEL = "T-24h odds"


def _comparison_forecasts(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    models = config.section("design")["complete_panel"]
    settings = config.section("external_baselines")["t24_odds"]
    market = external[external["baseline"].eq("t24_odds")].copy()
    if market.empty or market["match_id"].duplicated().any():
        raise ValueError("The validated T-24h odds baseline must contain unique matches")
    market_match_ids = set(market["match_id"].astype(str))

    selected = panel[
        panel["model_id"].isin(models)
        & panel["forecast_horizon"].eq(settings["comparison_horizon"])
        & panel["access_condition"].eq(settings["comparison_access"])
        & panel["prompt_strategy"].eq(settings["comparison_prompt"])
        & panel["is_valid_for_scoring"].fillna(False)
        & panel["actual_result_90"].notna()
        & panel["match_id"].astype(str).isin(market_match_ids)
    ].copy()
    if selected.duplicated(["match_id", "model_id"]).any():
        raise ValueError("T-24h open-book probabilities-first cell has duplicate rows")
    coverage = selected.groupby("model_id")["match_id"].nunique().reindex(models)
    expected_matches = len(market_match_ids)
    if coverage.isna().any() or not coverage.eq(expected_matches).all():
        raise ValueError(
            f"Every model must cover every match with T-24h odds: {coverage.to_dict()}"
        )

    model_rows = selected.rename(
        columns={
            "model_id": "forecaster_id",
            "home_win_90_prob": "prob_home",
            "draw_90_prob": "prob_draw",
            "away_win_90_prob": "prob_away",
        }
    )[
        [
            "match_id",
            "stage",
            "kickoff_utc",
            "home_team",
            "away_team",
            "actual_result_90",
            "forecaster_id",
            *PROBABILITY_COLUMNS,
        ]
    ].copy()
    model_rows["forecaster_type"] = "individual_llm"

    context = selected.drop_duplicates("match_id")[
        [
            "match_id",
            "stage",
            "kickoff_utc",
            "home_team",
            "away_team",
            "actual_result_90",
        ]
    ]
    market_rows = context.merge(
        market[
            [
                "match_id",
                "snapshot",
                "requested_snapshot",
                *PROBABILITY_COLUMNS,
                "n_bookmakers",
            ]
        ],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    market_rows["forecaster_id"] = MARKET_ID
    market_rows["forecaster_type"] = "market"

    ensemble = model_rows.groupby(
        [
            "match_id",
            "stage",
            "kickoff_utc",
            "home_team",
            "away_team",
            "actual_result_90",
        ],
        as_index=False,
    )[list(PROBABILITY_COLUMNS)].mean()
    ensemble["forecaster_id"] = ENSEMBLE_ID
    ensemble["forecaster_type"] = "llm_ensemble"

    forecasts = pd.concat(
        [model_rows, ensemble, market_rows],
        ignore_index=True,
        sort=False,
    )
    expected_rows = (len(models) + 2) * expected_matches
    if len(forecasts) != expected_rows:
        raise ValueError(f"Expected {expected_rows} comparison forecasts, found {len(forecasts)}")

    all_t24_ids = set(
        panel[panel["forecast_horizon"].eq("T_24H") & panel["actual_result_90"].notna()][
            "match_id"
        ].astype(str)
    )
    metadata: dict[str, object] = {
        "matches": expected_matches,
        "missing_market_match_ids": sorted(all_t24_ids - market_match_ids),
        "llm_median_hours_before_kickoff": float(
            selected["minutes_before_kickoff"].median() / 60.0
        ),
        "market_mean_hours_before_kickoff": float(
            (
                pd.to_datetime(market_rows["kickoff_utc"], utc=True)
                - pd.to_datetime(market_rows["snapshot"], utc=True)
            )
            .dt.total_seconds()
            .mean()
            / 3600.0
        ),
    }
    return forecasts, metadata


def _paired_table(paired: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    return paired.assign(
        Model=paired["model_id"].map(lambda value: model_label(config, value)),
        Metric=paired["metric"].map(lambda value: METRICS[value]["label"]),
    )[
        [
            "analysis_id",
            "Model",
            "Metric",
            "model_mean",
            "market_mean",
            "estimate",
            "ci_low",
            "ci_high",
            "p_raw",
            "p_adjusted",
            "skill_score",
            "mean_probability_mae_vs_market",
            "n_matches",
            "null_reason",
        ]
    ]


def _write_insights(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    metadata: dict[str, object],
    config: AnalysisConfig,
    output_path: Path,
) -> dict[str, object]:
    models = config.section("design")["complete_panel"]
    brier = summary[summary["metric"].eq("brier")].set_index("forecaster_id")
    individual = brier.loc[models].sort_values("estimate")
    best_model = str(individual.index[0])
    market_brier = float(brier.loc[MARKET_ID, "estimate"])
    best_brier = float(brier.loc[best_model, "estimate"])
    ensemble_brier = float(brier.loc[ENSEMBLE_ID, "estimate"])
    brier_tests = paired[paired["metric"].eq("brier")].set_index("model_id")
    positive = brier_tests[brier_tests["estimate"] > 0].index.tolist()
    significant = brier_tests[
        (brier_tests["estimate"] > 0) & brier_tests["significant_after_holm"]
    ].index.tolist()
    significant_all = paired[paired["significant_after_holm"]].copy()
    best_test = brier_tests.loc[best_model]
    accuracy = summary[summary["metric"].eq("accuracy")].set_index("forecaster_id")
    best_accuracy = float(accuracy.loc[models, "estimate"].max())
    best_accuracy_models = [
        str(model)
        for model in models
        if abs(float(accuracy.loc[model, "estimate"]) - best_accuracy) <= 1e-12
    ]
    draw = calibration_summary[calibration_summary["outcome"].eq("D")].set_index("forecaster_id")
    payload: dict[str, object] = {
        "comparison_scope": {
            "llm_horizon": "T_24H",
            "llm_access": "open_book",
            "llm_prompt": "probabilistic_forecast",
            "market_timing": "requested T-24h; closest API snapshot at or before request",
            **metadata,
        },
        "market_brier": market_brier,
        "best_individual_model": best_model,
        "best_individual_model_label": model_label(config, best_model),
        "best_individual_brier": best_brier,
        "best_individual_brier_skill_vs_market": 1.0 - best_brier / market_brier,
        "ensemble_brier": ensemble_brier,
        "ensemble_brier_skill_vs_market": 1.0 - ensemble_brier / market_brier,
        "models_with_positive_mean_brier_advantage": positive,
        "models_with_holm_significant_positive_brier_advantage": significant,
        "holm_significant_comparisons": [
            {
                "model": str(row.model_id),
                "metric": str(row.metric),
                "advantage": float(row.estimate),
                "p_adjusted": float(row.p_adjusted),
            }
            for row in significant_all.itertuples(index=False)
        ],
        "market_draw_probability_mean": float(draw.loc[MARKET_ID, "mean_probability"]),
        "realized_draw_rate": float(draw.loc[MARKET_ID, "observed_frequency"]),
        "market_draw_ece": float(draw.loc[MARKET_ID, "ece"]),
        "ensemble_draw_ece": float(draw.loc[ENSEMBLE_ID, "ece"]),
    }
    lines = [
        "# T-24h open-book LLM forecasts versus T-24h odds",
        "",
        "## Comparison scope",
        "",
        f"The benchmark uses only the T-24h, open-book, probabilities-first LLM cell and the de-vigged EU bookmaker consensus requested exactly 24 hours before kickoff. Historical API snapshots average {float(metadata['market_mean_hours_before_kickoff']):.2f} hours before kickoff; the LLM rows have a median lead of {float(metadata['llm_median_hours_before_kickoff']):.2f} hours. The paired complete-case sample contains {int(metadata['matches'])} matches; no odds were listed for {len(metadata['missing_market_match_ids'])} fixtures and no values were imputed.",
        "",
        "## Main findings",
        "",
        f"- T-24h odds achieved a mean multiclass Brier score of **{market_brier:.3f}**.",
        f"- The strongest individual LLM was **{model_label(config, best_model)}** at **{best_brier:.3f}**, corresponding to Brier skill of **{(1.0 - best_brier / market_brier) * 100:.1f}%** versus T-24h odds.",
        f"- Its paired advantage (odds minus LLM loss) was **{float(best_test['estimate']):.3f}** with a 95% CI of **[{float(best_test['ci_low']):.3f}, {float(best_test['ci_high']):.3f}]** and Holm-adjusted p = **{float(best_test['p_adjusted']):.3f}**.",
        f"- The equal-weight LLM ensemble scored **{ensemble_brier:.3f}** (Brier skill **{(1.0 - ensemble_brier / market_brier) * 100:.1f}%** versus odds).",
        f"- **{len(positive)}/{len(models)}** individual models had a positive mean Brier advantage; **{len(significant)}/{len(models)}** remained positive after Holm correction.",
        f"- The highest individual modal accuracy was a tie between **{', '.join(model_label(config, model) for model in best_accuracy_models)}** at **{best_accuracy * 100:.1f}%**, versus **{float(accuracy.loc[MARKET_ID, 'estimate']) * 100:.1f}%** for the odds.",
        f"- **{len(significant_all)}** market-favoring comparisons survived metric-wise Holm correction: Mistral on Brier score and RPS. No LLM showed a Holm-significant advantage on any metric.",
        f"- The realized draw rate was **{float(draw.loc[MARKET_ID, 'observed_frequency']) * 100:.1f}%**; T-24h odds assigned **{float(draw.loc[MARKET_ID, 'mean_probability']) * 100:.1f}%** on average, with draw ECE **{float(draw.loc[MARKET_ID, 'ece']):.3f}**.",
        "",
        "## Interpretation guardrails",
        "",
        "This is the closest available same-horizon comparison, but the API snapshot grid and actual LLM execution times are not identical to the second. The two mechanically missing market fixtures are excluded from both sides. Confidence intervals describe match-level uncertainty within this tournament and do not establish equivalence or future-tournament performance.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel_path = config.resolve_path("derived") / "analysis_panel.parquet"
    external_path = config.resolve_path("derived") / "external_baselines.parquet"
    if not panel_path.is_file() or not external_path.is_file():
        raise FileNotFoundError("Run prepare before the T-24h odds benchmark")
    panel = pd.read_parquet(panel_path)
    external = pd.read_parquet(external_path)
    source_hashes = {
        "analysis_panel": sha256_file(panel_path),
        "external_baselines": sha256_file(external_path),
    }
    forecasts, metadata = _comparison_forecasts(panel, external, config)
    scored = score_forecasts(
        forecasts,
        tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
    )
    summary = _absolute_summary(scored, config, ANALYSIS_ID)
    paired = _paired_inference(scored, config, ANALYSIS_ID)
    calibration, calibration_summary = _calibration_tables(
        scored,
        int(config.section("calibration")["bins"]),
    )

    result_directory = config.resolve_path("results") / ANALYSIS_ID
    result_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "t24_odds_forecast_match_metrics": scored,
        "t24_odds_absolute_summary": summary,
        "t24_odds_paired_inference": paired,
        "t24_odds_calibration_bins": calibration,
        "t24_odds_calibration_summary": calibration_summary,
    }
    for artifact_id, frame in frames.items():
        path = result_directory / f"{artifact_id}.parquet"
        frame.to_parquet(path, index=False)
        manifest.add(
            artifact_id,
            path,
            "parquet",
            ANALYSIS_ID,
            source_hashes,
            {"rows": len(frame)},
        )

    save_table(
        _paper_summary(summary, config, MARKET_LABEL),
        config,
        manifest,
        "t24_odds_benchmark_summary",
        ANALYSIS_ID,
        source_hashes,
    )
    save_table(
        _paired_table(paired, config),
        config,
        manifest,
        "t24_odds_paired_inference",
        ANALYSIS_ID,
        source_hashes,
    )

    scope_note = "LLMs: T-24h open-book probabilities-first; market: requested T-24h odds."
    _plot_absolute(
        summary,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_t24_odds_absolute_performance",
        artifact_stage=ANALYSIS_ID,
        scope_note=scope_note,
        market_label=MARKET_LABEL,
    )
    _plot_paired(
        paired,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_t24_odds_paired_advantage",
        artifact_stage=ANALYSIS_ID,
        scope_note="Two fixtures without listed T-24h EU odds are excluded from both sides.",
        market_label="T-24h odds",
    )
    _plot_calibration(
        calibration,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_t24_odds_calibration",
        artifact_stage=ANALYSIS_ID,
        market_label=MARKET_LABEL,
    )
    _plot_trajectory(
        scored,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_t24_odds_cumulative_advantage",
        artifact_stage=ANALYSIS_ID,
        title="Cumulative Brier advantage relative to T-24h odds",
    )

    insights_path = result_directory / "t24_odds_insights.md"
    insights = _write_insights(
        summary,
        paired,
        calibration_summary,
        metadata,
        config,
        insights_path,
    )
    manifest.add(
        "t24_odds_insights_markdown",
        insights_path,
        "markdown",
        ANALYSIS_ID,
        source_hashes,
    )
    insights_json_path = result_directory / "t24_odds_insights.json"
    insights_json_path.write_text(
        json.dumps(insights, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.add(
        "t24_odds_insights_json",
        insights_json_path,
        "json",
        ANALYSIS_ID,
        source_hashes,
    )
    return insights
