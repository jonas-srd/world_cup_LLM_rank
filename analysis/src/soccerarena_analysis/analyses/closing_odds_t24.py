from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    _paired_inference,
    _paper_summary,
    _plot_absolute,
    _plot_paired,
    _plot_trajectory,
    score_forecasts,
)
from .common import model_match_scores

ANALYSIS_ID = "closing_odds_t24"
CONTEXT_COLUMNS = (
    "match_id",
    "stage",
    "kickoff_utc",
    "home_team",
    "away_team",
    "actual_result_90",
)


def aggregate_scored_cells(scored: pd.DataFrame) -> pd.DataFrame:
    """Average forecasts and scores over the two probabilities-first access cells."""
    required = {
        *CONTEXT_COLUMNS,
        "forecaster_id",
        "forecaster_type",
        *PROBABILITY_COLUMNS,
        *METRICS,
    }
    if not required.issubset(scored.columns):
        raise KeyError(f"Missing scored-cell columns: {sorted(required - set(scored.columns))}")
    group_columns = [*CONTEXT_COLUMNS, "forecaster_id", "forecaster_type"]
    aggregated = scored.groupby(group_columns, as_index=False).agg(
        **{column: (column, "mean") for column in (*PROBABILITY_COLUMNS, *METRICS)},
        condition_rows=("forecaster_id", "size"),
    )
    if not aggregated["condition_rows"].eq(2).all():
        counts = aggregated.loc[
            ~aggregated["condition_rows"].eq(2),
            ["match_id", "forecaster_id", "condition_rows"],
        ]
        raise ValueError(
            "T-24h comparison requires two probabilities-first access cells per match/forecaster: "
            f"{counts.head().to_dict('records')}"
        )
    aggregated["score_aggregation"] = "equal_mean_across_two_t24_probabilities_first_access_cells"
    return aggregated


def _selected_t24_panel(panel: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    design = config.section("design")
    models = design["complete_panel"]
    horizon = design["primary_horizon"]
    if horizon != "T_24H":
        raise ValueError(f"The paper primary horizon must be T_24H, found {horizon}")
    selected = panel[
        panel["model_id"].isin(models)
        & panel["forecast_horizon"].eq(horizon)
        & panel["prompt_strategy"].eq("probabilistic_forecast")
        & panel["is_valid_for_scoring"].fillna(False)
        & panel["actual_result_90"].notna()
    ].copy()
    expected_access = set(design["access_conditions"])
    observed_access = set(selected["access_condition"].drop_duplicates())
    if observed_access != expected_access:
        raise ValueError(
            f"T-24h probabilities-first access mismatch: expected {sorted(expected_access)}, "
            f"found {sorted(observed_access)}"
        )
    counts = selected.groupby(["model_id", "match_id"]).size()
    if len(counts) != len(models) * 104 or not counts.eq(2).all():
        raise ValueError(
            "T-24h benchmark requires two valid probabilities-first rows for every model-match"
        )
    return selected


def _t24_forecasts(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    selected = _selected_t24_panel(panel, config)
    model_cells = selected.rename(
        columns={
            "model_id": "forecaster_id",
            "home_win_90_prob": "prob_home",
            "draw_90_prob": "prob_draw",
            "away_win_90_prob": "prob_away",
        }
    )[
        [
            *CONTEXT_COLUMNS,
            "access_condition",
            "prompt_strategy",
            "forecaster_id",
            *PROBABILITY_COLUMNS,
        ]
    ].copy()
    model_cells["forecaster_type"] = "individual_llm"
    scored_model_cells = score_forecasts(
        model_cells,
        tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
    )
    model_scores = aggregate_scored_cells(scored_model_cells)

    existing_overall = model_match_scores(selected, config).rename(
        columns={
            "model_id": "forecaster_id",
            "brier_90_recomputed": "brier_expected",
            "log_loss_90_recomputed": "log_loss_expected",
            "top_outcome_accuracy_90_recomputed": "accuracy_expected",
        }
    )
    parity = model_scores.merge(
        existing_overall[
            [
                "match_id",
                "forecaster_id",
                "brier_expected",
                "log_loss_expected",
                "accuracy_expected",
            ]
        ],
        on=["match_id", "forecaster_id"],
        validate="one_to_one",
    )
    for metric in ("brier", "log_loss", "accuracy"):
        if not np.allclose(parity[metric], parity[f"{metric}_expected"], atol=1e-12):
            raise ValueError(f"T-24h {metric} does not reproduce the paper overall estimand")

    ensemble_cells = (
        model_cells.groupby(
            [*CONTEXT_COLUMNS, "access_condition", "prompt_strategy"],
            as_index=False,
        )[list(PROBABILITY_COLUMNS)]
        .mean()
        .assign(
            forecaster_id=ENSEMBLE_ID,
            forecaster_type="llm_ensemble",
        )
    )
    ensemble_scores = aggregate_scored_cells(
        score_forecasts(
            ensemble_cells,
            tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
        )
    )

    context = selected.drop_duplicates("match_id")[[*CONTEXT_COLUMNS]]
    market = external[external["baseline"].eq("closing_odds")].copy()
    market = context.merge(
        market[["match_id", *PROBABILITY_COLUMNS]],
        on="match_id",
        how="inner",
        validate="one_to_one",
    ).assign(
        forecaster_id=MARKET_ID,
        forecaster_type="market",
        condition_rows=1,
        score_aggregation="single_closing_snapshot",
    )
    market_scores = score_forecasts(
        market,
        tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
    )
    forecasts = pd.concat(
        [model_scores, ensemble_scores, market_scores],
        ignore_index=True,
        sort=False,
    )
    expected_rows = (len(config.section("design")["complete_panel"]) + 2) * 104
    if len(forecasts) != expected_rows:
        raise ValueError(f"Expected {expected_rows} T-24h comparison rows, found {len(forecasts)}")
    return forecasts


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
    significant_secondary = paired[
        paired["metric"].ne("brier") & paired["significant_after_holm"]
    ].copy()
    best_test = brier_tests.loc[best_model]
    payload: dict[str, object] = {
        "comparison_scope": {
            "llm_horizon": "T_24H",
            "llm_aggregation": "equal mean of scores across closed/open book in the probabilities-first setting",
            "market_timing": "closest available historical snapshot before kickoff",
            "matches": 104,
            "models": len(models),
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
        "holm_significant_secondary_comparisons": [
            {
                "model": str(row.model_id),
                "metric": str(row.metric),
                "advantage": float(row.estimate),
                "p_adjusted": float(row.p_adjusted),
            }
            for row in significant_secondary.itertuples(index=False)
        ],
        "inference": "paired match-level stratified bootstrap-t confidence intervals and sign-flip tests; Holm adjustment across seven models within each metric",
        "interpretation": "timing-sensitivity benchmark; closing odds contain information arriving after the LLM T-24h cutoff",
    }
    lines = [
        "# T-24h LLM forecasts versus closing odds",
        "",
        "## Comparison scope",
        "",
        "Each LLM's match-level score is the equal mean across the closed- and open-book probabilities-first T-24h cells, exactly reproducing the point estimates in the paper overall leaderboard. The market comparator is the de-vigged closing-odds consensus from the closest available pre-kickoff snapshot. This is a timing-sensitivity benchmark, not a simultaneous-information comparison.",
        "",
        "## Main findings",
        "",
        f"- Closing odds achieved a mean multiclass Brier score of **{market_brier:.3f}**.",
        f"- The strongest T-24h individual LLM was **{model_label(config, best_model)}** at **{best_brier:.3f}**, corresponding to Brier skill of **{(1.0 - best_brier / market_brier) * 100:.1f}%** versus closing odds.",
        f"- Its paired advantage (market minus LLM) was **{float(best_test['estimate']):.3f}** with a 95% CI of **[{float(best_test['ci_low']):.3f}, {float(best_test['ci_high']):.3f}]** and Holm-adjusted p = **{float(best_test['p_adjusted']):.3f}**.",
        f"- The equal-weight LLM ensemble scored **{ensemble_brier:.3f}** (Brier skill **{(1.0 - ensemble_brier / market_brier) * 100:.1f}%** versus market).",
        f"- **{len(positive)}/{len(models)}** individual models had a positive mean Brier advantage; **{len(significant)}/{len(models)}** remained positive after Holm correction.",
        f"- Across the secondary metric families, **{len(significant_secondary)}** market-favoring comparisons survived metric-wise Holm correction. No modal-accuracy comparison survived correction.",
        "",
        "## Interpretation guardrails",
        "",
        "The LLM forecasts are frozen roughly 24 hours before kickoff, whereas closing odds incorporate later public information and betting activity. A market advantage therefore measures both forecasting quality and information-timing advantage. The sample is one 104-match tournament and does not establish equivalence or generalize automatically to future tournaments.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel_path = config.resolve_path("derived") / "analysis_panel.parquet"
    external_path = config.resolve_path("derived") / "external_baselines.parquet"
    if not panel_path.is_file() or not external_path.is_file():
        raise FileNotFoundError("Run prepare before the T-24h closing-odds benchmark")
    panel = pd.read_parquet(panel_path)
    external = pd.read_parquet(external_path)
    source_hashes = {
        "analysis_panel": sha256_file(panel_path),
        "external_baselines": sha256_file(external_path),
    }
    scored = _t24_forecasts(panel, external, config)
    summary = _absolute_summary(scored, config, ANALYSIS_ID)
    paired = _paired_inference(scored, config, ANALYSIS_ID)

    result_directory = config.resolve_path("results") / ANALYSIS_ID
    result_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "closing_odds_t24_match_metrics": scored,
        "closing_odds_t24_absolute_summary": summary,
        "closing_odds_t24_paired_inference": paired,
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
        _paper_summary(summary, config),
        config,
        manifest,
        "closing_odds_t24_benchmark_summary",
        ANALYSIS_ID,
        source_hashes,
    )
    save_table(
        _paired_table(paired, config),
        config,
        manifest,
        "closing_odds_t24_paired_inference",
        ANALYSIS_ID,
        source_hashes,
    )

    scope_note = (
        "LLMs: probabilities-first T-24h, averaged across access; market: closing snapshot."
    )
    _plot_absolute(
        summary,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_closing_odds_t24_absolute_performance",
        artifact_stage=ANALYSIS_ID,
        scope_note=scope_note,
    )
    _plot_paired(
        paired,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_closing_odds_t24_paired_advantage",
        artifact_stage=ANALYSIS_ID,
        scope_note="Closing odds contain post-T-24h information.",
    )
    _plot_trajectory(
        scored,
        config,
        manifest,
        source_hashes,
        artifact_id="fig_closing_odds_t24_cumulative_advantage",
        artifact_stage=ANALYSIS_ID,
        title="Cumulative T-24h Brier advantage relative to closing odds",
    )

    insights_path = result_directory / "closing_odds_t24_insights.md"
    insights = _write_insights(summary, paired, config, insights_path)
    manifest.add(
        "closing_odds_t24_insights_markdown",
        insights_path,
        "markdown",
        ANALYSIS_ID,
        source_hashes,
    )
    insights_json_path = result_directory / "closing_odds_t24_insights.json"
    insights_json_path.write_text(
        json.dumps(insights, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.add(
        "closing_odds_t24_insights_json",
        insights_json_path,
        "json",
        ANALYSIS_ID,
        source_hashes,
    )
    return insights
