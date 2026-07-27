from __future__ import annotations

import json
from dataclasses import dataclass
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
from .t24_odds import _paired_table

ANALYSIS_ID = "open_direct_odds"


@dataclass(frozen=True)
class OddsScenario:
    artifact_prefix: str
    baseline: str
    market_label: str
    market_timing: str


T2_CLOSING = OddsScenario(
    artifact_prefix="open_direct_t2_closing_odds",
    baseline="closing_odds",
    market_label="Closing odds",
    market_timing="closest historical API snapshot before kickoff",
)
T24_ODDS = OddsScenario(
    artifact_prefix="open_direct_t24_odds",
    baseline="t24_odds",
    market_label="T-24h odds",
    market_timing="requested T-24h; closest API snapshot at or before request",
)
SCENARIOS = (T24_ODDS, T2_CLOSING)


def _comparison_forecasts(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    config: AnalysisConfig,
    scenario: OddsScenario,
) -> tuple[pd.DataFrame, dict[str, object]]:
    models = config.section("design")["complete_panel"]
    cell = config.section("odds_benchmarks")["open_direct"]
    baseline_settings = config.section("external_baselines")[scenario.baseline]
    horizon = str(baseline_settings["comparison_horizon"])
    access = str(cell["access_condition"])
    prompt = str(cell["prompt_strategy"])
    if access != "open_book" or prompt != "direct_score":
        raise ValueError("The open-direct odds benchmark must use open_book and direct_score")

    market = external[external["baseline"].eq(scenario.baseline)].copy()
    if market.empty or market["match_id"].duplicated().any():
        raise ValueError(f"The validated {scenario.baseline} baseline must contain unique matches")
    market_match_ids = set(market["match_id"].astype(str))

    selected = panel[
        panel["model_id"].isin(models)
        & panel["forecast_horizon"].eq(horizon)
        & panel["access_condition"].eq(access)
        & panel["prompt_strategy"].eq(prompt)
        & panel["is_valid_for_scoring"].fillna(False)
        & panel["actual_result_90"].notna()
        & panel["match_id"].astype(str).isin(market_match_ids)
    ].copy()
    if selected.duplicated(["match_id", "model_id"]).any():
        raise ValueError(
            f"{horizon} open-book direct-score cell contains duplicate model-match rows"
        )
    expected_matches = len(market_match_ids)
    coverage = selected.groupby("model_id")["match_id"].nunique().reindex(models)
    if coverage.isna().any() or not coverage.eq(expected_matches).all():
        raise ValueError(
            f"Every model must cover every {scenario.market_label} match: {coverage.to_dict()}"
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
        market[["match_id", "snapshot", *PROBABILITY_COLUMNS, "n_bookmakers"]],
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
        [
            model_rows,
            ensemble,
            market_rows.drop(columns=["snapshot", "n_bookmakers"]),
        ],
        ignore_index=True,
        sort=False,
    )
    expected_rows = (len(models) + 2) * expected_matches
    if len(forecasts) != expected_rows:
        raise ValueError(f"Expected {expected_rows} comparison forecasts, found {len(forecasts)}")

    all_horizon_ids = set(
        panel[panel["forecast_horizon"].eq(horizon) & panel["actual_result_90"].notna()][
            "match_id"
        ].astype(str)
    )
    metadata: dict[str, object] = {
        "horizon": horizon,
        "access_condition": access,
        "prompt_strategy": prompt,
        "matches": expected_matches,
        "missing_market_match_ids": sorted(all_horizon_ids - market_match_ids),
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
        "minimum_bookmakers": int(market_rows["n_bookmakers"].min()),
        "maximum_bookmakers": int(market_rows["n_bookmakers"].max()),
    }
    return forecasts, metadata


def _write_insights(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    metadata: dict[str, object],
    scenario: OddsScenario,
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
    significant_positive = brier_tests[
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
    significant_descriptions = [
        {
            "model": str(row.model_id),
            "model_label": model_label(config, str(row.model_id)),
            "metric": str(row.metric),
            "advantage": float(row.estimate),
            "direction": "LLM" if float(row.estimate) > 0 else "market",
            "p_adjusted": float(row.p_adjusted),
        }
        for row in significant_all.itertuples(index=False)
    ]
    payload: dict[str, object] = {
        "comparison_scope": {
            **metadata,
            "market_baseline": scenario.baseline,
            "market_label": scenario.market_label,
            "market_timing": scenario.market_timing,
            "probability_source": "H/D/A probabilities reported in the direct-score prompt cell",
        },
        "market_brier": market_brier,
        "best_individual_model": best_model,
        "best_individual_model_label": model_label(config, best_model),
        "best_individual_brier": best_brier,
        "best_individual_brier_skill_vs_market": 1.0 - best_brier / market_brier,
        "ensemble_brier": ensemble_brier,
        "ensemble_brier_skill_vs_market": 1.0 - ensemble_brier / market_brier,
        "models_with_positive_mean_brier_advantage": positive,
        "models_with_holm_significant_positive_brier_advantage": significant_positive,
        "best_model_paired_brier_advantage": float(best_test["estimate"]),
        "best_model_paired_brier_ci": [
            float(best_test["ci_low"]),
            float(best_test["ci_high"]),
        ],
        "best_model_brier_p_adjusted": float(best_test["p_adjusted"]),
        "best_accuracy_models": best_accuracy_models,
        "best_accuracy": best_accuracy,
        "market_accuracy": float(accuracy.loc[MARKET_ID, "estimate"]),
        "holm_significant_comparisons": significant_descriptions,
        "market_draw_probability_mean": float(draw.loc[MARKET_ID, "mean_probability"]),
        "realized_draw_rate": float(draw.loc[MARKET_ID, "observed_frequency"]),
        "market_draw_ece": float(draw.loc[MARKET_ID, "ece"]),
        "ensemble_draw_ece": float(draw.loc[ENSEMBLE_ID, "ece"]),
    }

    missing_count = len(metadata["missing_market_match_ids"])
    if missing_count:
        coverage_sentence = (
            f"The paired complete-case sample contains {int(metadata['matches'])} matches; "
            f"{missing_count} fixtures without listed odds are excluded from both sides "
            "without imputation."
        )
    else:
        coverage_sentence = (
            f"All {int(metadata['matches'])} matches are included with no imputation."
        )
    if significant_descriptions:
        significant_sentence = "; ".join(
            f"{item['model_label']} on {METRICS[str(item['metric'])]['label']} "
            f"({item['direction']} favored, Holm p={float(item['p_adjusted']):.3f})"
            for item in significant_descriptions
        )
    else:
        significant_sentence = "No comparison survived metric-wise Holm correction."

    display_horizon = "T-24h" if metadata["horizon"] == "T_24H" else "T-2h"
    lines = [
        f"# Open-book direct-score forecasts versus {scenario.market_label}",
        "",
        "## Comparison scope",
        "",
        f"Only the {display_horizon}, open-book, direct-score LLM cell is used. "
        "The benchmark scores the H/D/A probabilities reported in that cell; it does "
        "not convert the modal scoreline into a deterministic match outcome. "
        f"The LLM rows have a median lead of "
        f"{float(metadata['llm_median_hours_before_kickoff']):.2f} hours, while the "
        f"market snapshots average {float(metadata['market_mean_hours_before_kickoff']):.2f} "
        f"hours before kickoff. {coverage_sentence}",
        "",
        "## Main findings",
        "",
        f"- {scenario.market_label} achieved a mean multiclass Brier score of "
        f"**{market_brier:.3f}**.",
        f"- The strongest individual LLM was **{model_label(config, best_model)}** at "
        f"**{best_brier:.3f}**, corresponding to Brier skill of "
        f"**{(1.0 - best_brier / market_brier) * 100:.1f}%** versus the market.",
        f"- Its paired Brier advantage (market minus LLM loss) was "
        f"**{float(best_test['estimate']):.3f}** with a 95% CI of "
        f"**[{float(best_test['ci_low']):.3f}, {float(best_test['ci_high']):.3f}]** "
        f"and Holm-adjusted p = **{float(best_test['p_adjusted']):.3f}**.",
        f"- The equal-weight LLM ensemble scored **{ensemble_brier:.3f}** "
        f"(Brier skill **{(1.0 - ensemble_brier / market_brier) * 100:.1f}%**).",
        f"- **{len(positive)}/{len(models)}** individual models had a positive mean "
        f"Brier advantage; **{len(significant_positive)}/{len(models)}** remained "
        "positive after Holm correction.",
        f"- The highest modal accuracy was **{best_accuracy * 100:.1f}%** for "
        f"**{', '.join(model_label(config, model) for model in best_accuracy_models)}**, "
        f"versus **{float(accuracy.loc[MARKET_ID, 'estimate']) * 100:.1f}%** for the market.",
        f"- {significant_sentence}",
        "",
        "## Interpretation guardrails",
        "",
        "Brier score is primary; log loss, RPS, modal accuracy, calibration, and "
        "cumulative trajectories are secondary diagnostics. Confidence intervals describe "
        "match-level uncertainty within this tournament. Closing odds have a timing "
        "advantage over T-2h forecasts; the T-24h comparison is aligned by nominal horizon "
        "but not to the exact second.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def _run_scenario(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
    scenario: OddsScenario,
) -> dict[str, object]:
    forecasts, metadata = _comparison_forecasts(panel, external, config, scenario)
    scored = score_forecasts(
        forecasts,
        tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
    )
    summary = _absolute_summary(scored, config, scenario.artifact_prefix)
    paired = _paired_inference(scored, config, scenario.artifact_prefix)
    calibration, calibration_summary = _calibration_tables(
        scored,
        int(config.section("calibration")["bins"]),
    )

    result_directory = config.resolve_path("results") / scenario.artifact_prefix
    result_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        f"{scenario.artifact_prefix}_forecast_match_metrics": scored,
        f"{scenario.artifact_prefix}_absolute_summary": summary,
        f"{scenario.artifact_prefix}_paired_inference": paired,
        f"{scenario.artifact_prefix}_calibration_bins": calibration,
        f"{scenario.artifact_prefix}_calibration_summary": calibration_summary,
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

    paper_summary = _paper_summary(summary, config, scenario.market_label)
    paired_table = _paired_table(paired, config)
    save_table(
        paper_summary,
        config,
        manifest,
        f"{scenario.artifact_prefix}_benchmark_summary",
        ANALYSIS_ID,
        source_hashes,
    )
    save_table(
        paired_table,
        config,
        manifest,
        f"{scenario.artifact_prefix}_paired_inference",
        ANALYSIS_ID,
        source_hashes,
    )

    missing_count = len(metadata["missing_market_match_ids"])
    display_horizon = "T-24h" if metadata["horizon"] == "T_24H" else "T-2h"
    scope_note = (
        f"LLMs: {display_horizon} open-book direct-score probabilities; "
        f"market: {scenario.market_label}."
    )
    paired_note = (
        f"{missing_count} fixtures without listed odds are excluded from both sides."
        if missing_count
        else "All tournament fixtures are included."
    )
    _plot_absolute(
        summary,
        config,
        manifest,
        source_hashes,
        artifact_id=f"fig_{scenario.artifact_prefix}_absolute_performance",
        artifact_stage=ANALYSIS_ID,
        scope_note=scope_note,
        market_label=scenario.market_label,
    )
    _plot_paired(
        paired,
        config,
        manifest,
        source_hashes,
        artifact_id=f"fig_{scenario.artifact_prefix}_paired_advantage",
        artifact_stage=ANALYSIS_ID,
        scope_note=paired_note,
        market_label=scenario.market_label.lower(),
    )
    _plot_calibration(
        calibration,
        config,
        manifest,
        source_hashes,
        artifact_id=f"fig_{scenario.artifact_prefix}_calibration",
        artifact_stage=ANALYSIS_ID,
        market_label=scenario.market_label,
    )
    _plot_trajectory(
        scored,
        config,
        manifest,
        source_hashes,
        artifact_id=f"fig_{scenario.artifact_prefix}_cumulative_advantage",
        artifact_stage=ANALYSIS_ID,
        title=f"Cumulative Brier advantage relative to {scenario.market_label}",
    )

    insights_path = result_directory / f"{scenario.artifact_prefix}_insights.md"
    insights = _write_insights(
        summary,
        paired,
        calibration_summary,
        metadata,
        scenario,
        config,
        insights_path,
    )
    manifest.add(
        f"{scenario.artifact_prefix}_insights_markdown",
        insights_path,
        "markdown",
        ANALYSIS_ID,
        source_hashes,
    )
    insights_json_path = result_directory / f"{scenario.artifact_prefix}_insights.json"
    insights_json_path.write_text(
        json.dumps(insights, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.add(
        f"{scenario.artifact_prefix}_insights_json",
        insights_json_path,
        "json",
        ANALYSIS_ID,
        source_hashes,
    )
    return {
        "summary": summary,
        "paper_summary": paper_summary,
        "paired": paired,
        "paired_table": paired_table,
        "insights": insights,
    }


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel_path = config.resolve_path("derived") / "analysis_panel.parquet"
    external_path = config.resolve_path("derived") / "external_baselines.parquet"
    if not panel_path.is_file() or not external_path.is_file():
        raise FileNotFoundError("Run prepare before the open-direct odds benchmarks")
    panel = pd.read_parquet(panel_path)
    external = pd.read_parquet(external_path)
    source_hashes = {
        "analysis_panel": sha256_file(panel_path),
        "external_baselines": sha256_file(external_path),
    }

    results = {
        scenario.artifact_prefix: _run_scenario(
            panel,
            external,
            config,
            manifest,
            source_hashes,
            scenario,
        )
        for scenario in SCENARIOS
    }
    overview = pd.concat(
        [
            result["paper_summary"].assign(Comparison=scenario.market_label)
            for scenario, result in zip(SCENARIOS, results.values())
        ],
        ignore_index=True,
    )
    overview = overview[["Comparison", *[c for c in overview if c != "Comparison"]]]
    save_table(
        overview,
        config,
        manifest,
        "open_direct_odds_benchmark_overview",
        ANALYSIS_ID,
        source_hashes,
    )
    manifest.write()
    return {key: value["insights"] for key, value in results.items()}
