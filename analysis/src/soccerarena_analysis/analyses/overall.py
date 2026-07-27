from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator, PercentFormatter

from ..config import AnalysisConfig, sha256_file
from ..manifest import Manifest
from ..reporting.figures import (
    add_numeric_grid,
    add_panel_label,
    apply_style,
    model_color,
    model_label,
    save_figure,
)
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table
from ..statistics.bootstrap import studentized_cluster_bootstrap
from .closing_odds import MARKET_ID, score_forecasts
from .common import load_panel, model_match_scores, paired_model_results, primary_panel

OVERALL_PROMPT_STRATEGY = "probabilistic_forecast"

MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "deepseek/deepseek-v4-pro": "deepseek-color.png",
    "google/gemini-3.1-pro-preview": "google-color.png",
    "x-ai/grok-4.3": "xai.png",
    "openai/gpt-5.5": "openai.png",
    "mistralai/mistral-large-2512": "mistral-color.png",
    "qwen/qwen3.7-max": "alibaba-color.png",
}


def _add_model_icon_legend(
    figure: plt.Figure,
    model_order: list[str],
    config: AnalysisConfig,
    style: dict[str, object],
) -> None:
    icon_directory = Path(__file__).resolve().parents[3] / "assets" / "brand_icons"
    icon_paths = {model: icon_directory / MODEL_PROVIDER_ICONS[model] for model in model_order}
    missing_icons = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing_icons:
        raise FileNotFoundError(f"Missing model provider icons: {missing_icons}")

    x_positions = np.linspace(
        float(style["overall_icon_legend_x_min"]),
        float(style["overall_icon_legend_x_max"]),
        len(model_order),
    )
    line_y = float(style["overall_icon_legend_line_y"])
    line_half_width = float(style["overall_icon_legend_line_half_width"])
    for x_position, model in zip(x_positions, model_order):
        figure.add_artist(
            Line2D(
                [x_position - line_half_width, x_position + line_half_width],
                [line_y, line_y],
                transform=figure.transFigure,
                color=model_color(config, model),
                linewidth=float(style["overall_icon_legend_line_width"]),
                solid_capstyle="round",
            )
        )
        figure.add_artist(
            AnnotationBbox(
                OffsetImage(
                    plt.imread(icon_paths[model]),
                    zoom=float(style["overall_icon_legend_zoom"]),
                    interpolation="lanczos",
                ),
                (x_position, float(style["overall_icon_legend_y"])),
                xycoords=figure.transFigure,
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _overall_probability_panel(panel: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Return the complete T-24h probabilities-first panel, retaining both access cells."""
    selected = primary_panel(panel, config)
    selected = selected[selected["prompt_strategy"].eq(OVERALL_PROMPT_STRATEGY)].copy()
    models = config.section("design")["complete_panel"]
    counts = selected.groupby(["model_id", "match_id"]).size()
    if len(counts) != len(models) * 104 or not counts.eq(2).all():
        raise ValueError(
            "Overall leaderboard requires closed- and open-book probabilities-first "
            "forecasts for every complete-panel model-match"
        )
    return selected


def _closing_odds_summary(
    scoped_panel: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, dict[str, object]]:
    external_path = config.resolve_path("derived") / "external_baselines.parquet"
    if not external_path.is_file():
        raise FileNotFoundError("Run prepare before generating the odds-augmented leaderboard")
    external = pd.read_parquet(external_path)
    market = external[external["baseline"].eq("closing_odds")].copy()
    if len(market) != 104 or market["match_id"].nunique() != 104:
        raise ValueError("The closing-odds baseline must contain all 104 matches")
    context = scoped_panel.drop_duplicates("match_id")[["match_id", "stage", "actual_result_90"]]
    forecasts = context.merge(
        market[["match_id", "prob_home", "prob_draw", "prob_away"]],
        on="match_id",
        validate="one_to_one",
    ).assign(forecaster_id=MARKET_ID)
    scored = score_forecasts(
        forecasts,
        tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
    )
    results: dict[str, dict[str, object]] = {}
    for metric in ("brier", "log_loss", "accuracy"):
        result = studentized_cluster_bootstrap(
            scored,
            metric,
            "stage",
            config,
            f"overall.{MARKET_ID}.{metric}",
            test_null=False,
        )
        results[metric] = result.as_dict()
    return results


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel, source_hashes = load_panel(config, manifest)
    scoped_panel = _overall_probability_panel(panel, config)
    scores = model_match_scores(scoped_panel, config)
    external_path = config.resolve_path("derived") / "external_baselines.parquet"
    source_hashes["external_baselines"] = sha256_file(external_path)
    metrics = [config.section("metrics")["primary"], *config.section("metrics")["secondary"]]
    metrics = [metric for metric in metrics if metric in scores.columns]
    leaderboard_rows: list[dict[str, object]] = []
    headline_records: list[dict[str, object]] = []
    for model, group in scores.groupby("model_id", sort=False):
        for metric in metrics:
            result = studentized_cluster_bootstrap(
                group, metric, "stage", config, f"overall.{model}.{metric}", test_null=False
            )
            row = {"model_id": model, "metric": metric, **result.as_dict()}
            leaderboard_rows.append(row)
            headline_records.append(
                headline_record(
                    config,
                    "overall_headlines",
                    result.analysis_id,
                    f"Mean {metric} after equal averaging of the two probabilities-first "
                    "T-24h access cells within match",
                    source_hashes,
                    **result.as_dict(),
                    p_adjusted=None,
                    n_predictions=int((scoped_panel["model_id"] == model).sum()),
                    units=metric,
                    aggregation="two probabilities-first access cells within model-match, then matches",
                )
            )
    leaderboard = pd.DataFrame(leaderboard_rows)
    result_dir = config.resolve_path("results") / "overall"
    result_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = result_dir / "leaderboard_long.parquet"
    leaderboard.to_parquet(leaderboard_path, index=False)
    manifest.add(
        "overall_leaderboard_long",
        leaderboard_path,
        "parquet",
        "overall",
        source_hashes,
        {"rows": len(leaderboard)},
    )

    table_style = config.section("reporting")["style"]
    decimals = int(table_style["table_float_decimals"])
    percentage_decimals = int(table_style["table_percentage_decimals"])
    points_decimals = int(table_style["table_points_decimals"])

    def estimate_ci(
        row: pd.Series, percentage: bool = False, displayed_decimals: int | None = None
    ) -> str:
        scale = 100.0 if percentage else 1.0
        precision = decimals if displayed_decimals is None else displayed_decimals
        return (
            f"{float(row['estimate']) * scale:.{precision}f} "
            f"[{float(row['ci_low']) * scale:.{precision}f}, "
            f"{float(row['ci_high']) * scale:.{precision}f}]"
        )

    primary_metric_name = config.section("metrics")["primary"]
    ordered_models = (
        leaderboard[leaderboard["metric"] == primary_metric_name]
        .sort_values("estimate")["model_id"]
        .tolist()
    )
    odds = _closing_odds_summary(scoped_panel, config)
    paper_rows: list[dict[str, object]] = [
        {
            "Model": "Closing odds",
            "Brier [95% CI]": estimate_ci(pd.Series(odds["brier"])),
            "Log loss [95% CI]": estimate_ci(pd.Series(odds["log_loss"])),
            "Modal H/D/A accuracy, % [95% CI]": estimate_ci(
                pd.Series(odds["accuracy"]),
                percentage=True,
                displayed_decimals=percentage_decimals,
            ),
            "Exact score, % [95% CI]": "--",
            "Kicktipp [95% CI]": "--",
            "n": int(odds["brier"]["n_matches"]),
        }
    ]
    for model in ordered_models:
        rows = leaderboard[leaderboard["model_id"] == model].set_index("metric")
        paper_rows.append(
            {
                "Model": model_label(config, model),
                "Brier [95% CI]": estimate_ci(rows.loc["brier_90_recomputed"]),
                "Log loss [95% CI]": estimate_ci(rows.loc["log_loss_90_recomputed"]),
                "Modal H/D/A accuracy, % [95% CI]": estimate_ci(
                    rows.loc["top_outcome_accuracy_90_recomputed"],
                    percentage=True,
                    displayed_decimals=percentage_decimals,
                ),
                "Exact score, % [95% CI]": estimate_ci(
                    rows.loc["exact_score_90_correct_recomputed"],
                    percentage=True,
                    displayed_decimals=percentage_decimals,
                ),
                "Kicktipp [95% CI]": estimate_ci(
                    rows.loc["kicktipp_points_90_recomputed"],
                    displayed_decimals=points_decimals,
                ),
                "n": int(rows.loc[primary_metric_name, "n_matches"]),
            }
        )
    save_table(
        pd.DataFrame(paper_rows), config, manifest, "overall_leaderboard", "overall", source_hashes
    )

    pair_results, _ = paired_model_results(scoped_panel, config)
    pair_path = result_dir / "model_pairs.parquet"
    pair_results.to_parquet(pair_path, index=False)
    manifest.add(
        "overall_model_pairs",
        pair_path,
        "parquet",
        "overall",
        source_hashes,
        {"rows": len(pair_results)},
    )

    figure_scores = model_match_scores(panel, config)
    chronological = figure_scores.sort_values(["kickoff_utc", "match_id"]).copy()
    chronological["match_number"] = chronological.groupby("model_id").cumcount() + 1
    chronological["cumulative_brier"] = (
        chronological.groupby("model_id")["brier_90_recomputed"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    chronological["cumulative_accuracy"] = (
        chronological.groupby("model_id")["top_outcome_accuracy_90_recomputed"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    trajectory_path = result_dir / "cumulative_trajectories.parquet"
    chronological.to_parquet(trajectory_path, index=False)
    manifest.add(
        "overall_cumulative_trajectories",
        trajectory_path,
        "parquet",
        "overall",
        source_hashes,
        {"rows": len(chronological)},
    )

    stage_scores = figure_scores.groupby(["match_id", "stage"], as_index=False)[
        "brier_90_recomputed"
    ].mean()
    stage_path = result_dir / "stage_difficulty.parquet"
    stage_scores.to_parquet(stage_path, index=False)
    manifest.add(
        "overall_stage_difficulty",
        stage_path,
        "parquet",
        "overall",
        source_hashes,
        {"rows": len(stage_scores)},
    )

    fable = config.section("design")["partial_models"]["fable"]
    fable_rows = panel[
        (panel["model_id"] == fable)
        & (panel["forecast_horizon"] == config.section("design")["primary_horizon"])
    ]
    reliability = panel.groupby("model_id", as_index=False).agg(
        planned_rows=("prediction_id", "size"),
        valid_rows=("is_valid_for_scoring", lambda values: int(values.fillna(False).sum())),
        api_errors=("validation_status", lambda values: int((values == "api_error").sum())),
        repaired_rows=("repair_attempted", lambda values: int(values.fillna(False).sum())),
        total_cost_usd=("cost_usd", "sum"),
        median_latency_ms=("latency_ms", "median"),
    )
    reliability["valid_rate"] = reliability["valid_rows"] / reliability["planned_rows"]
    reliability_paper = reliability.copy()
    reliability_paper["model_id"] = reliability_paper["model_id"].map(
        lambda value: model_label(config, value)
    )
    reliability_paper = reliability_paper.rename(
        columns={
            "model_id": "Model",
            "planned_rows": "Planned forecasts",
            "valid_rows": "Valid forecasts",
            "api_errors": "API errors",
            "repaired_rows": "Repaired",
            "total_cost_usd": "Total cost (USD)",
            "median_latency_ms": "Median latency (ms)",
            "valid_rate": "Valid rate",
        }
    )
    save_table(
        reliability_paper,
        config,
        manifest,
        "overall_operational_reliability",
        "overall",
        source_hashes,
    )

    common_models = config.section("design")["complete_panel"]
    fable_valid = fable_rows[
        fable_rows["is_valid_for_scoring"].fillna(False) & fable_rows["actual_result_90"].notna()
    ]
    common_matches = sorted(fable_valid["match_id"].unique())
    sensitivity_rows: list[dict[str, object]] = []
    for model in [*common_models, fable]:
        subset = panel[
            panel["model_id"].eq(model)
            & panel["match_id"].isin(common_matches)
            & panel["forecast_horizon"].eq(config.section("design")["primary_horizon"])
            & panel["is_valid_for_scoring"].fillna(False)
        ]
        by_match = subset.groupby(["match_id", "stage"], as_index=False)[
            "brier_90_recomputed"
        ].mean()
        sensitivity_rows.append(
            {
                "model_id": model,
                "n_matches": by_match["match_id"].nunique(),
                "mean_brier": by_match["brier_90_recomputed"].mean()
                if not by_match.empty
                else np.nan,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_paper = sensitivity.assign(
        Model=sensitivity["model_id"].map(lambda value: model_label(config, value))
    )[["Model", "n_matches", "mean_brier"]].rename(
        columns={"n_matches": "Common matches", "mean_brier": "Mean Brier score"}
    )
    save_table(
        sensitivity_paper,
        config,
        manifest,
        "overall_fable_common_match_sensitivity",
        "overall",
        source_hashes,
    )

    apply_style(config)
    reporting = config.section("reporting")
    style = reporting["style"]
    width = float(reporting["figure_width_double"])
    height = float(style["overall_height"])
    figure = plt.figure(figsize=(width, height))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=style["overall_row_height_ratios"],
        hspace=float(style["overall_hspace"]),
        wspace=float(style["overall_wspace"]),
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]
    model_order = [
        model
        for model in config.section("design")["complete_panel"]
        if model in set(chronological["model_id"])
    ]
    for model in model_order:
        group = chronological[chronological["model_id"] == model]
        axes[0].plot(
            group["match_number"],
            group["cumulative_brier"],
            label=model_label(config, model),
            color=model_color(config, model),
            solid_capstyle="round",
        )
    axes[0].set_title(
        "Cumulative Brier score",
        fontsize=float(style["overall_title_size"]),
        pad=float(style["overall_title_pad"]),
    )
    axes[0].set_xlabel(
        "Completed matches",
        fontsize=float(style["overall_axis_label_size"]),
    )
    axes[0].set_ylabel(
        "Mean Brier score",
        fontsize=float(style["overall_axis_label_size"]),
    )
    axes[0].tick_params(axis="both", labelsize=float(style["overall_tick_label_size"]))
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    add_numeric_grid(axes[0], config, "y")
    for model in model_order:
        group = chronological[chronological["model_id"] == model]
        axes[1].plot(
            group["match_number"],
            group["cumulative_accuracy"],
            label=model_label(config, model),
            color=model_color(config, model),
            solid_capstyle="round",
        )
    axes[1].set_title(
        "Cumulative modal H/D/A accuracy",
        fontsize=float(style["overall_title_size"]),
        pad=float(style["overall_title_pad"]),
    )
    axes[1].set_xlabel(
        "Completed matches",
        fontsize=float(style["overall_axis_label_size"]),
    )
    axes[1].set_ylabel(
        "Accuracy (fractional tie credit)",
        fontsize=float(style["overall_axis_label_size"]),
    )
    axes[1].tick_params(axis="both", labelsize=float(style["overall_tick_label_size"]))
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    add_numeric_grid(axes[1], config, "y")
    order = [
        stage
        for stage in config.section("design")["stage_order"]
        if stage in set(stage_scores["stage"])
    ]
    stage_counts = stage_scores.groupby("stage")["match_id"].nunique().to_dict()
    sns.boxplot(
        data=stage_scores,
        x="stage",
        y="brier_90_recomputed",
        order=order,
        ax=axes[2],
        showfliers=False,
        linewidth=float(style["zero_line_width"]),
        boxprops={
            "facecolor": reporting["palette"]["neutral_light"],
            "edgecolor": reporting["palette"]["neutral"],
            "alpha": 0.65,
        },
        whiskerprops={"color": reporting["palette"]["neutral"]},
        capprops={"color": reporting["palette"]["neutral"]},
        medianprops={
            "color": reporting["palette"]["text"],
            "linewidth": float(style["line_width"]),
        },
    )
    sns.swarmplot(
        data=stage_scores,
        x="stage",
        y="brier_90_recomputed",
        order=order,
        ax=axes[2],
        color=reporting["palette"]["text"],
        size=float(style["box_point_size"]),
        alpha=float(style["box_point_alpha"]),
    )
    stage_labels = reporting["labels"]["stages"]
    axes[2].set_xticks(
        range(len(order)), [f"{stage_labels[stage]}\n(n={stage_counts[stage]})" for stage in order]
    )
    axes[2].set_title(
        "Match-level forecast error by tournament stage",
        fontsize=float(style["overall_title_size"]),
        pad=float(style["overall_title_pad"]),
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel(
        "Mean Brier score across models\n(lower is better)",
        fontsize=float(style["overall_axis_label_size"]),
    )
    axes[2].tick_params(
        axis="x",
        labelsize=float(style["overall_stage_tick_label_size"]),
    )
    axes[2].tick_params(
        axis="y",
        labelsize=float(style["overall_tick_label_size"]),
    )
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=6))
    add_numeric_grid(axes[2], config, "y")
    add_panel_label(
        axes[0],
        "A",
        config,
        x=float(style["overall_panel_label_x"]),
        y=float(style["overall_panel_label_y"]),
        font_size=float(style["overall_panel_label_size"]),
    )
    add_panel_label(
        axes[1],
        "B",
        config,
        x=float(style["overall_panel_label_x"]),
        y=float(style["overall_panel_label_y"]),
        font_size=float(style["overall_panel_label_size"]),
    )
    add_panel_label(
        axes[2],
        "C",
        config,
        x=float(style["overall_wide_panel_label_x"]),
        y=float(style["overall_panel_label_y"]),
        font_size=float(style["overall_panel_label_size"]),
    )
    _add_model_icon_legend(figure, model_order, config, style)
    figure.subplots_adjust(**style["overall_margins"])
    save_figure(figure, config, manifest, "fig_overall_performance", "overall", source_hashes)
    write_headlines(
        config, manifest, "overall_headlines", headline_records, "overall", source_hashes
    )
    manifest.write()
    return {"leaderboard": leaderboard, "model_pairs": pair_results, "reliability": reliability}
