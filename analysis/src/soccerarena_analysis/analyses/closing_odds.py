from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from ..reporting.tables import save_table
from ..statistics.bootstrap import studentized_cluster_bootstrap
from ..statistics.multiplicity import holm_adjust

MARKET_ID = "closing_odds_consensus"
ENSEMBLE_ID = "llm_equal_weight_ensemble"
METRICS = {
    "brier": {"label": "Multiclass Brier score", "higher_is_better": False},
    "log_loss": {"label": "Log loss (nats)", "higher_is_better": False},
    "rps": {"label": "Ranked probability score", "higher_is_better": False},
    "accuracy": {"label": "Modal H/D/A accuracy", "higher_is_better": True},
}
OUTCOMES = ("H", "D", "A")
PROBABILITY_COLUMNS = ("prob_home", "prob_draw", "prob_away")
MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "deepseek/deepseek-v4-pro": "deepseek-color.png",
    "google/gemini-3.1-pro-preview": "google-color.png",
    "x-ai/grok-4.3": "xai.png",
    "openai/gpt-5.5": "openai.png",
    "mistralai/mistral-large-2512": "mistral-color.png",
    "qwen/qwen3.7-max": "alibaba-color.png",
}


def ranked_probability_score(probabilities: tuple[float, float, float], actual: str) -> float:
    if actual not in OUTCOMES:
        raise ValueError(f"Unsupported actual outcome: {actual}")
    probability_array = np.asarray(probabilities, dtype=float)
    truth = np.asarray([1.0 if outcome == actual else 0.0 for outcome in OUTCOMES])
    return float(np.square(np.cumsum(probability_array)[:-1] - np.cumsum(truth)[:-1]).sum() / 2.0)


def score_forecasts(frame: pd.DataFrame, tie_tolerance: float = 1e-6) -> pd.DataFrame:
    required = {
        "match_id",
        "stage",
        "forecaster_id",
        "actual_result_90",
        *PROBABILITY_COLUMNS,
    }
    if not required.issubset(frame.columns):
        raise KeyError(f"Missing forecast columns: {sorted(required - set(frame.columns))}")
    scored = frame.copy()
    probabilities = scored[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    probability_sums = probabilities.sum(axis=1)
    if (
        (~np.isfinite(probabilities)).any()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probability_sums, 1.0, atol=1e-6)
    ):
        raise ValueError("Forecast probabilities must be finite, in [0, 1], and sum to one")
    actual_indices = scored["actual_result_90"].map(
        {outcome: index for index, outcome in enumerate(OUTCOMES)}
    )
    if actual_indices.isna().any():
        raise ValueError("Every comparison row requires a realized H/D/A outcome")
    indices = actual_indices.to_numpy(dtype=int)
    truth = np.eye(3, dtype=float)[indices]
    scored["brier"] = np.square(probabilities - truth).sum(axis=1)
    actual_probability = probabilities[np.arange(len(probabilities)), indices]
    scored["log_loss"] = -np.log(np.clip(actual_probability, 1e-15, 1.0 - 1e-15))
    cumulative_difference = (
        np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(truth, axis=1)[:, :-1]
    )
    scored["rps"] = np.square(cumulative_difference).sum(axis=1) / 2.0
    maxima = probabilities.max(axis=1, keepdims=True)
    tied = np.isclose(probabilities, maxima, atol=tie_tolerance, rtol=0.0)
    scored["accuracy"] = tied[np.arange(len(tied)), indices] / tied.sum(axis=1)
    return scored


def _comparison_panel(panel: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    settings = config.section("external_baselines")["closing_odds"]
    models = config.section("design")["complete_panel"]
    selected = panel[
        panel["model_id"].isin(models)
        & panel["forecast_horizon"].eq(settings["comparison_horizon"])
        & panel["access_condition"].eq(settings["comparison_access"])
        & panel["prompt_strategy"].eq(settings["comparison_prompt"])
        & panel["is_valid_for_scoring"].fillna(False)
        & panel["actual_result_90"].notna()
    ].copy()
    duplicates = selected.duplicated(["match_id", "model_id"])
    if duplicates.any():
        raise ValueError("Closing-odds comparison cell must contain one row per match/model")
    coverage = selected.groupby("model_id")["match_id"].nunique().reindex(models)
    if coverage.isna().any() or not (coverage == 104).all():
        raise ValueError(
            f"Closing-odds comparison requires 104 matches per model: {coverage.to_dict()}"
        )
    return selected


def _build_forecasts(
    panel: pd.DataFrame, external: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    selected = _comparison_panel(panel, config)
    models = config.section("design")["complete_panel"]
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
    market = external[external["baseline"].eq("closing_odds")].copy()
    if len(market) != 104 or market["match_id"].nunique() != 104:
        raise ValueError("The validated closing-odds baseline must contain all 104 matches")
    market = context.merge(
        market[["match_id", *PROBABILITY_COLUMNS, "n_bookmakers"]],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    market["forecaster_id"] = MARKET_ID
    market["forecaster_type"] = "market"

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
        [model_rows, ensemble, market.drop(columns="n_bookmakers")],
        ignore_index=True,
        sort=False,
    )
    expected = (len(models) + 2) * 104
    if len(forecasts) != expected:
        raise ValueError(f"Expected {expected} comparison forecasts, found {len(forecasts)}")
    return forecasts


def _absolute_summary(
    scored: pd.DataFrame,
    config: AnalysisConfig,
    analysis_prefix: str = "closing_odds",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for forecaster_id, group in scored.groupby("forecaster_id", sort=False):
        for metric, definition in METRICS.items():
            result = studentized_cluster_bootstrap(
                group[["match_id", "stage", metric]],
                metric,
                "stage",
                config,
                f"{analysis_prefix}.absolute.{forecaster_id}.{metric}",
                test_null=False,
            )
            records.append(
                {
                    "forecaster_id": forecaster_id,
                    "forecaster_type": group["forecaster_type"].iloc[0],
                    "metric": metric,
                    "higher_is_better": definition["higher_is_better"],
                    **result.as_dict(),
                }
            )
    summary = pd.DataFrame(records)
    market_means = summary[summary["forecaster_id"].eq(MARKET_ID)].set_index("metric")["estimate"]
    summary["skill_vs_market"] = summary.apply(
        lambda row: (
            float(row["estimate"] - market_means[row["metric"]])
            if bool(row["higher_is_better"])
            else float(1.0 - row["estimate"] / market_means[row["metric"]])
        ),
        axis=1,
    )
    return summary


def _paired_inference(
    scored: pd.DataFrame,
    config: AnalysisConfig,
    analysis_prefix: str = "closing_odds",
) -> pd.DataFrame:
    models = config.section("design")["complete_panel"]
    records: list[dict[str, object]] = []
    market = scored[scored["forecaster_id"].eq(MARKET_ID)].set_index("match_id")
    for model in models:
        model_rows = scored[scored["forecaster_id"].eq(model)].set_index("match_id")
        common = (
            market[["stage", *METRICS]]
            .join(model_rows[list(METRICS)], how="inner", lsuffix="_market", rsuffix="_model")
            .reset_index()
        )
        probabilities = market[list(PROBABILITY_COLUMNS)].join(
            model_rows[list(PROBABILITY_COLUMNS)],
            how="inner",
            lsuffix="_market",
            rsuffix="_model",
        )
        mean_probability_mae = float(
            np.abs(
                probabilities[[f"{column}_model" for column in PROBABILITY_COLUMNS]].to_numpy()
                - probabilities[[f"{column}_market" for column in PROBABILITY_COLUMNS]].to_numpy()
            ).mean()
        )
        for metric, definition in METRICS.items():
            if definition["higher_is_better"]:
                common["advantage"] = common[f"{metric}_model"] - common[f"{metric}_market"]
            else:
                common["advantage"] = common[f"{metric}_market"] - common[f"{metric}_model"]
            analysis_id = f"{analysis_prefix}.paired.{model}.{metric}"
            try:
                result_payload = studentized_cluster_bootstrap(
                    common[["match_id", "stage", "advantage"]],
                    "advantage",
                    "stage",
                    config,
                    analysis_id,
                ).as_dict()
                null_reason = None
            except ValueError as error:
                if "Non-positive standard error" not in str(error):
                    raise
                estimate = float(common["advantage"].mean())
                result_payload = {
                    "analysis_id": analysis_id,
                    "estimate": estimate,
                    "ci_low": estimate,
                    "ci_high": estimate,
                    "p_raw": 1.0,
                    "median": float(common["advantage"].median()),
                    "standard_error": 0.0,
                    "n_matches": len(common),
                    "replicates": int(config.section("statistics")["bootstrap_replicates"]),
                    "permutation_replicates": int(
                        config.section("statistics")["permutation_replicates"]
                    ),
                    "alternative": "two-sided",
                }
                null_reason = "paired differences had zero stratified standard error; p set conservatively to 1"
            market_mean = float(common[f"{metric}_market"].mean())
            model_mean = float(common[f"{metric}_model"].mean())
            records.append(
                {
                    "model_id": model,
                    "metric": metric,
                    "model_mean": model_mean,
                    "market_mean": market_mean,
                    "skill_score": (
                        model_mean - market_mean
                        if definition["higher_is_better"]
                        else 1.0 - model_mean / market_mean
                    ),
                    "mean_probability_mae_vs_market": mean_probability_mae,
                    "null_reason": null_reason,
                    **result_payload,
                }
            )
    frame = pd.DataFrame(records)
    frame["p_adjusted"] = np.nan
    for metric, indices in frame.groupby("metric").groups.items():
        adjusted = holm_adjust({str(index): float(frame.loc[index, "p_raw"]) for index in indices})
        for index in indices:
            frame.loc[index, "p_adjusted"] = adjusted[str(index)]
    frame["significant_after_holm"] = frame["p_adjusted"] < float(
        config.section("statistics")["alpha"]
    )
    return frame


def _calibration_tables(scored: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome_columns = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for (forecaster_id, forecaster_type), group in scored.groupby(
        ["forecaster_id", "forecaster_type"], sort=False
    ):
        for outcome, probability_column in outcome_columns.items():
            work = pd.DataFrame(
                {
                    "probability": group[probability_column].astype(float),
                    "observed": group["actual_result_90"].eq(outcome).astype(float),
                }
            )
            work["bin"] = pd.cut(
                work["probability"], edges, labels=False, include_lowest=True
            ).clip(upper=bins - 1)
            calibration = work.groupby("bin", observed=True).agg(
                n=("observed", "size"),
                mean_probability=("probability", "mean"),
                observed_frequency=("observed", "mean"),
            )
            calibration["absolute_gap"] = (
                calibration["mean_probability"] - calibration["observed_frequency"]
            ).abs()
            ece = float((calibration["n"] / len(work) * calibration["absolute_gap"]).sum())
            for bin_index, row in calibration.iterrows():
                bin_records.append(
                    {
                        "forecaster_id": forecaster_id,
                        "forecaster_type": forecaster_type,
                        "outcome": outcome,
                        "bin": int(bin_index),
                        **row.to_dict(),
                    }
                )
            summary_records.append(
                {
                    "forecaster_id": forecaster_id,
                    "forecaster_type": forecaster_type,
                    "outcome": outcome,
                    "ece": ece,
                    "mean_probability": float(work["probability"].mean()),
                    "observed_frequency": float(work["observed"].mean()),
                }
            )
    return pd.DataFrame(bin_records), pd.DataFrame(summary_records)


def _forecaster_label(
    config: AnalysisConfig,
    forecaster_id: str,
    market_label: str = "Closing odds",
) -> str:
    if forecaster_id == MARKET_ID:
        return market_label
    if forecaster_id == ENSEMBLE_ID:
        return "LLM ensemble"
    return model_label(config, forecaster_id)


def _compact_absolute_label(forecaster_id: str) -> str:
    if forecaster_id == MARKET_ID:
        return "Odds"
    if forecaster_id == ENSEMBLE_ID:
        return "Ensemble"
    return ""


def _forecaster_color(config: AnalysisConfig, forecaster_id: str) -> str:
    palette = config.section("reporting")["palette"]
    if forecaster_id == MARKET_ID:
        return palette["text"]
    if forecaster_id == ENSEMBLE_ID:
        return palette["primary"]
    return model_color(config, forecaster_id)


def _add_provider_icons(
    axis: plt.Axes,
    forecaster_order: list[str],
    style: dict[str, object],
) -> None:
    icon_directory = Path(__file__).resolve().parents[3] / "assets" / "brand_icons"
    model_positions = [
        (position, forecaster_id)
        for position, forecaster_id in enumerate(forecaster_order)
        if forecaster_id in MODEL_PROVIDER_ICONS
    ]
    icon_paths = {
        forecaster_id: icon_directory / MODEL_PROVIDER_ICONS[forecaster_id]
        for _, forecaster_id in model_positions
    }
    missing_icons = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing_icons:
        raise FileNotFoundError(f"Missing model provider icons: {missing_icons}")

    axis.tick_params(
        axis="y",
        labelsize=float(style["closing_absolute_tick_label_size"]),
        pad=float(style["closing_absolute_label_pad"]),
        length=0,
    )
    for y_position, forecaster_id in model_positions:
        icon = OffsetImage(
            plt.imread(icon_paths[forecaster_id]),
            zoom=float(style["closing_absolute_icon_zoom"]),
            interpolation="lanczos",
        )
        icon_box = AnnotationBbox(
            icon,
            (float(style["closing_absolute_icon_x"]), y_position),
            xycoords=("axes fraction", "data"),
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
        axis.add_artist(icon_box)


def _add_provider_icon_legend(
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
        float(style["closing_trajectory_legend_x_min"]),
        float(style["closing_trajectory_legend_x_max"]),
        len(model_order),
    )
    icon_y = float(style["closing_trajectory_legend_icon_y"])
    line_y = float(style["closing_trajectory_legend_line_y"])
    line_half_width = float(style["closing_trajectory_legend_line_half_width"])
    for x_position, model in zip(x_positions, model_order):
        figure.add_artist(
            Line2D(
                [x_position - line_half_width, x_position + line_half_width],
                [line_y, line_y],
                transform=figure.transFigure,
                color=model_color(config, model),
                linewidth=float(style["closing_trajectory_line_width"]),
                solid_capstyle="round",
            )
        )
        icon = OffsetImage(
            plt.imread(icon_paths[model]),
            zoom=float(style["closing_trajectory_icon_zoom"]),
            interpolation="lanczos",
        )
        figure.add_artist(
            AnnotationBbox(
                icon,
                (x_position, icon_y),
                xycoords=figure.transFigure,
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _plot_absolute(
    summary: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
    *,
    artifact_id: str = "fig_closing_odds_absolute_performance",
    artifact_stage: str = "closing_odds",
    scope_note: str | None = None,
    market_label: str = "Closing odds",
) -> None:
    apply_style(config)
    reporting = config.section("reporting")
    style = reporting["style"]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            float(reporting["figure_width_double"]),
            float(style["closing_absolute_height"]),
        ),
    )
    market_and_models = summary[summary["forecaster_id"].ne(ENSEMBLE_ID)]
    n_matches = int(summary["n_matches"].min())
    order = (
        market_and_models[market_and_models["metric"].eq("brier")]
        .sort_values("estimate")["forecaster_id"]
        .tolist()
    )
    order.append(ENSEMBLE_ID)
    for panel_label, (metric, definition), axis in zip("ABCD", METRICS.items(), axes.flat):
        values = summary[summary["metric"].eq(metric)].set_index("forecaster_id").loc[order]
        y = np.arange(len(values))
        scale = 100.0 if metric == "accuracy" else 1.0
        for position, (forecaster_id, row) in enumerate(values.iterrows()):
            estimate = float(row["estimate"]) * scale
            low = float(row["ci_low"]) * scale
            high = float(row["ci_high"]) * scale
            marker = (
                "D"
                if forecaster_id == MARKET_ID
                else ("s" if forecaster_id == ENSEMBLE_ID else "o")
            )
            axis.errorbar(
                estimate,
                position,
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                color=_forecaster_color(config, forecaster_id),
                capsize=2.5,
                zorder=3,
            )
        axis.set_yticks(
            y,
            [_compact_absolute_label(value) for value in order],
        )
        _add_provider_icons(axis, order, style)
        axis.tick_params(
            axis="x",
            labelsize=float(style["closing_absolute_tick_label_size"]),
        )
        axis.invert_yaxis()
        axis.set_title(
            str(definition["label"]),
            fontsize=float(style["closing_absolute_title_size"]),
            pad=float(style["closing_absolute_title_pad"]),
        )
        direction = "higher is better" if definition["higher_is_better"] else "lower is better"
        axis.set_xlabel(
            f"Mean across {n_matches} matches\n({direction})",
            fontsize=float(style["closing_absolute_axis_label_size"]),
        )
        if metric == "accuracy":
            axis.xaxis.set_major_formatter(PercentFormatter(100.0, decimals=0))
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        add_numeric_grid(axis, config, "x")
        add_panel_label(
            axis,
            panel_label,
            config,
            x=float(style["closing_absolute_panel_label_x"]),
            font_size=float(style["closing_absolute_panel_label_size"]),
        )
    if scope_note:
        figure.text(0.5, 0.005, scope_note, ha="center", fontsize=7)
        figure.tight_layout(
            rect=[0, 0.035, 1, 1],
            h_pad=float(style["closing_absolute_h_pad"]),
            w_pad=float(style["closing_absolute_w_pad"]),
        )
    else:
        figure.tight_layout(
            h_pad=float(style["closing_absolute_h_pad"]),
            w_pad=float(style["closing_absolute_w_pad"]),
        )
    save_figure(
        figure,
        config,
        manifest,
        artifact_id,
        artifact_stage,
        source_hashes,
    )


def _plot_paired(
    paired: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
    *,
    artifact_id: str = "fig_closing_odds_paired_advantage",
    artifact_stage: str = "closing_odds",
    scope_note: str | None = None,
    market_label: str = "closing odds",
) -> None:
    apply_style(config)
    reporting = config.section("reporting")
    style = reporting["style"]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            float(reporting["figure_width_double"]),
            float(style["closing_paired_height"]),
        ),
    )
    order = (
        paired[paired["metric"].eq("brier")]
        .sort_values("estimate", ascending=False)["model_id"]
        .tolist()
    )
    for panel_label, (metric, definition), axis in zip("ABCD", METRICS.items(), axes.flat):
        values = paired[paired["metric"].eq(metric)].set_index("model_id").loc[order]
        scale = 100.0 if metric == "accuracy" else 1.0
        for position, (model_id, row) in enumerate(values.iterrows()):
            estimate = float(row["estimate"]) * scale
            low = float(row["ci_low"]) * scale
            high = float(row["ci_high"]) * scale
            axis.errorbar(
                estimate,
                position,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=model_color(config, model_id),
                capsize=2.5,
                zorder=3,
            )
            if bool(row["significant_after_holm"]):
                axis.annotate(
                    "*",
                    (high, position),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=float(style["closing_paired_significance_size"]),
                )
        axis.axvline(
            0.0,
            color=reporting["palette"]["neutral"],
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )
        axis.set_yticks(np.arange(len(values)), [""] * len(values))
        _add_provider_icons(axis, order, style)
        axis.tick_params(
            axis="x",
            labelsize=float(style["closing_absolute_tick_label_size"]),
        )
        axis.invert_yaxis()
        axis.set_title(
            str(definition["label"]),
            fontsize=float(style["closing_absolute_title_size"]),
            pad=float(style["closing_absolute_title_pad"]),
        )
        units = "percentage points" if metric == "accuracy" else "score units"
        axis.set_xlabel(
            f"Paired advantage vs {market_label}\n({units})",
            fontsize=float(style["closing_absolute_axis_label_size"]),
        )
        add_numeric_grid(axis, config, "x")
        add_panel_label(
            axis,
            panel_label,
            config,
            x=float(style["closing_absolute_panel_label_x"]),
            font_size=float(style["closing_absolute_panel_label_size"]),
        )
    footer = (
        "Positive values favor the LLM. Bars show stratified bootstrap-t 95% CIs.\n"
        "* Holm-adjusted p < 0.05."
    )
    if scope_note:
        footer = f"{footer}\n{scope_note}"
    figure.text(
        0.5,
        float(style["closing_paired_footer_y"]),
        footer,
        ha="center",
        va="bottom",
        fontsize=float(style["closing_paired_footer_size"]),
    )
    figure.tight_layout(
        rect=[0, float(style["closing_paired_bottom_margin"]), 1, 1],
        h_pad=float(style["closing_absolute_h_pad"]),
        w_pad=float(style["closing_absolute_w_pad"]),
    )
    save_figure(
        figure,
        config,
        manifest,
        artifact_id,
        artifact_stage,
        source_hashes,
    )


def _plot_calibration(
    calibration: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
    *,
    artifact_id: str = "fig_closing_odds_calibration",
    artifact_stage: str = "closing_odds",
    market_label: str = "Closing odds",
) -> None:
    apply_style(config)
    reporting = config.section("reporting")
    palette = reporting["palette"]
    figure, axes = plt.subplots(
        1, 3, figsize=(float(reporting["figure_width_double"]), 3.0), sharex=True, sharey=True
    )
    outcome_labels = {"H": "Home win", "D": "Draw", "A": "Away win"}
    models = config.section("design")["complete_panel"]
    display_min_bin_n = 5
    n_matches = int(calibration.groupby(["forecaster_id", "outcome"])["n"].sum().min())
    for panel_label, outcome, axis in zip("ABC", OUTCOMES, axes):
        axis.plot([0, 1], [0, 1], color=palette["neutral"], linestyle="--", linewidth=0.8)
        for model in models:
            values = calibration[
                calibration["forecaster_id"].eq(model)
                & calibration["outcome"].eq(outcome)
                & calibration["n"].ge(display_min_bin_n)
            ].sort_values("mean_probability")
            axis.plot(
                values["mean_probability"],
                values["observed_frequency"],
                color=palette["neutral_light"],
                linewidth=0.8,
                alpha=0.8,
                marker="o",
                markersize=2.1,
            )
        for forecaster_id, marker, linestyle in (
            (MARKET_ID, "D", "-"),
            (ENSEMBLE_ID, "s", "-"),
        ):
            values = calibration[
                calibration["forecaster_id"].eq(forecaster_id)
                & calibration["outcome"].eq(outcome)
                & calibration["n"].ge(display_min_bin_n)
            ].sort_values("mean_probability")
            axis.plot(
                values["mean_probability"],
                values["observed_frequency"],
                color=_forecaster_color(config, forecaster_id),
                linewidth=1.5,
                marker=marker,
                markersize=3.4,
                linestyle=linestyle,
            )
        axis.set_title(outcome_labels[outcome])
        axis.set_xlabel("Mean forecast probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        add_numeric_grid(axis, config, "both")
        add_panel_label(axis, panel_label, config, x=-0.22)
    axes[0].set_ylabel("Observed frequency")
    legend = [
        Line2D([0], [0], color=palette["text"], marker="D", label=market_label),
        Line2D([0], [0], color=palette["primary"], marker="s", label="LLM ensemble"),
        Line2D([0], [0], color=palette["neutral_light"], marker="o", label="Individual LLMs"),
        Line2D([0], [0], color=palette["neutral"], linestyle="--", label="Perfect calibration"),
    ]
    figure.legend(
        legend,
        [item.get_label() for item in legend],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
    )
    figure.text(
        0.5,
        0.01,
        f"Five equal-width bins; plotted bins require n >= {display_min_bin_n}. "
        f"ECE uses all {n_matches} matches.",
        ha="center",
        fontsize=7,
    )
    figure.tight_layout(rect=[0, 0.22, 1, 1], w_pad=1.05)
    save_figure(
        figure,
        config,
        manifest,
        artifact_id,
        artifact_stage,
        source_hashes,
    )


def _plot_trajectory(
    scored: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
    *,
    artifact_id: str = "fig_closing_odds_cumulative_advantage",
    artifact_stage: str = "closing_odds",
    title: str = "Cumulative Brier advantage relative to closing odds",
) -> None:
    apply_style(config)
    reporting = config.section("reporting")
    style = reporting["style"]
    models = config.section("design")["complete_panel"]
    market = scored[scored["forecaster_id"].eq(MARKET_ID)][["match_id", "brier"]].rename(
        columns={"brier": "market_brier"}
    )
    figure, axis = plt.subplots(
        figsize=(
            float(reporting["figure_width_double"]),
            float(style["closing_trajectory_height"]),
        )
    )
    for model in models:
        group = (
            scored[scored["forecaster_id"].eq(model)]
            .merge(market, on="match_id", validate="one_to_one")
            .sort_values(["kickoff_utc", "match_id"])
            .copy()
        )
        group["match_number"] = np.arange(1, len(group) + 1)
        group["cumulative_advantage"] = (group["market_brier"] - group["brier"]).expanding().mean()
        axis.plot(
            group["match_number"],
            group["cumulative_advantage"],
            color=model_color(config, model),
            linewidth=float(style["closing_trajectory_line_width"]),
        )
    axis.axhline(
        0,
        color=reporting["palette"]["neutral"],
        linestyle="--",
        linewidth=float(style["closing_trajectory_zero_line_width"]),
    )
    axis.set_xlabel(
        "Completed matches",
        fontsize=float(style["closing_absolute_axis_label_size"]),
    )
    axis.set_ylabel(
        "Market Brier - LLM Brier\n(positive favors LLM)",
        fontsize=float(style["closing_absolute_axis_label_size"]),
    )
    axis.tick_params(
        axis="both",
        labelsize=float(style["closing_absolute_tick_label_size"]),
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    add_numeric_grid(axis, config, "y")
    _add_provider_icon_legend(figure, models, config, style)
    figure.tight_layout(rect=[0, float(style["closing_trajectory_bottom_margin"]), 1, 1])
    save_figure(
        figure,
        config,
        manifest,
        artifact_id,
        artifact_stage,
        source_hashes,
    )


def _format_ci(row: pd.Series, scale: float = 1.0, decimals: int = 3) -> str:
    return (
        f"{float(row['estimate']) * scale:.{decimals}f} "
        f"[{float(row['ci_low']) * scale:.{decimals}f}, "
        f"{float(row['ci_high']) * scale:.{decimals}f}]"
    )


def _paper_summary(
    summary: pd.DataFrame,
    config: AnalysisConfig,
    market_label: str = "Closing odds",
) -> pd.DataFrame:
    brier_order = summary[summary["metric"].eq("brier")].sort_values("estimate")
    rows: list[dict[str, object]] = []
    for forecaster_id in brier_order["forecaster_id"]:
        metrics = summary[summary["forecaster_id"].eq(forecaster_id)].set_index("metric")
        rows.append(
            {
                "Forecaster": _forecaster_label(config, forecaster_id, market_label),
                "Type": metrics.iloc[0]["forecaster_type"],
                "Brier [95% CI]": _format_ci(metrics.loc["brier"]),
                "Brier skill vs market (%)": float(metrics.loc["brier", "skill_vs_market"]) * 100,
                "Log loss [95% CI]": _format_ci(metrics.loc["log_loss"]),
                "RPS [95% CI]": _format_ci(metrics.loc["rps"]),
                "Accuracy, % [95% CI]": _format_ci(
                    metrics.loc["accuracy"], scale=100.0, decimals=1
                ),
                "n": int(metrics.loc["brier", "n_matches"]),
            }
        )
    return pd.DataFrame(rows)


def _write_insights(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    calibration_summary: pd.DataFrame,
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
    positive_models = brier_tests[brier_tests["estimate"] > 0].index.tolist()
    significant_positive = brier_tests[
        (brier_tests["estimate"] > 0) & brier_tests["significant_after_holm"]
    ].index.tolist()
    draw = calibration_summary[calibration_summary["outcome"].eq("D")].set_index("forecaster_id")
    payload: dict[str, object] = {
        "comparison_scope": {
            "forecast_horizon": "T_2H",
            "access_condition": "open_book",
            "prompt_strategy": "probabilistic_forecast",
            "matches": 104,
            "models": len(models),
            "market_aggregation": "per-bookmaker overround removal, outcome-wise median, final renormalization",
        },
        "market_brier": market_brier,
        "best_individual_model": best_model,
        "best_individual_model_label": model_label(config, best_model),
        "best_individual_brier": best_brier,
        "best_individual_brier_skill_vs_market": 1.0 - best_brier / market_brier,
        "ensemble_brier": ensemble_brier,
        "ensemble_brier_skill_vs_market": 1.0 - ensemble_brier / market_brier,
        "models_with_positive_mean_brier_advantage": positive_models,
        "models_with_holm_significant_positive_brier_advantage": significant_positive,
        "market_draw_probability_mean": float(draw.loc[MARKET_ID, "mean_probability"]),
        "realized_draw_rate": float(draw.loc[MARKET_ID, "observed_frequency"]),
        "market_draw_ece": float(draw.loc[MARKET_ID, "ece"]),
        "ensemble_draw_ece": float(draw.loc[ENSEMBLE_ID, "ece"]),
        "inference": "paired match-level stratified bootstrap-t confidence intervals and sign-flip tests; Holm adjustment across seven models within each metric",
        "publication_constraint": "confirm vendor attribution and derived-data publication terms before submission; raw bookmaker snapshots are not included",
    }
    best_test = brier_tests.loc[best_model]
    lines = [
        "# Closing-odds benchmark: paper-ready findings",
        "",
        "## Comparison scope",
        "",
        "The comparison uses the prespecified like-for-like cell: open-book, probabilities-first forecasts at T−2h for the seven-model complete panel. All 104 matches have one valid forecast per model and a timestamped bookmaker consensus. Odds are converted to probabilities after removing each bookmaker's overround; outcome-wise bookmaker medians are then renormalized.",
        "",
        "## Main findings",
        "",
        f"- The closing-odds consensus achieved a mean multiclass Brier score of **{market_brier:.3f}**.",
        f"- The descriptively strongest individual LLM was **{model_label(config, best_model)}** at **{best_brier:.3f}**, a Brier skill score of **{(1.0 - best_brier / market_brier) * 100:.1f}%** relative to the market.",
        f"- Its paired Brier advantage (market minus LLM) was **{float(best_test['estimate']):.3f}** with a 95% CI of **[{float(best_test['ci_low']):.3f}, {float(best_test['ci_high']):.3f}]** and Holm-adjusted p = **{float(best_test['p_adjusted']):.3f}**. This is a post-hoc descriptive best-model label; inference remains model-specific and multiplicity-adjusted.",
        f"- The equal-weight LLM ensemble scored **{ensemble_brier:.3f}** (Brier skill **{(1.0 - ensemble_brier / market_brier) * 100:.1f}%** vs market).",
        f"- **{len(positive_models)}/{len(models)}** individual models had a positive mean Brier advantage over closing odds; **{len(significant_positive)}/{len(models)}** remained positive with Holm-adjusted p < 0.05.",
        f"- The realized draw rate was **{float(draw.loc[MARKET_ID, 'observed_frequency']) * 100:.1f}%**. The market assigned **{float(draw.loc[MARKET_ID, 'mean_probability']) * 100:.1f}%** on average; its five-bin draw ECE was **{float(draw.loc[MARKET_ID, 'ece']):.3f}**, compared with **{float(draw.loc[ENSEMBLE_ID, 'ece']):.3f}** for the LLM ensemble.",
        "",
        "## Interpretation guardrails",
        "",
        "Brier score is the primary metric; log loss, RPS, modal accuracy, calibration, and cumulative trajectories are robustness and diagnostic views. The sample is one 104-match tournament, so confidence intervals quantify match-level sampling uncertainty within this tournament design, not performance across future tournaments. Closing odds are a strong information-rich reference, not ground truth. Raw subscription odds are retained locally and are not part of the paper artifacts.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel_path = config.resolve_path("derived") / "analysis_panel.parquet"
    external_path = config.resolve_path("derived") / "external_baselines.parquet"
    if not panel_path.is_file() or not external_path.is_file():
        raise FileNotFoundError("Run prepare before the closing-odds benchmark")
    panel = pd.read_parquet(panel_path)
    external = pd.read_parquet(external_path)
    source_hashes = {
        "analysis_panel": sha256_file(panel_path),
        "external_baselines": sha256_file(external_path),
    }
    forecasts = _build_forecasts(panel, external, config)
    scored = score_forecasts(
        forecasts,
        tie_tolerance=float(config.section("metrics")["top_outcome"]["tie_tolerance"]),
    )
    summary = _absolute_summary(scored, config)
    paired = _paired_inference(scored, config)
    calibration, calibration_summary = _calibration_tables(
        scored, int(config.section("calibration")["bins"])
    )

    result_directory = config.resolve_path("results") / "closing_odds"
    result_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "closing_odds_forecast_match_metrics": scored,
        "closing_odds_absolute_summary": summary,
        "closing_odds_paired_inference": paired,
        "closing_odds_calibration_bins": calibration,
        "closing_odds_calibration_summary": calibration_summary,
    }
    for artifact_id, frame in frames.items():
        path = result_directory / f"{artifact_id}.parquet"
        frame.to_parquet(path, index=False)
        manifest.add(
            artifact_id,
            path,
            "parquet",
            "closing_odds",
            source_hashes,
            {"rows": len(frame)},
        )

    save_table(
        _paper_summary(summary, config),
        config,
        manifest,
        "closing_odds_benchmark_summary",
        "closing_odds",
        source_hashes,
    )
    paired_table = paired.assign(
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
    save_table(
        paired_table,
        config,
        manifest,
        "closing_odds_paired_inference",
        "closing_odds",
        source_hashes,
    )

    _plot_absolute(summary, config, manifest, source_hashes)
    _plot_paired(paired, config, manifest, source_hashes)
    _plot_calibration(calibration, config, manifest, source_hashes)
    _plot_trajectory(scored, config, manifest, source_hashes)

    insights_path = result_directory / "closing_odds_insights.md"
    insights = _write_insights(summary, paired, calibration_summary, config, insights_path)
    manifest.add(
        "closing_odds_insights_markdown",
        insights_path,
        "markdown",
        "closing_odds",
        source_hashes,
    )
    insights_json_path = result_directory / "closing_odds_insights.json"
    insights_json_path.write_text(json.dumps(insights, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add(
        "closing_odds_insights_json",
        insights_json_path,
        "json",
        "closing_odds",
        source_hashes,
    )
    manifest.write()
    return {
        "summary": summary,
        "paired": paired,
        "calibration": calibration,
        "insights": insights,
    }
