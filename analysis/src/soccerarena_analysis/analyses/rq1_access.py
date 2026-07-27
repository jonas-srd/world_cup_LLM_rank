from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..reporting.figures import (
    add_numeric_grid,
    add_panel_label,
    apply_style,
    condition_color,
    forest_plot,
    model_label,
    save_figure,
)
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table
from ..statistics.bootstrap import leave_one_match_out, studentized_cluster_bootstrap
from ..statistics.multiplicity import holm_adjust
from .common import factorial_results, load_panel, paired_model_results, primary_panel

MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "deepseek/deepseek-v4-pro": "deepseek-color.png",
    "google/gemini-3.1-pro-preview": "google-color.png",
    "x-ai/grok-4.3": "xai.png",
    "openai/gpt-5.5": "openai.png",
    "mistralai/mistral-large-2512": "mistral-color.png",
    "qwen/qwen3.7-max": "alibaba-color.png",
}


def _add_model_provider_icons(
    axis: plt.Axes,
    model_order: list[str],
    style: dict[str, object],
) -> None:
    icon_directory = Path(__file__).resolve().parents[3] / "assets" / "brand_icons"
    icon_paths = {model: icon_directory / MODEL_PROVIDER_ICONS[model] for model in model_order}
    missing_icons = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing_icons:
        raise FileNotFoundError(f"Missing model provider icons: {missing_icons}")

    axis.tick_params(
        axis="y",
        labelsize=float(style["model_access_tick_label_size"]),
        pad=float(style["model_access_label_pad"]),
    )
    for y_position, model in enumerate(model_order):
        icon = OffsetImage(
            plt.imread(icon_paths[model]),
            zoom=float(style["model_access_icon_zoom"]),
            interpolation="lanczos",
        )
        icon_box = AnnotationBbox(
            icon,
            (float(style["model_access_icon_x"]), y_position),
            xycoords=("axes fraction", "data"),
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
        axis.add_artist(icon_box)


def _snapshot_contrasts(panel: pd.DataFrame, config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    design = config.section("design")
    subset = panel[
        panel["model_id"].isin(design["complete_panel"])
        & panel["is_valid_for_scoring"].fillna(False)
        & panel["actual_result_90"].notna()
    ]
    cells = subset.groupby(
        [
            "match_id",
            "stage",
            "model_id",
            "access_condition",
            "prompt_strategy",
            "forecast_horizon",
        ],
        as_index=False,
    )["brier_90_recomputed"].mean()
    pivot = cells.pivot_table(
        index=["match_id", "stage", "model_id", "access_condition", "prompt_strategy"],
        columns="forecast_horizon",
        values="brier_90_recomputed",
    ).dropna(subset=design["horizons"])
    pivot["stage_to_t24"] = pivot["STAGE_OPENING"] - pivot["T_24H"]
    pivot["t24_to_t2"] = pivot["T_24H"] - pivot["T_2H"]
    reset = pivot.reset_index()
    outputs: dict[str, pd.DataFrame] = {}
    for access, prefix in (("open_book", "open"), ("closed_book", "closed")):
        access_rows = reset[reset["access_condition"] == access]
        for column, suffix in (("stage_to_t24", "stage_to_t24"), ("t24_to_t2", "t24_to_t2")):
            outputs[f"{prefix}_{suffix}"] = (
                access_rows.groupby(["match_id", "stage"], as_index=False)[column]
                .mean()
                .rename(columns={column: "difference"})
            )
    return outputs


def _chronological_access_trend(access_match: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    required = {"match_id", "stage", "kickoff_utc", "difference"}
    missing = required - set(access_match.columns)
    if missing:
        raise ValueError(f"Chronological access trend is missing columns: {sorted(missing)}")
    ordered = access_match.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True).copy()
    if ordered["match_id"].duplicated().any():
        raise ValueError("Chronological access trend requires one access effect per match")
    ordered["match_number"] = np.arange(1, len(ordered) + 1, dtype=int)
    x_values = ordered["match_number"].to_numpy(dtype=float)
    y_values = ordered["difference"].to_numpy(dtype=float)
    trend_config = config.section("statistics")["descriptive_time_trend"]
    if trend_config["kernel"] != "gaussian":
        raise ValueError("Only the prespecified Gaussian time-trend kernel is supported")
    bandwidth = float(trend_config["bandwidth_matches"])
    if bandwidth <= 0:
        raise ValueError("The time-trend bandwidth must be positive")
    distances = (x_values[:, None] - x_values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * distances**2)
    kernel_sums = kernel.sum(axis=1)
    ordered["smoothed_difference"] = (kernel @ y_values) / kernel_sums
    ordered["effective_matches"] = kernel_sums**2 / (kernel**2).sum(axis=1)

    statistics = config.section("statistics")
    replicates = int(statistics["bootstrap_replicates"])
    counts = np.zeros((replicates, len(ordered)), dtype=float)
    assigned = np.zeros(len(ordered), dtype=bool)
    rng = np.random.default_rng(config.derived_seed("rq1.chronological_access_trend"))
    for stages in statistics["strata"].values():
        indices = np.flatnonzero(ordered["stage"].isin(stages).to_numpy())
        if len(indices) == 0:
            continue
        probabilities = np.full(len(indices), 1.0 / len(indices))
        counts[:, indices] = rng.multinomial(len(indices), probabilities, size=replicates)
        assigned[indices] = True
    if not assigned.all():
        missing_stages = sorted(ordered.loc[~assigned, "stage"].unique())
        raise ValueError(f"Time-trend bootstrap strata omit stages: {missing_stages}")
    numerator = (counts * y_values[None, :]) @ kernel.T
    denominator = counts @ kernel.T
    bootstrapped = numerator / denominator
    confidence = float(statistics["confidence_level"])
    tail = (1.0 - confidence) / 2.0
    ordered["band_low"] = np.quantile(bootstrapped, tail, axis=0)
    ordered["band_high"] = np.quantile(bootstrapped, 1.0 - tail, axis=0)
    ordered["bandwidth_matches"] = bandwidth
    ordered["bootstrap_replicates"] = replicates
    return ordered


def _draw_chronological_access(
    axis: plt.Axes,
    chronological_trend: pd.DataFrame,
    config: AnalysisConfig,
    *,
    compact: bool = False,
) -> None:
    reporting = config.section("reporting")
    style = reporting["style"]
    stage_order = list(config.section("design")["stage_order"])
    short_stage_labels = reporting["labels"]["stages_short"]
    stage_colors = sns.color_palette("cividis", n_colors=len(stage_order) + 2)[1:-1]
    stage_ranges: dict[str, tuple[float, float]] = {}
    for stage_index, stage in enumerate(stage_order):
        stage_rows = chronological_trend[chronological_trend["stage"] == stage]
        if stage_rows.empty:
            continue
        start = float(stage_rows["match_number"].min()) - 0.5
        end = float(stage_rows["match_number"].max()) + 0.5
        stage_ranges[stage] = (start, end)
        if compact and stage_index % 2 == 0:
            axis.axvspan(
                start,
                end,
                color=reporting["palette"]["neutral_light"],
                alpha=float(style["stage_band_alpha"]),
                linewidth=0,
                zorder=0,
            )
        elif not compact:
            axis.axvspan(
                start,
                end,
                color=stage_colors[stage_index],
                alpha=0.035,
                linewidth=0,
                zorder=0,
            )
        if start > 0.5:
            axis.axvline(
                start,
                color=reporting["palette"]["neutral_light"],
                linewidth=float(style["grid_line_width"]),
                zorder=1,
            )
    if compact:
        group_rows = chronological_trend[chronological_trend["stage"] == "group_stage"]
        knockout_rows = chronological_trend[chronological_trend["stage"] != "group_stage"]
        for label, rows in (("Group stage", group_rows), ("Knockout rounds", knockout_rows)):
            if rows.empty:
                continue
            axis.text(
                float(rows["match_number"].median()),
                float(style["time_stage_label_y"]),
                label,
                transform=axis.get_xaxis_transform(),
                color=reporting["palette"]["neutral"],
                fontsize=float(style["time_stage_label_size"]),
                ha="center",
                va="top",
                zorder=5,
            )
    else:
        stage_transform = axis.get_xaxis_transform()
        strip_y = 0.965
        strip_height = 0.035
        for stage_index, stage in enumerate(stage_order):
            if stage not in stage_ranges:
                continue
            start, end = stage_ranges[stage]
            axis.add_patch(
                Rectangle(
                    (start, strip_y),
                    end - start,
                    strip_height,
                    transform=stage_transform,
                    facecolor=stage_colors[stage_index],
                    edgecolor=axis.get_facecolor(),
                    linewidth=0.35,
                    alpha=0.96,
                    clip_on=True,
                    zorder=6,
                )
            )
        stage_handles = [
            Patch(
                facecolor=color,
                edgecolor=reporting["palette"]["neutral_light"],
                linewidth=0.4,
                label=short_stage_labels[stage],
            )
            for stage, color in zip(stage_order, stage_colors)
            if stage in stage_ranges
        ]
        legend_columns = int(style["time_standalone_legend_columns"])
        legend_rows = int(np.ceil(len(stage_handles) / legend_columns))
        stage_handles = [
            stage_handles[row * legend_columns + column]
            for column in range(legend_columns)
            for row in range(legend_rows)
            if row * legend_columns + column < len(stage_handles)
        ]
        axis.legend(
            handles=stage_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=legend_columns,
            frameon=False,
            fontsize=float(style["time_standalone_legend_font_size"]),
            handlelength=1.25,
            handletextpad=0.4,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
    x_values = chronological_trend["match_number"].to_numpy(dtype=float)
    axis.scatter(
        x_values,
        chronological_trend["difference"],
        s=float(style["time_scatter_size"]),
        alpha=float(style["time_scatter_alpha"]),
        color=reporting["palette"]["neutral"],
        edgecolors="none",
        zorder=2,
    )
    axis.fill_between(
        x_values,
        chronological_trend["band_low"],
        chronological_trend["band_high"],
        color=reporting["palette"]["primary"],
        alpha=float(style["time_band_alpha"]),
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        x_values,
        chronological_trend["smoothed_difference"],
        color=reporting["palette"]["primary"],
        zorder=3,
    )
    axis.axhline(
        0.0,
        color=reporting["palette"]["neutral"],
        linestyle="--",
        linewidth=float(style["zero_line_width"]),
        zorder=1,
    )
    add_numeric_grid(axis, config, "y")
    axis.margins(x=0.0, y=float(style["axis_margin_fraction"]))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=int(style["axis_max_major_ticks"])))
    if compact:
        axis.set_title(
            "Access effect over the match calendar",
            pad=float(style["time_compact_title_pad"]),
        )
        axis.set_xlabel("Chronological match number")
        axis.set_ylabel("Closed − open Brier")
    else:
        axis.set_title("")
        axis.tick_params(
            axis="both",
            labelsize=float(style["time_standalone_tick_label_size"]),
        )
        axis.set_xlabel(
            "Completed matches in chronological order",
            fontsize=float(style["time_standalone_axis_label_size"]),
        )
        axis.set_ylabel(
            "Δ Brier = closed − open\n(positive favors open book)",
            fontsize=float(style["time_standalone_axis_label_size"]),
        )


def _variance_decomposition(panel: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    primary = primary_panel(panel, config)
    cell = primary.groupby(
        ["match_id", "model_id", "access_condition", "prompt_strategy"], as_index=False
    )["brier_90_recomputed"].mean()
    models = list(config.section("design")["complete_panel"])
    conditions = pd.MultiIndex.from_product(
        [
            config.section("design")["access_conditions"],
            config.section("design")["prompt_strategies"],
        ],
        names=["access_condition", "prompt_strategy"],
    )
    match_ids = cell["match_id"].drop_duplicates().tolist()
    cube = np.empty((len(match_ids), len(models), len(conditions)), dtype=float)
    for match_index, match_id in enumerate(match_ids):
        matrix = (
            cell[cell["match_id"] == match_id]
            .pivot_table(
                index="model_id",
                columns=["access_condition", "prompt_strategy"],
                values="brier_90_recomputed",
            )
            .reindex(index=models, columns=conditions)
        )
        if matrix.isna().any().any():
            raise ValueError(f"Variance decomposition cell is incomplete for match {match_id}")
        cube[match_index] = matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(config.derived_seed("rq1.variance_decomposition"))
    replicates = int(config.section("statistics")["bootstrap_replicates"])

    def shares(matrix: np.ndarray) -> tuple[float, float, float]:
        grand = np.nanmean(matrix)
        model_means = np.nanmean(matrix, axis=1)
        condition_means = np.nanmean(matrix, axis=0)
        model_ss = matrix.shape[1] * np.nansum((model_means - grand) ** 2)
        condition_ss = matrix.shape[0] * np.nansum((condition_means - grand) ** 2)
        interaction = matrix - model_means[:, None] - condition_means[None, :] + grand
        interaction_ss = np.nansum(interaction**2)
        total = model_ss + condition_ss + interaction_ss
        return model_ss / total, condition_ss / total, interaction_ss / total

    estimates = shares(cube.mean(axis=0))
    sampled = rng.integers(0, len(match_ids), size=(replicates, len(match_ids)))
    matrices = cube[sampled].mean(axis=1)
    grand = matrices.mean(axis=(1, 2))
    model_means = matrices.mean(axis=2)
    condition_means = matrices.mean(axis=1)
    model_ss = matrices.shape[2] * ((model_means - grand[:, None]) ** 2).sum(axis=1)
    condition_ss = matrices.shape[1] * ((condition_means - grand[:, None]) ** 2).sum(axis=1)
    interaction = (
        matrices - model_means[:, :, None] - condition_means[:, None, :] + grand[:, None, None]
    )
    interaction_ss = (interaction**2).sum(axis=(1, 2))
    total = model_ss + condition_ss + interaction_ss
    boot = np.column_stack([model_ss / total, condition_ss / total, interaction_ss / total])
    names = ["model", "condition", "model_condition_interaction"]
    return pd.DataFrame(
        {
            "component": names,
            "share": estimates,
            "ci_low": np.quantile(boot, 0.025, axis=0),
            "ci_high": np.quantile(boot, 0.975, axis=0),
        }
    )


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel, source_hashes = load_panel(config, manifest)
    factorial = factorial_results(panel, config)
    access = factorial["access"]
    primary = primary_panel(panel, config)
    access_values = (
        primary.groupby(["match_id", "stage", "model_id", "access_condition"], as_index=False)[
            "brier_90_recomputed"
        ]
        .mean()
        .pivot_table(
            index=["match_id", "stage", "model_id"],
            columns="access_condition",
            values="brier_90_recomputed",
        )
        .dropna()
    )
    access_values["difference"] = access_values["closed_book"] - access_values["open_book"]
    access_match = (
        access_values.reset_index()
        .groupby(["match_id", "stage"], as_index=False)["difference"]
        .mean()
    )
    match_times = primary[["match_id", "kickoff_utc"]].drop_duplicates()
    if match_times["match_id"].duplicated().any():
        raise ValueError("A match has multiple kickoff times in the primary panel")
    access_match = access_match.merge(match_times, on="match_id", how="left", validate="one_to_one")
    chronological_trend = _chronological_access_trend(access_match, config)
    model_access_units = access_values.reset_index()[
        ["match_id", "stage", "model_id", "difference"]
    ].copy()
    model_access_summary = model_access_units.groupby("model_id", as_index=False).agg(
        mean_difference=("difference", "mean"),
        median_difference=("difference", "median"),
        standard_deviation=("difference", "std"),
        n_matches=("match_id", "nunique"),
    )
    loo = leave_one_match_out(access_match["difference"])

    stage_rows: list[dict[str, object]] = []
    minimum = int(config.section("statistics")["minimum_stage_matches_for_inference"])
    for stage, group in access_match.groupby("stage"):
        if len(group) >= minimum:
            result = studentized_cluster_bootstrap(
                group, "difference", "stage", config, f"rq1.stage.{stage}"
            )
            stage_rows.append({"stage": stage, **result.as_dict()})
        else:
            stage_rows.append(
                {
                    "stage": stage,
                    "estimate": group["difference"].mean(),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_raw": np.nan,
                    "median": group["difference"].median(),
                    "n_matches": len(group),
                }
            )
    stage_effects = pd.DataFrame(stage_rows)
    eligible = stage_effects["p_raw"].notna()
    if eligible.any():
        adjusted = holm_adjust(
            {str(index): value for index, value in stage_effects.loc[eligible, "p_raw"].items()}
        )
        stage_effects.loc[eligible, "p_adjusted"] = [
            adjusted[str(index)] for index in stage_effects.index[eligible]
        ]

    snapshots = _snapshot_contrasts(panel, config)
    snapshot_results = {
        name: studentized_cluster_bootstrap(
            values, "difference", "stage", config, f"rq1.snapshot.{name}"
        )
        for name, values in snapshots.items()
    }
    snapshot_adjusted = holm_adjust(
        {name: result.p_raw for name, result in snapshot_results.items()}
    )
    snapshot_table = pd.DataFrame(
        [
            {**result.as_dict(), "contrast": name, "p_adjusted": snapshot_adjusted[name]}
            for name, result in snapshot_results.items()
        ]
    )

    timing = config.section("timing")["T_24H"]
    on_time = panel[
        panel["model_id"].isin(config.section("design")["complete_panel"])
        & panel["forecast_horizon"].eq("T_24H")
        & panel["actual_result_90"].notna()
        & panel["is_valid_for_scoring"].fillna(False)
        & panel["minutes_before_kickoff"].between(
            timing["target_minutes"] - timing["tolerance_minutes"],
            timing["target_minutes"] + timing["tolerance_minutes"],
        )
    ]
    on_cells = (
        on_time.groupby(["match_id", "stage", "model_id", "access_condition"], as_index=False)[
            "brier_90_recomputed"
        ]
        .mean()
        .pivot_table(
            index=["match_id", "stage", "model_id"],
            columns="access_condition",
            values="brier_90_recomputed",
        )
        .dropna()
    )
    on_cells["difference"] = on_cells["closed_book"] - on_cells["open_book"]
    on_match = (
        on_cells.reset_index().groupby(["match_id", "stage"], as_index=False)["difference"].mean()
    )
    on_time_result = studentized_cluster_bootstrap(
        on_match, "difference", "stage", config, "rq1.access.on_time_t24"
    )

    model_pairs, pair_frames = paired_model_results(panel, config)
    g_rows: list[dict[str, object]] = []
    for row in model_pairs.itertuples(index=False):
        key = (row.model_a, row.model_b)
        pair = pair_frames[key].rename(columns={"difference": "model_difference"})
        merged = access_match.merge(pair, on=["match_id", "stage"], how="inner")
        sign = 1.0 if merged["model_difference"].mean() >= 0 else -1.0
        merged["g"] = merged["difference"] - sign * merged["model_difference"]
        result = studentized_cluster_bootstrap(
            merged,
            "g",
            "stage",
            config,
            f"rq1.dominance.{row.model_a}.vs.{row.model_b}",
            alternative="greater",
        )
        g_rows.append(
            {
                "model_a": row.model_a,
                "model_b": row.model_b,
                **result.as_dict(),
                "passes_component": result.p_raw < config.section("statistics")["alpha"],
            }
        )
    dominance = pd.DataFrame(g_rows)
    dominance_all_components_pass = bool(dominance["passes_component"].all())

    variance = _variance_decomposition(panel, config)
    complete_models = config.section("design")["complete_panel"]
    operational_panel = panel[panel["model_id"].isin(complete_models)].copy()
    operations = operational_panel.groupby(["model_id", "access_condition"], as_index=False).agg(
        predictions=("prediction_id", "size"),
        valid_rate=("is_valid_for_scoring", "mean"),
        search_observed_rate=("tool_calls_observed", "mean"),
        mean_tool_calls=("num_tool_calls", "mean"),
        mean_cost_usd=("cost_usd", "mean"),
        median_latency_ms=("latency_ms", "median"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
    )
    operation_units = operational_panel.groupby(
        [
            "match_id",
            "stage",
            "model_id",
            "forecast_horizon",
            "prompt_strategy",
            "access_condition",
        ],
        as_index=False,
    ).agg(
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        latency_ms=("latency_ms", "mean"),
        cost_usd=("cost_usd", "mean"),
    )
    operation_wide = (
        operation_units.pivot_table(
            index=["match_id", "stage", "model_id", "forecast_horizon", "prompt_strategy"],
            columns="access_condition",
            values=["input_tokens", "output_tokens", "latency_ms", "cost_usd"],
        )
        .dropna()
        .reset_index()
    )
    operation_wide.columns = [
        f"{first}__{second}" if second else first for first, second in operation_wide.columns
    ]
    incremental_records: list[dict[str, object]] = []
    for measure in ("input_tokens", "output_tokens", "latency_ms", "cost_usd"):
        values = operation_wide[["match_id", "stage"]].copy()
        values["difference"] = (
            operation_wide[f"{measure}__open_book"] - operation_wide[f"{measure}__closed_book"]
        )
        values = values.groupby(["match_id", "stage"], as_index=False)["difference"].mean()
        result = studentized_cluster_bootstrap(
            values, "difference", "stage", config, f"rq1.operational_increment.{measure}"
        )
        incremental_records.append({"measure": measure, **result.as_dict()})
    incremental_operations = pd.DataFrame(incremental_records)
    model_incremental = operation_wide[operation_wide["model_id"].isin(complete_models)].copy()
    for measure in ("input_tokens", "output_tokens", "latency_ms", "cost_usd"):
        model_incremental[f"delta_{measure}"] = (
            model_incremental[f"{measure}__open_book"]
            - model_incremental[f"{measure}__closed_book"]
        )
    model_incremental = model_incremental.groupby("model_id", as_index=False).agg(
        delta_input_tokens=("delta_input_tokens", "mean"),
        delta_output_tokens=("delta_output_tokens", "mean"),
        delta_latency_ms=("delta_latency_ms", "mean"),
        delta_cost_usd=("delta_cost_usd", "mean"),
    )

    quantiles = [float(value) for value in config.section("timing")["lead_time_quantiles"]]
    lead_records: list[dict[str, object]] = []
    for (horizon, access_condition), group in operational_panel.dropna(
        subset=["minutes_before_kickoff"]
    ).groupby(["forecast_horizon", "access_condition"]):
        values = group["minutes_before_kickoff"].astype(float)
        record: dict[str, object] = {
            "forecast_horizon": horizon,
            "access_condition": access_condition,
            "n_predictions": len(values),
            "n_matches": group["match_id"].nunique(),
        }
        for quantile, value in zip(quantiles, values.quantile(quantiles)):
            record[f"q{int(round(quantile * 100)):02d}_minutes"] = float(value)
        lead_records.append(record)
    lead_times = pd.DataFrame(lead_records)
    framing_sentence = (
        config.section("reporting")["conditions_vs_models_sentence"]
        if access["p_adjusted"] < config.section("statistics")["alpha"]
        and not (model_pairs["p_adjusted"] < config.section("statistics")["alpha"]).any()
        else None
    )

    result_dir = config.resolve_path("results") / "rq1"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "access_summary.json").write_text(
        json.dumps(
            {
                "access": access,
                "leave_one_match_out": loo,
                "on_time": on_time_result.as_dict(),
                "dominance_iut_all_components_pass": dominance_all_components_pass,
                "locked_sentence": framing_sentence,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stage_effects.to_parquet(result_dir / "stage_effects.parquet", index=False)
    chronological_trend.to_parquet(result_dir / "access_over_time.parquet", index=False)
    model_access_units.to_parquet(result_dir / "access_by_model.parquet", index=False)
    snapshot_table.to_parquet(result_dir / "snapshot_effects.parquet", index=False)
    dominance.to_parquet(result_dir / "dominance_iut.parquet", index=False)
    variance.to_parquet(result_dir / "variance_decomposition.parquet", index=False)
    for name in (
        "access_summary.json",
        "stage_effects.parquet",
        "access_over_time.parquet",
        "access_by_model.parquet",
        "snapshot_effects.parquet",
        "dominance_iut.parquet",
        "variance_decomposition.parquet",
    ):
        manifest.add(
            f"rq1_{Path(name).stem}",
            result_dir / name,
            Path(name).suffix.lstrip("."),
            "rq1_access",
            source_hashes,
        )
    operations.to_parquet(result_dir / "operational_by_model_access.parquet", index=False)
    operation_wide.to_parquet(result_dir / "operational_paired_units.parquet", index=False)
    manifest.add(
        "rq1_operational_by_model_access",
        result_dir / "operational_by_model_access.parquet",
        "parquet",
        "rq1_access",
        source_hashes,
        {"rows": len(operations)},
    )
    manifest.add(
        "rq1_operational_paired_units",
        result_dir / "operational_paired_units.parquet",
        "parquet",
        "rq1_access",
        source_hashes,
        {"rows": len(operation_wide)},
    )

    access_effects = pd.DataFrame(
        [
            {"contrast": "primary_access", **access},
            {"contrast": "on_time_only_t24", **on_time_result.as_dict(), "p_adjusted": np.nan},
        ]
    )
    access_display = access_effects.assign(
        Contrast=["All T−24h calls", "On-time T−24h calls (1,440 ± 90 min)"],
        **{
            "Mean Δ Brier": access_effects["estimate"],
            "95% CI lower": access_effects["ci_low"],
            "95% CI upper": access_effects["ci_high"],
            "Raw p": access_effects["p_raw"],
            "Holm p": access_effects["p_adjusted"],
            "Median Δ": access_effects["median"],
            "Matches": access_effects["n_matches"],
        },
    )[
        [
            "Contrast",
            "Mean Δ Brier",
            "95% CI lower",
            "95% CI upper",
            "Raw p",
            "Holm p",
            "Median Δ",
            "Matches",
        ]
    ]
    save_table(
        access_display,
        config,
        manifest,
        "rq1_access_effects",
        "rq1_access",
        source_hashes,
        headline_frame=access_effects,
    )

    model_access_display = model_access_summary.assign(
        Model=model_access_summary["model_id"].map(lambda value: model_label(config, value)),
        **{
            "Mean Δ Brier": model_access_summary["mean_difference"],
            "Median Δ Brier": model_access_summary["median_difference"],
            "SD across matches": model_access_summary["standard_deviation"],
            "Matches": model_access_summary["n_matches"],
        },
    )[["Model", "Mean Δ Brier", "Median Δ Brier", "SD across matches", "Matches"]]
    save_table(
        model_access_display,
        config,
        manifest,
        "rq1_access_by_model",
        "rq1_access",
        source_hashes,
    )

    snapshot_labels = config.section("reporting")["labels"]["snapshots"]
    snapshot_display = snapshot_table.assign(
        Contrast=snapshot_table["contrast"].map(snapshot_labels),
        **{
            "Mean Δ Brier": snapshot_table["estimate"],
            "95% CI lower": snapshot_table["ci_low"],
            "95% CI upper": snapshot_table["ci_high"],
            "Holm p": snapshot_table["p_adjusted"],
            "Median Δ": snapshot_table["median"],
            "Matches": snapshot_table["n_matches"],
        },
    )[["Contrast", "Mean Δ Brier", "95% CI lower", "95% CI upper", "Holm p", "Median Δ", "Matches"]]
    save_table(
        snapshot_display,
        config,
        manifest,
        "rq1_snapshot_effects",
        "rq1_access",
        source_hashes,
        headline_frame=snapshot_table,
    )

    variance_trace = variance.rename(columns={"share": "estimate"}).assign(
        analysis_id=lambda values: "rq1.variance." + values["component"],
        p_raw=np.nan,
        p_adjusted=np.nan,
        median=np.nan,
        n_matches=access["n_matches"],
    )
    variance_display = variance.assign(
        Component=variance["component"].map(
            {
                "model": "Model",
                "condition": "Condition",
                "model_condition_interaction": "Model × condition",
            }
        ),
        **{
            "Variance share, %": variance["share"] * 100.0,
            "95% CI lower, %": variance["ci_low"] * 100.0,
            "95% CI upper, %": variance["ci_high"] * 100.0,
        },
    )[["Component", "Variance share, %", "95% CI lower, %", "95% CI upper, %"]]
    save_table(
        variance_display,
        config,
        manifest,
        "rq1_variance_decomposition",
        "rq1_access",
        source_hashes,
        headline_frame=variance_trace,
    )

    open_operations = operations[
        operations["model_id"].isin(complete_models)
        & operations["access_condition"].eq("open_book")
    ].merge(model_incremental, on="model_id", validate="one_to_one")
    operational_display = open_operations.assign(
        Model=open_operations["model_id"].map(lambda value: model_label(config, value)),
        **{
            "Observed search, %": open_operations["search_observed_rate"] * 100.0,
            "Mean tool calls": open_operations["mean_tool_calls"],
            "Open−closed total tokens": open_operations["delta_input_tokens"]
            + open_operations["delta_output_tokens"],
            "Open−closed latency (s)": open_operations["delta_latency_ms"] / 1000.0,
            "Open−closed cost (USD)": open_operations["delta_cost_usd"],
        },
    )[
        [
            "Model",
            "Observed search, %",
            "Mean tool calls",
            "Open−closed total tokens",
            "Open−closed latency (s)",
            "Open−closed cost (USD)",
        ]
    ]
    save_table(
        operational_display, config, manifest, "rq1_operational_costs", "rq1_access", source_hashes
    )

    measure_labels = {
        "input_tokens": ("Input tokens", 1.0, "tokens"),
        "output_tokens": ("Output tokens", 1.0, "tokens"),
        "latency_ms": ("Latency", 0.001, "seconds"),
        "cost_usd": ("Cost", 1.0, "USD"),
    }
    incremental_display_rows = []
    for row in incremental_operations.itertuples(index=False):
        label, scale, unit = measure_labels[row.measure]
        incremental_display_rows.append(
            {
                "Measure": label,
                "Open−closed estimate": row.estimate * scale,
                "95% CI lower": row.ci_low * scale,
                "95% CI upper": row.ci_high * scale,
                "Unit": unit,
                "Matches": row.n_matches,
            }
        )
    save_table(
        pd.DataFrame(incremental_display_rows),
        config,
        manifest,
        "rq1_incremental_operational_costs",
        "rq1_access",
        source_hashes,
        headline_frame=incremental_operations,
    )

    access_labels = config.section("reporting")["labels"]["access"]
    lead_display = lead_times.assign(
        Horizon=lead_times["forecast_horizon"],
        Access=lead_times["access_condition"].map(access_labels),
        Predictions=lead_times["n_predictions"],
        Matches=lead_times["n_matches"],
        **{
            "P10 lead time (hours)": lead_times["q10_minutes"] / 60.0,
            "Median lead time (hours)": lead_times["q50_minutes"] / 60.0,
            "P90 lead time (hours)": lead_times["q90_minutes"] / 60.0,
        },
    )[
        [
            "Horizon",
            "Access",
            "Predictions",
            "Matches",
            "P10 lead time (hours)",
            "Median lead time (hours)",
            "P90 lead time (hours)",
        ]
    ]
    save_table(
        lead_display,
        config,
        manifest,
        "rq1_actual_lead_time_distributions",
        "rq1_access",
        source_hashes,
    )

    apply_style(config)
    reporting = config.section("reporting")
    style = reporting["style"]
    figure = plt.figure(
        figsize=(float(reporting["figure_width_double"]), float(style["rq1_height"]))
    )
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=style["rq1_row_height_ratios"],
        width_ratios=style["rq1_column_width_ratios"],
        hspace=float(style["rq1_hspace"]),
        wspace=float(style["rq1_wspace"]),
    )
    access_axis = figure.add_subplot(grid[0, 0])
    time_axis = figure.add_subplot(grid[0, 1])
    snapshot_axis = figure.add_subplot(grid[1, 0])
    operational_grid = grid[1, 1].subgridspec(1, 2, wspace=float(style["rq1_operational_wspace"]))
    search_axis = figure.add_subplot(operational_grid[0, 0])
    cost_axis = figure.add_subplot(operational_grid[0, 1])
    ordered_stages = (
        stage_effects.set_index("stage")
        .reindex(config.section("design")["stage_order"])
        .dropna(subset=["estimate"])
        .reset_index()
    )
    access_forest = pd.concat(
        [
            pd.DataFrame([{"stage": "overall", **access}]),
            ordered_stages,
        ],
        ignore_index=True,
        sort=False,
    )
    stage_labels = reporting["labels"]["stages"]
    forest_labels = [
        "Overall"
        if row.stage == "overall"
        else (
            f"{stage_labels[row.stage]} (n={int(row.n_matches)}; desc.)"
            if pd.isna(row.ci_low)
            else f"{stage_labels[row.stage]} (n={int(row.n_matches)})"
        )
        for row in access_forest.itertuples(index=False)
    ]
    forest_plot(
        access_axis,
        access_forest,
        forest_labels,
        config,
        colors=[reporting["palette"]["primary"]]
        + [reporting["palette"]["neutral"]] * len(ordered_stages),
        markers=["D"] + ["o"] * len(ordered_stages),
    )
    access_axis.set_title("Access effect by tournament stage")
    access_axis.set_xlabel("Δ Brier (closed − open)\nPositive values favor open-book forecasts")
    access_axis.xaxis.set_major_locator(MaxNLocator(nbins=int(style["axis_max_major_ticks"])))

    _draw_chronological_access(time_axis, chronological_trend, config, compact=True)

    snapshot_plot = snapshot_table.copy()
    snapshot_labels = reporting["labels"]["snapshots"]
    snapshot_colors = [
        condition_color(
            config, "access", "open_book" if value.startswith("open_") else "closed_book"
        )
        for value in snapshot_plot["contrast"]
    ]
    snapshot_markers = [
        "o" if value.startswith("open_") else "s" for value in snapshot_plot["contrast"]
    ]
    forest_plot(
        snapshot_axis,
        snapshot_plot,
        [snapshot_labels[value] for value in snapshot_plot["contrast"]],
        config,
        colors=snapshot_colors,
        markers=snapshot_markers,
    )
    snapshot_axis.set_title("Forecast change across snapshots")
    snapshot_axis.set_xlabel("Δ Brier (earlier − later)\nPositive values favor the later snapshot")
    snapshot_axis.xaxis.set_major_locator(MaxNLocator(nbins=int(style["axis_max_major_ticks"])))

    open_ops = open_operations.sort_values("search_observed_rate")
    y_positions = np.arange(len(open_ops))
    search_axis.hlines(
        y_positions,
        0,
        open_ops["search_observed_rate"],
        color=reporting["palette"]["neutral_light"],
        linewidth=float(reporting["style"]["line_width"]),
    )
    search_axis.scatter(
        open_ops["search_observed_rate"],
        y_positions,
        color=condition_color(config, "access", "open_book"),
        zorder=3,
    )
    search_axis.set_yticks(
        y_positions, [model_label(config, value) for value in open_ops["model_id"]]
    )
    search_axis.set_xlim(-0.04, 1.05)
    search_axis.set_xticks([0.0, 0.5, 1.0])
    search_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    search_axis.set_title("Operational burden", loc="left")
    search_axis.set_xlabel("Observed search use")
    add_numeric_grid(search_axis, config, "x")

    cost_axis.hlines(
        y_positions,
        0,
        open_ops["delta_cost_usd"],
        color=reporting["palette"]["neutral_light"],
        linewidth=float(style["line_width"]),
    )
    cost_axis.scatter(
        open_ops["delta_cost_usd"],
        y_positions,
        color=reporting["palette"]["secondary"],
        marker="D",
        zorder=3,
    )
    cost_axis.set_yticks(y_positions, [""] * len(y_positions))
    cost_axis.margins(x=float(style["axis_margin_fraction"]))
    cost_axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
    cost_axis.xaxis.set_major_formatter(StrMethodFormatter("${x:.2f}"))
    cost_axis.set_title("")
    cost_axis.set_xlabel("Added cost\nper forecast")
    add_numeric_grid(cost_axis, config, "x")
    add_panel_label(access_axis, "A", config)
    add_panel_label(time_axis, "B", config)
    add_panel_label(snapshot_axis, "C", config)
    add_panel_label(
        search_axis,
        "D",
        config,
        x=float(style["rq1_operational_panel_label_x"]),
    )
    save_figure(figure, config, manifest, "fig_rq1_access", "rq1_access", source_hashes)

    time_figure, time_axis = plt.subplots(
        figsize=(float(reporting["figure_width_double"]), float(style["rq1_time_height"]))
    )
    _draw_chronological_access(time_axis, chronological_trend, config)
    save_figure(
        time_figure,
        config,
        manifest,
        "fig_rq1_access_over_time",
        "rq1_access",
        source_hashes,
    )

    model_figure, model_axis = plt.subplots(
        figsize=(float(reporting["figure_width_double"]), float(style["rq1_model_height"]))
    )
    model_order = model_access_summary.sort_values("mean_difference", ascending=False)[
        "model_id"
    ].tolist()
    sns.boxplot(
        data=model_access_units,
        x="difference",
        y="model_id",
        order=model_order,
        ax=model_axis,
        color=reporting["palette"]["neutral_light"],
        width=float(style["model_access_box_width"]),
        showfliers=False,
    )
    jitter_rng = np.random.default_rng(config.derived_seed("rq1.model_access_plot"))
    for y_position, model in enumerate(model_order):
        values = model_access_units.loc[
            model_access_units["model_id"].eq(model), "difference"
        ].to_numpy(dtype=float)
        jitter = jitter_rng.uniform(
            -float(style["model_access_jitter"]),
            float(style["model_access_jitter"]),
            size=len(values),
        )
        model_axis.scatter(
            values,
            y_position + jitter,
            s=float(style["box_point_size"]),
            alpha=float(style["box_point_alpha"]),
            color=reporting["palette"]["neutral"],
            edgecolors="none",
            zorder=2,
        )
        mean_value = float(
            model_access_summary.loc[
                model_access_summary["model_id"].eq(model), "mean_difference"
            ].iloc[0]
        )
        model_axis.plot(
            mean_value,
            y_position,
            marker="D",
            markersize=float(style["model_access_mean_marker_size"]),
            color=reporting["palette"]["primary"],
            linestyle="none",
            zorder=4,
        )
    model_axis.set_yticks(
        np.arange(len(model_order)), [model_label(config, model) for model in model_order]
    )
    _add_model_provider_icons(model_axis, model_order, style)
    model_axis.axvline(
        0.0,
        color=reporting["palette"]["neutral"],
        linestyle="--",
        linewidth=float(style["zero_line_width"]),
        zorder=1,
    )
    add_numeric_grid(model_axis, config, "x")
    model_axis.set_title("")
    model_axis.tick_params(
        axis="x",
        labelsize=float(style["model_access_tick_label_size"]),
    )
    model_axis.set_xlabel(
        "Δ Brier = closed − open (positive favors open book)",
        fontsize=float(style["model_access_axis_label_size"]),
    )
    model_axis.set_ylabel("")
    save_figure(
        model_figure,
        config,
        manifest,
        "fig_rq1_access_by_model",
        "rq1_access",
        source_hashes,
    )
    headline = headline_record(
        config,
        "rq1_headlines",
        "rq1.access.primary",
        "Mean T−24h closed-book minus open-book Brier",
        source_hashes,
        **access,
        n_predictions=int(len(primary_panel(panel, config))),
        units="Brier score",
        aggregation="models and prompts within match",
        extra={"leave_one_match_out": loo, "locked_sentence": framing_sentence},
    )
    write_headlines(config, manifest, "rq1_headlines", [headline], "rq1_access", source_hashes)
    manifest.write()
    return {
        "access": access,
        "stages": stage_effects,
        "snapshots": snapshot_table,
        "chronological_trend": chronological_trend,
        "model_access": model_access_summary,
        "dominance": dominance,
        "dominance_all_components_pass": dominance_all_components_pass,
        "variance": variance,
        "lead_times": lead_times,
        "incremental_operations": incremental_operations,
        "locked_sentence": framing_sentence,
    }
