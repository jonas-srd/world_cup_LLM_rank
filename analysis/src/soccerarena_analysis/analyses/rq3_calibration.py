from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..reporting.figures import add_numeric_grid, add_panel_label, apply_style, save_figure
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table
from ..statistics.bootstrap import studentized_cluster_bootstrap
from .common import load_panel, primary_panel

PROBABILITY_COLUMNS = {
    "home": "home_win_90_prob",
    "draw": "draw_90_prob",
    "away": "away_win_90_prob",
}
ACTUAL_LABELS = {"home": "H", "draw": "D", "away": "A"}


def _estimate_interval(
    estimate: float, lower: float, upper: float, decimals: int, scale: float = 1.0
) -> str:
    return (
        f"{estimate * scale:.{decimals}f} "
        f"[{lower * scale:.{decimals}f}, {upper * scale:.{decimals}f}]"
    )


def _format_p(value: float, decimals: int) -> str:
    threshold = 10.0 ** (-decimals)
    return f"<{threshold:.{decimals}f}" if value < threshold else f"{value:.{decimals}f}"


def _aggregate_cells(panel: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "stage", "access_condition", "prompt_strategy"]
    columns = list(PROBABILITY_COLUMNS.values()) + ["brier_90_recomputed", "confidence"]
    cells = panel.groupby(keys, as_index=False)[columns].mean()
    outcomes = panel.groupby(keys, as_index=False)["actual_result_90"].first()
    return cells.merge(outcomes, on=keys, validate="one_to_one")


def _bin_assignments(frame: pd.DataFrame, probability: str, bins: int) -> pd.Series:
    ranked = frame[probability].rank(method="first")
    return pd.qcut(ranked, q=min(bins, len(frame)), labels=False, duplicates="drop").astype(int)


def _reliability(cells: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    bins = int(config.section("calibration")["bins"])
    confidence = float(config.section("statistics")["confidence_level"])
    replicates = int(config.section("statistics")["bootstrap_replicates"])
    alpha = (1.0 - confidence) / 2.0
    records: list[dict[str, object]] = []
    for outcome in config.section("calibration")["outcomes"]:
        probability = PROBABILITY_COLUMNS[outcome]
        work = cells.copy()
        work["bin"] = _bin_assignments(work, probability, bins)
        work["observed"] = (work["actual_result_90"] == ACTUAL_LABELS[outcome]).astype(float)
        point = work.groupby("bin", as_index=False).agg(
            predicted=(probability, "mean"),
            observed=("observed", "mean"),
            n_matches=("match_id", "nunique"),
            n_cells=("match_id", "size"),
        )
        rng = np.random.default_rng(config.derived_seed(f"rq3.reliability.{outcome}"))
        bin_ids = sorted(int(value) for value in point["bin"])
        total_sums = np.zeros((replicates, len(bin_ids)), dtype=float)
        total_counts = np.zeros((replicates, len(bin_ids)), dtype=float)
        for stages in config.section("statistics")["strata"].values():
            stratum = work[work["stage"].isin(stages)]
            match_ids = stratum["match_id"].drop_duplicates().tolist()
            if not match_ids:
                continue
            sums = (
                stratum.pivot_table(
                    index="match_id", columns="bin", values="observed", aggfunc="sum", fill_value=0
                )
                .reindex(index=match_ids, columns=bin_ids, fill_value=0)
                .to_numpy(dtype=float)
            )
            counts = (
                stratum.assign(_count=1)
                .pivot_table(
                    index="match_id", columns="bin", values="_count", aggfunc="sum", fill_value=0
                )
                .reindex(index=match_ids, columns=bin_ids, fill_value=0)
                .to_numpy(dtype=float)
            )
            sampled = rng.integers(0, len(match_ids), size=(replicates, len(match_ids)))
            total_sums += sums[sampled].sum(axis=1)
            total_counts += counts[sampled].sum(axis=1)
        rates = np.divide(
            total_sums, total_counts, out=np.full_like(total_sums, np.nan), where=total_counts > 0
        )
        for row in point.itertuples(index=False):
            distribution = rates[:, bin_ids.index(int(row.bin))]
            distribution = distribution[np.isfinite(distribution)]
            records.append(
                {
                    "outcome": outcome,
                    "bin": int(row.bin) + 1,
                    "predicted": float(row.predicted),
                    "observed": float(row.observed),
                    "ci_low": float(np.quantile(distribution, alpha)),
                    "ci_high": float(np.quantile(distribution, 1.0 - alpha)),
                    "n_matches": int(row.n_matches),
                    "n_cells": int(row.n_cells),
                }
            )
    return pd.DataFrame(records)


def _clustered_mean_interval(
    frame: pd.DataFrame,
    value_column: str,
    config: AnalysisConfig,
    analysis_id: str,
) -> tuple[float, float, float]:
    replicates = int(config.section("statistics")["bootstrap_replicates"])
    confidence = float(config.section("statistics")["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(config.derived_seed(analysis_id))
    sampled_sums = np.zeros(replicates, dtype=float)
    sampled_counts = np.zeros(replicates, dtype=float)
    for stages in config.section("statistics")["strata"].values():
        stratum = frame[frame["stage"].isin(stages)]
        clusters = stratum.groupby("match_id")[value_column].agg(["sum", "count"])
        if clusters.empty:
            continue
        sampled = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
        sampled_sums += clusters["sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
        sampled_counts += clusters["count"].to_numpy(dtype=float)[sampled].sum(axis=1)
    distribution = sampled_sums / sampled_counts
    return (
        float(frame[value_column].mean()),
        float(np.quantile(distribution, alpha)),
        float(np.quantile(distribution, 1.0 - alpha)),
    )


def _within_cell_confidence_groups(frame: pd.DataFrame, config: AnalysisConfig) -> pd.Series:
    groups = int(config.section("calibration")["confidence_groups"])
    tie_method = str(config.section("calibration")["confidence_rank_tie_method"])
    percentile_rank = frame.groupby(
        ["model_id", "access_condition", "prompt_strategy"], group_keys=False
    )["confidence"].transform(lambda values: values.rank(method=tie_method, pct=True))
    return pd.Series(
        np.minimum(
            np.ceil(percentile_rank * groups).astype(int) - 1,
            groups - 1,
        ),
        index=frame.index,
    )


def _confidence_association(panel: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    work = panel.dropna(
        subset=["confidence", "brier_90_recomputed", "top_outcome_accuracy_90_recomputed"]
    ).copy()
    work["confidence_group"] = _within_cell_confidence_groups(work, config)
    records: list[dict[str, object]] = []
    for group_id, group in work.groupby("confidence_group"):
        brier_estimate, brier_low, brier_high = _clustered_mean_interval(
            group,
            "brier_90_recomputed",
            config,
            f"rq3.confidence_group.{int(group_id)}.brier",
        )
        accuracy_estimate, accuracy_low, accuracy_high = _clustered_mean_interval(
            group,
            "top_outcome_accuracy_90_recomputed",
            config,
            f"rq3.confidence_group.{int(group_id)}.accuracy",
        )
        records.append(
            {
                "confidence_group": int(group_id) + 1,
                "mean_confidence": float(group["confidence"].mean()),
                "mean_brier": brier_estimate,
                "brier_ci_low": brier_low,
                "brier_ci_high": brier_high,
                "accuracy": accuracy_estimate,
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "n_predictions": len(group),
                "n_matches": group["match_id"].nunique(),
            }
        )
    return pd.DataFrame(records)


def _draw_diagnostics(
    cells: pd.DataFrame, config: AnalysisConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    draw = cells.assign(
        observed_draw=(cells["actual_result_90"] == ACTUAL_LABELS["draw"]).astype(float),
        draw_brier_contribution=(
            cells[PROBABILITY_COLUMNS["draw"]]
            - (cells["actual_result_90"] == ACTUAL_LABELS["draw"]).astype(float)
        )
        ** 2,
        probability_sum=cells[list(PROBABILITY_COLUMNS.values())].sum(axis=1),
    )
    by_match = draw.groupby(["match_id", "stage"], as_index=False).agg(
        mean_draw_probability=(PROBABILITY_COLUMNS["draw"], "mean"),
        observed_draw=("observed_draw", "first"),
    )
    by_match["draw_probability_gap"] = by_match["mean_draw_probability"] - by_match["observed_draw"]
    gap = studentized_cluster_bootstrap(
        by_match,
        "draw_probability_gap",
        "stage",
        config,
        "rq3.draw_probability_gap",
    ).as_dict()
    summary = pd.DataFrame(
        [
            {
                "mean_draw_probability": draw[PROBABILITY_COLUMNS["draw"]].mean(),
                "observed_draw_frequency": draw["observed_draw"].mean(),
                "draw_probability_gap": gap["estimate"],
                "draw_probability_gap_ci_low": gap["ci_low"],
                "draw_probability_gap_ci_high": gap["ci_high"],
                "draw_probability_gap_p": gap["p_raw"],
                "mean_draw_brier_contribution": draw["draw_brier_contribution"].mean(),
                "mean_probability_sum": draw["probability_sum"].mean(),
                "max_absolute_sum_error": (draw["probability_sum"] - 1.0).abs().max(),
                "n_matches": draw["match_id"].nunique(),
                "n_cells": len(draw),
            }
        ]
    )
    return summary, gap


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    frame, source_hashes = load_panel(config, manifest)
    panel = primary_panel(frame, config)
    cells = _aggregate_cells(panel)
    reliability = _reliability(cells, config)
    confidence = _confidence_association(panel, config)
    draw_summary, draw_gap = _draw_diagnostics(cells, config)
    reporting = config.section("reporting")
    style = reporting["style"]
    percentage_decimals = int(style["table_percentage_decimals"])
    float_decimals = int(style["table_float_decimals"])
    p_decimals = int(style["table_p_decimals"])

    reliability_trace = reliability.rename(columns={"observed": "estimate"}).assign(
        analysis_id=lambda values: (
            "rq3.reliability." + values["outcome"] + ".bin" + values["bin"].astype(str)
        ),
        p_raw=np.nan,
        p_adjusted=np.nan,
        median=np.nan,
        n_predictions=reliability["n_cells"],
        units="proportion",
        aggregation="one probability vector per match and design cell; match-clustered interval",
    )
    reliability_display = reliability.assign(
        Outcome=reliability["outcome"].str.title(),
        Bin=reliability["bin"],
        **{
            "Mean predicted, %": reliability["predicted"].mul(100.0).round(percentage_decimals),
            "Observed [95% CI], %": reliability.apply(
                lambda row: _estimate_interval(
                    float(row["observed"]),
                    float(row["ci_low"]),
                    float(row["ci_high"]),
                    percentage_decimals,
                    100.0,
                ),
                axis=1,
            ),
            "Unique matches": reliability["n_matches"],
            "Cells": reliability["n_cells"],
        },
    )[
        [
            "Outcome",
            "Bin",
            "Mean predicted, %",
            "Observed [95% CI], %",
            "Unique matches",
            "Cells",
        ]
    ]
    save_table(
        reliability_display,
        config,
        manifest,
        "rq3_reliability",
        "rq3",
        source_hashes,
        headline_frame=reliability_trace,
    )

    draw_row = draw_summary.iloc[0]
    draw_display = pd.DataFrame(
        [
            {
                "Quantity": "Mean predicted draw probability",
                "Estimate": f"{float(draw_row['mean_draw_probability']) * 100.0:.{percentage_decimals}f}%",
                "95% CI": "--",
                "p": "--",
            },
            {
                "Quantity": "Observed draw frequency",
                "Estimate": f"{float(draw_row['observed_draw_frequency']) * 100.0:.{percentage_decimals}f}%",
                "95% CI": "--",
                "p": "--",
            },
            {
                "Quantity": "Predicted minus observed draw rate",
                "Estimate": f"{float(draw_row['draw_probability_gap']) * 100.0:.{percentage_decimals}f} pp",
                "95% CI": (
                    f"[{float(draw_row['draw_probability_gap_ci_low']) * 100.0:.{percentage_decimals}f}, "
                    f"{float(draw_row['draw_probability_gap_ci_high']) * 100.0:.{percentage_decimals}f}] pp"
                ),
                "p": _format_p(float(draw_row["draw_probability_gap_p"]), p_decimals),
            },
            {
                "Quantity": "Mean draw-class Brier contribution",
                "Estimate": f"{float(draw_row['mean_draw_brier_contribution']):.{float_decimals}f}",
                "95% CI": "--",
                "p": "--",
            },
            {
                "Quantity": "Mean H/D/A probability sum",
                "Estimate": f"{float(draw_row['mean_probability_sum']):.{float_decimals}f}",
                "95% CI": "--",
                "p": "--",
            },
            {
                "Quantity": "Maximum absolute probability-sum error",
                "Estimate": f"{float(draw_row['max_absolute_sum_error']):.2e}",
                "95% CI": "--",
                "p": "--",
            },
        ]
    )
    draw_gap_trace = pd.DataFrame(
        [
            {
                **draw_gap,
                "units": "probability",
                "aggregation": "four design cells averaged within each match",
                "n_predictions": int(draw_row["n_cells"]),
            }
        ]
    )
    save_table(
        draw_display,
        config,
        manifest,
        "rq3_draw_summary",
        "rq3",
        source_hashes,
        headline_frame=draw_gap_trace,
    )

    confidence_trace_rows: list[dict[str, object]] = []
    for row in confidence.itertuples(index=False):
        confidence_trace_rows.extend(
            [
                {
                    "analysis_id": f"rq3.confidence_group.{row.confidence_group}.brier",
                    "estimate": row.mean_brier,
                    "ci_low": row.brier_ci_low,
                    "ci_high": row.brier_ci_high,
                    "p_raw": np.nan,
                    "p_adjusted": np.nan,
                    "median": np.nan,
                    "n_matches": row.n_matches,
                },
                {
                    "analysis_id": f"rq3.confidence_group.{row.confidence_group}.accuracy",
                    "estimate": row.accuracy,
                    "ci_low": row.accuracy_ci_low,
                    "ci_high": row.accuracy_ci_high,
                    "p_raw": np.nan,
                    "p_adjusted": np.nan,
                    "median": np.nan,
                    "n_matches": row.n_matches,
                },
            ]
        )
    confidence_display = confidence.assign(
        **{
            "Within-cell confidence rank": confidence["confidence_group"],
            "Mean confidence, %": confidence["mean_confidence"]
            .mul(100.0)
            .round(percentage_decimals),
            "Mean Brier [95% CI]": confidence.apply(
                lambda row: _estimate_interval(
                    float(row["mean_brier"]),
                    float(row["brier_ci_low"]),
                    float(row["brier_ci_high"]),
                    float_decimals,
                ),
                axis=1,
            ),
            "Modal accuracy [95% CI], %": confidence.apply(
                lambda row: _estimate_interval(
                    float(row["accuracy"]),
                    float(row["accuracy_ci_low"]),
                    float(row["accuracy_ci_high"]),
                    percentage_decimals,
                    100.0,
                ),
                axis=1,
            ),
            "Predictions": confidence["n_predictions"],
            "Matches": confidence["n_matches"],
        }
    )[
        [
            "Within-cell confidence rank",
            "Mean confidence, %",
            "Mean Brier [95% CI]",
            "Modal accuracy [95% CI], %",
            "Predictions",
            "Matches",
        ]
    ]
    save_table(
        confidence_display,
        config,
        manifest,
        "rq3_confidence_accuracy_association",
        "rq3",
        source_hashes,
        headline_frame=pd.DataFrame(confidence_trace_rows),
    )

    apply_style(config)
    figure = plt.figure(
        figsize=(float(reporting["figure_width_double"]), float(style["rq3_height"]))
    )
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[float(value) for value in style["rq3_row_height_ratios"]],
        hspace=float(style["rq3_outer_hspace"]),
    )
    reliability_grid = outer_grid[0].subgridspec(
        1,
        3,
        wspace=float(style["rq3_reliability_wspace"]),
    )
    confidence_grid = outer_grid[1].subgridspec(
        1,
        2,
        wspace=float(style["rq3_confidence_wspace"]),
    )
    reliability_axes = [
        figure.add_subplot(reliability_grid[0, 0]),
        figure.add_subplot(reliability_grid[0, 1]),
        figure.add_subplot(reliability_grid[0, 2]),
    ]
    brier_axis = figure.add_subplot(confidence_grid[0, 0])
    accuracy_axis = figure.add_subplot(confidence_grid[0, 1], sharex=brier_axis)
    outcome_colors = reporting["colors"]["outcome"]
    for axis, outcome in zip(reliability_axes, config.section("calibration")["outcomes"]):
        part = reliability[reliability["outcome"] == outcome].sort_values("bin")
        axis.errorbar(
            part["predicted"],
            part["observed"],
            yerr=[part["observed"] - part["ci_low"], part["ci_high"] - part["observed"]],
            marker="o",
            color=outcome_colors[outcome],
            capsize=float(style["ci_cap_size"]),
        )
        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color=reporting["palette"]["neutral"],
            linestyle="--",
            linewidth=float(style["zero_line_width"]),
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.tick_params(
            axis="both",
            labelsize=float(style["rq3_tick_label_size"]),
        )
        axis.set_title(
            f"{outcome.title()} outcome",
            fontsize=float(style["rq3_title_size"]),
            pad=float(style["rq3_title_pad"]),
        )
        add_numeric_grid(axis, config, "both")
    reliability_axes[0].set_ylabel(
        "Observed frequency",
        fontsize=float(style["rq3_axis_label_size"]),
    )
    reliability_axes[1].set_xlabel(
        "Mean predicted probability",
        fontsize=float(style["rq3_axis_label_size"]),
    )
    for axis in reliability_axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    confidence_color = reporting["colors"]["confidence"]
    brier_axis.errorbar(
        confidence["mean_confidence"],
        confidence["mean_brier"],
        yerr=[
            confidence["mean_brier"] - confidence["brier_ci_low"],
            confidence["brier_ci_high"] - confidence["mean_brier"],
        ],
        marker="o",
        color=confidence_color,
        capsize=float(style["ci_cap_size"]),
    )
    brier_axis.set_title(
        "Brier score vs confidence",
        x=0.03,
        fontsize=float(style["rq3_title_size"]),
        pad=float(style["rq3_title_pad"]),
    )
    brier_axis.set_xlabel(
        "Mean elicited confidence",
        fontsize=float(style["rq3_axis_label_size"]),
    )
    brier_axis.set_ylabel(
        "Mean Brier score\n(lower is better)",
        fontsize=float(style["rq3_axis_label_size"]),
    )
    brier_axis.tick_params(
        axis="both",
        labelsize=float(style["rq3_tick_label_size"]),
    )
    brier_axis.xaxis.set_major_locator(MaxNLocator(nbins=int(style["axis_max_major_ticks"])))
    brier_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    add_numeric_grid(brier_axis, config, "both")

    accuracy_axis.errorbar(
        confidence["mean_confidence"],
        confidence["accuracy"],
        yerr=[
            confidence["accuracy"] - confidence["accuracy_ci_low"],
            confidence["accuracy_ci_high"] - confidence["accuracy"],
        ],
        marker="s",
        color=confidence_color,
        capsize=float(style["ci_cap_size"]),
    )
    accuracy_axis.set_title(
        "Accuracy vs confidence",
        x=0.03,
        fontsize=float(style["rq3_title_size"]),
        pad=float(style["rq3_title_pad"]),
    )
    accuracy_axis.set_xlabel(
        "Mean elicited confidence",
        fontsize=float(style["rq3_axis_label_size"]),
    )
    accuracy_axis.set_ylabel(
        "Modal accuracy\n(higher is better)",
        fontsize=float(style["rq3_axis_label_size"]),
    )
    accuracy_axis.tick_params(
        axis="both",
        labelsize=float(style["rq3_tick_label_size"]),
    )
    accuracy_axis.xaxis.set_major_locator(MaxNLocator(nbins=int(style["axis_max_major_ticks"])))
    accuracy_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    add_numeric_grid(accuracy_axis, config, "both")
    label_offset = tuple(float(value) for value in style["rq3_confidence_label_offset_points"])
    for x_value, y_value, group_id in zip(
        confidence["mean_confidence"],
        confidence["mean_brier"],
        confidence["confidence_group"],
    ):
        brier_axis.annotate(
            str(int(group_id)),
            (float(x_value), float(y_value)),
            xytext=label_offset,
            textcoords="offset points",
            color=reporting["palette"]["text"],
            fontsize=float(style["rq3_annotation_size"]),
            va="bottom",
        )

    accuracy_label_offsets = style["rq3_accuracy_label_offsets_points"]
    for x_value, y_value, group_id in zip(
        confidence["mean_confidence"],
        confidence["accuracy"],
        confidence["confidence_group"],
    ):
        group_number = int(group_id)
        offset = tuple(float(value) for value in accuracy_label_offsets[group_number])
        accuracy_axis.annotate(
            str(group_number),
            (float(x_value), float(y_value)),
            xytext=offset,
            textcoords="offset points",
            color=reporting["palette"]["text"],
            fontsize=float(style["rq3_annotation_size"]),
            ha="right" if offset[0] < 0 else "left",
            va="bottom",
        )
    rq3_panel_x = float(style["rq3_panel_label_x"])
    rq3_panel_y = float(style["rq3_panel_label_y"])
    rq3_panel_size = float(style["rq3_panel_label_size"])
    rq3_bottom_panel_x = float(style["rq3_bottom_panel_label_x"])
    rq3_accuracy_panel_x = float(style["rq3_accuracy_panel_label_x"])
    rq3_bottom_panel_y = float(style["rq3_bottom_panel_label_y"])
    add_panel_label(
        reliability_axes[0],
        "A",
        config,
        x=rq3_panel_x,
        y=rq3_panel_y,
        font_size=rq3_panel_size,
    )
    add_panel_label(
        brier_axis,
        "B",
        config,
        x=rq3_bottom_panel_x,
        y=rq3_bottom_panel_y,
        font_size=rq3_panel_size,
    )
    add_panel_label(
        accuracy_axis,
        "C",
        config,
        x=rq3_accuracy_panel_x,
        y=rq3_bottom_panel_y,
        font_size=rq3_panel_size,
    )
    figure.subplots_adjust(**style["rq3_margins"])
    save_figure(figure, config, manifest, "rq3_calibration", "rq3", source_hashes)

    summary = draw_summary.iloc[0]
    records = [
        headline_record(
            config,
            "rq3_headlines",
            "rq3",
            "mean elicited draw probability",
            source_hashes,
            estimate=float(summary["mean_draw_probability"]),
            ci_low=None,
            ci_high=None,
            p_raw=None,
            p_adjusted=None,
            median=None,
            n_matches=int(summary["n_matches"]),
            n_predictions=int(summary["n_cells"]),
            units="probability",
            aggregation="one probability vector per match and design cell",
            extra={"null_reason": "descriptive estimand"},
        ),
        headline_record(
            config,
            "rq3_headlines",
            "rq3.observed_draw_frequency",
            "observed draw frequency",
            source_hashes,
            estimate=float(summary["observed_draw_frequency"]),
            ci_low=None,
            ci_high=None,
            p_raw=None,
            p_adjusted=None,
            median=None,
            n_matches=int(summary["n_matches"]),
            n_predictions=int(summary["n_cells"]),
            units="proportion",
            aggregation="one observed outcome per match, repeated only for balanced cell summary",
            extra={"null_reason": "descriptive estimand"},
        ),
        headline_record(
            config,
            "rq3_headlines",
            "rq3.draw_brier_contribution",
            "mean draw-class Brier contribution",
            source_hashes,
            estimate=float(summary["mean_draw_brier_contribution"]),
            ci_low=None,
            ci_high=None,
            p_raw=None,
            p_adjusted=None,
            median=None,
            n_matches=int(summary["n_matches"]),
            n_predictions=int(summary["n_cells"]),
            units="squared probability error",
            aggregation="one probability vector per match and design cell",
            extra={"null_reason": "descriptive estimand"},
        ),
    ]
    write_headlines(config, manifest, "rq3_headlines", records, "rq3", source_hashes)
    return {"reliability": reliability, "draw": draw_summary, "confidence": confidence}
