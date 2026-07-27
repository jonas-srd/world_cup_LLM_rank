from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..reporting.figures import add_panel_label, apply_style, forest_plot, model_label, save_figure
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table
from ..statistics.bootstrap import studentized_cluster_bootstrap
from ..statistics.metrics import brier_score
from ..statistics.multiplicity import holm_adjust
from .common import load_panel, primary_panel

PROBABILITY_COLUMNS = ["home_win_90_prob", "draw_90_prob", "away_win_90_prob"]

MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "deepseek/deepseek-v4-pro": "deepseek-color.png",
    "google/gemini-3.1-pro-preview": "google-color.png",
    "x-ai/grok-4.3": "xai.png",
    "openai/gpt-5.5": "openai.png",
    "mistralai/mistral-large-2512": "mistral-color.png",
    "qwen/qwen3.7-max": "alibaba-color.png",
}


def _model_icon_paths(model_order: list[str]) -> dict[str, Path]:
    icon_directory = Path(__file__).resolve().parents[3] / "assets" / "brand_icons"
    icon_paths = {model: icon_directory / MODEL_PROVIDER_ICONS[model] for model in model_order}
    missing_icons = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing_icons:
        raise FileNotFoundError(f"Missing model provider icons: {missing_icons}")
    return icon_paths


def _add_heatmap_model_icons(
    axis: plt.Axes,
    model_order: list[str],
    style: dict[str, object],
) -> None:
    icon_paths = _model_icon_paths(model_order)
    axis.set_xticklabels([""] * len(model_order))
    axis.set_yticklabels([""] * len(model_order))
    axis.tick_params(axis="both", length=0)
    for position, model in enumerate(model_order):
        center = position + 0.5
        icon_data = plt.imread(icon_paths[model])
        axis.add_artist(
            AnnotationBbox(
                OffsetImage(
                    icon_data,
                    zoom=float(style["rq4_icon_zoom"]),
                    interpolation="lanczos",
                ),
                (center, float(style["rq4_heatmap_x_icon_y"])),
                xycoords=("data", "axes fraction"),
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )
        axis.add_artist(
            AnnotationBbox(
                OffsetImage(
                    icon_data,
                    zoom=float(style["rq4_icon_zoom"]),
                    interpolation="lanczos",
                ),
                (float(style["rq4_heatmap_y_icon_x"]), center),
                xycoords=("axes fraction", "data"),
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _add_forest_model_icons(
    axis: plt.Axes,
    comparison_order: list[str],
    style: dict[str, object],
) -> None:
    model_order = [value for value in comparison_order if value != "average_member"]
    icon_paths = _model_icon_paths(model_order)
    axis.tick_params(
        axis="y",
        labelsize=float(style["rq4_tick_label_size"]),
        pad=float(style["rq4_forest_label_pad"]),
        length=0,
    )
    for y_position, model in enumerate(comparison_order):
        if model == "average_member":
            continue
        axis.add_artist(
            AnnotationBbox(
                OffsetImage(
                    plt.imread(icon_paths[model]),
                    zoom=float(style["rq4_icon_zoom"]),
                    interpolation="lanczos",
                ),
                (float(style["rq4_forest_icon_x"]), y_position),
                xycoords=("axes fraction", "data"),
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _estimate_interval(estimate: float, lower: float, upper: float, decimals: int) -> str:
    return f"{estimate:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def _format_p(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "--"
    threshold = 10.0 ** (-decimals)
    return f"<{threshold:.{decimals}f}" if value < threshold else f"{value:.{decimals}f}"


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = (first + second) / 2.0
    terms = []
    for values in (first, second):
        valid = values > 0
        terms.append(float(np.sum(values[valid] * np.log(values[valid] / midpoint[valid]))))
    return float((terms[0] + terms[1]) / 2.0)


def _pairwise(panel: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["match_id", "stage", "access_condition", "prompt_strategy", "model_id"]
    cells = panel.groupby(keys, as_index=False)[PROBABILITY_COLUMNS].mean()
    js_records: list[dict[str, object]] = []
    correlation_records: list[dict[str, object]] = []
    for first, second in combinations(config.section("design")["complete_panel"], 2):
        left = cells[cells["model_id"] == first].drop(columns="model_id")
        right = cells[cells["model_id"] == second].drop(columns="model_id")
        paired = left.merge(right, on=keys[:-1], suffixes=("_a", "_b"), validate="one_to_one")
        for row in paired.itertuples(index=False):
            a = np.asarray(
                [getattr(row, f"{column}_a") for column in PROBABILITY_COLUMNS], dtype=float
            )
            b = np.asarray(
                [getattr(row, f"{column}_b") for column in PROBABILITY_COLUMNS], dtype=float
            )
            js_records.append(
                {
                    "match_id": row.match_id,
                    "stage": row.stage,
                    "access_condition": row.access_condition,
                    "prompt_strategy": row.prompt_strategy,
                    "model_a": first,
                    "model_b": second,
                    "js_divergence": _js_divergence(a, b),
                }
            )
        for access in config.section("design")["access_conditions"]:
            for prompt in config.section("design")["prompt_strategies"]:
                subset = paired[
                    (paired["access_condition"] == access) & (paired["prompt_strategy"] == prompt)
                ]
                correlations: list[float] = []
                for column in PROBABILITY_COLUMNS:
                    correlation = subset[f"{column}_a"].corr(subset[f"{column}_b"])
                    clip = float(config.section("diversity")["fisher_correlation_clip"])
                    correlations.append(float(np.clip(correlation, -clip, clip)))
                fisher = float(np.tanh(np.mean(np.arctanh(correlations))))
                correlation_records.append(
                    {
                        "model_a": first,
                        "model_b": second,
                        "access_condition": access,
                        "prompt_strategy": prompt,
                        "fisher_z_combined_correlation": fisher,
                    }
                )
    return pd.DataFrame(js_records), pd.DataFrame(correlation_records)


def _ensemble(panel: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_keys = ["match_id", "stage", "forecast_horizon", "access_condition", "prompt_strategy"]
    records: list[dict[str, object]] = []
    for keys, group in panel.groupby(cell_keys, sort=False):
        if set(group["model_id"]) != set(config.section("design")["complete_panel"]):
            continue
        probabilities = group.groupby("model_id")[PROBABILITY_COLUMNS].mean()
        ensemble = probabilities.mean(axis=0).to_numpy(dtype=float)
        actual = group["actual_result_90"].iloc[0]
        ensemble_brier = brier_score(tuple(ensemble), actual)
        member_scores = group.groupby("model_id")["brier_90_recomputed"].mean()
        dispersion = float(
            np.mean(
                np.sum(
                    (probabilities.to_numpy(dtype=float) - ensemble) ** 2,
                    axis=1,
                )
            )
        )
        average_member_difference = ensemble_brier - float(member_scores.mean())
        tolerance = float(config.section("diversity")["average_member_identity_tolerance"])
        if not np.isclose(
            average_member_difference,
            -dispersion,
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError("Same-cell Brier ensemble identity failed")
        base = dict(zip(cell_keys, keys))
        for model_id, score in member_scores.items():
            records.append(
                {
                    **base,
                    "member": model_id,
                    "difference": ensemble_brier - float(score),
                    "outcome_independent_dispersion": np.nan,
                }
            )
        records.append(
            {
                **base,
                "member": "average_member",
                "difference": average_member_difference,
                "outcome_independent_dispersion": dispersion,
            }
        )
    differences = pd.DataFrame(records)
    raw: dict[str, object] = {}
    results: list[dict[str, object]] = []
    for member, group in differences.groupby("member", sort=False):
        units = group.groupby(["match_id", "stage"], as_index=False)["difference"].mean()
        result = studentized_cluster_bootstrap(
            units, "difference", "stage", config, f"rq4.ensemble.{member}"
        )
        raw[member] = result
    declared_members = set(config.section("design")["complete_panel"])
    inferential_members = {member for member in raw if member != "average_member"}
    if inferential_members != declared_members:
        raise ValueError("RQ4 ensemble Holm family does not match the complete model panel")
    adjusted = holm_adjust(
        {member: raw[member].p_raw for member in config.section("design")["complete_panel"]}
    )
    for member, result in raw.items():
        record = {"comparison": member, **result.as_dict()}
        if member == "average_member":
            record["p_raw"] = np.nan
            record["p_adjusted"] = np.nan
            record["comparison_type"] = "algebraic_identity"
        else:
            record["p_adjusted"] = adjusted[member]
            record["comparison_type"] = "paired_model_comparison"
        results.append(record)
    return differences, pd.DataFrame(results)


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, pd.DataFrame]:
    manifest.discard_generated(
        [
            "rq4_pairwise_js_csv",
            "rq4_pairwise_js_companion",
            "rq4_pairwise_correlations_csv",
            "rq4_pairwise_correlations_tex",
            "rq4_pairwise_correlations_companion",
        ]
    )
    frame, source_hashes = load_panel(config, manifest)
    panel = primary_panel(frame, config)
    js, correlations = _pairwise(panel, config)
    diversity = js.groupby(["match_id", "stage", "access_condition"], as_index=False)[
        "js_divergence"
    ].mean()
    pivot = (
        diversity.pivot_table(
            index=["match_id", "stage"], columns="access_condition", values="js_divergence"
        )
        .dropna()
        .reset_index()
    )
    pivot["difference"] = pivot["open_book"] - pivot["closed_book"]
    diversity_result = studentized_cluster_bootstrap(
        pivot, "difference", "stage", config, "rq4.open_vs_closed_diversity"
    )
    differences, ensemble_results = _ensemble(panel, config)
    reporting = config.section("reporting")
    style = reporting["style"]
    js_scale = float(style["heatmap_js_scale"])
    effect_decimals = int(style["rq4_effect_decimals"])
    p_decimals = int(style["table_p_decimals"])

    result_dir = config.resolve_path("results") / "rq4"
    result_dir.mkdir(parents=True, exist_ok=True)
    js_path = result_dir / "pairwise_js_cells.parquet"
    correlation_path = result_dir / "pairwise_correlations_cells.parquet"
    differences_path = result_dir / "ensemble_cell_differences.parquet"
    js.to_parquet(js_path, index=False)
    correlations.to_parquet(correlation_path, index=False)
    differences.to_parquet(differences_path, index=False)
    for artifact_id, path, rows in (
        ("rq4_pairwise_js_cells", js_path, len(js)),
        ("rq4_pairwise_correlations_cells", correlation_path, len(correlations)),
        ("rq4_ensemble_cell_differences", differences_path, len(differences)),
    ):
        manifest.add(artifact_id, path, "parquet", "rq4", source_hashes, {"rows": rows})

    pairwise_js = js.groupby(["model_a", "model_b"], as_index=False).agg(
        mean_js_divergence=("js_divergence", "mean"),
        same_cell_comparisons=("js_divergence", "size"),
    )
    clip = float(config.section("diversity")["fisher_correlation_clip"])
    pairwise_correlation = correlations.groupby(["model_a", "model_b"], as_index=False).agg(
        fisher_z_combined_correlation=(
            "fisher_z_combined_correlation",
            lambda values: float(np.tanh(np.mean(np.arctanh(np.clip(values, -clip, clip))))),
        )
    )
    pairwise_summary = pairwise_js.merge(
        pairwise_correlation,
        on=["model_a", "model_b"],
        validate="one_to_one",
    )
    pairwise_display = pairwise_summary.assign(
        **{
            "Model A": pairwise_summary["model_a"].map(lambda value: model_label(config, value)),
            "Model B": pairwise_summary["model_b"].map(lambda value: model_label(config, value)),
            "Mean JS divergence (×10⁻³ nats)": pairwise_summary["mean_js_divergence"]
            .mul(js_scale)
            .round(int(style["table_percentage_decimals"])),
            "Fisher-z combined correlation": pairwise_summary[
                "fisher_z_combined_correlation"
            ].round(int(style["table_float_decimals"])),
            "Same-cell comparisons": pairwise_summary["same_cell_comparisons"],
        }
    )[
        [
            "Model A",
            "Model B",
            "Mean JS divergence (×10⁻³ nats)",
            "Fisher-z combined correlation",
            "Same-cell comparisons",
        ]
    ]
    save_table(pairwise_display, config, manifest, "rq4_pairwise_summary", "rq4", source_hashes)

    diversity_trace = pd.DataFrame(
        [{"contrast": "open_minus_closed_js", **diversity_result.as_dict()}]
    ).assign(
        units="natural-log JS divergence",
        aggregation="21 model pairs and two prompt cells averaged within match and access",
        n_predictions=len(js),
    )
    diversity_display = pd.DataFrame(
        [
            {
                "Contrast": "Open-book − closed-book mean pairwise JS",
                "Estimate [95% CI] (×10⁻³ nats)": _estimate_interval(
                    diversity_result.estimate * js_scale,
                    diversity_result.ci_low * js_scale,
                    diversity_result.ci_high * js_scale,
                    int(style["table_percentage_decimals"]),
                ),
                "p": _format_p(diversity_result.p_raw, p_decimals),
                "Median (×10⁻³ nats)": round(
                    diversity_result.median * js_scale,
                    int(style["table_percentage_decimals"]),
                ),
                "n": diversity_result.n_matches,
            }
        ]
    )
    save_table(
        diversity_display,
        config,
        manifest,
        "rq4_access_diversity_effect",
        "rq4",
        source_hashes,
        headline_frame=diversity_trace,
    )

    comparison_order = ["average_member", *config.section("design")["complete_panel"]]
    ensemble_report = (
        ensemble_results.set_index("comparison")
        .reindex(comparison_order)
        .dropna(subset=["estimate"])
        .reset_index()
    )
    comparison_counts = differences.groupby("member").size()
    ensemble_report = ensemble_report.assign(
        units="Brier score",
        aggregation="four access-prompt cells averaged within each match",
        n_predictions=ensemble_report["comparison"].map(comparison_counts).astype(int),
        null_reason=ensemble_report["comparison"].map(
            lambda value: (
                "average-member difference is an algebraic Brier identity; no null test"
                if value == "average_member"
                else None
            )
        ),
    )
    ensemble_display = ensemble_report.assign(
        Comparator=ensemble_report["comparison"].map(
            lambda value: (
                "Average member (Brier identity)"
                if value == "average_member"
                else model_label(config, value)
            )
        ),
        **{
            "Ensemble − comparator Δ Brier [95% CI]": ensemble_report.apply(
                lambda row: _estimate_interval(
                    float(row["estimate"]),
                    float(row["ci_low"]),
                    float(row["ci_high"]),
                    effect_decimals,
                ),
                axis=1,
            ),
            "Raw p": ensemble_report["p_raw"].map(
                lambda value: _format_p(float(value), p_decimals)
            ),
            "Holm p": ensemble_report["p_adjusted"].map(
                lambda value: _format_p(float(value), p_decimals)
            ),
            "Median Δ": ensemble_report["median"].round(effect_decimals),
            "n": ensemble_report["n_matches"],
        },
    )[
        [
            "Comparator",
            "Ensemble − comparator Δ Brier [95% CI]",
            "Raw p",
            "Holm p",
            "Median Δ",
            "n",
        ]
    ]
    save_table(
        ensemble_display,
        config,
        manifest,
        "rq4_ensemble_effects",
        "rq4",
        source_hashes,
        headline_frame=ensemble_report,
    )

    apply_style(config)
    figure = plt.figure(
        figsize=(float(reporting["figure_width_double"]), float(style["rq4_height"]))
    )
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[float(value) for value in style["rq4_row_height_ratios"]],
        hspace=float(style["rq4_hspace"]),
        wspace=float(style["rq4_wspace"]),
    )
    js_axis = figure.add_subplot(grid[0, 0])
    correlation_axis = figure.add_subplot(grid[0, 1])
    ensemble_axis = figure.add_subplot(grid[1, :])
    models = list(config.section("design")["complete_panel"])
    js_matrix = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    correlation_matrix = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    for row in pairwise_summary.itertuples(index=False):
        js_matrix.loc[row.model_a, row.model_b] = row.mean_js_divergence * float(
            style["heatmap_js_scale"]
        )
        js_matrix.loc[row.model_b, row.model_a] = row.mean_js_divergence * float(
            style["heatmap_js_scale"]
        )
        correlation_matrix.loc[row.model_a, row.model_b] = row.fisher_z_combined_correlation
        correlation_matrix.loc[row.model_b, row.model_a] = row.fisher_z_combined_correlation
    mask = np.triu(np.ones_like(js_matrix, dtype=bool))
    heatmap_common = {
        "mask": mask,
        "annot": True,
        "square": True,
        "linewidths": float(style["heatmap_line_width"]),
        "linecolor": figure.get_facecolor(),
        "annot_kws": {"fontsize": float(style["rq4_heatmap_annotation_size"])},
    }
    colorbar_kwargs = {
        "shrink": float(style["heatmap_colorbar_shrink"]),
        "pad": float(style["rq4_colorbar_pad"]),
        "aspect": int(style["rq4_colorbar_aspect"]),
    }
    js_heatmap = sns.heatmap(
        js_matrix,
        ax=js_axis,
        cmap=str(style["rq4_js_colormap"]),
        fmt=".1f",
        vmin=float(style["heatmap_js_min"]),
        vmax=float(style["heatmap_js_max"]),
        cbar_kws=colorbar_kwargs,
        **heatmap_common,
    )
    js_colorbar = js_heatmap.collections[0].colorbar
    js_colorbar.ax.set_title(
        r"$10^{-3}$ nats",
        fontsize=float(style["rq4_note_size"]),
        pad=4,
    )
    js_colorbar.ax.tick_params(labelsize=float(style["rq4_tick_label_size"]))
    _add_heatmap_model_icons(js_axis, models, style)
    js_axis.set_title(
        "Pairwise JS divergence",
        fontsize=float(style["rq4_title_size"]),
        pad=float(style["rq4_title_pad"]),
    )
    js_axis.text(
        float(style["rq4_access_note_x"]),
        float(style["rq4_access_note_y"]),
        "Access contrast (open − closed)\n"
        f"Δ = {diversity_result.estimate * js_scale:.1f} "
        f"[{diversity_result.ci_low * js_scale:.1f}, "
        f"{diversity_result.ci_high * js_scale:.1f}], "
        f"p = {diversity_result.p_raw:.3f}",
        transform=js_axis.transAxes,
        ha="right",
        va="top",
        color=reporting["palette"]["neutral"],
        fontsize=float(style["rq4_note_size"]),
    )
    js_axis.set_xlabel("")
    js_axis.set_ylabel("")

    correlation_heatmap = sns.heatmap(
        correlation_matrix,
        ax=correlation_axis,
        cmap=str(style["rq4_correlation_colormap"]),
        fmt=".2f",
        vmin=float(style["heatmap_correlation_min"]),
        vmax=float(style["heatmap_correlation_max"]),
        cbar_kws=colorbar_kwargs,
        **heatmap_common,
    )
    correlation_colorbar = correlation_heatmap.collections[0].colorbar
    correlation_colorbar.ax.set_title(
        "Mean r",
        fontsize=float(style["rq4_note_size"]),
        pad=4,
    )
    correlation_colorbar.ax.tick_params(labelsize=float(style["rq4_tick_label_size"]))
    _add_heatmap_model_icons(correlation_axis, models, style)
    correlation_axis.set_title(
        "Pairwise probability correlation",
        fontsize=float(style["rq4_title_size"]),
        pad=float(style["rq4_title_pad"]),
    )
    correlation_axis.text(
        float(style["rq4_access_note_x"]),
        float(style["rq4_access_note_y"]),
        "Fisher-z mean across H/D/A",
        transform=correlation_axis.transAxes,
        ha="right",
        va="top",
        color=reporting["palette"]["neutral"],
        fontsize=float(style["rq4_note_size"]),
    )
    correlation_axis.set_xlabel("")
    correlation_axis.set_ylabel("")

    ordered = ensemble_report
    comparison_labels = [
        "Average member (identity)" if value == "average_member" else ""
        for value in ordered["comparison"]
    ]
    forest_plot(
        ensemble_axis,
        ordered,
        comparison_labels,
        config,
        colors=[reporting["palette"]["primary"]]
        + [reporting["palette"]["neutral"]] * (len(ordered) - 1),
        markers=["D"] + ["o"] * (len(ordered) - 1),
    )
    _add_forest_model_icons(
        ensemble_axis,
        ordered["comparison"].astype(str).tolist(),
        style,
    )
    ensemble_axis.tick_params(axis="x", labelsize=float(style["rq4_tick_label_size"]))
    ensemble_axis.set_title(
        "Same-cell ensemble comparisons",
        fontsize=float(style["rq4_title_size"]),
        pad=float(style["rq4_title_pad"]),
    )
    ensemble_axis.set_xlabel(
        "Δ Brier (ensemble − comparator)\nNegative values favor the ensemble",
        fontsize=float(style["rq4_axis_label_size"]),
    )
    add_panel_label(
        js_axis,
        "A",
        config,
        x=float(style["rq4_a_panel_label_x"]),
        y=float(style["rq4_top_panel_label_y"]),
        font_size=float(style["rq4_panel_label_size"]),
    )
    add_panel_label(
        correlation_axis,
        "B",
        config,
        x=float(style["rq4_top_panel_label_x"]),
        y=float(style["rq4_top_panel_label_y"]),
        font_size=float(style["rq4_panel_label_size"]),
    )
    add_panel_label(
        ensemble_axis,
        "C",
        config,
        x=float(style["rq4_bottom_panel_label_x"]),
        y=float(style["rq4_bottom_panel_label_y"]),
        font_size=float(style["rq4_panel_label_size"]),
    )
    figure.subplots_adjust(**style["rq4_margins"])
    save_figure(figure, config, manifest, "rq4_diversity_herding", "rq4", source_hashes)

    min_js_row = pairwise_summary.loc[pairwise_summary["mean_js_divergence"].idxmin()]
    max_js_row = pairwise_summary.loc[pairwise_summary["mean_js_divergence"].idxmax()]
    min_correlation_row = pairwise_summary.loc[
        pairwise_summary["fisher_z_combined_correlation"].idxmin()
    ]
    max_correlation_row = pairwise_summary.loc[
        pairwise_summary["fisher_z_combined_correlation"].idxmax()
    ]
    records = [
        headline_record(
            config,
            "rq4_headlines",
            "rq4",
            "open-book minus closed-book pairwise JS divergence",
            source_hashes,
            estimate=diversity_result.estimate,
            ci_low=diversity_result.ci_low,
            ci_high=diversity_result.ci_high,
            p_raw=diversity_result.p_raw,
            p_adjusted=None,
            median=diversity_result.median,
            n_matches=diversity_result.n_matches,
            n_predictions=len(js),
            units="natural-log JS divergence",
            aggregation="pairwise model divergence averaged within match and access condition",
        ),
        headline_record(
            config,
            "rq4_headlines",
            "rq4.mean_pairwise_js",
            "mean pairwise same-cell JS divergence",
            source_hashes,
            estimate=float(pairwise_summary["mean_js_divergence"].mean()),
            ci_low=None,
            ci_high=None,
            p_raw=None,
            p_adjusted=None,
            median=None,
            n_matches=panel["match_id"].nunique(),
            n_predictions=len(js),
            units="natural-log JS divergence",
            aggregation="mean over 21 model pairs and 404 match-condition cells",
            extra={
                "null_reason": "descriptive estimand",
                "minimum": float(min_js_row["mean_js_divergence"]),
                "minimum_pair": [min_js_row["model_a"], min_js_row["model_b"]],
                "maximum": float(max_js_row["mean_js_divergence"]),
                "maximum_pair": [max_js_row["model_a"], max_js_row["model_b"]],
            },
        ),
        headline_record(
            config,
            "rq4_headlines",
            "rq4.mean_pairwise_correlation",
            "mean Fisher-z combined pairwise probability correlation",
            source_hashes,
            estimate=float(pairwise_summary["fisher_z_combined_correlation"].mean()),
            ci_low=None,
            ci_high=None,
            p_raw=None,
            p_adjusted=None,
            median=None,
            n_matches=panel["match_id"].nunique(),
            n_predictions=len(correlations),
            units="Pearson correlation",
            aggregation="Fisher-z mean over H/D/A and four access-prompt cells",
            extra={
                "null_reason": "descriptive estimand",
                "minimum": float(min_correlation_row["fisher_z_combined_correlation"]),
                "minimum_pair": [
                    min_correlation_row["model_a"],
                    min_correlation_row["model_b"],
                ],
                "maximum": float(max_correlation_row["fisher_z_combined_correlation"]),
                "maximum_pair": [
                    max_correlation_row["model_a"],
                    max_correlation_row["model_b"],
                ],
            },
        ),
    ]
    write_headlines(config, manifest, "rq4_headlines", records, "rq4", source_hashes)
    return {
        "js": js,
        "correlations": correlations,
        "ensemble": ensemble_results,
        "diversity": pivot,
    }
