from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..reporting.figures import (
    add_numeric_grid,
    add_panel_label,
    apply_style,
    forest_plot,
    model_label,
    save_figure,
)
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table
from ..statistics.bootstrap import studentized_cluster_bootstrap
from ..statistics.multiplicity import holm_adjust
from .common import factorial_results, load_panel, primary_panel


def _reverse_effect(result: dict[str, object]) -> dict[str, object]:
    """Reverse an effect and its interval while preserving inferential metadata."""
    reversed_result = dict(result)
    reversed_result["estimate"] = -float(result["estimate"])
    reversed_result["ci_low"] = -float(result["ci_high"])
    reversed_result["ci_high"] = -float(result["ci_low"])
    reversed_result["median"] = -float(result["median"])
    return reversed_result


def _estimate_interval(estimate: float, ci_low: float, ci_high: float, decimals: int) -> str:
    return f"{estimate:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def _format_p(value: float, decimals: int) -> str:
    threshold = 10.0 ** (-decimals)
    if value < threshold:
        return f"<{threshold:.{decimals}f}"
    return f"{value:.{decimals}f}"


def _secondary_prompt_results(primary: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    specifications = config.section("metrics")["rq2_robustness"]
    metrics = [item["metric"] for item in specifications]
    declared_family = config.section("statistics")["holm_families"]["rq2_secondary_metrics"]
    if metrics != declared_family:
        raise ValueError("RQ2 robustness metrics must match their declared Holm family in order")

    cells = primary.groupby(
        ["match_id", "stage", "model_id", "access_condition", "prompt_strategy"],
        as_index=False,
    )[metrics].mean()
    results: list[dict[str, object]] = []
    prompt_strategies = config.section("design")["prompt_strategies"]
    for specification in specifications:
        metric = specification["metric"]
        pivot = cells.pivot_table(
            index=["match_id", "stage", "model_id", "access_condition"],
            columns="prompt_strategy",
            values=metric,
        ).dropna(subset=prompt_strategies)
        pivot["raw_difference"] = pivot["probabilistic_forecast"] - pivot["direct_score"]
        values = (
            pivot.reset_index()
            .groupby(["match_id", "stage"], as_index=False)["raw_difference"]
            .mean()
        )
        result = studentized_cluster_bootstrap(
            values,
            "raw_difference",
            "stage",
            config,
            f"rq2.secondary_prompt.{metric}",
        ).as_dict()
        if not bool(specification["higher_is_better"]):
            result = _reverse_effect(result)
        results.append(
            {
                "metric": metric,
                "label": specification["label"],
                "unit": specification["unit"],
                "display_scale": float(specification["display_scale"]),
                **result,
            }
        )

    adjusted = holm_adjust({row["metric"]: float(row["p_raw"]) for row in results})
    for row in results:
        row["p_adjusted"] = adjusted[row["metric"]]
    return pd.DataFrame(results)


def _derive_draw_structure(primary: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    frame = primary.copy()
    score_available = frame[["predicted_home_90", "predicted_away_90"]].notna().all(axis=1)
    frame["score_implied_draw"] = np.where(
        score_available,
        frame["predicted_home_90"].eq(frame["predicted_away_90"]),
        np.nan,
    )
    frame["exact_1_1_score"] = np.where(
        score_available,
        frame["predicted_home_90"].eq(1) & frame["predicted_away_90"].eq(1),
        np.nan,
    )
    frame["elicited_draw_probability"] = frame["draw_90_prob"]
    probabilities = frame[["home_win_90_prob", "draw_90_prob", "away_win_90_prob"]].to_numpy(
        dtype=float
    )
    maxima = probabilities.max(axis=1)
    tolerance = float(config.section("metrics")["top_outcome"]["tie_tolerance"])
    draw_is_modal = np.isclose(
        probabilities[:, 1],
        maxima,
        atol=tolerance,
        rtol=0.0,
    )
    frame["score_draw_probability_nondraw"] = np.where(
        score_available,
        frame["score_implied_draw"].fillna(False).astype(bool) & ~draw_is_modal,
        np.nan,
    )
    return frame


def _expected_goals_modal_diagnostic(
    structure: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Check score compatibility with an explicitly exploratory Poisson mapping."""
    settings = config.section("metrics")["rq2_expected_goals_diagnostic"]
    if settings["distribution"] != "independent_poisson":
        raise ValueError("RQ2 expected-goals diagnostic requires independent_poisson")
    tolerance = float(settings["integer_tolerance"])
    frame = structure.copy()

    def component_is_modal(score: pd.Series, rate: pd.Series) -> np.ndarray:
        valid = score.notna() & rate.notna() & rate.ge(0.0)
        rate_values = rate.fillna(0.0).to_numpy(dtype=float)
        score_values = score.fillna(0.0).to_numpy(dtype=float)
        lower_mode = np.floor(rate_values)
        integer_rate = np.isclose(
            rate_values,
            lower_mode,
            atol=tolerance,
            rtol=0.0,
        )
        modal = np.isclose(score_values, lower_mode, atol=tolerance, rtol=0.0) | (
            integer_rate
            & (lower_mode > 0.0)
            & np.isclose(score_values, lower_mode - 1.0, atol=tolerance, rtol=0.0)
        )
        return np.where(valid.to_numpy(), modal, False)

    frame["poisson_modal_score"] = component_is_modal(
        frame["predicted_home_90"], frame["expected_home_goals_90"]
    ) & component_is_modal(frame["predicted_away_90"], frame["expected_away_goals_90"])
    rows: list[dict[str, object]] = []
    for prompt_strategy in config.section("design")["prompt_strategies"]:
        prompt = frame[frame["prompt_strategy"] == prompt_strategy]
        divergent = prompt[prompt["score_draw_probability_nondraw"].eq(1.0)]
        rows.append(
            {
                "prompt_strategy": prompt_strategy,
                "distribution": settings["distribution"],
                "poisson_modal_score_rate": float(prompt["poisson_modal_score"].mean()),
                "n_predictions": int(len(prompt)),
                "divergent_poisson_modal_score_rate": (
                    float(divergent["poisson_modal_score"].mean()) if len(divergent) else np.nan
                ),
                "n_divergent_predictions": int(len(divergent)),
            }
        )
    return pd.DataFrame(rows)


def _draw_structure_results(structure: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    specifications = config.section("metrics")["rq2_draw_structure"]
    metrics = [item["metric"] for item in specifications]
    declared_family = config.section("statistics")["holm_families"]["rq2_draw_structure"]
    if metrics != declared_family:
        raise ValueError("RQ2 draw-structure metrics must match their Holm family in order")
    cells = structure.groupby(
        ["match_id", "stage", "model_id", "access_condition", "prompt_strategy"],
        as_index=False,
    )[metrics].mean()
    results: list[dict[str, object]] = []
    prompt_strategies = config.section("design")["prompt_strategies"]
    for specification in specifications:
        metric = specification["metric"]
        pivot = cells.pivot_table(
            index=["match_id", "stage", "model_id", "access_condition"],
            columns="prompt_strategy",
            values=metric,
        ).dropna(subset=prompt_strategies)
        pivot["difference"] = pivot["probabilistic_forecast"] - pivot["direct_score"]
        values = (
            pivot.reset_index().groupby(["match_id", "stage"], as_index=False)["difference"].mean()
        )
        result = studentized_cluster_bootstrap(
            values,
            "difference",
            "stage",
            config,
            f"rq2.draw_structure.{metric}",
        ).as_dict()
        results.append(
            {
                "metric": metric,
                "label": specification["label"],
                "plot_label": specification["plot_label"],
                "units": "proportion",
                "aggregation": "models and access cells within match",
                **result,
            }
        )
    adjusted = holm_adjust({row["metric"]: float(row["p_raw"]) for row in results})
    for row in results:
        row["p_adjusted"] = adjusted[row["metric"]]
    return pd.DataFrame(results)


def _draw_structure_by_model(structure: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    metrics = [item["metric"] for item in config.section("metrics")["rq2_draw_structure"]]
    cells = structure.groupby(["model_id", "prompt_strategy"], as_index=False)[metrics].mean()
    rows: list[dict[str, object]] = []
    for model_id in config.section("design")["complete_panel"]:
        model = cells[cells["model_id"] == model_id].set_index("prompt_strategy")
        row: dict[str, object] = {"model_id": model_id}
        for metric in metrics:
            score_first = float(model.loc["direct_score", metric])
            probabilities_first = float(model.loc["probabilistic_forecast", metric])
            row[f"{metric}_score_first"] = score_first
            row[f"{metric}_probabilities_first"] = probabilities_first
            row[f"{metric}_difference"] = probabilities_first - score_first
        rows.append(row)
    return pd.DataFrame(rows)


def _save_result_frames(
    frames: dict[str, pd.DataFrame],
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
) -> None:
    result_dir = config.resolve_path("results") / "rq2"
    result_dir.mkdir(parents=True, exist_ok=True)
    for artifact_id, frame in frames.items():
        path = result_dir / f"{artifact_id}.parquet"
        frame.to_parquet(path, index=False)
        manifest.add(
            artifact_id,
            path,
            "parquet",
            "rq2_elicitation",
            source_hashes,
            {"rows": len(frame)},
        )
    manifest.write()


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel, source_hashes = load_panel(config, manifest)
    factorial = factorial_results(panel, config)
    primary = primary_panel(panel, config)
    structure = _derive_draw_structure(primary, config)
    draw_structure = _draw_structure_results(structure, config)
    draw_structure_by_model = _draw_structure_by_model(structure, config)
    expected_goals_diagnostic = _expected_goals_modal_diagnostic(structure, config)
    reporting = config.section("reporting")
    style = reporting["style"]
    diagnostics = structure.groupby(["access_condition", "prompt_strategy"], as_index=False).agg(
        score_probability_consistent_recomputed=(
            "score_probability_consistent_recomputed",
            "mean",
        ),
        score_implied_draw=("score_implied_draw", "mean"),
        exact_1_1_score=("exact_1_1_score", "mean"),
        elicited_draw_probability=("elicited_draw_probability", "mean"),
        score_draw_probability_nondraw=("score_draw_probability_nondraw", "mean"),
        repair_attempted=("repair_attempted", "mean"),
        normalization_applied=("normalization_applied", "mean"),
        n_predictions=("prediction_id", "size"),
    )
    secondary = _secondary_prompt_results(primary, config)

    # Express both Brier effects as advantages: positive values favor probabilities-first.
    effects = pd.DataFrame(
        [
            {"contrast": "prompt", **_reverse_effect(factorial["prompt"])},
            {
                "contrast": "access_prompt_interaction",
                **_reverse_effect(factorial["access_prompt_interaction"]),
            },
        ]
    )
    effect_labels = {
        "prompt": "Overall probabilities-first advantage",
        "access_prompt_interaction": "Additional advantage under open book",
    }
    effect_decimals = int(style["rq2_effect_decimals"])
    effect_display = effects.assign(
        Contrast=effects["contrast"].map(effect_labels),
        **{
            "Brier advantage [95% CI]": effects.apply(
                lambda row: _estimate_interval(
                    float(row["estimate"]),
                    float(row["ci_low"]),
                    float(row["ci_high"]),
                    effect_decimals,
                ),
                axis=1,
            ),
            "p": effects["p_raw"].map(
                lambda value: _format_p(float(value), int(style["table_p_decimals"]))
            ),
            "Holm p": effects["p_adjusted"].map(
                lambda value: _format_p(float(value), int(style["table_p_decimals"]))
            ),
            "Median": effects["median"].round(effect_decimals),
            "n": effects["n_matches"],
        },
    )[["Contrast", "Brier advantage [95% CI]", "p", "Holm p", "Median", "n"]]
    save_table(
        effect_display,
        config,
        manifest,
        "rq2_elicitation_effects",
        "rq2_elicitation",
        source_hashes,
        headline_frame=effects,
    )

    secondary_decimals = int(style["rq2_secondary_decimals"])
    scaled_estimate = secondary["estimate"] * secondary["display_scale"]
    scaled_low = secondary["ci_low"] * secondary["display_scale"]
    scaled_high = secondary["ci_high"] * secondary["display_scale"]
    secondary_display = secondary.assign(
        **{
            "Metric": secondary["label"],
            "Advantage [95% CI]": [
                _estimate_interval(estimate, low, high, secondary_decimals)
                for estimate, low, high in zip(scaled_estimate, scaled_low, scaled_high)
            ],
            "p": secondary["p_raw"].map(
                lambda value: _format_p(float(value), int(style["table_p_decimals"]))
            ),
            "Holm p": secondary["p_adjusted"].map(
                lambda value: _format_p(float(value), int(style["table_p_decimals"]))
            ),
            "Median": (secondary["median"] * secondary["display_scale"]).round(secondary_decimals),
            "Units": secondary.apply(
                lambda row: (
                    "percentage points" if float(row["display_scale"]) != 1.0 else row["unit"]
                ),
                axis=1,
            ),
            "n": secondary["n_matches"],
        }
    )[["Metric", "Advantage [95% CI]", "p", "Holm p", "Median", "Units", "n"]]
    save_table(
        secondary_display,
        config,
        manifest,
        "rq2_secondary_metric_robustness",
        "rq2_elicitation",
        source_hashes,
        headline_frame=secondary,
    )

    structure_scale = float(style["rq2_structure_display_scale"])
    structure_decimals = int(style["rq2_secondary_decimals"])
    structure_display = draw_structure.assign(
        **{
            "Structural quantity": draw_structure["label"],
            "Prob.-first − score-first [95% CI]": draw_structure.apply(
                lambda row: _estimate_interval(
                    float(row["estimate"]) * structure_scale,
                    float(row["ci_low"]) * structure_scale,
                    float(row["ci_high"]) * structure_scale,
                    structure_decimals,
                ),
                axis=1,
            ),
            "p": draw_structure["p_raw"].map(
                lambda value: _format_p(float(value), int(style["table_p_decimals"]))
            ),
            "Holm p": draw_structure["p_adjusted"].map(
                lambda value: _format_p(float(value), int(style["table_p_decimals"]))
            ),
            "Median": (draw_structure["median"] * structure_scale).round(structure_decimals),
            "Units": "percentage points",
            "n": draw_structure["n_matches"],
        }
    )[
        [
            "Structural quantity",
            "Prob.-first − score-first [95% CI]",
            "p",
            "Holm p",
            "Median",
            "Units",
            "n",
        ]
    ]
    save_table(
        structure_display,
        config,
        manifest,
        "rq2_draw_structure_effects",
        "rq2_elicitation",
        source_hashes,
        headline_frame=draw_structure,
    )

    access_labels = reporting["labels"]["access"]
    prompt_labels = reporting["labels"]["prompts"]
    diagnostic_display = diagnostics.assign(
        Access=diagnostics["access_condition"].map(access_labels),
        Prompt=diagnostics["prompt_strategy"].map(prompt_labels),
        **{
            "Modal agreement, %": diagnostics["score_probability_consistent_recomputed"]
            .mul(100.0)
            .round(int(style["table_percentage_decimals"])),
            "Repair, %": diagnostics["repair_attempted"]
            .mul(100.0)
            .round(int(style["table_percentage_decimals"])),
            "Normalized, %": diagnostics["normalization_applied"]
            .mul(100.0)
            .round(int(style["table_percentage_decimals"])),
            "n": diagnostics["n_predictions"],
        },
    )[
        [
            "Access",
            "Prompt",
            "Modal agreement, %",
            "Repair, %",
            "Normalized, %",
            "n",
        ]
    ]
    save_table(
        diagnostic_display,
        config,
        manifest,
        "rq2_coherence_diagnostics",
        "rq2_elicitation",
        source_hashes,
    )

    percentage_decimals = int(style["table_percentage_decimals"])
    structure_cell_display = diagnostics.assign(
        Access=diagnostics["access_condition"].map(access_labels),
        Prompt=diagnostics["prompt_strategy"].map(prompt_labels),
        **{
            "Score draw, %": diagnostics["score_implied_draw"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Exact 1–1, %": diagnostics["exact_1_1_score"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Mean draw probability, %": diagnostics["elicited_draw_probability"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Draw-score/modal divergence, %": diagnostics["score_draw_probability_nondraw"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "n": diagnostics["n_predictions"],
        },
    )[
        [
            "Access",
            "Prompt",
            "Score draw, %",
            "Exact 1–1, %",
            "Mean draw probability, %",
            "Draw-score/modal divergence, %",
            "n",
        ]
    ]
    save_table(
        structure_cell_display,
        config,
        manifest,
        "rq2_draw_structure_cells",
        "rq2_elicitation",
        source_hashes,
    )

    model_structure_display = draw_structure_by_model.assign(
        Model=draw_structure_by_model["model_id"].map(lambda value: model_label(config, value)),
        **{
            "Score-first draws, %": draw_structure_by_model["score_implied_draw_score_first"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Prob.-first draws, %": draw_structure_by_model[
                "score_implied_draw_probabilities_first"
            ]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Δ draws, pp": draw_structure_by_model["score_implied_draw_difference"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Δ exact 1–1, pp": draw_structure_by_model["exact_1_1_score_difference"]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Δ draw probability, pp": draw_structure_by_model[
                "elicited_draw_probability_difference"
            ]
            .mul(structure_scale)
            .round(percentage_decimals),
            "Δ draw-score/modal divergence, pp": draw_structure_by_model[
                "score_draw_probability_nondraw_difference"
            ]
            .mul(structure_scale)
            .round(percentage_decimals),
        },
    )[
        [
            "Model",
            "Score-first draws, %",
            "Prob.-first draws, %",
            "Δ draws, pp",
            "Δ exact 1–1, pp",
            "Δ draw probability, pp",
            "Δ draw-score/modal divergence, pp",
        ]
    ]
    save_table(
        model_structure_display,
        config,
        manifest,
        "rq2_draw_structure_by_model",
        "rq2_elicitation",
        source_hashes,
    )

    expected_goals_display = expected_goals_diagnostic.assign(
        Prompt=expected_goals_diagnostic["prompt_strategy"].map(prompt_labels),
        **{
            "Selected score is Poisson-modal, %": expected_goals_diagnostic[
                "poisson_modal_score_rate"
            ]
            .mul(structure_scale)
            .round(percentage_decimals),
            "n forecasts": expected_goals_diagnostic["n_predictions"],
            "Among modal-divergence cases, %": expected_goals_diagnostic[
                "divergent_poisson_modal_score_rate"
            ]
            .mul(structure_scale)
            .round(percentage_decimals),
            "n divergence cases": expected_goals_diagnostic["n_divergent_predictions"],
        },
    )[
        [
            "Prompt",
            "Selected score is Poisson-modal, %",
            "n forecasts",
            "Among modal-divergence cases, %",
            "n divergence cases",
        ]
    ]
    save_table(
        expected_goals_display,
        config,
        manifest,
        "rq2_expected_goals_modal_diagnostic",
        "rq2_elicitation",
        source_hashes,
    )

    _save_result_frames(
        {
            "rq2_effects": effects,
            "rq2_diagnostics": diagnostics,
            "rq2_secondary_prompt_results": secondary,
            "rq2_draw_structure_results": draw_structure,
            "rq2_draw_structure_by_model": draw_structure_by_model,
            "rq2_expected_goals_modal_diagnostic": expected_goals_diagnostic,
        },
        config,
        manifest,
        source_hashes,
    )

    apply_style(config)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(float(reporting["figure_width_double"]), float(style["rq2_height"])),
        gridspec_kw={
            "width_ratios": style["rq2_column_width_ratios"],
            "wspace": float(style["rq2_wspace"]),
        },
    )
    forest_plot(
        axes[0],
        effects,
        [
            {
                "prompt": "Overall prompt effect",
                "access_prompt_interaction": "Open-book interaction",
            }[value]
            for value in effects["contrast"]
        ],
        config,
        colors=[reporting["palette"]["primary"], reporting["palette"]["secondary"]],
        markers=["o", "D"],
    )
    axes[0].set_xlabel(
        "Brier advantage\n(positive favors probabilities first)",
        fontsize=float(style["rq2_plot_axis_label_size"]),
    )

    structure_plot = draw_structure.copy()
    for column in ("estimate", "ci_low", "ci_high", "median"):
        structure_plot[column] = structure_plot[column] * structure_scale
    structure_plot_labels = {
        "score_implied_draw": "Score-implied draw",
        "exact_1_1_score": "Exact 1-1",
        "elicited_draw_probability": "Draw probability",
        "score_draw_probability_nondraw": "Modal divergence",
    }
    forest_plot(
        axes[1],
        structure_plot,
        structure_plot["metric"].map(structure_plot_labels).tolist(),
        config,
        colors=[reporting["colors"]["prompt"]["probabilistic_forecast"]] * len(structure_plot),
        markers=style["rq2_structure_markers"],
    )
    axes[1].set_xlabel(
        "Probabilities first - score first\n(percentage points)",
        fontsize=float(style["rq2_plot_axis_label_size"]),
    )
    for axis in axes:
        axis.tick_params(
            axis="both",
            labelsize=float(style["rq2_plot_tick_label_size"]),
        )
        axis.xaxis.set_major_locator(MaxNLocator(nbins=int(style["rq2_plot_max_x_ticks"])))
    for label, axis in zip(("A", "B"), axes):
        add_panel_label(
            axis,
            label,
            config,
            x=float(style["rq2_plot_panel_label_x"]),
            y=float(style["rq2_plot_panel_label_y"]),
            font_size=float(style["rq2_plot_panel_label_size"]),
        )
    figure.subplots_adjust(
        left=float(style["rq2_plot_left_margin"]),
        right=float(style["rq2_plot_right_margin"]),
        bottom=float(style["rq2_plot_bottom_margin"]),
        top=float(style["rq2_plot_top_margin"]),
        wspace=float(style["rq2_wspace"]),
    )
    save_figure(figure, config, manifest, "fig_rq2_elicitation", "rq2_elicitation", source_hashes)

    coherence_figure, coherence_axis = plt.subplots(
        figsize=(
            float(reporting["figure_width_single"]),
            float(style["rq2_coherence_height"]),
        )
    )
    coherence = diagnostics.pivot(
        index="access_condition",
        columns="prompt_strategy",
        values="score_probability_consistent_recomputed",
    ).loc[config.section("design")["access_conditions"]]
    y_positions = np.arange(len(coherence), dtype=float)
    direct_values = coherence["direct_score"].to_numpy(dtype=float)
    probability_values = coherence["probabilistic_forecast"].to_numpy(dtype=float)
    coherence_axis.hlines(
        y_positions,
        np.minimum(direct_values, probability_values),
        np.maximum(direct_values, probability_values),
        color=reporting["palette"]["neutral_light"],
        linewidth=float(style["line_width"]),
    )
    prompt_colors = reporting["colors"]["prompt"]
    marker_size = float(style["rq2_coherence_marker_size"])
    coherence_axis.plot(
        direct_values,
        y_positions,
        linestyle="none",
        marker="s",
        markersize=marker_size,
        color=prompt_colors["direct_score"],
        label=prompt_labels["direct_score"],
        zorder=3,
    )
    coherence_axis.plot(
        probability_values,
        y_positions,
        linestyle="none",
        marker="o",
        markersize=marker_size,
        color=prompt_colors["probabilistic_forecast"],
        label=prompt_labels["probabilistic_forecast"],
        zorder=3,
    )
    value_label_offset = float(style["rq2_value_label_offset_points"])
    for row_index, (y_position, direct_value, probability_value) in enumerate(
        zip(y_positions, direct_values, probability_values)
    ):
        vertical_offset = (
            -value_label_offset if row_index < len(y_positions) / 2.0 else value_label_offset
        )
        vertical_alignment = "top" if vertical_offset < 0.0 else "bottom"
        horizontal_offset = float(style["rq2_coherence_label_horizontal_offset"])
        direct_is_left = direct_value <= probability_value
        direct_horizontal_offset = -horizontal_offset if direct_is_left else horizontal_offset
        probability_horizontal_offset = horizontal_offset if direct_is_left else -horizontal_offset
        coherence_axis.annotate(
            f"{direct_value:.1%}",
            xy=(direct_value, y_position),
            xytext=(direct_horizontal_offset, vertical_offset),
            textcoords="offset points",
            ha="right" if direct_is_left else "left",
            va=vertical_alignment,
            fontsize=float(style["rq2_coherence_annotation_size"]),
        )
        coherence_axis.annotate(
            f"{probability_value:.1%}",
            xy=(probability_value, y_position),
            xytext=(probability_horizontal_offset, vertical_offset),
            textcoords="offset points",
            ha="left" if direct_is_left else "right",
            va=vertical_alignment,
            fontsize=float(style["rq2_coherence_annotation_size"]),
        )
    coherence_axis.set_yticks(
        y_positions,
        [access_labels[value] for value in coherence.index],
    )
    coherence_axis.invert_yaxis()
    coherence_axis.set_xlim(
        float(style["rq2_coherence_axis_min"]),
        float(style["rq2_coherence_axis_max"]),
    )
    coherence_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    coherence_axis.set_xticks(style["rq2_coherence_xticks"])
    coherence_axis.tick_params(
        axis="both",
        labelsize=float(style["rq2_coherence_tick_label_size"]),
    )
    coherence_axis.set_xlabel(
        "Share with score tendency matching\na modal H/D/A outcome",
        fontsize=float(style["rq2_coherence_axis_label_size"]),
    )
    coherence_axis.legend(
        loc="center",
        bbox_to_anchor=(0.5, float(style["rq2_midnote_y"])),
        bbox_transform=coherence_figure.transFigure,
        ncol=int(style["rq2_coherence_legend_columns"]),
        frameon=False,
        fontsize=float(style["rq2_coherence_legend_size"]),
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    add_numeric_grid(coherence_axis, config, "x")
    coherence_figure.subplots_adjust(
        left=float(style["rq2_coherence_left_margin"]),
        right=float(style["rq2_coherence_right_margin"]),
        bottom=float(style["rq2_coherence_bottom_margin"]),
        top=float(style["rq2_coherence_top_margin"]),
    )
    save_figure(
        coherence_figure,
        config,
        manifest,
        "fig_rq2_coherence",
        "rq2_elicitation",
        source_hashes,
    )

    records = [
        headline_record(
            config,
            "rq2_headlines",
            f"rq2.{row['contrast']}",
            "Mean paired T−24h Brier advantage",
            source_hashes,
            **row,
            n_predictions=int(len(primary)),
            units="Brier score",
            aggregation="models and complementary design cells within match",
        )
        for row in effects.to_dict(orient="records")
    ]
    write_headlines(config, manifest, "rq2_headlines", records, "rq2_elicitation", source_hashes)
    return {
        "effects": effects,
        "diagnostics": diagnostics,
        "secondary": secondary,
        "draw_structure": draw_structure,
        "draw_structure_by_model": draw_structure_by_model,
    }
