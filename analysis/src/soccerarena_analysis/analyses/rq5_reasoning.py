from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import PercentFormatter

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..reporting.figures import (
    add_numeric_grid,
    apply_style,
    condition_color,
    condition_label,
    model_label,
    save_figure,
)
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table
from ..stages.annotations import annotation_corpus

MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "anthropic/claude-fable-5": "anthropic.png",
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
        labelsize=float(style["rq5_length_tick_label_size"]),
        pad=float(style["rq5_length_label_pad"]),
        length=0,
    )
    for y_position, model in enumerate(model_order):
        icon = OffsetImage(
            plt.imread(icon_paths[model]),
            zoom=float(style["rq5_length_icon_zoom"]),
            interpolation="lanczos",
        )
        axis.add_artist(
            AnnotationBbox(
                icon,
                (float(style["rq5_length_icon_x"]), y_position),
                xycoords=("axes fraction", "data"),
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _scope_label(value: str) -> str:
    labels = {
        "full_corpus": "All benchmark forecasts",
        "complete_panel": "Complete model panel",
        "full_nonempty_corpus": "All non-empty rationales",
        "primary_rq5_cell": "T−24h, probabilities-first, complete panel",
    }
    return labels[value]


def _median_iqr(row: pd.Series, decimals: int) -> str:
    return (
        f"{float(row['median_words']):.{decimals}f} "
        f"[{float(row['lower_words']):.{decimals}f}–"
        f"{float(row['upper_words']):.{decimals}f}]"
    )


def _add_scope(frame: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    complete_models = set(config.section("design")["complete_panel"])
    scoped = frame.copy()
    scoped["model_panel"] = np.where(
        scoped["model_id"].isin(complete_models), "complete_panel", "partial_model"
    )
    return scoped


def _objective_frames(
    tools: pd.DataFrame, rationale: pd.DataFrame, config: AnalysisConfig
) -> dict[str, pd.DataFrame]:
    scoped_tools = _add_scope(tools, config)
    open_mask = scoped_tools["access_condition"].eq("open_book")
    scoped_tools["observed_search"] = np.where(
        open_mask,
        scoped_tools["open_book_compliance"].eq("observed_search").astype(float),
        np.nan,
    )
    scoped_tools["explicit_no_search"] = np.where(
        open_mask,
        scoped_tools["open_book_compliance"].eq("no_observed_search").astype(float),
        np.nan,
    )
    scoped_tools["unknown_search_status"] = np.where(
        open_mask,
        scoped_tools["open_book_compliance"].eq("unknown").astype(float),
        np.nan,
    )
    operational_fields = list(config.section("annotation")["operational_fields"])
    scoped_tools["operational_metadata_complete"] = (
        scoped_tools[operational_fields].notna().all(axis=1)
    )
    operations = scoped_tools.groupby(
        ["model_panel", "model_id", "access_condition"], as_index=False
    ).agg(
        forecasts=("prediction_id", "size"),
        matches=("match_id", "nunique"),
        tools_enabled_rate=("tools_enabled", "mean"),
        observed_search_rate=("observed_search", "mean"),
        explicit_no_search_rate=("explicit_no_search", "mean"),
        unknown_search_status_rate=("unknown_search_status", "mean"),
        trace_available_rate=("tool_trace_available", "mean"),
        mean_tool_calls=("num_tool_calls", "mean"),
        operational_metadata_complete_rate=("operational_metadata_complete", "mean"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
        mean_cost_usd=("cost_usd", "mean"),
    )

    summary_rows: list[dict[str, object]] = []
    complete_models = set(config.section("design")["complete_panel"])
    for scope, subset in (
        ("full_corpus", scoped_tools),
        ("complete_panel", scoped_tools[scoped_tools["model_id"].isin(complete_models)]),
    ):
        for access, group in subset.groupby("access_condition", sort=False):
            summary_rows.append(
                {
                    "scope": scope,
                    "access_condition": access,
                    "forecasts": len(group),
                    "matches": group["match_id"].nunique(),
                    "models": group["model_id"].nunique(),
                    "tools_enabled_rate": float(group["tools_enabled"].mean()),
                    "observed_search_rate": (
                        float(group["observed_search"].mean()) if access == "open_book" else np.nan
                    ),
                    "explicit_no_search_rate": (
                        float(group["explicit_no_search"].mean())
                        if access == "open_book"
                        else np.nan
                    ),
                    "unknown_search_status_rate": (
                        float(group["unknown_search_status"].mean())
                        if access == "open_book"
                        else np.nan
                    ),
                    "trace_available_rate": float(group["tool_trace_available"].mean()),
                    "operational_metadata_complete_rate": float(
                        group["operational_metadata_complete"].mean()
                    ),
                }
            )
    operations_summary = pd.DataFrame(summary_rows)

    rationale_ids = set(rationale["prediction_id"])
    availability_source = scoped_tools.assign(
        rationale_available=scoped_tools["prediction_id"].isin(rationale_ids)
    )
    availability = availability_source.groupby(
        ["model_panel", "model_id", "access_condition"], as_index=False
    ).agg(
        forecasts=("prediction_id", "size"),
        nonempty_rationales=("rationale_available", "sum"),
        rationale_available_rate=("rationale_available", "mean"),
    )

    quantiles = list(config.section("annotation")["rationale_length_quantiles"])
    lower_quantile, upper_quantile = (float(value) for value in quantiles)
    lengths = _add_scope(rationale, config).assign(
        words=rationale["rationale_text"].str.split().str.len(),
        characters=rationale["rationale_text"].str.len(),
    )
    length = lengths.groupby(["model_panel", "model_id", "access_condition"], as_index=False).agg(
        rationales=("prediction_id", "size"),
        matches=("match_id", "nunique"),
        mean_words=("words", "mean"),
        median_words=("words", "median"),
        lower_words=("words", lambda values: values.quantile(lower_quantile)),
        upper_words=("words", lambda values: values.quantile(upper_quantile)),
        mean_characters=("characters", "mean"),
    )
    return {
        "operations": operations,
        "operations_summary": operations_summary,
        "availability": availability,
        "rationale_lengths": lengths,
        "length": length,
    }


def _write_result_parquet(
    frame: pd.DataFrame,
    path: Path,
    manifest: Manifest,
    artifact_id: str,
    source_hashes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    manifest.add(artifact_id, path, "parquet", "rq5", source_hashes, {"rows": len(frame)})


def _plot_rationale_length(
    length: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
) -> None:
    reporting = config.section("reporting")
    style = reporting["style"]
    apply_style(config)
    figure, axis = plt.subplots(
        figsize=(float(reporting["figure_width_double"]), float(style["rq5_length_height"]))
    )
    model_order = [
        *config.section("design")["complete_panel"],
        config.section("design")["partial_models"]["fable"],
    ]
    access_order = list(config.section("design")["access_conditions"])
    offset = float(style["rq5_length_y_offset"])
    y_positions = np.arange(len(model_order), dtype=float)
    for y_position, model_id in zip(y_positions, model_order):
        model_rows = length[length["model_id"].eq(model_id)].set_index("access_condition")
        if set(access_order).issubset(model_rows.index):
            axis.plot(
                [model_rows.loc[value, "median_words"] for value in access_order],
                [y_position - offset, y_position + offset],
                color=reporting["palette"]["neutral_light"],
                linewidth=float(style["zero_line_width"]),
                zorder=1,
            )
        for access_index, (access, marker) in enumerate(
            zip(access_order, config.section("annotation")["access_markers"])
        ):
            if access not in model_rows.index:
                continue
            row = model_rows.loc[access]
            y_value = y_position + (-offset if access_index == 0 else offset)
            axis.errorbar(
                row["median_words"],
                y_value,
                xerr=[
                    [row["median_words"] - row["lower_words"]],
                    [row["upper_words"] - row["median_words"]],
                ],
                fmt=marker,
                color=condition_color(config, "access", access),
                ecolor=condition_color(config, "access", access),
                capsize=float(style["ci_cap_size"]),
                markersize=float(style["rq5_marker_size"]),
                label=condition_label(config, "access", access) if y_position == 0 else None,
                zorder=3,
            )
    axis.set_yticks(y_positions, [model_label(config, model) for model in model_order])
    _add_model_provider_icons(axis, model_order, style)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.tick_params(
        axis="x",
        labelsize=float(style["rq5_length_tick_label_size"]),
    )
    axis.set_xlabel(
        "Rationale length (words)",
        fontsize=float(style["rq5_length_axis_label_size"]),
    )
    axis.set_title("")
    axis.axhline(
        len(model_order) - 1.5,
        color=reporting["palette"]["neutral_light"],
        linewidth=float(style["grid_line_width"]),
        zorder=0,
    )
    axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, float(style["rq5_legend_anchor_y"])),
        ncol=int(style["rq5_legend_columns"]),
        fontsize=float(style["rq5_length_legend_font_size"]),
    )
    add_numeric_grid(axis, config, "x")
    figure.subplots_adjust(**style["rq5_length_margins"])
    save_figure(
        figure,
        config,
        manifest,
        "fig_rq5_rationale_length_appendix",
        "rq5",
        source_hashes,
    )


def run_objective(config: AnalysisConfig, manifest: Manifest) -> dict[str, pd.DataFrame]:
    tool_record = manifest.require("derived_tool_events")
    rationale_record = manifest.require("derived_rationale_corpus")
    tools = pd.read_parquet(tool_record.path)
    rationale = pd.read_parquet(rationale_record.path)
    source_hashes = {
        "tool_events": tool_record.sha256,
        "rationale_corpus": rationale_record.sha256,
    }
    manifest.discard_generated(
        [
            "rq5_tool_metadata_full_corpus_tex",
            "rq5_rationale_availability_tex",
        ]
    )
    frames = _objective_frames(tools, rationale, config)
    result_dir = config.resolve_path("results") / "rq5"
    for name in ("operations", "operations_summary", "availability", "length"):
        _write_result_parquet(
            frames[name],
            result_dir / f"{name}.parquet",
            manifest,
            f"rq5_{name}",
            source_hashes,
        )

    reporting = config.section("reporting")
    access_labels = reporting["labels"]["access"]
    operations_display = frames["operations"].assign(
        Panel=frames["operations"]["model_panel"].map(
            {"complete_panel": "Complete", "partial_model": "Partial"}
        ),
        Model=frames["operations"]["model_id"].map(lambda value: model_label(config, value)),
        Access=frames["operations"]["access_condition"].map(access_labels),
        **{
            "Forecasts": frames["operations"]["forecasts"],
            "Tools enabled, %": frames["operations"]["tools_enabled_rate"] * 100.0,
            "Observed search, %": frames["operations"]["observed_search_rate"] * 100.0,
            "Explicit no search, %": frames["operations"]["explicit_no_search_rate"] * 100.0,
            "Search status unknown, %": frames["operations"]["unknown_search_status_rate"] * 100.0,
            "Trace available, %": frames["operations"]["trace_available_rate"] * 100.0,
            "Mean tool calls": frames["operations"]["mean_tool_calls"],
            "Operational fields complete, %": frames["operations"][
                "operational_metadata_complete_rate"
            ]
            * 100.0,
            "Mean input tokens": frames["operations"]["mean_input_tokens"],
            "Mean output tokens": frames["operations"]["mean_output_tokens"],
            "Mean latency (s)": frames["operations"]["mean_latency_ms"] / 1000.0,
            "Mean cost (USD)": frames["operations"]["mean_cost_usd"],
        },
    )[
        [
            "Panel",
            "Model",
            "Access",
            "Forecasts",
            "Tools enabled, %",
            "Observed search, %",
            "Explicit no search, %",
            "Search status unknown, %",
            "Trace available, %",
            "Mean tool calls",
            "Operational fields complete, %",
            "Mean input tokens",
            "Mean output tokens",
            "Mean latency (s)",
            "Mean cost (USD)",
        ]
    ]
    save_table(
        operations_display,
        config,
        manifest,
        "rq5_tool_metadata_full_corpus",
        "rq5",
        source_hashes,
        latex=False,
    )

    percent_decimals = int(reporting["style"]["table_percentage_decimals"])
    model_order = [
        *config.section("design")["complete_panel"],
        config.section("design")["partial_models"]["fable"],
    ]
    open_operations = (
        frames["operations"]
        .query("access_condition == 'open_book'")
        .merge(
            frames["availability"].query("access_condition == 'open_book'")[
                ["model_panel", "model_id", "rationale_available_rate"]
            ],
            on=["model_panel", "model_id"],
            how="left",
            validate="one_to_one",
        )
    )
    open_operations["_order"] = open_operations["model_id"].map(
        {model_id: index for index, model_id in enumerate(model_order)}
    )
    open_operations = open_operations.sort_values("_order")
    operations_paper = pd.DataFrame(
        {
            "Panel": open_operations["model_panel"].map(
                {"complete_panel": "Complete", "partial_model": "Partial"}
            ),
            "Model": open_operations["model_id"].map(lambda value: model_label(config, value)),
            "Forecasts": open_operations["forecasts"].astype(int),
            "Search trace (%)": open_operations["observed_search_rate"] * 100.0,
            "No search trace (%)": open_operations["explicit_no_search_rate"] * 100.0,
            "Unknown (%)": open_operations["unknown_search_status_rate"] * 100.0,
            "Rationale available (%)": open_operations["rationale_available_rate"] * 100.0,
        }
    )
    save_table(
        operations_paper,
        config,
        manifest,
        "rq5_tool_metadata_appendix",
        "rq5",
        source_hashes,
        latex_formats={
            column: f"{{:.{percent_decimals}f}}"
            for column in (
                "Search trace (%)",
                "No search trace (%)",
                "Unknown (%)",
                "Rationale available (%)",
            )
        },
    )

    summary_display = frames["operations_summary"].assign(
        Scope=frames["operations_summary"]["scope"].map(_scope_label),
        Access=frames["operations_summary"]["access_condition"].map(access_labels),
        **{
            "Forecasts": frames["operations_summary"]["forecasts"],
            "Models": frames["operations_summary"]["models"],
            "Search trace (%)": frames["operations_summary"]["observed_search_rate"] * 100.0,
            "No search trace (%)": frames["operations_summary"]["explicit_no_search_rate"] * 100.0,
            "Unknown (%)": frames["operations_summary"]["unknown_search_status_rate"] * 100.0,
        },
    )[
        [
            "Scope",
            "Access",
            "Forecasts",
            "Models",
            "Search trace (%)",
            "No search trace (%)",
            "Unknown (%)",
        ]
    ]
    save_table(
        summary_display,
        config,
        manifest,
        "rq5_tool_status_summary",
        "rq5",
        source_hashes,
        latex_formats={
            column: f"{{:.{percent_decimals}f}}"
            for column in ("Search trace (%)", "No search trace (%)", "Unknown (%)")
        },
    )

    availability_display = frames["availability"].assign(
        Panel=frames["availability"]["model_panel"].map(
            {"complete_panel": "Complete", "partial_model": "Partial"}
        ),
        Model=frames["availability"]["model_id"].map(lambda value: model_label(config, value)),
        Access=frames["availability"]["access_condition"].map(access_labels),
        **{
            "Forecasts": frames["availability"]["forecasts"],
            "Non-empty rationales": frames["availability"]["nonempty_rationales"],
            "Rationale available, %": frames["availability"]["rationale_available_rate"] * 100.0,
        },
    )[
        [
            "Panel",
            "Model",
            "Access",
            "Forecasts",
            "Non-empty rationales",
            "Rationale available, %",
        ]
    ]
    save_table(
        availability_display,
        config,
        manifest,
        "rq5_rationale_availability",
        "rq5",
        source_hashes,
        latex=False,
    )

    length_display = frames["length"].assign(
        Panel=frames["length"]["model_panel"].map(
            {"complete_panel": "Complete", "partial_model": "Partial"}
        ),
        Model=frames["length"]["model_id"].map(lambda value: model_label(config, value)),
        Access=frames["length"]["access_condition"].map(access_labels),
        **{
            "Rationales": frames["length"]["rationales"],
            "Mean words": frames["length"]["mean_words"],
            "Median words": frames["length"]["median_words"],
            "Lower word quartile": frames["length"]["lower_words"],
            "Upper word quartile": frames["length"]["upper_words"],
            "Mean characters": frames["length"]["mean_characters"],
        },
    )[
        [
            "Panel",
            "Model",
            "Access",
            "Rationales",
            "Mean words",
            "Median words",
            "Lower word quartile",
            "Upper word quartile",
            "Mean characters",
        ]
    ]
    save_table(
        length_display,
        config,
        manifest,
        "rq5_rationale_length_full",
        "rq5",
        source_hashes,
        latex=False,
    )
    length_decimals = int(reporting["style"]["rq5_length_table_decimals"])
    length_rows: list[dict[str, object]] = []
    for model_id in model_order:
        model_rows = frames["length"][frames["length"]["model_id"].eq(model_id)].set_index(
            "access_condition"
        )
        if not set(config.section("design")["access_conditions"]).issubset(model_rows.index):
            continue
        closed = model_rows.loc["closed_book"]
        open_book = model_rows.loc["open_book"]
        length_rows.append(
            {
                "Model": model_label(config, model_id),
                "Closed: median [IQR]": _median_iqr(closed, length_decimals),
                "Open: median [IQR]": _median_iqr(open_book, length_decimals),
                "n (closed/open)": f"{int(closed['rationales'])}/{int(open_book['rationales'])}",
            }
        )
    save_table(
        pd.DataFrame(length_rows),
        config,
        manifest,
        "rq5_rationale_length",
        "rq5",
        source_hashes,
    )
    _plot_rationale_length(frames["length"], config, manifest, source_hashes)

    full_open = (
        frames["operations_summary"]
        .query("scope == 'full_corpus' and access_condition == 'open_book'")
        .iloc[0]
    )
    complete_open = (
        frames["operations_summary"]
        .query("scope == 'complete_panel' and access_condition == 'open_book'")
        .iloc[0]
    )
    records = []
    for analysis_id, row in (
        ("all_models_open_book_observed_search", full_open),
        ("complete_panel_open_book_observed_search", complete_open),
    ):
        records.append(
            headline_record(
                config,
                "rq5_objective_headlines",
                analysis_id,
                "share of open-book forecasts with an observed search trace",
                source_hashes,
                estimate=float(row["observed_search_rate"]),
                ci_low=None,
                ci_high=None,
                p_raw=None,
                p_adjusted=None,
                median=None,
                n_matches=int(row["matches"]),
                n_predictions=int(row["forecasts"]),
                units="share",
                aggregation="objective full-corpus tool metadata",
                extra={"null_reason": "descriptive operational metadata"},
            )
        )
    write_headlines(
        config,
        manifest,
        "rq5_objective_headlines",
        records,
        "rq5",
        source_hashes,
    )
    manifest.write()
    return frames


def _resolved_prevalence(
    rationale: pd.DataFrame, resolved: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    categories = list(config.section("annotation")["categories"])
    resolved_columns = [f"resolved__{category}" for category in categories]
    scoped = annotation_corpus(rationale, config)
    labeled = scoped.merge(
        resolved[["prediction_id", *resolved_columns]],
        on="prediction_id",
        how="inner",
        validate="one_to_one",
    )
    if len(labeled) != len(scoped):
        raise ValueError("Resolved rationale labels do not cover the complete RQ5.2 cell")
    rows: list[dict[str, object]] = []
    for access, group in labeled.groupby("access_condition", sort=False):
        for category in categories:
            rows.append(
                {
                    "scope": "primary_rq5_cell",
                    "access_condition": access,
                    "category": category,
                    "prevalence": float(group[f"resolved__{category}"].mean()),
                    "n_rationales": len(group),
                    "n_matches": group["match_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def _model_prevalence(
    rationale: pd.DataFrame, resolved: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    categories = list(config.section("annotation")["categories"])
    resolved_columns = [f"resolved__{category}" for category in categories]
    labeled = annotation_corpus(rationale, config).merge(
        resolved[["prediction_id", *resolved_columns]],
        on="prediction_id",
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for (model_id, access), group in labeled.groupby(["model_id", "access_condition"], sort=False):
        for category in categories:
            rows.append(
                {
                    "model_id": model_id,
                    "access_condition": access,
                    "category": category,
                    "prevalence": float(group[f"resolved__{category}"].mean()),
                    "n_rationales": len(group),
                    "n_matches": group["match_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def _prevalence_wide(prevalence: pd.DataFrame, scope: str) -> pd.DataFrame:
    selected = prevalence[prevalence["scope"].eq(scope)]
    rates = selected.pivot(index="category", columns="access_condition", values="prevalence")
    counts = selected.pivot(index="category", columns="access_condition", values="n_rationales")
    matches = selected.pivot(index="category", columns="access_condition", values="n_matches")
    wide = rates.reset_index()
    wide["closed_n"] = counts["closed_book"].to_numpy()
    wide["open_n"] = counts["open_book"].to_numpy()
    wide["n_matches"] = int(matches.max().max())
    wide["difference_percentage_points"] = (wide["open_book"] - wide["closed_book"]) * 100.0
    wide["scope"] = scope
    return wide


def _truncate(text: str, maximum: int) -> str:
    if len(text) <= maximum:
        return text
    shortened = text[:maximum].rsplit(" ", 1)[0].rstrip()
    return f"{shortened}…"


def _illustrative_excerpts(
    corpus: pd.DataFrame,
    resolved: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    keys = ["match_id", "model_id", "forecast_horizon", "prompt_strategy"]
    maximum = int(config.section("annotation")["illustrative_quote_max_characters"])
    count = int(config.section("annotation")["illustrative_quote_count"])
    label_columns = [column for column in resolved if column.startswith("resolved__")]
    labeled = annotation_corpus(corpus, config).merge(
        resolved[["prediction_id", *label_columns]],
        on="prediction_id",
        how="inner",
        validate="one_to_one",
    )
    closed = labeled[labeled["access_condition"].eq("closed_book")].copy()
    open_book = labeled[labeled["access_condition"].eq("open_book")].copy()
    paired = closed.merge(
        open_book, on=keys, suffixes=("__closed", "__open"), validate="one_to_one"
    )
    if paired.empty:
        return pd.DataFrame(columns=["contrast", "excerpt", "selection_basis"])
    closed_labels = [f"{column}__closed" for column in label_columns]
    open_labels = [f"{column}__open" for column in label_columns]
    paired["evidence_category_contrast"] = (
        paired[closed_labels].to_numpy(dtype=bool) != paired[open_labels].to_numpy(dtype=bool)
    ).sum(axis=1)
    paired["length_contrast"] = (
        paired["rationale_text__closed"].str.len() - paired["rationale_text__open"].str.len()
    ).abs()
    paired = paired.sort_values(
        ["evidence_category_contrast", "length_contrast", *keys],
        ascending=[False, False, True, True, True, True],
    )
    chosen = paired.iloc[0]
    basis = (
        "Same model/match/horizon/prompt pair with the largest resolved evidence-category "
        "contrast; selected without outcomes or forecast scores"
    )
    records = [
        {
            "contrast": "Closed-book rationale (illustrative)",
            "excerpt": _truncate(str(chosen["rationale_text__closed"]), maximum),
            "selection_basis": basis,
        },
        {
            "contrast": "Open-book rationale (illustrative)",
            "excerpt": _truncate(str(chosen["rationale_text__open"]), maximum),
            "selection_basis": basis,
        },
    ]
    if count > len(records) and len(paired) > 1:
        second = paired.iloc[1]
        records.append(
            {
                "contrast": "Open-book rationale (second illustrative contrast)",
                "excerpt": _truncate(str(second["rationale_text__open"]), maximum),
                "selection_basis": (
                    "Second-ranked same-cell evidence-category contrast; selected without "
                    "outcomes or forecast scores"
                ),
            }
        )
    return pd.DataFrame(records[:count])


def _plot_evidence_prevalence(
    primary: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
    *,
    categories: list[str],
    artifact_id: str,
    title: str,
    width: float,
    height: float,
    width_ratios: list[float],
    margins: dict[str, float],
    label_map: dict[str, str],
    subtitle_prefix: str | None,
    include_n_in_subtitle: bool,
    tick_label_size: float | None = None,
    axis_label_size: float | None = None,
    x_ticks: list[float] | None = None,
) -> None:
    reporting = config.section("reporting")
    style = reporting["style"]
    annotation = config.section("annotation")
    tick_size = (
        float(style["tick_label_size"]) if tick_label_size is None else float(tick_label_size)
    )
    label_size = (
        float(style["axes_label_size"]) if axis_label_size is None else float(axis_label_size)
    )
    plot = primary.set_index("category").reindex(categories).reset_index()
    apply_style(config)
    figure, (axis, delta_axis) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(width, height),
        gridspec_kw={
            "width_ratios": width_ratios,
            "wspace": float(style["rq5_evidence_wspace"]),
        },
    )
    y_positions = np.arange(len(plot), dtype=float)
    for y_position, row in zip(y_positions, plot.itertuples(index=False)):
        difference = float(row.difference_percentage_points)
        arrow_color = (
            reporting["palette"]["positive"]
            if difference >= 0
            else reporting["palette"]["negative"]
        )
        axis.add_patch(
            FancyArrowPatch(
                (float(row.closed_book), y_position),
                (float(row.open_book), y_position),
                arrowstyle="-|>",
                mutation_scale=float(style["rq5_arrow_mutation_scale"]),
                linewidth=float(style["rq5_arrow_line_width"]),
                color=arrow_color,
                shrinkA=float(style["rq5_arrow_shrink_points"]),
                shrinkB=float(style["rq5_arrow_shrink_points"]),
                zorder=2,
            )
        )
        axis.plot(
            float(row.closed_book),
            y_position,
            marker=annotation["access_markers"][0],
            markersize=float(style["rq5_marker_size"]),
            color=condition_color(config, "access", "closed_book"),
            linestyle="none",
            zorder=3,
        )
        delta_axis.text(
            float(style["rq5_delta_text_x"]),
            y_position,
            f"{difference:+.1f} pp",
            ha="left",
            va="center",
            fontsize=tick_size,
            color=arrow_color,
            fontweight="bold",
        )
    axis.set_yticks(y_positions, plot["category"].map(label_map))
    axis.tick_params(axis="both", labelsize=tick_size)
    axis.invert_yaxis()
    axis.set_xlim(float(style["rq5_evidence_axis_min"]), float(style["rq5_evidence_axis_max"]))
    if x_ticks is not None:
        axis.set_xticks(x_ticks)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel(
        "Share of rationales mentioning category",
        fontsize=label_size,
    )
    if title:
        axis.set_title(title, pad=float(style["rq5_title_pad"]), loc="left")
    if subtitle_prefix is not None:
        n_rationales = int(plot["closed_n"].iloc[0] + plot["open_n"].iloc[0])
        axis.text(
            0.0,
            float(style["rq5_subtitle_y"]),
            (
                f"{subtitle_prefix} (n={n_rationales:,})"
                if include_n_in_subtitle
                else subtitle_prefix
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=tick_size,
            color=reporting["palette"]["neutral"],
        )
    delta_axis.set_xlim(0.0, 1.0)
    delta_axis.tick_params(
        axis="both", which="both", left=False, labelleft=False, bottom=False, labelbottom=False
    )
    for spine in delta_axis.spines.values():
        spine.set_visible(False)
    delta_axis.grid(False)
    add_numeric_grid(axis, config, "x")
    figure.subplots_adjust(**{key: float(value) for key, value in margins.items()})
    save_figure(
        figure,
        config,
        manifest,
        artifact_id,
        "rq5",
        source_hashes,
    )


def run_final(config: AnalysisConfig, manifest: Manifest) -> dict[str, pd.DataFrame]:
    objective = run_objective(config, manifest)
    rationale_record = manifest.require("derived_rationale_corpus")
    resolved_record = manifest.require("resolved_annotations")
    agreement_record = manifest.require("annotation_automated_agreement")
    human_agreement_record = manifest.require("annotation_human_agreement")
    tsne_record = manifest.require("rq5_tsne_coordinates")
    rationale = pd.read_parquet(rationale_record.path)
    resolved = pd.read_parquet(resolved_record.path)
    agreement = pd.read_parquet(agreement_record.path)
    human_agreement = pd.read_parquet(human_agreement_record.path)
    source_hashes = {
        "rationale_corpus": rationale_record.sha256,
        "resolved_annotations": resolved_record.sha256,
        "automated_agreement": agreement_record.sha256,
        "human_audit_agreement": human_agreement_record.sha256,
        "tsne_coordinates": tsne_record.sha256,
    }
    manifest.discard_generated(
        [
            "rq5_evidence_categories_by_model_appendix_companion",
            "rq5_evidence_categories_by_model_appendix_csv",
            "rq5_evidence_categories_by_model_appendix_tex",
            "rq5_illustrative_excerpts_tex",
        ]
    )
    prevalence = _resolved_prevalence(rationale, resolved, config)
    model_prevalence = _model_prevalence(rationale, resolved, config)
    scopes = ["primary_rq5_cell"]
    wide = pd.concat([_prevalence_wide(prevalence, scope) for scope in scopes], ignore_index=True)
    excerpts = _illustrative_excerpts(rationale, resolved, config)

    result_dir = config.resolve_path("results") / "rq5"
    for name, frame in (
        ("resolved_prevalence", prevalence),
        ("resolved_prevalence_wide", wide),
        ("resolved_prevalence_by_model", model_prevalence),
        ("illustrative_excerpts", excerpts),
    ):
        _write_result_parquet(
            frame,
            result_dir / f"{name}.parquet",
            manifest,
            f"rq5_{name}",
            source_hashes,
        )

    reporting = config.section("reporting")
    category_labels = reporting["labels"]["annotation_categories"]
    percent_decimals = int(reporting["style"]["table_percentage_decimals"])
    category_order = list(config.section("annotation")["category_order"])
    evidence_rows = wide.set_index("category").reindex(category_order).reset_index()
    evidence_display = pd.DataFrame(
        {
            "Category": evidence_rows["category"].map(category_labels),
            "Closed (%)": evidence_rows["closed_book"] * 100.0,
            "Open (%)": evidence_rows["open_book"] * 100.0,
            "Δ (pp)": evidence_rows["difference_percentage_points"],
        }
    )
    save_table(
        evidence_display,
        config,
        manifest,
        "rq5_evidence_categories_by_access",
        "rq5",
        source_hashes,
        latex_formats={
            "Closed (%)": f"{{:.{percent_decimals}f}}",
            "Open (%)": f"{{:.{percent_decimals}f}}",
            "Δ (pp)": f"{{:+.{percent_decimals}f}}",
        },
    )

    agreement_decimals = int(reporting["style"]["table_points_decimals"])
    agreement_rows = agreement.set_index("category").reindex(category_order).reset_index()
    agreement_display = pd.DataFrame(
        {
            "Category": agreement_rows["category"].map(category_labels),
            "Keyword (%)": agreement_rows["first_prevalence"] * 100.0,
            "GLM (%)": agreement_rows["second_prevalence"] * 100.0,
            "Agreement (%)": agreement_rows["raw_agreement"] * 100.0,
            "κ": agreement_rows["cohen_kappa"],
        }
    )
    save_table(
        agreement_display,
        config,
        manifest,
        "rq5_automated_coder_agreement",
        "rq5",
        source_hashes,
        latex_formats={
            "Keyword (%)": f"{{:.{percent_decimals}f}}",
            "GLM (%)": f"{{:.{percent_decimals}f}}",
            "Agreement (%)": f"{{:.{percent_decimals}f}}",
            "κ": f"{{:.{agreement_decimals}f}}",
        },
    )
    human_agreement_rows = human_agreement.assign(
        comparison_order=np.where(human_agreement["first_coder"].eq("llm"), 0, 1),
        category_order=human_agreement["category"].map(
            {category: index for index, category in enumerate(category_order)}
        ),
    ).sort_values(["comparison_order", "category_order"])
    human_agreement_display = pd.DataFrame(
        {
            "Coder pair": np.where(
                human_agreement_rows["first_coder"].eq("llm"),
                "GLM–human",
                "Keyword–human",
            ),
            "Category": human_agreement_rows["category"].map(category_labels),
            "Automated (%)": human_agreement_rows["first_prevalence"] * 100.0,
            "Human (%)": human_agreement_rows["second_prevalence"] * 100.0,
            "Agreement (%)": human_agreement_rows["raw_agreement"] * 100.0,
            "κ": human_agreement_rows["cohen_kappa"],
        }
    )
    save_table(
        human_agreement_display,
        config,
        manifest,
        "rq5_human_audit_agreement",
        "rq5",
        source_hashes,
        latex_formats={
            "Automated (%)": f"{{:.{percent_decimals}f}}",
            "Human (%)": f"{{:.{percent_decimals}f}}",
            "Agreement (%)": f"{{:.{percent_decimals}f}}",
            "κ": f"{{:.{agreement_decimals}f}}",
        },
    )
    model_display = model_prevalence.assign(
        Model=model_prevalence["model_id"].map(lambda value: model_label(config, value)),
        Access=model_prevalence["access_condition"].map(reporting["labels"]["access"]),
        Category=model_prevalence["category"].map(category_labels),
        **{
            "Rationales mentioning category, %": model_prevalence["prevalence"] * 100.0,
            "Rationales": model_prevalence["n_rationales"],
        },
    )[["Model", "Access", "Category", "Rationales mentioning category, %", "Rationales"]]
    save_table(
        model_display,
        config,
        manifest,
        "rq5_evidence_categories_by_model_full",
        "rq5",
        source_hashes,
        latex=False,
    )
    model_rates = model_prevalence.pivot(
        index=["model_id", "category"],
        columns="access_condition",
        values="prevalence",
    ).reset_index()
    model_rates["difference_percentage_points"] = (
        model_rates["open_book"] - model_rates["closed_book"]
    ) * 100.0
    main_categories = list(reporting["rq5_main_figure_categories"])
    model_shifts = model_rates[model_rates["category"].isin(main_categories)].pivot(
        index="model_id",
        columns="category",
        values="difference_percentage_points",
    )
    complete_models = list(config.section("design")["complete_panel"])
    model_shifts = model_shifts.reindex(complete_models)
    short_labels = reporting["labels"]["annotation_categories_short"]
    model_shift_display = pd.DataFrame(
        {
            "Model": [model_label(config, model_id) for model_id in model_shifts.index],
            **{
                short_labels[category]: model_shifts[category].tolist()
                for category in main_categories
            },
        }
    )
    save_table(
        model_shift_display,
        config,
        manifest,
        "rq5_evidence_shifts_by_model_appendix",
        "rq5",
        source_hashes,
        latex_formats={
            short_labels[category]: f"{{:+.{percent_decimals}f}}" for category in main_categories
        },
    )
    save_table(
        excerpts,
        config,
        manifest,
        "rq5_illustrative_excerpts",
        "rq5",
        source_hashes,
        latex=False,
    )

    primary_scope = str(config.section("annotation")["primary_reporting_scope"])
    primary = wide[wide["scope"].eq(primary_scope)].copy()
    _plot_evidence_prevalence(
        primary,
        config,
        manifest,
        source_hashes,
        categories=main_categories,
        artifact_id="fig_rq5_evidence_by_access",
        title="",
        width=float(reporting["figure_width_single"]),
        height=float(reporting["style"]["rq5_evidence_main_height"]),
        width_ratios=[
            float(value) for value in reporting["style"]["rq5_evidence_main_width_ratios"]
        ],
        margins=reporting["style"]["rq5_evidence_main_margins"],
        label_map=reporting["labels"]["annotation_categories_figure"],
        subtitle_prefix=None,
        include_n_in_subtitle=False,
        tick_label_size=float(reporting["style"]["rq5_evidence_main_tick_label_size"]),
        axis_label_size=float(reporting["style"]["rq5_evidence_main_axis_label_size"]),
        x_ticks=[float(value) for value in reporting["style"]["rq5_evidence_main_x_ticks"]],
    )
    _plot_evidence_prevalence(
        primary,
        config,
        manifest,
        source_hashes,
        categories=list(config.section("annotation")["category_order"]),
        artifact_id="fig_rq5_evidence_by_access_appendix",
        title="",
        width=float(reporting["figure_width_double"]),
        height=float(reporting["style"]["rq5_evidence_appendix_height"]),
        width_ratios=[
            float(value) for value in reporting["style"]["rq5_evidence_appendix_width_ratios"]
        ],
        margins=reporting["style"]["rq5_evidence_appendix_margins"],
        label_map=category_labels,
        subtitle_prefix=None,
        include_n_in_subtitle=False,
        tick_label_size=float(reporting["style"]["rq5_evidence_appendix_tick_label_size"]),
        axis_label_size=float(reporting["style"]["rq5_evidence_appendix_axis_label_size"]),
        x_ticks=[float(value) for value in reporting["style"]["rq5_evidence_appendix_x_ticks"]],
    )

    records = []
    for row in primary.itertuples(index=False):
        records.append(
            headline_record(
                config,
                "rq5_headlines",
                f"evidence_access_difference.{row.category}",
                "open-book minus closed-book resolved category prevalence",
                source_hashes,
                estimate=float(row.difference_percentage_points),
                ci_low=None,
                ci_high=None,
                p_raw=None,
                p_adjusted=None,
                median=None,
                n_matches=int(row.n_matches),
                n_predictions=int(row.closed_n + row.open_n),
                units="percentage points",
                aggregation="descriptive rates over resolved non-empty rationales",
                extra={"null_reason": "descriptive finite-corpus content analysis without a test"},
            )
        )
    write_headlines(config, manifest, "rq5_headlines", records, "rq5", source_hashes)
    manifest.write()
    return {
        **objective,
        "prevalence": prevalence,
        "prevalence_wide": wide,
        "agreement": agreement,
        "human_agreement": human_agreement,
        "model_prevalence": model_prevalence,
        "excerpts": excerpts,
    }


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, pd.DataFrame]:
    return run_final(config, manifest)
