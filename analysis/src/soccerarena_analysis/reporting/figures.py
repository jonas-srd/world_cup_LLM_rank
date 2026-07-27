from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..manifest import Manifest
from .headline_json import headline_record, write_headlines


def apply_style(config: AnalysisConfig) -> None:
    reporting = config.section("reporting")
    palette = reporting["palette"]
    style = reporting["style"]
    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.sans-serif": [style["font_family"], "Liberation Sans", "DejaVu Sans"],
            "font.size": style["font_size"],
            "axes.titlesize": style["axes_title_size"],
            "axes.labelsize": style["axes_label_size"],
            "legend.fontsize": style["legend_font_size"],
            "xtick.labelsize": style["tick_label_size"],
            "ytick.labelsize": style["tick_label_size"],
            "axes.titleweight": style["axes_title_weight"],
            "axes.titlelocation": "left",
            "axes.edgecolor": palette["neutral"],
            "axes.labelcolor": palette["text"],
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": palette["text"],
            "xtick.color": palette["text"],
            "ytick.color": palette["text"],
            "lines.linewidth": style["line_width"],
            "lines.markersize": style["marker_size"],
            "legend.frameon": False,
            "savefig.facecolor": "white",
            # Keep text as editable, embedded TrueType glyphs in publication exports.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.prop_cycle": plt.cycler(
                color=[
                    palette["primary"],
                    palette["secondary"],
                    palette["positive"],
                    palette["neutral"],
                    palette["negative"],
                ]
            ),
        }
    )


def model_label(config: AnalysisConfig, model_id: str) -> str:
    return config.section("reporting")["labels"]["models"].get(model_id, model_id)


def model_color(config: AnalysisConfig, model_id: str) -> str:
    reporting = config.section("reporting")
    return reporting["colors"]["model"].get(model_id, reporting["palette"]["neutral"])


def condition_label(config: AnalysisConfig, family: str, value: str) -> str:
    return config.section("reporting")["labels"][family].get(value, value)


def condition_color(config: AnalysisConfig, family: str, value: str) -> str:
    reporting = config.section("reporting")
    return reporting["colors"][family].get(value, reporting["palette"]["neutral"])


def metric_label(config: AnalysisConfig, metric: str) -> str:
    return config.section("reporting")["labels"]["metrics"].get(metric, metric)


def add_panel_label(
    axis: plt.Axes,
    label: str,
    config: AnalysisConfig,
    *,
    x: float | None = None,
    y: float | None = None,
    font_size: float | None = None,
) -> None:
    style = config.section("reporting")["style"]
    axis.text(
        float(style["panel_label_x"]) if x is None else float(x),
        float(style["panel_label_y"]) if y is None else float(y),
        label,
        transform=axis.transAxes,
        fontsize=(float(style["panel_label_size"]) if font_size is None else float(font_size)),
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_zero_reference(
    axis: plt.Axes, config: AnalysisConfig, orientation: str = "vertical"
) -> None:
    reporting = config.section("reporting")
    kwargs = {
        "color": reporting["palette"]["neutral"],
        "linestyle": "--",
        "linewidth": float(reporting["style"]["zero_line_width"]),
        "zorder": 0,
    }
    if orientation == "vertical":
        axis.axvline(0.0, **kwargs)
    else:
        axis.axhline(0.0, **kwargs)


def add_numeric_grid(axis: plt.Axes, config: AnalysisConfig, axis_name: str = "x") -> None:
    reporting = config.section("reporting")
    axis.grid(
        axis=axis_name,
        color=reporting["palette"]["grid"],
        linewidth=float(reporting["style"]["grid_line_width"]),
        alpha=float(reporting["style"]["grid_alpha"]),
    )


def forest_plot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    labels: Sequence[str],
    config: AnalysisConfig,
    *,
    estimate: str = "estimate",
    ci_low: str = "ci_low",
    ci_high: str = "ci_high",
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
) -> None:
    reporting = config.section("reporting")
    style = reporting["style"]
    y_positions = np.arange(len(frame), dtype=float)
    chosen_colors = (
        list(colors) if colors is not None else [reporting["palette"]["primary"]] * len(frame)
    )
    chosen_markers = list(markers) if markers is not None else ["o"] * len(frame)
    for y_position, (_, row), color, marker in zip(
        y_positions, frame.iterrows(), chosen_colors, chosen_markers
    ):
        value = float(row[estimate])
        low = row.get(ci_low)
        high = row.get(ci_high)
        if pd.notna(low) and pd.notna(high):
            axis.errorbar(
                value,
                y_position,
                xerr=[[value - float(low)], [float(high) - value]],
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=float(style["ci_cap_size"]),
                zorder=3,
            )
        else:
            axis.plot(value, y_position, marker=marker, color=color, linestyle="none", zorder=3)
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    add_zero_reference(axis, config)
    add_numeric_grid(axis, config, "x")


def save_figure(
    figure: plt.Figure,
    config: AnalysisConfig,
    manifest: Manifest,
    artifact_id: str,
    module: str,
    source_hashes: dict[str, str],
) -> list[Path]:
    directory = config.resolve_path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    formats = sorted(
        config.section("reporting")["formats"], key=lambda extension: extension.lower() != "png"
    )
    for extension in formats:
        canonical_path = directory / f"{artifact_id}.{extension}"
        path = canonical_path
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = int(config.section("reporting")["dpi"])
        metadata: dict[str, str] = {}
        try:
            figure.savefig(path, **kwargs)
        except PermissionError:
            fallback_directory = directory / "versioned"
            fallback_directory.mkdir(parents=True, exist_ok=True)
            path = fallback_directory / f"{artifact_id}.{config.digest}.{extension}"
            figure.savefig(path, **kwargs)
            metadata = {
                "canonical_path": str(canonical_path.resolve()),
                "fallback_reason": "canonical output was locked during generation",
            }
        manifest.add(
            f"{artifact_id}_{extension}",
            path,
            "figure",
            module,
            source_hashes,
            metadata,
        )
        paths.append(path)
    plt.close(figure)
    manifest.write()
    companion_id = f"{artifact_id}_companion"
    output_hashes = {
        path.suffix.lstrip("."): getattr(
            manifest.records[f"{artifact_id}_{path.suffix.lstrip('.')}"], "sha256"
        )
        for path in paths
    }
    record = headline_record(
        config,
        companion_id,
        artifact_id,
        artifact_id,
        source_hashes,
        estimate=None,
        ci_low=None,
        ci_high=None,
        p_raw=None,
        p_adjusted=None,
        median=None,
        n_matches=None,
        n_predictions=None,
        units=None,
        aggregation="visual display",
        extra={
            "null_reason": "figure estimands are stored in the analysis headline JSON and plotted source tables",
            "output_hashes": output_hashes,
        },
    )
    write_headlines(config, manifest, companion_id, [record], module, source_hashes)
    return paths
