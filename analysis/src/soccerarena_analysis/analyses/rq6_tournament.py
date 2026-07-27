from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import PercentFormatter

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..reporting.figures import (
    add_numeric_grid,
    apply_style,
    model_label,
    save_figure,
)
from ..reporting.headline_json import headline_record, write_headlines
from ..reporting.tables import save_table

COUNTRY_FLAG_FILES = {
    "Argentina": "ar.png",
    "Belgium": "be.png",
    "Brazil": "br.png",
    "Colombia": "co.png",
    "England": "gb-eng.png",
    "France": "fr.png",
    "Germany": "de.png",
    "Mexico": "mx.png",
    "Netherlands": "nl.png",
    "Spain": "es.png",
    "Switzerland": "ch.png",
    "United States": "us.png",
}

MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "deepseek/deepseek-v4-pro": "deepseek-color.png",
    "google/gemini-3.1-pro-preview": "google-color.png",
    "x-ai/grok-4.3": "xai.png",
    "openai/gpt-5.5": "openai.png",
    "mistralai/mistral-large-2512": "mistral-color.png",
    "qwen/qwen3.7-max": "alibaba-color.png",
}


def _add_sum_audit_model_icons(
    axis: plt.Axes,
    model_order: list[str],
    style: dict[str, object],
) -> None:
    icon_directory = Path(__file__).resolve().parents[3] / "assets" / "brand_icons"
    icon_paths = {model: icon_directory / MODEL_PROVIDER_ICONS[model] for model in model_order}
    missing_icons = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing_icons:
        raise FileNotFoundError(f"Missing model provider icons: {missing_icons}")

    axis.tick_params(axis="y", length=0)
    for y_position, model in enumerate(model_order):
        axis.add_artist(
            AnnotationBbox(
                OffsetImage(
                    plt.imread(icon_paths[model]),
                    zoom=float(style["rq6_sum_icon_zoom"]),
                    interpolation="lanczos",
                ),
                (float(style["rq6_sum_icon_x"]), y_position),
                xycoords=("axes fraction", "data"),
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _analysis_prediction_panel(
    predictions: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    settings = config.section("special_questions")
    model_panel = set(config.section("design")["complete_panel"])
    valid = predictions[
        predictions["is_valid_for_scoring"].fillna(False)
        & predictions["model_id"].isin(model_panel)
    ].copy()
    observed_models = set(valid["model_id"].dropna().astype(str))
    if observed_models != model_panel:
        raise ValueError(
            "Tournament analysis model panel mismatch: "
            f"expected={sorted(model_panel)}, observed={sorted(observed_models)}"
        )
    counts = valid.groupby("question_id")["prediction_id"].nunique()
    expected = int(settings["analysis_forecasts_per_question"])
    if len(counts) != int(settings["expected_questions"]) or not counts.eq(expected).all():
        raise ValueError(
            f"Unexpected complete-panel forecast count by tournament question: {counts.to_dict()}"
        )
    return valid


def _add_country_flags(
    axis: plt.Axes,
    plot_rows: list[dict[str, object]],
    style: dict[str, object],
    reporting: dict[str, object],
) -> None:
    flag_directory = Path(__file__).resolve().parents[3] / "assets" / "country_flags"
    countries = {
        str(row["flag_country"]) for row in plot_rows if row.get("flag_country") is not None
    }
    flag_paths = {country: flag_directory / COUNTRY_FLAG_FILES[country] for country in countries}
    missing_flags = [str(path) for path in flag_paths.values() if not path.is_file()]
    if missing_flags:
        raise FileNotFoundError(f"Missing country flags: {missing_flags}")

    axis.tick_params(
        axis="y",
        labelsize=float(style["rq6_forecast_tick_label_size"]),
        pad=float(style["rq6_forecast_label_pad"]),
    )
    for y_position, plot_row in enumerate(plot_rows):
        country = plot_row.get("flag_country")
        if country is None:
            continue
        flag = OffsetImage(
            plt.imread(flag_paths[str(country)]),
            zoom=float(style["rq6_flag_zoom"]),
            interpolation="lanczos",
        )
        flag_box = AnnotationBbox(
            flag,
            (float(style["rq6_flag_x"]), y_position),
            xycoords=("axes fraction", "data"),
            box_alignment=(0.5, 0.5),
            frameon=True,
            pad=0.02,
            bboxprops={
                "edgecolor": reporting["palette"]["neutral_light"],
                "facecolor": "white",
                "linewidth": 0.4,
                "boxstyle": "square,pad=0.02",
            },
            annotation_clip=False,
        )
        axis.add_artist(flag_box)


def _validate_normalized_inputs(
    predictions: pd.DataFrame,
    options: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    settings = config.section("special_questions")
    tolerance = float(config.section("validation")["probability_tolerance"])
    expected_questions = int(settings["expected_questions"])
    expected_forecasts = int(settings["expected_forecasts_per_question"])
    question_count = predictions["question_id"].nunique()
    if question_count != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} normalized special questions, found {question_count}"
        )
    counts = predictions.groupby("question_id")["prediction_id"].nunique()
    if not counts.eq(expected_forecasts).all():
        raise ValueError(f"Unexpected forecast count by tournament question: {counts.to_dict()}")
    stable_columns = [
        "question_id",
        "model_id",
        "model_version",
        "access_condition",
        "prompt_strategy",
        "forecast_horizon",
        "sample_id",
    ]
    if predictions.duplicated(stable_columns).any():
        raise ValueError("Duplicate stable forecast cells in normalized tournament questions")
    if options.duplicated(["prediction_id", "candidate_id"]).any():
        raise ValueError("Duplicate candidate within a normalized tournament forecast")
    if (
        options["probability"].isna().any()
        or ((options["probability"] < 0.0) | (options["probability"] > 1.0)).any()
    ):
        raise ValueError("Tournament-question probabilities must lie in [0, 1]")

    prediction_types = predictions.set_index("prediction_id")["prediction_type"]
    option_audit = options.groupby("prediction_id").agg(
        candidates=("candidate_id", "nunique"),
        probability_sum=("probability", "sum"),
        final_picks=("is_final_pick", "sum"),
    )
    option_audit["prediction_type"] = option_audit.index.map(prediction_types)
    single = option_audit[option_audit["prediction_type"].eq("single_choice")]
    if not single["final_picks"].eq(int(settings["expected_single_choice_pick_count"])).all():
        raise ValueError(
            "Each single-choice tournament forecast must contain exactly one final pick"
        )
    if not np.isclose(
        single["probability_sum"].to_numpy(dtype=float),
        1.0,
        atol=tolerance,
        rtol=0.0,
    ).all():
        raise ValueError("Single-choice tournament probabilities must sum to one")
    semifinal = option_audit[option_audit["prediction_type"].eq("multi_choice_fixed_k")]
    if not semifinal["final_picks"].eq(int(settings["semifinal_k"])).all():
        raise ValueError("Each semifinal forecast must contain exactly four final picks")
    if not semifinal["candidates"].eq(int(settings["expected_tournament_candidates"])).all():
        raise ValueError("Each semifinal forecast must cover the full tournament candidate set")

    group_ids = predictions.loc[
        predictions["question_id"].str.startswith(settings["group_question_prefix"]),
        "prediction_id",
    ]
    if (
        not option_audit.loc[group_ids, "candidates"]
        .eq(int(settings["expected_group_candidates"]))
        .all()
    ):
        raise ValueError("Each group-winner forecast must contain four candidates")
    tournament_ids = predictions.loc[
        predictions["question_id"].isin(
            [settings["champion_question_id"], settings["top_scorer_question_id"]]
        ),
        "prediction_id",
    ]
    if (
        not option_audit.loc[tournament_ids, "candidates"]
        .eq(int(settings["expected_tournament_candidates"]))
        .all()
    ):
        raise ValueError("Tournament-wide single-choice forecasts must cover all candidates")

    option_universes = options.groupby(["question_id", "prediction_id"])["candidate_id"].agg(
        lambda values: frozenset(values.astype(str))
    )
    if option_universes.groupby(level="question_id").nunique().gt(1).any():
        raise ValueError("Candidate universes differ across forecasts for the same question")
    for question_id, group in outcomes.groupby("question_id"):
        universe = option_universes.xs(question_id, level="question_id").iloc[0]
        realized = set(group.loc[group["is_realized"], "candidate_id"].astype(str))
        if not realized.issubset(universe):
            raise ValueError(f"Realized candidate is absent from option universe: {question_id}")


def _single_choice(
    question: pd.DataFrame, outcomes: pd.DataFrame, epsilon: float, tolerance: float
) -> pd.DataFrame:
    realized = set(outcomes.loc[outcomes["is_realized"], "candidate_id"].astype(str))
    if len(realized) != 1:
        return pd.DataFrame()
    actual = next(iter(realized))
    records: list[dict[str, object]] = []
    for prediction_id, group in question.groupby("prediction_id"):
        probabilities = group.set_index("candidate_id")["probability"].astype(float)
        if actual not in probabilities.index:
            raise ValueError(f"Realized candidate absent from forecast {prediction_id}")
        if probabilities.isna().any() or not np.isclose(
            probabilities.sum(), 1.0, atol=tolerance, rtol=0.0
        ):
            raise ValueError(f"Invalid single-choice probability vector: {prediction_id}")
        target = probabilities.index.to_series().astype(str).eq(actual).astype(float).to_numpy()
        actual_probability = float(probabilities.loc[actual])
        final_picks = (
            group.loc[group["is_final_pick"].fillna(False), "candidate_id"].astype(str).tolist()
        )
        if len(final_picks) != 1:
            raise ValueError(f"Single-choice forecast lacks exactly one pick: {prediction_id}")
        records.append(
            {
                "prediction_id": prediction_id,
                "actual_candidate": actual,
                "final_pick_candidate": final_picks[0],
                "accuracy": final_picks[0] == actual,
                "brier": float(np.sum((probabilities.to_numpy() - target) ** 2)),
                "log_loss": float(-math.log(max(min(actual_probability, 1.0 - epsilon), epsilon))),
                "realized_option_probability": actual_probability,
            }
        )
    return pd.DataFrame(records)


def _uniform_random_reference(
    options: pd.DataFrame,
    question_id: str,
    pick_count: int,
) -> float:
    """Return the per-candidate probability under uniform random selection.

    The candidate universe is derived from the normalized forecasts rather than
    hard-coded. For a fixed-k question, a coherent uniform forecast assigns
    k / n marginal probability to each of n candidates.
    """
    question = options.loc[options["question_id"].eq(question_id)]
    if question.empty:
        raise ValueError(f"No normalized options for tournament question: {question_id}")
    candidate_counts = question.groupby("prediction_id")["candidate_id"].nunique()
    if candidate_counts.nunique() != 1:
        raise ValueError(f"Candidate counts differ across forecasts for: {question_id}")
    candidate_count = int(candidate_counts.iloc[0])
    if pick_count < 1 or pick_count > candidate_count:
        raise ValueError(
            f"Invalid uniform-reference pick count {pick_count} for "
            f"{candidate_count} candidates in {question_id}"
        )
    return float(pick_count / candidate_count)


def _semifinalists(question: pd.DataFrame, outcomes: pd.DataFrame, expected_k: int) -> pd.DataFrame:
    realized = set(outcomes.loc[outcomes["is_realized"], "candidate_id"].astype(str))
    if len(realized) != expected_k:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for prediction_id, group in question.groupby("prediction_id"):
        candidates = group["candidate_id"].astype(str)
        probabilities = group["probability"].astype(float)
        selected = set(group.loc[group["is_final_pick"].fillna(False), "candidate_id"].astype(str))
        if len(selected) != expected_k:
            raise ValueError(
                f"Semifinal forecast lacks exactly {expected_k} picks: {prediction_id}"
            )
        target = candidates.isin(realized).astype(float).to_numpy()
        selected_vector = candidates.isin(selected).astype(float).to_numpy()
        probability_sum = float(probabilities.sum())
        records.append(
            {
                "prediction_id": prediction_id,
                "actual_candidates": " | ".join(sorted(realized)),
                "exact_set_accuracy": selected == realized,
                "correct_count": len(selected & realized),
                "hamming_loss": float(np.mean(selected_vector != target)),
                "marginal_brier": float(np.mean((probabilities.to_numpy() - target) ** 2)),
                "semifinal_probability_sum": probability_sum,
                "absolute_sum_from_expected_k": float(abs(probability_sum - expected_k)),
            }
        )
    return pd.DataFrame(records)


def _question_label(question_id: str, settings: dict[str, object]) -> str:
    if question_id == settings["semifinal_question_id"]:
        return "Semifinalists"
    if question_id == settings["champion_question_id"]:
        return "World champion"
    if question_id == settings["top_scorer_question_id"]:
        return "Top-scorer team"
    group = question_id.removeprefix(str(settings["group_question_prefix"]))
    return f"Group {group} winner"


def _summary_tables(
    summary: pd.DataFrame, config: AnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = config.section("special_questions")
    single = summary[summary["question_id"].ne(settings["semifinal_question_id"])].copy()
    single_display = single.assign(
        Question=single["question_id"].map(lambda value: _question_label(value, settings)),
        Status=single["status"].str.title(),
        **{
            "Realized outcome": single.get("realized_candidate"),
            "Correct picks, %": single.get("single_choice_accuracy") * 100.0,
            "Mean probability on outcome, %": single.get("realized_option_probability") * 100.0,
            "Brier score": single.get("brier"),
            "Log loss": single.get("log_loss"),
            "Forecasts": single.get("n_forecasts"),
        },
    )[
        [
            "Question",
            "Status",
            "Realized outcome",
            "Correct picks, %",
            "Mean probability on outcome, %",
            "Brier score",
            "Log loss",
            "Forecasts",
        ]
    ]

    semifinal = summary[summary["question_id"].eq(settings["semifinal_question_id"])].copy()
    if semifinal.empty:
        semifinal_display = pd.DataFrame(
            columns=[
                "Question",
                "Status",
                "Realized teams",
                "Exact sets, %",
                "Mean correct of four",
                "Correct share, %",
                "Hamming loss, %",
                "Marginal Brier",
                "Forecasts",
            ]
        )
    else:
        semifinal_display = semifinal.assign(
            Question="Semifinalists",
            Status=semifinal["status"].str.title(),
            **{
                "Realized teams": semifinal.get("realized_candidate"),
                "Exact sets, %": semifinal.get("exact_set_accuracy") * 100.0,
                "Mean correct of four": semifinal.get("correct_count"),
                "Correct share, %": semifinal.get("correct_count")
                / int(settings["semifinal_k"])
                * 100.0,
                "Hamming loss, %": semifinal.get("hamming_loss") * 100.0,
                "Marginal Brier": semifinal.get("marginal_brier"),
                "Forecasts": semifinal.get("n_forecasts"),
            },
        )[
            [
                "Question",
                "Status",
                "Realized teams",
                "Exact sets, %",
                "Mean correct of four",
                "Correct share, %",
                "Hamming loss, %",
                "Marginal Brier",
                "Forecasts",
            ]
        ]

    groups = summary[
        summary["question_id"].str.startswith(str(settings["group_question_prefix"]))
        & summary["status"].eq("resolved")
    ]
    uniform = float(settings["group_uniform_probability"])
    aggregate_display = pd.DataFrame(
        [
            {
                "Metric": "Correct final pick",
                "Equal-weight group mean": groups["single_choice_accuracy"].mean() * 100.0,
                "Uniform four-team reference": uniform * 100.0,
                "Unit": "%",
                "Questions": len(groups),
            },
            {
                "Metric": "Brier score",
                "Equal-weight group mean": groups["brier"].mean(),
                "Uniform four-team reference": 1.0 - uniform,
                "Unit": "score",
                "Questions": len(groups),
            },
            {
                "Metric": "Log loss",
                "Equal-weight group mean": groups["log_loss"].mean(),
                "Uniform four-team reference": -math.log(uniform),
                "Unit": "nats",
                "Questions": len(groups),
            },
            {
                "Metric": "Probability on realized winner",
                "Equal-weight group mean": groups["realized_option_probability"].mean() * 100.0,
                "Uniform four-team reference": uniform * 100.0,
                "Unit": "%",
                "Questions": len(groups),
            },
        ]
    )
    return single_display, semifinal_display, aggregate_display


def _plot_main(
    summary: pd.DataFrame,
    detailed: pd.DataFrame,
    options: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
) -> None:
    settings = config.section("special_questions")
    reporting = config.section("reporting")
    style = reporting["style"]
    apply_style(config)
    resolved = summary[summary["status"].eq("resolved")].copy()
    groups = resolved[
        resolved["question_id"].str.startswith(str(settings["group_question_prefix"]))
    ].copy()
    groups["group_label"] = groups["question_id"].str.removeprefix(
        str(settings["group_question_prefix"])
    )
    groups = groups.sort_values("group_label")

    plot_rows: list[dict[str, object]] = []
    for row in groups.itertuples(index=False):
        values = detailed.loc[
            detailed["question_id"].eq(row.question_id),
            "realized_option_probability",
        ].astype(float)
        plot_rows.append(
            {
                "question_id": row.question_id,
                "label": f"Group {row.group_label}",
                "flag_country": str(row.realized_candidate),
                "values": values,
                "mean": float(row.realized_option_probability),
                "correct": f"{int(row.correct_picks)}/{int(row.n_forecasts)}",
                "random_reference": _uniform_random_reference(
                    options, str(row.question_id), pick_count=1
                ),
            }
        )

    semifinal_id = str(settings["semifinal_question_id"])
    semifinal_row = resolved[resolved["question_id"].eq(semifinal_id)]
    if not semifinal_row.empty:
        realized_semifinalists = sorted(
            outcomes.loc[
                outcomes["question_id"].eq(semifinal_id) & outcomes["is_realized"],
                "candidate_id",
            ].astype(str)
        )
        semifinal_values = (
            options.loc[
                options["question_id"].eq(semifinal_id)
                & options["candidate_id"].astype(str).isin(realized_semifinalists)
            ]
            .groupby("prediction_id")["probability"]
            .mean()
            .astype(float)
        )
        row = semifinal_row.iloc[0]
        plot_rows.append(
            {
                "question_id": semifinal_id,
                "label": "Semifinalists",
                "flag_country": None,
                "values": semifinal_values,
                "mean": float(semifinal_values.mean()),
                "correct": f"{int(row['exact_sets'])}/{int(row['n_forecasts'])} exact",
                "random_reference": _uniform_random_reference(
                    options,
                    semifinal_id,
                    pick_count=int(settings["semifinal_k"]),
                ),
            }
        )

    for question_id, question_label, show_realized_candidate in (
        (str(settings["top_scorer_question_id"]), "Top-scorer team", False),
        (str(settings["champion_question_id"]), "Winner", False),
    ):
        question_row = summary[summary["question_id"].eq(question_id)].iloc[0]
        if question_row["status"] == "resolved":
            values = detailed.loc[
                detailed["question_id"].eq(question_id),
                "realized_option_probability",
            ].astype(float)
            row_label = question_label
            if show_realized_candidate:
                row_label += f" · {question_row['realized_candidate']}"
            plot_rows.append(
                {
                    "question_id": question_id,
                    "label": row_label,
                    "flag_country": str(question_row["realized_candidate"]),
                    "values": values,
                    "mean": float(question_row["realized_option_probability"]),
                    "correct": (
                        f"{int(question_row['correct_picks'])}/{int(question_row['n_forecasts'])}"
                    ),
                    "random_reference": _uniform_random_reference(
                        options, question_id, pick_count=1
                    ),
                }
            )
        else:
            plot_rows.append(
                {
                    "question_id": question_id,
                    "label": question_label,
                    "flag_country": None,
                    "values": pd.Series(dtype=float),
                    "mean": np.nan,
                    "correct": "Pending",
                    "random_reference": _uniform_random_reference(
                        options, question_id, pick_count=1
                    ),
                }
            )

    forecast_figure, group_axis = plt.subplots(
        figsize=(
            float(reporting["figure_width_double"]),
            float(style["rq6_forecast_height"]),
        )
    )
    y_positions = np.arange(len(plot_rows), dtype=float)
    for y_position, plot_row in zip(y_positions, plot_rows):
        group_axis.vlines(
            float(plot_row["random_reference"]),
            y_position - 0.38,
            y_position + 0.38,
            color=reporting["palette"]["neutral"],
            linestyle="--",
            linewidth=float(style["zero_line_width"]),
            zorder=1,
        )
        question_values = pd.Series(plot_row["values"], dtype=float)
        if not question_values.empty:
            generator = np.random.default_rng(
                config.derived_seed(f"rq6.jitter.{plot_row['question_id']}")
            )
            jitter = generator.uniform(
                -float(style["rq6_raw_point_jitter"]),
                float(style["rq6_raw_point_jitter"]),
                len(question_values),
            )
            group_axis.scatter(
                question_values,
                y_position + jitter,
                s=float(style["rq6_raw_point_size"]),
                color=reporting["palette"]["neutral"],
                alpha=float(style["rq6_raw_point_alpha"]),
                linewidths=0.0,
                zorder=2,
            )
            group_axis.plot(
                float(plot_row["mean"]),
                y_position,
                marker="D",
                color=reporting["palette"]["primary"],
                markersize=float(style["rq6_mean_marker_size"]),
                linestyle="none",
                zorder=4,
            )
        else:
            group_axis.text(
                0.02,
                y_position,
                "Outcome pending",
                ha="left",
                va="center",
                fontsize=float(style["rq6_forecast_annotation_size"]),
                color=reporting["palette"]["neutral"],
                fontstyle="italic",
            )
        group_axis.text(
            float(style["rq6_correct_column_x"]),
            y_position,
            str(plot_row["correct"]),
            transform=group_axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=float(style["rq6_forecast_annotation_size"]),
        )

    group_count = len(groups)
    group_axis.axhline(
        group_count - 0.5,
        color=reporting["palette"]["neutral_light"],
        linewidth=float(style["grid_line_width"]),
    )
    group_axis.text(
        float(style["rq6_correct_column_x"]),
        float(style["rq6_correct_header_y"]),
        "Correct picks",
        transform=group_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=float(style["rq6_forecast_annotation_size"]),
        fontweight="bold",
    )
    group_axis.set_yticks(y_positions, [str(row["label"]) for row in plot_rows])
    _add_country_flags(group_axis, plot_rows, style, reporting)
    group_axis.invert_yaxis()
    group_axis.set_xlim(0.0, 1.0)
    group_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    group_axis.tick_params(
        axis="x",
        labelsize=float(style["rq6_forecast_tick_label_size"]),
    )
    group_axis.set_xlabel(
        "Probability assigned to realized outcome",
        fontsize=float(style["rq6_forecast_axis_label_size"]),
    )
    group_axis.set_title("")
    add_numeric_grid(group_axis, config, "x")

    group_reference = float(plot_rows[0]["random_reference"])
    semifinal_reference = float(
        next(row["random_reference"] for row in plot_rows if row["question_id"] == semifinal_id)
    )
    tournament_reference = float(
        next(
            row["random_reference"]
            for row in plot_rows
            if row["question_id"] == str(settings["champion_question_id"])
        )
    )
    group_axis.text(
        group_reference,
        float(style["rq6_group_baseline_y"]),
        f"Random guess ({group_reference:.0%})",
        ha="center",
        va="bottom",
        fontsize=float(style["rq6_forecast_annotation_size"]),
        color=reporting["palette"]["neutral"],
    )
    group_axis.text(
        semifinal_reference + 0.01,
        group_count - 0.42,
        f"{semifinal_reference:.1%}",
        ha="left",
        va="center",
        fontsize=float(style["rq6_forecast_annotation_size"]),
        color=reporting["palette"]["neutral"],
    )
    group_axis.text(
        tournament_reference + 0.01,
        group_count + 0.58,
        f"{tournament_reference:.1%}",
        ha="left",
        va="center",
        fontsize=float(style["rq6_forecast_annotation_size"]),
        color=reporting["palette"]["neutral"],
    )
    forecast_figure.subplots_adjust(bottom=float(style["rq6_forecast_bottom_margin"]))
    save_figure(
        forecast_figure,
        config,
        manifest,
        "rq6_tournament_questions",
        "rq6",
        source_hashes,
    )

    recovery_figure, semifinal_axis = plt.subplots(
        figsize=(
            float(reporting["figure_width_single"]),
            float(style["rq6_recovery_height"]),
        )
    )
    semifinal = resolved[resolved["question_id"].eq(settings["semifinal_question_id"])]
    if not semifinal.empty:
        row = semifinal.iloc[0]
        exact = float(row["exact_set_accuracy"])
        recovery = float(row["correct_count"]) / int(settings["semifinal_k"])
        bar_positions = np.arange(2, dtype=float)
        bars = semifinal_axis.barh(
            bar_positions,
            [exact, recovery],
            color=[reporting["palette"]["primary"], reporting["palette"]["secondary"]],
        )
        value_labels = [
            f"{exact:.1%} · {int(row.exact_sets)}/{int(row.n_forecasts)}",
            f"{recovery:.1%} · {float(row.correct_count):.2f}/4",
        ]
        for bar, value, label in zip(bars, (exact, recovery), value_labels):
            semifinal_axis.text(
                value + float(style["rq6_bar_label_offset"]),
                bar.get_y() + bar.get_height() / 2.0,
                label,
                ha="left",
                va="center",
                fontsize=float(style["tick_label_size"]),
            )
        semifinal_axis.set_yticks(
            bar_positions,
            ["Exact 4-team set", "Teams recovered"],
        )
        semifinal_axis.invert_yaxis()
        semifinal_axis.set_title("Semifinal forecast recovery", pad=float(style["rq6_title_pad"]))
        semifinal_axis.set_xlabel("Share")
        semifinal_axis.set_xlim(0.0, float(style["rq6_recovery_right_limit"]))
        semifinal_axis.set_xticks(np.linspace(0.0, 1.0, 5))
        semifinal_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        add_numeric_grid(semifinal_axis, config, "x")
    semifinal_axis.text(
        0.0,
        float(style["rq6_subtitle_y"]),
        "32 pre-tournament forecasts",
        transform=semifinal_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=float(style["tick_label_size"]),
        color=reporting["palette"]["neutral"],
    )
    recovery_figure.subplots_adjust(bottom=float(style["rq6_recovery_bottom_margin"]))
    save_figure(
        recovery_figure,
        config,
        manifest,
        "rq6_semifinal_recovery",
        "rq6",
        source_hashes,
    )


def _plot_sum_audit(
    audit: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
) -> pd.DataFrame:
    reporting = config.section("reporting")
    style = reporting["style"]
    settings = config.section("special_questions")
    model_order = list(config.section("design")["complete_panel"])
    model_summary = audit.groupby("model_id", as_index=False).agg(
        forecasts=("prediction_id", "size"),
        mean_probability_sum=("semifinal_probability_sum", "mean"),
        minimum_probability_sum=("semifinal_probability_sum", "min"),
        maximum_probability_sum=("semifinal_probability_sum", "max"),
        mean_absolute_deviation=("absolute_sum_from_expected_k", "mean"),
    )
    apply_style(config)
    figure, axis = plt.subplots(
        figsize=(
            float(reporting["figure_width_double"]),
            float(style["rq6_sum_audit_height"]),
        )
    )
    y_positions = np.arange(len(model_order), dtype=float)
    for y_position, model_id in zip(y_positions, model_order):
        values = audit.loc[
            audit["model_id"].eq(model_id), "semifinal_probability_sum"
        ].sort_values()
        offsets = np.linspace(
            -float(style["rq6_sum_point_jitter"]),
            float(style["rq6_sum_point_jitter"]),
            len(values),
        )
        axis.scatter(
            values,
            y_position + offsets,
            s=float(style["rq6_sum_point_size"]),
            color=reporting["palette"]["neutral"],
            alpha=float(style["rq6_sum_point_alpha"]),
            linewidths=0.0,
            zorder=2,
        )
        mean_value = model_summary.loc[
            model_summary["model_id"].eq(model_id), "mean_probability_sum"
        ].iloc[0]
        axis.plot(
            mean_value,
            y_position,
            marker="D",
            color=reporting["palette"]["primary"],
            markersize=float(style["rq6_sum_mean_marker_size"]),
            linestyle="none",
            zorder=4,
        )
    axis.axvline(
        float(settings["semifinal_k"]),
        color=reporting["palette"]["neutral"],
        linestyle="--",
        linewidth=float(style["zero_line_width"]),
    )
    axis.set_yticks(y_positions, [""] * len(model_order))
    _add_sum_audit_model_icons(axis, model_order, style)
    axis.invert_yaxis()
    axis.set_xlim(left=0.0)
    axis.tick_params(
        axis="x",
        labelsize=float(style["rq6_sum_tick_label_size"]),
    )
    axis.set_xlabel(
        "Sum across 48 semifinalist inclusion probabilities",
        fontsize=float(style["rq6_sum_axis_label_size"]),
    )
    axis.set_title("")
    axis.text(
        float(settings["semifinal_k"]) + float(style["rq6_sum_target_label_offset"]),
        0.02,
        "Coherent target = 4",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=float(style["rq6_sum_annotation_size"]),
        color=reporting["palette"]["neutral"],
    )
    add_numeric_grid(axis, config, "x")
    figure.subplots_adjust(**style["rq6_sum_margins"])
    save_figure(
        figure,
        config,
        manifest,
        "fig_rq6_semifinal_probability_sums_appendix",
        "rq6",
        source_hashes,
    )
    return model_summary


def run(config: AnalysisConfig, manifest: Manifest) -> dict[str, pd.DataFrame]:
    manifest.discard_generated(
        [
            "rq6_forecast_level_diagnostics_csv",
            "rq6_forecast_level_diagnostics_companion",
        ]
    )
    prediction_record = manifest.require("derived_special_predictions")
    option_record = manifest.require("derived_special_options")
    outcome_record = manifest.require("derived_special_outcomes")
    predictions = pd.read_parquet(prediction_record.path)
    options = pd.read_parquet(option_record.path)
    outcomes = pd.read_parquet(outcome_record.path)
    source_hashes = {
        "special_predictions": prediction_record.sha256,
        "special_options": option_record.sha256,
        "special_outcomes": outcome_record.sha256,
    }
    _validate_normalized_inputs(predictions, options, outcomes, config)
    valid = _analysis_prediction_panel(predictions, config)
    joined = options.merge(
        valid[["prediction_id", "question_id"]],
        on=["prediction_id", "question_id"],
        how="inner",
        validate="many_to_one",
    )
    settings = config.section("special_questions")
    observed_questions = sorted(predictions["question_id"].dropna().unique())
    epsilon = float(config.section("statistics")["log_loss_epsilon"])
    tolerance = float(config.section("validation")["probability_tolerance"])
    expected_k = int(settings["semifinal_k"])
    summary_records: list[dict[str, object]] = []
    prediction_metrics: list[pd.DataFrame] = []
    for question_id in observed_questions:
        question_options = joined[joined["question_id"].eq(question_id)]
        question_outcomes = outcomes[outcomes["question_id"].eq(question_id)]
        realized = question_outcomes.loc[question_outcomes["is_realized"], "candidate_id"].astype(
            str
        )
        required_realized = expected_k if question_id == settings["semifinal_question_id"] else 1
        if len(realized) != required_realized:
            summary_records.append(
                {"question_id": question_id, "status": "unresolved", "analysis_unit_weight": 1.0}
            )
            continue
        if question_id == settings["semifinal_question_id"]:
            metrics = _semifinalists(question_options, question_outcomes, expected_k)
            metrics["question_id"] = question_id
            prediction_metrics.append(metrics)
            summary_records.append(
                {
                    "question_id": question_id,
                    "status": "resolved",
                    "analysis_unit_weight": 1.0,
                    "realized_candidate": " | ".join(sorted(realized)),
                    "exact_set_accuracy": metrics["exact_set_accuracy"].mean(),
                    "exact_sets": int(metrics["exact_set_accuracy"].sum()),
                    "correct_count": metrics["correct_count"].mean(),
                    "hamming_loss": metrics["hamming_loss"].mean(),
                    "marginal_brier": metrics["marginal_brier"].mean(),
                    "n_forecasts": len(metrics),
                }
            )
        else:
            metrics = _single_choice(question_options, question_outcomes, epsilon, tolerance)
            metrics["question_id"] = question_id
            prediction_metrics.append(metrics)
            summary_records.append(
                {
                    "question_id": question_id,
                    "status": "resolved",
                    "analysis_unit_weight": 1.0,
                    "realized_candidate": realized.iloc[0],
                    "single_choice_accuracy": metrics["accuracy"].mean(),
                    "correct_picks": int(metrics["accuracy"].sum()),
                    "brier": metrics["brier"].mean(),
                    "log_loss": metrics["log_loss"].mean(),
                    "realized_option_probability": metrics["realized_option_probability"].mean(),
                    "n_forecasts": len(metrics),
                }
            )
    summary = pd.DataFrame(summary_records)
    detailed = (
        pd.concat(prediction_metrics, ignore_index=True) if prediction_metrics else pd.DataFrame()
    )
    if not detailed.empty:
        metadata = valid[
            [
                "prediction_id",
                "model_id",
                "access_condition",
                "prompt_strategy",
                "forecast_horizon",
                "sample_id",
            ]
        ]
        detailed = detailed.merge(metadata, on="prediction_id", how="left", validate="one_to_one")
    resolved_count = int(summary["status"].eq("resolved").sum())
    expected = int(settings["expected_questions"])
    if config.is_final and resolved_count != expected:
        unresolved = summary.loc[summary["status"].ne("resolved"), "question_id"].tolist()
        raise ValueError(
            f"Final mode requires all {expected} tournament questions to be resolved: {unresolved}"
        )

    result_dir = config.resolve_path("results") / "rq6"
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("question_summary", summary), ("forecast_level_diagnostics", detailed)):
        path = result_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        manifest.add(f"rq6_{name}", path, "parquet", "rq6", source_hashes, {"rows": len(frame)})

    single_display, semifinal_display, aggregate_display = _summary_tables(summary, config)
    save_table(
        single_display,
        config,
        manifest,
        "rq6_question_level_results",
        "rq6",
        source_hashes,
    )
    save_table(
        semifinal_display,
        config,
        manifest,
        "rq6_semifinal_set_results",
        "rq6",
        source_hashes,
    )
    save_table(
        aggregate_display,
        config,
        manifest,
        "rq6_group_winner_aggregate",
        "rq6",
        source_hashes,
    )

    semifinal_audit = detailed[detailed["question_id"].eq(settings["semifinal_question_id"])][
        [
            "prediction_id",
            "model_id",
            "access_condition",
            "prompt_strategy",
            "semifinal_probability_sum",
            "absolute_sum_from_expected_k",
        ]
    ].copy()
    model_audit = pd.DataFrame()
    if not semifinal_audit.empty:
        audit_path = result_dir / "semifinal_probability_sum_audit.parquet"
        semifinal_audit.to_parquet(audit_path, index=False)
        manifest.add(
            "rq6_semifinal_probability_sum_audit_raw",
            audit_path,
            "parquet",
            "rq6",
            source_hashes,
            {"rows": len(semifinal_audit)},
        )
        semifinal_audit_summary = pd.DataFrame(
            [
                {
                    "Forecasts": len(semifinal_audit),
                    "Mean probability sum": semifinal_audit["semifinal_probability_sum"].mean(),
                    "Median probability sum": semifinal_audit["semifinal_probability_sum"].median(),
                    "Minimum probability sum": semifinal_audit["semifinal_probability_sum"].min(),
                    "Maximum probability sum": semifinal_audit["semifinal_probability_sum"].max(),
                    "Mean absolute deviation from four": semifinal_audit[
                        "absolute_sum_from_expected_k"
                    ].mean(),
                    "Maximum absolute deviation from four": semifinal_audit[
                        "absolute_sum_from_expected_k"
                    ].max(),
                }
            ]
        )
        save_table(
            semifinal_audit_summary,
            config,
            manifest,
            "rq6_semifinal_probability_sum_audit",
            "rq6",
            source_hashes,
        )
        model_audit = _plot_sum_audit(semifinal_audit, config, manifest, source_hashes)
        model_audit_display = model_audit.assign(
            Model=model_audit["model_id"].map(lambda value: model_label(config, value)),
            **{
                "Forecasts": model_audit["forecasts"],
                "Mean probability sum": model_audit["mean_probability_sum"],
                "Minimum probability sum": model_audit["minimum_probability_sum"],
                "Maximum probability sum": model_audit["maximum_probability_sum"],
                "Mean absolute deviation from four": model_audit["mean_absolute_deviation"],
            },
        )[
            [
                "Model",
                "Forecasts",
                "Mean probability sum",
                "Minimum probability sum",
                "Maximum probability sum",
                "Mean absolute deviation from four",
            ]
        ]
        save_table(
            model_audit_display,
            config,
            manifest,
            "rq6_semifinal_probability_sums_by_model",
            "rq6",
            source_hashes,
        )

    _plot_main(summary, detailed, joined, outcomes, config, manifest, source_hashes)

    records = [
        headline_record(
            config,
            "rq6_headlines",
            "resolved_questions",
            "resolved tournament questions among 15 prespecified analysis units",
            source_hashes,
            estimate=resolved_count,
            ci_low=None,
            ci_high=None,
            p_raw=None,
            p_adjusted=None,
            median=None,
            n_matches=None,
            n_predictions=len(valid),
            units="questions",
            aggregation="one equal-weight analysis unit per question",
            extra={"null_reason": "prospective case-study status count"},
        )
    ]
    groups = summary[
        summary["question_id"].str.startswith(str(settings["group_question_prefix"]))
        & summary["status"].eq("resolved")
    ]
    for metric, unit in (
        ("single_choice_accuracy", "share"),
        ("brier", "score"),
        ("log_loss", "nats"),
        ("realized_option_probability", "share"),
    ):
        records.append(
            headline_record(
                config,
                "rq6_headlines",
                f"group_winners.{metric}",
                f"equal-weight mean question-level {metric}",
                source_hashes,
                estimate=float(groups[metric].mean()),
                ci_low=None,
                ci_high=None,
                p_raw=None,
                p_adjusted=None,
                median=float(groups[metric].median()),
                n_matches=None,
                n_predictions=int(groups["n_forecasts"].sum()),
                units=unit,
                aggregation="equal-weight mean over resolved group-winner questions",
                extra={
                    "n_questions": len(groups),
                    "null_reason": "descriptive prospective case study",
                },
            )
        )
    semifinal = summary[
        summary["question_id"].eq(settings["semifinal_question_id"])
        & summary["status"].eq("resolved")
    ]
    if not semifinal.empty:
        row = semifinal.iloc[0]
        for metric, unit in (
            ("exact_set_accuracy", "share"),
            ("correct_count", "teams of four"),
            ("hamming_loss", "share of 48 candidates"),
            ("marginal_brier", "score"),
        ):
            records.append(
                headline_record(
                    config,
                    "rq6_headlines",
                    f"semifinalists.{metric}",
                    f"semifinalist-set {metric}",
                    source_hashes,
                    estimate=float(row[metric]),
                    ci_low=None,
                    ci_high=None,
                    p_raw=None,
                    p_adjusted=None,
                    median=None,
                    n_matches=None,
                    n_predictions=int(row["n_forecasts"]),
                    units=unit,
                    aggregation="descriptive mean over semifinal-set forecasts",
                    extra={
                        "n_questions": 1,
                        "null_reason": "descriptive prospective case study",
                    },
                )
            )
    if not semifinal_audit.empty:
        records.append(
            headline_record(
                config,
                "rq6_headlines",
                "semifinalists.probability_sum",
                "sum of 48 supplied semifinalist marginal probabilities",
                source_hashes,
                estimate=float(semifinal_audit["semifinal_probability_sum"].mean()),
                ci_low=None,
                ci_high=None,
                p_raw=None,
                p_adjusted=None,
                median=float(semifinal_audit["semifinal_probability_sum"].median()),
                n_matches=None,
                n_predictions=len(semifinal_audit),
                units="expected teams; target sum four",
                aggregation="descriptive mean, median, and range over 32 forecasts",
                extra={
                    "minimum": float(semifinal_audit["semifinal_probability_sum"].min()),
                    "maximum": float(semifinal_audit["semifinal_probability_sum"].max()),
                    "null_reason": "descriptive coherence audit; no inferential interval",
                },
            )
        )
    write_headlines(config, manifest, "rq6_headlines", records, "rq6", source_hashes)
    manifest.write()
    return {
        "summary": summary,
        "details": detailed,
        "semifinal_audit": semifinal_audit,
        "model_audit": model_audit,
    }
