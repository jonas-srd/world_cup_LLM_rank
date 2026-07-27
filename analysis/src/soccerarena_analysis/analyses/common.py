from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from ..config import AnalysisConfig
from ..manifest import Manifest
from ..statistics.bootstrap import BootstrapResult, studentized_cluster_bootstrap
from ..statistics.multiplicity import holm_adjust


def load_panel(config: AnalysisConfig, manifest: Manifest) -> tuple[pd.DataFrame, dict[str, str]]:
    record = manifest.require("derived_analysis_panel")
    return pd.read_parquet(Path(record.path)), {"analysis_panel": record.sha256}


def primary_panel(
    frame: pd.DataFrame, config: AnalysisConfig, include_fable: bool = False
) -> pd.DataFrame:
    design = config.section("design")
    models = list(design["complete_panel"])
    if include_fable:
        models.append(design["partial_models"]["fable"])
    return frame[
        frame["model_id"].isin(models)
        & (frame["forecast_horizon"] == design["primary_horizon"])
        & frame["is_valid_for_scoring"].fillna(False)
        & frame["actual_result_90"].notna()
    ].copy()


def match_level_factorial_contrasts(
    frame: pd.DataFrame, config: AnalysisConfig
) -> dict[str, pd.DataFrame]:
    primary = primary_panel(frame, config)
    cells = primary.groupby(
        ["match_id", "stage", "model_id", "access_condition", "prompt_strategy"], as_index=False
    )["brier_90_recomputed"].mean()
    pivot = cells.pivot_table(
        index=["match_id", "stage", "model_id"],
        columns=["access_condition", "prompt_strategy"],
        values="brier_90_recomputed",
    )
    closed_direct = pivot[("closed_book", "direct_score")]
    closed_prob = pivot[("closed_book", "probabilistic_forecast")]
    open_direct = pivot[("open_book", "direct_score")]
    open_prob = pivot[("open_book", "probabilistic_forecast")]
    unit = pd.DataFrame(
        {
            "access": ((closed_direct + closed_prob) / 2.0) - ((open_direct + open_prob) / 2.0),
            "prompt": ((closed_prob + open_prob) / 2.0) - ((closed_direct + open_direct) / 2.0),
            "access_prompt_interaction": (open_prob - open_direct) - (closed_prob - closed_direct),
        }
    ).reset_index()
    return {
        name: unit.groupby(["match_id", "stage"], as_index=False)[name].mean()
        for name in ("access", "prompt", "access_prompt_interaction")
    }


def factorial_results(frame: pd.DataFrame, config: AnalysisConfig) -> dict[str, dict[str, object]]:
    contrasts = match_level_factorial_contrasts(frame, config)
    raw: dict[str, BootstrapResult] = {
        name: studentized_cluster_bootstrap(values, name, "stage", config, f"factorial.{name}")
        for name, values in contrasts.items()
    }
    adjusted = holm_adjust({name: result.p_raw for name, result in raw.items()})
    return {
        name: {**result.as_dict(), "p_adjusted": adjusted[name]} for name, result in raw.items()
    }


def model_match_scores(frame: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    primary = primary_panel(frame, config)
    return primary.groupby(["match_id", "stage", "kickoff_utc", "model_id"], as_index=False).agg(
        brier_90_recomputed=("brier_90_recomputed", "mean"),
        log_loss_90_recomputed=("log_loss_90_recomputed", "mean"),
        top_outcome_accuracy_90_recomputed=("top_outcome_accuracy_90_recomputed", "mean"),
        exact_score_90_correct_recomputed=("exact_score_90_correct_recomputed", "mean"),
        kicktipp_points_90_recomputed=("kicktipp_points_90_recomputed", "mean"),
    )


def paired_model_results(
    frame: pd.DataFrame, config: AnalysisConfig
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    scores = model_match_scores(frame, config)
    pivot = scores.pivot_table(
        index=["match_id", "stage"], columns="model_id", values="brier_90_recomputed"
    )
    raw_results: list[dict[str, object]] = []
    difference_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for first, second in combinations(config.section("design")["complete_panel"], 2):
        clean = pivot[[first, second]].dropna().copy()
        clean["difference"] = clean[first] - clean[second]
        values = clean.reset_index()[["match_id", "stage", "difference"]]
        result = studentized_cluster_bootstrap(
            values, "difference", "stage", config, f"model_pair.{first}.vs.{second}"
        )
        raw_results.append({"model_a": first, "model_b": second, **result.as_dict()})
        difference_frames[(first, second)] = values
    results = pd.DataFrame(raw_results)
    adjusted = holm_adjust({str(index): value for index, value in results["p_raw"].items()})
    results["p_adjusted"] = [adjusted[str(index)] for index in results.index]
    return results, difference_frames
