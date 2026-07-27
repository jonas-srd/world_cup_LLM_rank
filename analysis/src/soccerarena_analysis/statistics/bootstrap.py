from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import AnalysisConfig


@dataclass(frozen=True)
class BootstrapResult:
    analysis_id: str
    estimate: float
    ci_low: float
    ci_high: float
    p_raw: float | None
    median: float
    standard_error: float
    n_matches: int
    replicates: int
    permutation_replicates: int
    alternative: str

    def as_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def _stratum(stage: str, config: AnalysisConfig) -> str:
    strata = config.section("statistics")["strata"]
    for name, stages in strata.items():
        if stage in stages:
            return name
    raise ValueError(f"Stage {stage!r} is not assigned to a bootstrap stratum")


def _stratified_se(values: np.ndarray, labels: np.ndarray) -> float:
    n_total = len(values)
    variance = 0.0
    for label in np.unique(labels):
        subset = values[labels == label]
        if len(subset) <= 1:
            continue
        weight = len(subset) / n_total
        variance += (weight**2) * np.var(subset, ddof=1) / len(subset)
    return float(np.sqrt(max(variance, 0.0)))


def studentized_cluster_bootstrap(
    frame: pd.DataFrame,
    value_col: str,
    stage_col: str,
    config: AnalysisConfig,
    analysis_id: str,
    alternative: str = "two-sided",
    test_null: bool = True,
) -> BootstrapResult:
    """Bootstrap-t CI plus a sign-flip test of the identical match-level mean."""
    required = {"match_id", value_col, stage_col}
    if not required.issubset(frame.columns):
        raise KeyError(f"Missing bootstrap columns: {sorted(required - set(frame.columns))}")
    clean = frame[["match_id", value_col, stage_col]].dropna().copy()
    if clean.empty:
        raise ValueError(f"No observations for {analysis_id}")
    if clean["match_id"].duplicated().any():
        raise ValueError(f"{analysis_id} must contain exactly one value per match")
    clean["_stratum"] = clean[stage_col].map(lambda value: _stratum(str(value), config))
    values = clean[value_col].to_numpy(dtype=float)
    labels = clean["_stratum"].to_numpy(dtype=str)
    estimate = float(np.mean(values))
    standard_error = _stratified_se(values, labels)
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError(f"Non-positive standard error for {analysis_id}")

    settings = config.section("statistics")
    replicates = int(settings["bootstrap_replicates"])
    rng = np.random.default_rng(config.derived_seed(analysis_id))
    t_star = np.empty(replicates, dtype=float)
    unique_labels = np.unique(labels)
    index_by_label = {label: np.flatnonzero(labels == label) for label in unique_labels}

    for replicate in range(replicates):
        sampled_parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in index_by_label.values()
        ]
        sampled_index = np.concatenate(sampled_parts)
        sampled_values = values[sampled_index]
        sampled_labels = labels[sampled_index]
        sampled_se = _stratified_se(sampled_values, sampled_labels)
        t_star[replicate] = (
            0.0 if sampled_se <= 0 else (np.mean(sampled_values) - estimate) / sampled_se
        )

    confidence = float(settings["confidence_level"])
    tail = (1.0 - confidence) / 2.0
    q_low, q_high = np.quantile(t_star, [tail, 1.0 - tail])
    ci_low = float(estimate - q_high * standard_error)
    ci_high = float(estimate - q_low * standard_error)
    permutation_replicates = int(settings["permutation_replicates"])
    p_raw: float | None = None
    if test_null:
        permutation_rng = np.random.default_rng(config.derived_seed(f"{analysis_id}.sign_flip"))
        permuted = np.empty(permutation_replicates, dtype=float)
        for replicate in range(permutation_replicates):
            signs = permutation_rng.choice(np.asarray([-1.0, 1.0]), size=len(values), replace=True)
            permuted[replicate] = float(np.mean(values * signs))
        if alternative == "two-sided":
            exceedances = int(np.count_nonzero(np.abs(permuted) >= abs(estimate)))
        elif alternative == "greater":
            exceedances = int(np.count_nonzero(permuted >= estimate))
        elif alternative == "less":
            exceedances = int(np.count_nonzero(permuted <= estimate))
        else:
            raise ValueError("alternative must be two-sided, greater, or less")
        p_raw = float((1.0 + exceedances) / (permutation_replicates + 1.0))
    return BootstrapResult(
        analysis_id=analysis_id,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_raw=p_raw,
        median=float(np.median(values)),
        standard_error=standard_error,
        n_matches=len(values),
        replicates=replicates,
        permutation_replicates=permutation_replicates,
        alternative=alternative,
    )


def leave_one_match_out(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) <= 1:
        raise ValueError("Leave-one-match-out requires at least two matches")
    total = float(clean.sum())
    estimates = (total - clean) / (len(clean) - 1)
    return {
        "n_matches": len(clean),
        "minimum": float(estimates.min()),
        "maximum": float(estimates.max()),
    }
