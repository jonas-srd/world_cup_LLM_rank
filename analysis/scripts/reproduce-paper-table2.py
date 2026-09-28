#!/usr/bin/env python3
"""Reproduce the frozen July 21 World Cup paper's Table 2 without network access.

Requires Python 3.12 and numpy==2.5.1. Reads the byte-exact original CSV, verifies
its SHA-256, recomputes metrics, validates the complete design, and checks every
displayed Table 2 value (including the 95% confidence limits).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import sys
from pathlib import Path

CSV_SHA256 = "745c6ff9c38c258204ed4ea4839cd5e1ca4dd0794b18983d6bc7bb3f0f24a681"
NUMPY_VERSION = "2.5.1"
MASTER_SEED = 20260715
REPLICATES = 10000
EPSILON = 1e-15
TIE_TOLERANCE = 1e-6
MODELS = {
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "openai/gpt-5.5": "GPT-5.5",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "x-ai/grok-4.3": "Grok 4.3",
    "qwen/qwen3.7-max": "Qwen 3.7 Max",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "mistralai/mistral-large-2512": "Mistral Large 2512",
}
# Metric order and names preserve the original analysis' independent RNG streams.
METRICS = (
    ("brier_90_recomputed", "Brier [95% CI]", 1, 3),
    ("log_loss_90_recomputed", "Log loss [95% CI]", 1, 3),
    ("top_outcome_accuracy_90_recomputed", "Modal H/D/A accuracy, % [95% CI]", 100, 1),
    ("exact_score_90_correct_recomputed", "Exact score, % [95% CI]", 100, 1),
    ("kicktipp_points_90_recomputed", "Scoring System [95% CI]", 1, 2),
)
# Transcribed from the frozen paper source, main.tex, label tab:leaderboard.
# These are verification targets only; they never enter metric or CI calculations.
PAPER_VALUES = {
    "google/gemini-3.1-pro-preview": ["0.506 [0.433, 0.587]", "0.853 [0.757, 0.963]", "63.7 [54.0, 72.6]", "15.4 [9.8, 22.8]", "1.36 [1.10, 1.68]"],
    "openai/gpt-5.5": ["0.517 [0.454, 0.584]", "0.873 [0.789, 0.963]", "62.0 [52.4, 70.9]", "15.4 [10.0, 22.5]", "1.39 [1.12, 1.70]"],
    "deepseek/deepseek-v4-pro": ["0.520 [0.452, 0.596]", "0.877 [0.781, 0.980]", "63.7 [54.2, 72.4]", "13.9 [9.0, 20.6]", "1.27 [1.02, 1.56]"],
    "x-ai/grok-4.3": ["0.524 [0.455, 0.600]", "0.876 [0.781, 0.979]", "63.5 [53.6, 72.4]", "11.5 [7.0, 18.1]", "1.19 [0.96, 1.48]"],
    "qwen/qwen3.7-max": ["0.527 [0.463, 0.597]", "0.887 [0.803, 0.978]", "61.1 [51.1, 70.1]", "13.5 [8.6, 20.3]", "1.24 [0.99, 1.54]"],
    "anthropic/claude-opus-4.8": ["0.528 [0.465, 0.597]", "0.888 [0.803, 0.982]", "63.0 [53.2, 71.8]", "14.9 [9.5, 22.3]", "1.25 [0.98, 1.57]"],
    "mistralai/mistral-large-2512": ["0.546 [0.493, 0.604]", "0.913 [0.839, 0.989]", "59.6 [50.2, 68.1]", "12.5 [7.6, 19.2]", "1.17 [0.93, 1.47]"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(directory: Path) -> None:
    """Verify standard SHA256SUMS files, rejecting absolute/traversing entries."""
    root = directory.resolve()
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = (root / relative).resolve()
        require(path.is_relative_to(root), f"Checksum path outside release: {relative}")
        require(len(digest) == 64 and sha256(path) == digest.lower(), f"SHA-256 mismatch: {relative}")
        count += 1
    require(count > 0, "SHA256SUMS is empty")
    print(f"Verified SHA-256 for {count} release files.")


def read_panel(path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    actual_sha = sha256(path)
    require(actual_sha == CSV_SHA256, f"Frozen CSV SHA-256 mismatch: {actual_sha}")
    reader = csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig")))
    headers = next(reader)
    require(len(headers) == 63, "Expected the original 63-column CSV header")
    rows = []
    record_ids: set[str] = set()
    special_count = 0
    for number, values in enumerate(reader, start=2):
        require(len(values) >= 2, f"Malformed record {number}")
        require(values[1] not in record_ids, f"Duplicate prediction ID at record {number}")
        record_ids.add(values[1])
        if values[0] == "special_question_prediction":
            # Original export omitted the empty Actual advancer field in these
            # 480 rows. They do not participate in match performance / Table 2.
            require(len(values) == 62, f"Unexpected original special row width: {number}")
            special_count += 1
            continue
        require(values[0] == "match_prediction" and len(values) == 63, f"Malformed match record {number}")
        rows.append(dict(zip(headers, values)))
    require(len(rows) == 9984 and special_count == 480, "Frozen release row counts changed")
    panel = [row for row in rows if row["Predictor ID"] in MODELS]
    require(len(panel) == 8736, "Expected 8,736 main-panel forecasts")
    require(all(row["Valid for scoring"] == "true" for row in panel), "Invalid main-panel forecast")
    fixtures: dict[str, tuple[str, str, str, str, str, str]] = {}
    observed = set()
    for row in panel:
        fixture = tuple(row[key] for key in ("Stage", "Match date", "Home team", "Away team", "Actual home 90", "Actual away 90"))
        mid = row["Match ID"]
        require(mid not in fixtures or fixtures[mid] == fixture, f"Inconsistent fixture/outcome {mid}")
        fixtures[mid] = fixture
        key = (mid, row["Predictor ID"], row["Horizon"], row["Access"], row["Prompt"], row["Sample ID"])
        require(key not in observed, f"Duplicate design cell: {key}")
        observed.add(key)
        recompute(row)  # Reconcile every evaluated forecast, not only Table 2.
    expected = set(itertools.product(
        fixtures, MODELS, ("STAGE_OPENING", "T_24H", "T_2H"),
        ("Closed Book", "Open Book"), ("Direct Score", "Probabilistic Forecast"), ("1",),
    ))
    require(len(fixtures) == 104 and observed == expected, "Incomplete factorial panel")
    require(sum(value[0] == "Group Stage" for value in fixtures.values()) == 72, "Expected 72 group matches")
    return panel, {
        "csv_sha256": actual_sha, "all_records": len(record_ids),
        "match_attempts": len(rows), "tournament_records": special_count,
        "evaluated_match_records": len(panel), "matches": len(fixtures),
        "metric_reconciliation_absolute_tolerance": 1e-9,
        "probability_sum_tolerance": 1e-6,
        "special_csv_defect": "Original special rows omit empty Actual advancer column; excluded from Table 2. See corrected snapshot.csv.",
    }


def recompute(row: dict[str, str]) -> tuple[float, ...]:
    probabilities = tuple(float(row[key]) for key in ("Home win 90 prob", "Draw 90 prob", "Away win 90 prob"))
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), "Invalid probability")
    require(abs(sum(probabilities) - 1) <= 1e-6, "Probability vector does not sum to one")
    home, away, predicted_home, predicted_away = (
        int(row[key]) for key in ("Actual home 90", "Actual away 90", "Predicted home 90", "Predicted away 90")
    )
    require(min(home, away, predicted_home, predicted_away) >= 0, "Negative score")
    actual = 0 if home > away else 2 if home < away else 1
    brier = sum((p - int(i == actual)) ** 2 for i, p in enumerate(probabilities))
    logloss = -math.log(min(max(probabilities[actual], EPSILON), 1 - EPSILON))
    maximum = max(probabilities)
    # Paper's modal accuracy uses equal credit for ties within 1e-6. Stored export's
    # boolean argmax result is intentionally not used for this metric.
    tied = [abs(p - maximum) <= TIE_TOLERANCE for p in probabilities]
    modal = (1 / sum(tied)) if tied[actual] else 0.0
    exact = float((home, away) == (predicted_home, predicted_away))
    delta = home - away
    predicted_delta = predicted_home - predicted_away
    tendency = (delta > 0) - (delta < 0) == (predicted_delta > 0) - (predicted_delta < 0)
    points = 5.0 if exact else 2.0 if delta == predicted_delta else 1.0 if tendency else 0.0
    for column, value in (("Brier 90", brier), ("Log loss 90", logloss), ("Points 90", points)):
        require(abs(float(row[column]) - value) <= 1e-9, f"Stored metric mismatch for {row['Prediction ID']}: {column}")
    require((row["Exact score correct 90"] == "true") == bool(exact), "Stored exact-score metric mismatch")
    return brier, logloss, modal, exact, points


def bootstrap(values, group_mask, analysis_id: str, np) -> dict[str, float | int | str]:
    """Original stratified bootstrap-t, preserving within-match paired cells."""
    seed = int.from_bytes(hashlib.sha256(f"{MASTER_SEED}:{analysis_id}".encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    indices = (np.flatnonzero(group_mask), np.flatnonzero(~group_mask))
    sizes = tuple(len(index) for index in indices)

    def standard_error(parts):
        return float(np.sqrt(sum((len(part) / len(values)) ** 2 * np.var(part, ddof=1) / len(part) for part in parts)))

    estimate = float(np.mean(values))
    se = standard_error([values[index] for index in indices])
    require(se > 0, f"Non-positive standard error for {analysis_id}")
    t_star = np.empty(REPLICATES)
    for replicate in range(REPLICATES):
        sampled = np.concatenate([values[rng.choice(index, size=size, replace=True)] for index, size in zip(indices, sizes)])
        sampled_se = standard_error((sampled[:sizes[0]], sampled[sizes[0]:]))
        t_star[replicate] = 0 if sampled_se <= 0 else (np.mean(sampled) - estimate) / sampled_se
    # Match the original confidence-tail expression, including floating-point
    # arithmetic, and NumPy's default linear-interpolation quantile convention.
    tail = (1 - 0.95) / 2
    lower, upper = np.quantile(t_star, [tail, 1 - tail])
    return {
        "analysis_id": analysis_id, "estimate": estimate,
        "ci_low": float(estimate - upper * se), "ci_high": float(estimate - lower * se),
        "standard_error": se, "seed": seed, "n_matches": len(values),
    }


def reproduce(csv_path: Path, output: Path) -> None:
    import numpy as np

    require(np.__version__ == NUMPY_VERSION, f"Install numpy=={NUMPY_VERSION}; found {np.__version__}")
    panel, validation = read_panel(csv_path)
    table_rows = [row for row in panel if row["Horizon"] == "T_24H" and row["Prompt"] == "Probabilistic Forecast"]
    require(len(table_rows) == 1456, "Expected 1,456 Table 2 input forecasts")
    output.mkdir(parents=True, exist_ok=True)
    for name, cohort in (("evaluated-prediction-ids.txt", panel), ("table2-prediction-ids.txt", table_rows)):
        (output / name).write_bytes(("\n".join(sorted(row["Prediction ID"] for row in cohort)) + "\n").encode())
    summary, results, discrepancies = [], [], []
    for model, label in MODELS.items():
        by_match = collections.defaultdict(list)
        for row in table_rows:
            if row["Predictor ID"] == model:
                by_match[(row["Match ID"], row["Stage"])].append(recompute(row))
        keys = sorted(by_match)
        require(len(keys) == 104 and all(len(by_match[key]) == 2 for key in keys), "Table 2 must pair two access conditions per model-match")
        scores = np.array([np.mean(by_match[key], axis=0) for key in keys])
        group_mask = np.array([key[1] == "Group Stage" for key in keys])
        formatted = {"Model": label}
        for metric_index, (metric, column, scale, decimals) in enumerate(METRICS):
            result = bootstrap(scores[:, metric_index], group_mask, f"overall.{model}.{metric}", np)
            shown = f"{result['estimate'] * scale:.{decimals}f} [{result['ci_low'] * scale:.{decimals}f}, {result['ci_high'] * scale:.{decimals}f}]"
            formatted[column] = shown
            expected = PAPER_VALUES[model][metric_index]
            if shown != expected:
                discrepancies.append({"model_id": model, "metric": metric, "paper": expected, "recomputed": shown})
            results.append({"model_id": model, "metric": metric, **result, "display_scale": scale, "display_decimals": decimals, "display": shown, "paper_display": expected, "matches_paper": shown == expected})
        formatted["n"] = 104
        summary.append(formatted)
    with (output / "table2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    report = {
        "release_id": "worldcup2026-2026-07-21-v1",
        "validation": {**validation, "table2_records": len(table_rows), "display_cells_checked": 35, "numeric_estimates_and_ci_limits_checked": 105, "matches_paper_at_displayed_precision": not discrepancies},
        "method": {
            "horizon": "T_24H", "prompt": "Probabilistic Forecast",
            "aggregation": "Recompute each forecast metric; equal average across closed/open access within model-match; equal average across 104 matches.",
            "interval": "95% studentized (bootstrap-t), matches resampled within 72 group-stage and 32 knockout strata",
            "replicates": REPLICATES, "master_seed": MASTER_SEED,
            "derived_seed": "First four SHA-256 bytes of UTF-8 '20260715:overall.<model_id>.<metric>' interpreted as unsigned big-endian integer",
            "generator": "NumPy default_rng / PCG64", "numpy_version": NUMPY_VERSION,
            "match_order": "Lexicographically ascending Match ID; group stratum sampled before knockout stratum",
            "log_loss_epsilon": EPSILON, "modal_ties": "Maximum-probability ties within 1e-6 receive 1/k credit",
            "verification_tolerance": "Exact equality after rounding to paper precision: Brier/log loss 3 decimals, accuracy percentages 1 decimal, points 2 decimals",
        },
        "cohorts": {name: {"sha256": sha256(output / name), "records": size} for name, size in (("evaluated-prediction-ids.txt", 8736), ("table2-prediction-ids.txt", 1456))},
        "results": results, "discrepancies": discrepancies,
    }
    (output / "table2.json").write_bytes((json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    for row in summary:
        print(" | ".join(str(value) for value in row.values()))
    require(not discrepancies, f"{len(discrepancies)} Table 2 cells differ; see table2.json. No results were substituted.")
    print("PASS: 8,736 evaluated records; 1,456 Table 2 records; all 105 estimates/CI limits match the paper at displayed precision.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Byte-exact snapshot-original.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("table2-reproduced"))
    parser.add_argument("--verify-checksums", type=Path, metavar="RELEASE_DIR", help="Verify every file in RELEASE_DIR/SHA256SUMS using the standard library")
    args = parser.parse_args()
    if args.csv is None and args.verify_checksums is None:
        parser.error("provide --csv and/or --verify-checksums")
    try:
        if args.verify_checksums is not None:
            verify_checksums(args.verify_checksums)
        if args.csv is not None:
            reproduce(args.csv, args.output_dir)
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
