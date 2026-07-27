from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AnalysisConfig, sha256_file
from ..manifest import Manifest

PUBLIC_COLUMNS = {
    "Prediction ID": "prediction_id",
    "Match ID": "match_id_public",
    "Home win 90 prob": "home_win_90_prob_public",
    "Draw 90 prob": "draw_90_prob_public",
    "Away win 90 prob": "away_win_90_prob_public",
    "Brier 90": "brier_90_public",
    "Log loss 90": "log_loss_90_public",
    "Valid for scoring": "valid_for_scoring_public",
}


def reconcile_public_export(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    public_path = config.resolve_path("public_csv")
    panel_path = Path(manifest.require("derived_analysis_panel").path)
    public = pd.read_csv(public_path, low_memory=False)
    public = public.loc[public["Record type"] == "match_prediction", list(PUBLIC_COLUMNS)].rename(
        columns=PUBLIC_COLUMNS
    )
    panel = pd.read_parquet(
        panel_path,
        columns=[
            "prediction_id",
            "match_id",
            "home_win_90_prob",
            "draw_90_prob",
            "away_win_90_prob",
            "brier_90_recomputed",
            "log_loss_90_recomputed",
            "is_valid_for_scoring",
        ],
    )
    joined = panel.merge(
        public, on="prediction_id", how="outer", indicator=True, validate="one_to_one"
    )
    discrepancies: list[pd.DataFrame] = []
    missing = joined[joined["_merge"] != "both"].copy()
    if not missing.empty:
        missing["field"] = "record_presence"
        missing["derived_value"] = missing["_merge"].map(
            {"left_only": "derived_only", "right_only": "", "both": ""}
        )
        missing["public_value"] = missing["_merge"].map(
            {"right_only": "public_only", "left_only": "", "both": ""}
        )
        discrepancies.append(missing[["prediction_id", "field", "derived_value", "public_value"]])
    both = joined[joined["_merge"] == "both"].copy()
    tolerance = float(config.section("validation")["reconciliation_absolute_tolerance"])
    comparisons = [
        ("match_id", "match_id_public"),
        ("home_win_90_prob", "home_win_90_prob_public"),
        ("draw_90_prob", "draw_90_prob_public"),
        ("away_win_90_prob", "away_win_90_prob_public"),
        ("brier_90_recomputed", "brier_90_public"),
        ("log_loss_90_recomputed", "log_loss_90_public"),
    ]
    mismatch_counts: dict[str, int] = {}
    for derived_col, public_col in comparisons:
        numeric = derived_col not in {"match_id"}
        if numeric:
            comparable = both[derived_col].notna() | both[public_col].notna()
            mismatch = comparable & ~np.isclose(
                pd.to_numeric(both[derived_col], errors="coerce"),
                pd.to_numeric(both[public_col], errors="coerce"),
                atol=tolerance,
                rtol=0.0,
                equal_nan=True,
            )
        else:
            mismatch = both[derived_col].fillna("").astype(str) != both[public_col].fillna(
                ""
            ).astype(str)
        mismatch_counts[derived_col] = int(mismatch.sum())
        if mismatch.any():
            part = both.loc[mismatch, ["prediction_id", derived_col, public_col]].copy()
            part["field"] = derived_col
            part = part.rename(columns={derived_col: "derived_value", public_col: "public_value"})
            discrepancies.append(part[["prediction_id", "field", "derived_value", "public_value"]])
    discrepancy_frame = (
        pd.concat(discrepancies, ignore_index=True)
        if discrepancies
        else pd.DataFrame(columns=["prediction_id", "field", "derived_value", "public_value"])
    )
    output_dir = config.resolve_path("verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    discrepancy_path = output_dir / "public_export_discrepancies.csv"
    discrepancy_frame.to_csv(discrepancy_path, index=False)
    report = {
        "public_export_sha256": sha256_file(public_path),
        "derived_panel_sha256": sha256_file(panel_path),
        "public_match_prediction_rows": len(public),
        "derived_prediction_rows": len(panel),
        "matched_rows": len(both),
        "derived_only_rows": int((joined["_merge"] == "left_only").sum()),
        "public_only_rows": int((joined["_merge"] == "right_only").sum()),
        "mismatch_counts": mismatch_counts,
        "total_discrepancies": len(discrepancy_frame),
        "policy": "report_only_sqlite_remains_source_of_truth",
    }
    report_path = output_dir / "public_export_reconciliation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    source_hashes = {
        "public_csv": report["public_export_sha256"],
        "analysis_panel": report["derived_panel_sha256"],
    }
    manifest.add(
        "public_export_discrepancies",
        discrepancy_path,
        "csv",
        "reconcile_public_export",
        source_hashes,
        {"rows": len(discrepancy_frame)},
    )
    manifest.add(
        "public_export_reconciliation",
        report_path,
        "json",
        "reconcile_public_export",
        source_hashes,
        report,
    )
    manifest.write()
    return report
