from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..manifest import Manifest


def _source_contract(config: AnalysisConfig) -> dict[str, object]:
    source_root = config.root / "src" / "soccerarena_analysis"
    python_files = list(source_root.rglob("*.py"))
    forbidden_test_name = "wilco" + "xon"
    forbidden_hits = [
        str(path)
        for path in python_files
        if forbidden_test_name in path.read_text(encoding="utf-8").casefold()
    ]
    public_readers = []
    for path in python_files:
        text = path.read_text(encoding="utf-8")
        reader_token = "resolve_path(" + chr(34) + "public_csv" + chr(34) + ")"
        alternate_token = "resolve_path(" + chr(39) + "public_csv" + chr(39) + ")"
        if reader_token in text or alternate_token in text:
            public_readers.append(path.name)
    if forbidden_hits:
        raise ValueError(f"Prohibited rank-test reference found: {forbidden_hits}")
    if public_readers != ["reconcile_public_export.py"]:
        raise ValueError(f"Website CSV must have exactly one reader, found {public_readers}")
    return {"prohibited_rank_test_hits": 0, "website_csv_readers": public_readers}


def verify_pre_results(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    panel_record = manifest.require("derived_analysis_panel")
    special_record = manifest.require("derived_special_predictions")
    reconciliation = manifest.require("public_export_reconciliation")
    panel = pd.read_parquet(panel_record.path)
    specials = pd.read_parquet(special_record.path)
    key = config.section("validation")["stable_key"]
    duplicate_keys = int(panel.duplicated(key).sum())
    valid = panel[panel["is_valid_for_scoring"].fillna(False)]
    probabilities = valid[["home_win_90_prob", "draw_90_prob", "away_win_90_prob"]]
    tolerance = float(config.section("validation")["probability_tolerance"])
    bad_sums = int((~np.isclose(probabilities.sum(axis=1), 1.0, atol=tolerance, rtol=0.0)).sum())
    bad_ranges = int(
        ((probabilities < -tolerance) | (probabilities > 1.0 + tolerance)).any(axis=1).sum()
    )
    expected_questions = int(config.section("special_questions")["expected_questions"])
    question_count = int(specials["question_id"].nunique())
    failures = {
        "duplicate_stable_keys": duplicate_keys,
        "invalid_probability_sums": bad_sums,
        "invalid_probability_ranges": bad_ranges,
        "special_question_count_difference": question_count - expected_questions,
    }
    if any(failures.values()):
        raise ValueError(f"Pre-results acceptance failed: {failures}")
    source_contract = _source_contract(config)
    report = {
        "status": "passed",
        "stable_prediction_key": key,
        "analysis_rows": len(panel),
        "valid_prediction_rows": len(valid),
        "special_questions": question_count,
        "reconciliation_artifact": reconciliation.sha256,
        "checks": failures,
        "source_contract": source_contract,
    }
    path = config.resolve_path("verification") / "pre_results_acceptance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add(
        "pre_results_acceptance",
        path,
        "json",
        "acceptance",
        {"analysis_panel": panel_record.sha256, "special_predictions": special_record.sha256},
    )
    manifest.write()
    return report


def verify_final_outputs(config: AnalysisConfig, manifest: Manifest) -> dict[str, object]:
    required = [
        "overall_headlines",
        "rq1_headlines",
        "rq2_headlines",
        "rq3_headlines",
        "rq4_headlines",
        "rq5_headlines",
        "rq6_headlines",
        "resolved_annotations",
        "external_baseline_audit_json",
    ]
    verified = {artifact_id: manifest.require(artifact_id).sha256 for artifact_id in required}
    prohibited = [term.casefold() for term in config.section("reporting")["prohibited_claim_terms"]]
    text_artifacts = [
        record for record in manifest.records.values() if record.kind in {"json", "markdown", "tex"}
    ]
    claim_hits: list[dict[str, str]] = []
    for record in text_artifacts:
        text = Path(record.path).read_text(encoding="utf-8", errors="replace").casefold()
        for term in prohibited:
            if term in text:
                claim_hits.append({"artifact_id": record.artifact_id, "term": term})
    paper_number_index: dict[str, object] = {}
    for artifact_id, record in sorted(manifest.records.items()):
        if record.kind != "headline_json":
            continue
        for index, item in enumerate(json.loads(Path(record.path).read_text(encoding="utf-8"))):
            key = f"{artifact_id}::{item['analysis_id']}"
            paper_number_index[key] = {
                "artifact_id": artifact_id,
                "path": record.path,
                "record_index": index,
            }
    index_path = config.resolve_path("headlines") / "paper_number_index.json"
    index_path.write_text(
        json.dumps(paper_number_index, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest.add(
        "paper_number_index",
        index_path,
        "json",
        "acceptance",
        verified,
        {"records": len(paper_number_index)},
    )
    report = {
        "status": "passed" if not claim_hits else "review_required",
        "verified_artifacts": verified,
        "paper_number_records": len(paper_number_index),
        "vocabulary_hits": claim_hits,
    }
    path = config.resolve_path("verification") / "final_acceptance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add(
        "final_acceptance", path, "json", "acceptance", verified, {"status": report["status"]}
    )
    manifest.write()
    if claim_hits:
        raise ValueError(f"Claim-discipline review found prohibited vocabulary: {claim_hits}")
    return report
