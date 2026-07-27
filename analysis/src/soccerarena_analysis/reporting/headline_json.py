from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AnalysisConfig
from ..manifest import Manifest

HEADLINE_KEYS = {
    "artifact_id",
    "analysis_id",
    "estimand",
    "estimate",
    "ci_low",
    "ci_high",
    "p_raw",
    "p_adjusted",
    "median",
    "n_matches",
    "n_predictions",
    "units",
    "aggregation",
    "config_hash",
    "source_hashes",
    "manifest_key",
}


def headline_record(
    config: AnalysisConfig,
    artifact_id: str,
    analysis_id_value: str,
    estimand: str,
    source_hashes: dict[str, str],
    **values: Any,
) -> dict[str, Any]:
    record = {
        "artifact_id": artifact_id,
        "analysis_id": analysis_id_value,
        "estimand": estimand,
        "estimate": values.get("estimate"),
        "ci_low": values.get("ci_low"),
        "ci_high": values.get("ci_high"),
        "p_raw": values.get("p_raw"),
        "p_adjusted": values.get("p_adjusted"),
        "median": values.get("median"),
        "n_matches": values.get("n_matches"),
        "n_predictions": values.get("n_predictions"),
        "units": values.get("units"),
        "aggregation": values.get("aggregation"),
        "config_hash": config.digest,
        "source_hashes": source_hashes,
        "manifest_key": artifact_id,
    }
    extra = values.get("extra")
    if extra:
        record["extra"] = extra
    return record


def write_headlines(
    config: AnalysisConfig,
    manifest: Manifest,
    artifact_id: str,
    records: list[dict[str, Any]],
    module: str,
    source_hashes: dict[str, str],
) -> Path:
    for record in records:
        missing = HEADLINE_KEYS - set(record)
        if missing:
            raise ValueError(f"Headline record is missing fields: {sorted(missing)}")
    path = config.resolve_path("headlines") / f"{artifact_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add(
        artifact_id, path, "headline_json", module, source_hashes, {"records": len(records)}
    )
    manifest.write()
    return path
