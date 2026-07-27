from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import AnalysisConfig
from ..manifest import Manifest
from .headline_json import headline_record, write_headlines


def _json_scalar(value):
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def save_table(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    artifact_id: str,
    module: str,
    source_hashes: dict[str, str],
    latex: bool = True,
    headline_frame: pd.DataFrame | None = None,
    latex_formats: dict[str, str] | None = None,
) -> list[Path]:
    directory = config.resolve_path("tables")
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{artifact_id}.csv"
    frame.to_csv(csv_path, index=False)
    manifest.add(
        f"{artifact_id}_csv", csv_path, "table", module, source_hashes, {"rows": len(frame)}
    )
    paths = [csv_path]
    if latex:
        tex_path = directory / f"{artifact_id}.tex"
        decimals = int(config.section("reporting")["style"]["table_float_decimals"])
        formatters = {
            column: (lambda value, template=template: template.format(value))
            for column, template in (latex_formats or {}).items()
        }
        tex_path.write_text(
            frame.to_latex(
                index=False,
                escape=True,
                float_format=f"%.{decimals}f",
                formatters=formatters,
                na_rep="--",
            ),
            encoding="utf-8",
        )
        manifest.add(
            f"{artifact_id}_tex", tex_path, "table", module, source_hashes, {"rows": len(frame)}
        )
        paths.append(tex_path)
    manifest.write()
    output_hashes = {
        path.suffix.lstrip("."): getattr(
            manifest.records[f"{artifact_id}_{path.suffix.lstrip('.')}"], "sha256"
        )
        for path in paths
    }
    companion_id = f"{artifact_id}_companion"
    trace_frame = frame if headline_frame is None else headline_frame
    inferential = "estimate" in trace_frame.columns
    records = []
    if inferential:
        identifier_columns = [
            column
            for column in (
                "analysis_id",
                "contrast",
                "comparison",
                "model_id",
                "question_id",
                "stage",
                "component",
            )
            if column in trace_frame.columns
        ]
        for index, row in trace_frame.iterrows():
            identifier = (
                ".".join(str(row[column]) for column in identifier_columns)
                if identifier_columns
                else str(index)
            )
            record_extra = {"output_hashes": output_hashes}
            if "null_reason" in trace_frame.columns and pd.notna(row.get("null_reason")):
                record_extra["null_reason"] = _json_scalar(row.get("null_reason"))
            records.append(
                headline_record(
                    config,
                    companion_id,
                    f"{artifact_id}.{identifier}",
                    artifact_id,
                    source_hashes,
                    estimate=_json_scalar(row.get("estimate")),
                    ci_low=_json_scalar(row.get("ci_low")),
                    ci_high=_json_scalar(row.get("ci_high")),
                    p_raw=_json_scalar(row.get("p_raw")),
                    p_adjusted=_json_scalar(row.get("p_adjusted")),
                    median=_json_scalar(row.get("median")),
                    n_matches=_json_scalar(row.get("n_matches")),
                    n_predictions=_json_scalar(row.get("n_predictions")),
                    units=_json_scalar(row.get("units")),
                    aggregation=_json_scalar(row.get("aggregation")),
                    extra=record_extra,
                )
            )
    else:
        records.append(
            headline_record(
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
                n_predictions=len(frame),
                units=None,
                aggregation="rows in emitted table",
                extra={
                    "null_reason": "descriptive table without a single inferential estimand",
                    "output_hashes": output_hashes,
                },
            )
        )
    write_headlines(config, manifest, companion_id, records, module, source_hashes)
    return paths
