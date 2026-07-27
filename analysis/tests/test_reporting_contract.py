from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import matplotlib as mpl
import pandas as pd
import pytest

from soccerarena_analysis.config import canonical_json, sha256_bytes
from soccerarena_analysis.manifest import Manifest
from soccerarena_analysis.reporting.figures import apply_style
from soccerarena_analysis.reporting.headline_json import (
    HEADLINE_KEYS,
    headline_record,
    write_headlines,
)
from soccerarena_analysis.reporting.tables import save_table


def test_figure_style_uses_configured_publication_font(fast_config):
    apply_style(fast_config)

    assert mpl.rcParams["font.family"] == ["Arial"]
    assert mpl.rcParams["font.sans-serif"][0] == "Arial"
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42


def test_headline_json_contract(fast_config, tmp_path):
    fast_config.raw["paths"]["headlines"] = str(tmp_path / "headlines")
    fast_config.raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    manifest = Manifest(fast_config)
    record = headline_record(
        fast_config,
        "test_headline",
        "test.analysis",
        "mean paired difference",
        {"table": "abc"},
        estimate=0.1,
        ci_low=0.0,
        ci_high=0.2,
        p_raw=0.05,
        p_adjusted=0.1,
        median=0.08,
        n_matches=12,
        n_predictions=48,
        units="Brier",
        aggregation="within match",
    )
    assert HEADLINE_KEYS.issubset(record)
    path = write_headlines(
        fast_config, manifest, "test_headline", [record], "test", {"table": "abc"}
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored[0]["manifest_key"] == "test_headline"
    assert manifest.require("test_headline").sha256


def test_inferential_table_emits_traceable_companion(fast_config, tmp_path):
    fast_config.raw["paths"]["tables"] = str(tmp_path / "tables")
    fast_config.raw["paths"]["headlines"] = str(tmp_path / "headlines")
    fast_config.raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    manifest = Manifest(fast_config)
    frame = pd.DataFrame(
        [
            {
                "analysis_id": "paired.effect",
                "estimate": 0.02,
                "ci_low": 0.01,
                "ci_high": 0.03,
                "p_raw": 0.01,
                "p_adjusted": 0.02,
                "median": 0.018,
                "n_matches": 20,
            }
        ]
    )
    save_table(frame, fast_config, manifest, "effect_table", "test", {"source": "abc"}, latex=False)
    companion = manifest.require("effect_table_companion")
    payload = json.loads(open(companion.path, encoding="utf-8").read())
    assert payload[0]["estimate"] == 0.02
    assert (
        payload[0]["extra"]["output_hashes"]["csv"] == manifest.require("effect_table_csv").sha256
    )


def test_display_table_can_trace_separate_numeric_frame(fast_config, tmp_path):
    fast_config.raw["paths"]["tables"] = str(tmp_path / "tables")
    fast_config.raw["paths"]["headlines"] = str(tmp_path / "headlines")
    fast_config.raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    manifest = Manifest(fast_config)
    display = pd.DataFrame([{"Contrast": "Readable label", "95% CI": "[0.01, 0.03]"}])
    trace = pd.DataFrame(
        [
            {
                "analysis_id": "paired.effect",
                "estimate": 0.02,
                "ci_low": 0.01,
                "ci_high": 0.03,
                "p_raw": 0.01,
                "p_adjusted": 0.02,
                "median": 0.018,
                "n_matches": 20,
            }
        ]
    )
    save_table(
        display,
        fast_config,
        manifest,
        "readable_effect_table",
        "test",
        {"source": "abc"},
        latex=False,
        headline_frame=trace,
    )
    payload = json.loads(
        open(manifest.require("readable_effect_table_companion").path, encoding="utf-8").read()
    )
    assert payload[0]["estimate"] == 0.02


def test_latex_table_supports_column_specific_numeric_formats(fast_config, tmp_path):
    fast_config.raw["paths"]["tables"] = str(tmp_path / "tables")
    fast_config.raw["paths"]["headlines"] = str(tmp_path / "headlines")
    fast_config.raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    manifest = Manifest(fast_config)
    frame = pd.DataFrame([{"Label": "Example", "Rate (%)": 12.345, "Δ (pp)": 1.234}])
    paths = save_table(
        frame,
        fast_config,
        manifest,
        "formatted_table",
        "test",
        {"source": "abc"},
        latex_formats={"Rate (%)": "{:.1f}", "Δ (pp)": "{:+.1f}"},
    )
    latex = next(path for path in paths if path.suffix == ".tex").read_text(encoding="utf-8")
    assert "\\begin{tabular}{lrr}" in latex
    assert "12.3" in latex
    assert "+1.2" in latex


def test_reporting_change_reuses_only_reporting_independent_inputs(fast_config, tmp_path):
    fast_config.raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    manifest = Manifest(fast_config)
    derived_path = tmp_path / "derived.parquet"
    result_path = tmp_path / "result.json"
    derived_path.write_text("derived", encoding="utf-8")
    result_path.write_text("result", encoding="utf-8")
    manifest.add("derived_input", derived_path, "parquet", "derive")
    manifest.add("analysis_result", result_path, "json", "rq1_access")
    manifest.write()

    changed_raw = deepcopy(fast_config.raw)
    changed_raw["reporting"]["dpi"] += 1
    changed = replace(
        fast_config,
        raw=changed_raw,
        digest=sha256_bytes(canonical_json(changed_raw).encode("utf-8")),
    )
    changed_manifest = Manifest(changed)
    assert changed_manifest.require("derived_input").sha256
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        changed_manifest.require("analysis_result")
