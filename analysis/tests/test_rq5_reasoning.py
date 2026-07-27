from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd

from soccerarena_analysis.analyses.rq5_reasoning import _objective_frames
from soccerarena_analysis.config import canonical_json, sha256_bytes
from soccerarena_analysis.stages.annotations import annotator_prompt


def test_closed_book_is_not_mislabeled_as_failed_search(config):
    model = config.section("design")["complete_panel"][0]
    tools = pd.DataFrame(
        [
            {
                "prediction_id": "closed",
                "match_id": "m1",
                "model_id": model,
                "forecast_horizon": "T_24H",
                "access_condition": "closed_book",
                "prompt_strategy": "direct_score",
                "tools_enabled": False,
                "tool_calls_observed": pd.NA,
                "num_tool_calls": float("nan"),
                "tool_trace_available": False,
                "open_book_compliance": "not_applicable",
                "input_tokens": 10.0,
                "output_tokens": 2.0,
                "latency_ms": 4.0,
                "cost_usd": 0.01,
            },
            {
                "prediction_id": "open-observed",
                "match_id": "m1",
                "model_id": model,
                "forecast_horizon": "T_24H",
                "access_condition": "open_book",
                "prompt_strategy": "direct_score",
                "tools_enabled": True,
                "tool_calls_observed": True,
                "num_tool_calls": 1.0,
                "tool_trace_available": True,
                "open_book_compliance": "observed_search",
                "input_tokens": 20.0,
                "output_tokens": 3.0,
                "latency_ms": 8.0,
                "cost_usd": 0.02,
            },
            {
                "prediction_id": "open-unknown",
                "match_id": "m1",
                "model_id": model,
                "forecast_horizon": "T_24H",
                "access_condition": "open_book",
                "prompt_strategy": "probabilistic_forecast",
                "tools_enabled": True,
                "tool_calls_observed": False,
                "num_tool_calls": 0.0,
                "tool_trace_available": False,
                "open_book_compliance": "unknown",
                "input_tokens": 20.0,
                "output_tokens": 3.0,
                "latency_ms": 8.0,
                "cost_usd": 0.02,
            },
        ]
    )
    rationale = pd.DataFrame(
        [
            {
                "prediction_id": item,
                "match_id": "m1",
                "model_id": model,
                "forecast_horizon": "T_24H",
                "access_condition": access,
                "prompt_strategy": "direct_score",
                "rationale_text": "A short rationale.",
            }
            for item, access in (("closed", "closed_book"), ("open-observed", "open_book"))
        ]
    )
    frames = _objective_frames(tools, rationale, config)
    summary = frames["operations_summary"]
    closed = summary.query("scope == 'full_corpus' and access_condition == 'closed_book'").iloc[0]
    open_book = summary.query("scope == 'full_corpus' and access_condition == 'open_book'").iloc[0]
    assert pd.isna(closed["observed_search_rate"])
    assert open_book["observed_search_rate"] == 0.5
    assert open_book["unknown_search_status_rate"] == 0.5


def test_category_definition_changes_annotation_prompt_hash(config):
    _, first_hash = annotator_prompt(config)
    raw = deepcopy(config.raw)
    raw["annotation"]["category_definitions"]["tactics"] += " Additional rule."
    digest = sha256_bytes(canonical_json(raw).encode("utf-8"))
    changed = replace(config, raw=raw, digest=digest)
    _, second_hash = annotator_prompt(changed)
    assert first_hash != second_hash
