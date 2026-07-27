from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from soccerarena_analysis.config import canonical_json, sha256_bytes
from soccerarena_analysis.manifest import Manifest
from soccerarena_analysis.stages.annotations import (
    AdjudicationRequired,
    _keyword_labels,
    annotator_prompt,
    run,
)


def test_annotation_checkpoint_halts_and_resumes(config, tmp_path):
    raw = deepcopy(config.raw)
    raw["paths"]["annotations"] = str(tmp_path / "annotations")
    raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    raw["annotation"]["model"] = "test-annotator-v1"
    raw["annotation"]["analysis_scope"]["expected_rationales"] = 2
    raw["annotation"]["human_audit_per_model_access_cell"] = 1
    raw["design"]["complete_panel"] = ["model"]
    digest = sha256_bytes(canonical_json(raw).encode("utf-8"))
    test_config = replace(config, raw=raw, digest=digest)
    corpus = pd.DataFrame(
        [
            {
                "prediction_id": "p1",
                "match_id": "m1",
                "model_id": "model",
                "forecast_horizon": "T_24H",
                "access_condition": "closed_book",
                "prompt_strategy": "probabilistic_forecast",
                "rationale_text": "Recent form should win.",
            },
            {
                "prediction_id": "p2",
                "match_id": "m1",
                "model_id": "model",
                "forecast_horizon": "T_24H",
                "access_condition": "open_book",
                "prompt_strategy": "probabilistic_forecast",
                "rationale_text": "Odds and an injury report.",
            },
        ]
    )
    corpus_path = tmp_path / "rationale.parquet"
    corpus.to_parquet(corpus_path, index=False)
    manifest = Manifest(test_config)
    manifest.add("derived_rationale_corpus", corpus_path, "parquet", "test")
    manifest.write()
    categories = list(raw["annotation"]["categories"])
    _, prompt_hash = annotator_prompt(test_config)
    cache_path = test_config.resolve_path("annotations") / "llm_label_cache.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache_records = []
    for prediction_id, text in zip(corpus["prediction_id"], corpus["rationale_text"]):
        annotation_id = hashlib.sha256(
            f"{test_config.master_seed}|{prediction_id}".encode()
        ).hexdigest()[:20]
        labels = {category: False for category in categories}
        cache_records.append(
            {
                "annotation_id": annotation_id,
                "rationale_hash": hashlib.sha256(text.encode()).hexdigest(),
                "model": raw["annotation"]["model"],
                "prompt_hash": prompt_hash,
                "base_url": raw["annotation"]["base_url"],
                "temperature": raw["annotation"]["temperature"],
                "seed": raw["annotation"]["seed"],
                "labels": labels,
                "raw_batch_response_hash": "test",
            }
        )
    cache_path.write_text(
        "\n".join(json.dumps(item) for item in cache_records) + "\n", encoding="utf-8"
    )
    with pytest.raises(AdjudicationRequired):
        run(test_config, manifest)
    checkpoint = pd.read_csv(
        test_config.resolve_path("annotations") / "human_audit.csv",
        dtype=str,
        keep_default_na=False,
    )
    for column in [name for name in checkpoint if name.startswith("human__")]:
        checkpoint[column] = "false"
    checkpoint.to_csv(test_config.resolve_path("annotations") / "human_audit.csv", index=False)
    resolved = run(test_config, manifest)
    assert len(resolved) == len(corpus)
    assert all(
        column in resolved for column in [f"resolved__{category}" for category in categories]
    )


def test_keyword_lexicon_respects_boundaries_and_composite_unsupported(config):
    lexicon = config.section("annotation")["categories"]
    market_value = _keyword_labels("The squad has the higher market value.", lexicon)
    betting_market = _keyword_labels("Bookmaker odds imply a narrow favorite.", lexicon)
    generic = _keyword_labels("They should win because they are simply better.", lexicon)
    supported = _keyword_labels("They should win because of recent form.", lexicon)
    assert not market_value["markets_odds"]
    assert market_value["rankings_strength"]
    assert betting_market["markets_odds"]
    assert generic["unsupported_generic"]
    assert not supported["unsupported_generic"]
