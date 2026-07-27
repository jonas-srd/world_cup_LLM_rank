from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..config import AnalysisConfig, canonical_json, sha256_bytes, sha256_file
from ..manifest import Manifest


class AnnotationConfigurationRequired(RuntimeError):
    pass


class AdjudicationRequired(RuntimeError):
    pass


def _label_columns(prefix: str, categories: list[str]) -> list[str]:
    return [f"{prefix}__{category}" for category in categories]


def _annotation_id(config: AnalysisConfig, prediction_id: str) -> str:
    return hashlib.sha256(f"{config.master_seed}|{prediction_id}".encode()).hexdigest()[:20]


def _normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _keyword_labels(text: str, lexicon: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Apply the released, boundary-aware regular-expression lexicon."""
    normalized = _normalized_text(text)
    labels: dict[str, bool] = {}
    for category, rules in lexicon.items():
        included = any(re.search(pattern, normalized) for pattern in rules.get("include", []))
        excluded = any(re.search(pattern, normalized) for pattern in rules.get("exclude", []))
        labels[category] = included and not excluded
    for category, rules in lexicon.items():
        required_absent = rules.get("require_no_support_categories", [])
        if required_absent:
            labels[category] = labels[category] and not any(
                labels[support_category] for support_category in required_absent
            )
    return labels


def annotation_corpus(corpus: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Return and validate the prospectively defined RQ5.2 analysis cell."""
    annotation = config.section("annotation")
    scope = annotation["analysis_scope"]
    required = {
        "prediction_id",
        "match_id",
        "model_id",
        "forecast_horizon",
        "access_condition",
        "prompt_strategy",
        "rationale_text",
    }
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Rationale corpus lacks required fields: {sorted(missing)}")
    selected = corpus.copy()
    if scope["model_panel"] != "complete_panel":
        raise AnnotationConfigurationRequired(
            "annotation.analysis_scope.model_panel must be complete_panel"
        )
    models = list(config.section("design")["complete_panel"])
    selected = selected[
        selected["model_id"].isin(models)
        & selected["forecast_horizon"].isin(scope["forecast_horizons"])
        & selected["prompt_strategy"].isin(scope["prompt_strategies"])
        & selected["access_condition"].isin(scope["access_conditions"])
        & selected["rationale_text"].fillna("").str.strip().ne("")
    ].copy()
    if selected["prediction_id"].duplicated().any():
        raise ValueError("RQ5.2 corpus contains duplicate prediction IDs")
    expected = int(scope["expected_rationales"])
    if len(selected) != expected:
        raise ValueError(f"RQ5.2 corpus has {len(selected):,} rationales; expected {expected:,}")
    expected_cells = pd.MultiIndex.from_product(
        [models, scope["access_conditions"]], names=["model_id", "access_condition"]
    )
    observed_cells = selected.groupby(["model_id", "access_condition"]).size()
    missing_cells = expected_cells.difference(observed_cells.index)
    if len(missing_cells):
        raise ValueError(f"RQ5.2 corpus lacks model/access cells: {missing_cells.tolist()}")
    return selected.sort_values("prediction_id").reset_index(drop=True)


def annotator_prompt(config: AnalysisConfig) -> tuple[str, str]:
    """Return the exact coder instruction and hash all semantic annotation inputs."""
    settings = config.section("annotation")
    categories = list(settings["categories"])
    definitions = settings["category_definitions"]
    if set(definitions) != set(categories):
        raise AnnotationConfigurationRequired(
            "annotation.category_definitions must match annotation.categories exactly"
        )
    specification = {
        "instruction": settings["prompt"],
        "categories": categories,
        "category_definitions": definitions,
        "model": settings["model"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
    }
    system = (
        f"{settings['prompt']}\n\n"
        f"Category definitions: {json.dumps(definitions, ensure_ascii=False, sort_keys=True)}\n\n"
        "Return one annotation for every supplied annotation_id. "
        f"Required boolean labels: {json.dumps(categories)}"
    )
    return system, sha256_bytes(canonical_json(specification).encode("utf-8"))


def _response_schema(categories: list[str], batch_size: int) -> dict[str, Any]:
    properties: dict[str, Any] = {"annotation_id": {"type": "string"}}
    properties.update({category: {"type": "boolean"} for category in categories})
    item = {
        "type": "object",
        "properties": properties,
        "required": ["annotation_id", *categories],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rationale_annotations",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "annotations": {
                        "type": "array",
                        "items": item,
                        "minItems": batch_size,
                        "maxItems": batch_size,
                    }
                },
                "required": ["annotations"],
                "additionalProperties": False,
            },
        },
    }


def _load_credentials(config: AnalysisConfig) -> tuple[str, dict[str, str]]:
    settings = config.section("annotation")
    env_path = (config.root / str(settings["credential_env_file"])).resolve()
    load_dotenv(env_path, override=False)
    api_key = os.environ.get(str(settings["api_key_env"]))
    if not api_key:
        raise AnnotationConfigurationRequired(
            f"Missing API key environment variable {settings['api_key_env']}"
        )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    site_url = os.environ.get(str(settings["site_url_env"]))
    site_name = os.environ.get(str(settings["site_name_env"]))
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    return api_key, headers


def _parse_batch_payload(
    raw_text: str, categories: list[str], expected_ids: set[str]
) -> dict[str, dict[str, bool]]:
    payload = json.loads(raw_text)
    if set(payload) != {"annotations"} or not isinstance(payload["annotations"], list):
        raise ValueError("LLM batch annotation must contain only an annotations array")
    labels: dict[str, dict[str, bool]] = {}
    expected_keys = {"annotation_id", *categories}
    for item in payload["annotations"]:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError("LLM annotation item has an invalid schema")
        annotation_id = item["annotation_id"]
        if annotation_id in labels or annotation_id not in expected_ids:
            raise ValueError(f"LLM returned an unexpected or duplicate ID: {annotation_id}")
        if any(not isinstance(item[category], bool) for category in categories):
            raise ValueError(f"LLM returned a non-boolean label for {annotation_id}")
        labels[annotation_id] = {category: item[category] for category in categories}
    if set(labels) != expected_ids:
        raise ValueError("LLM batch response omitted one or more annotation IDs")
    return labels


def _call_annotator_batch(
    batch: list[dict[str, str]], config: AnalysisConfig, headers: dict[str, str]
) -> tuple[dict[str, dict[str, bool]], dict[str, Any], dict[str, Any]]:
    settings = config.section("annotation")
    model = str(settings["model"])
    if model == "CONFIGURE_EXACT_MODEL_VERSION":
        raise AnnotationConfigurationRequired("Set annotation.model to an exact model version")
    categories = list(settings["categories"])
    system, prompt_hash = annotator_prompt(config)
    request_body = {
        "model": model,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "max_tokens": int(settings["max_output_tokens"]),
        "include_reasoning": False,
        "response_format": _response_schema(categories, len(batch)),
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": canonical_json(
                    {
                        "rationales": [
                            {"annotation_id": x["annotation_id"], "text": x["text"]} for x in batch
                        ]
                    }
                ),
            },
        ],
    }
    attempts = int(settings["retry_attempts"])
    for attempt in range(attempts):
        try:
            response = httpx.post(
                f"{str(settings['base_url']).rstrip('/')}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=float(settings["timeout_seconds"]),
            )
            response.raise_for_status()
            raw = response.json()
            content = raw["choices"][0]["message"]["content"]
            expected_ids = {item["annotation_id"] for item in batch}
            labels = _parse_batch_payload(content, categories, expected_ids)
            request_provenance = {
                "request_hash": sha256_bytes(canonical_json(request_body).encode("utf-8")),
                "prompt_hash": prompt_hash,
                "annotation_ids": sorted(expected_ids),
            }
            return labels, raw, request_provenance
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(float(settings["retry_backoff_seconds"]) * (attempt + 1))
    raise RuntimeError("Unreachable annotation retry state")


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _usage_cost(raw: dict[str, Any], settings: dict[str, Any]) -> tuple[int, int, float]:
    usage = raw.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reported = usage.get("cost")
    if reported is not None:
        return prompt_tokens, completion_tokens, float(reported)
    estimated = (
        prompt_tokens * float(settings["listed_prompt_cost_per_million"])
        + completion_tokens * float(settings["listed_completion_cost_per_million"])
    ) / 1_000_000
    return prompt_tokens, completion_tokens, estimated


def _preflight_cost(items: list[dict[str, str]], settings: dict[str, Any]) -> float:
    batch_size = int(settings["batch_size"])
    batch_count = int(np.ceil(len(items) / batch_size))
    prompt_tokens = sum(len(item["text"]) for item in items) / float(
        settings["cost_estimation_characters_per_token"]
    ) + batch_count * int(settings["cost_estimation_request_overhead_tokens_per_batch"])
    completion_tokens = len(items) * int(settings["cost_estimation_output_tokens_per_item"])
    return (
        prompt_tokens * float(settings["listed_prompt_cost_per_million"])
        + completion_tokens * float(settings["listed_completion_cost_per_million"])
    ) / 1_000_000


def _validate_cached_item(
    cached: dict[str, Any], row: Any, config: AnalysisConfig, prompt_hash: str
) -> None:
    settings = config.section("annotation")
    expected = {
        "rationale_hash": sha256_bytes(row.rationale_text.encode("utf-8")),
        "model": settings["model"],
        "prompt_hash": prompt_hash,
        "base_url": settings["base_url"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
    }
    mismatch = [key for key, value in expected.items() if cached.get(key) != value]
    if mismatch:
        raise ValueError(
            f"Frozen annotation cache mismatch for {cached.get('annotation_id')}: {mismatch}"
        )


def _automated_labels(
    corpus: pd.DataFrame, config: AnalysisConfig, directory: Path
) -> tuple[pd.DataFrame, Path, Path]:
    settings = config.section("annotation")
    categories = list(settings["categories"])
    label_cache_path = directory / "llm_label_cache.jsonl"
    raw_batch_path = directory / "llm_raw_batches.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    _, prompt_hash = annotator_prompt(config)
    existing_records = _jsonl_records(label_cache_path)
    existing = {item["annotation_id"]: item for item in existing_records}
    if len(existing) != len(existing_records):
        raise ValueError("LLM label cache contains duplicate annotation IDs")

    rows: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for row in corpus.itertuples(index=False):
        annotation_id = _annotation_id(config, str(row.prediction_id))
        rows[annotation_id] = row
        cached = existing.get(annotation_id)
        if cached is not None:
            _validate_cached_item(cached, row, config, prompt_hash)
        else:
            missing.append({"annotation_id": annotation_id, "text": row.rationale_text})

    if missing:
        estimated_cost = _preflight_cost(missing, settings)
        if estimated_cost > float(settings["max_annotation_cost_usd"]):
            raise RuntimeError(
                f"Estimated annotation cost ${estimated_cost:.2f} exceeds the configured cap"
            )
        _, headers = _load_credentials(config)
        size = int(settings["batch_size"])
        batches = [missing[start : start + size] for start in range(0, len(missing), size)]
        total_cost = 0.0
        with (
            raw_batch_path.open("a", encoding="utf-8", newline="\n") as raw_handle,
            label_cache_path.open("a", encoding="utf-8", newline="\n") as label_handle,
            ThreadPoolExecutor(max_workers=int(settings["max_concurrency"])) as executor,
        ):
            futures = {
                executor.submit(_call_annotator_batch, batch, config, headers): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_index = futures[future]
                labels, raw, provenance = future.result()
                raw_hash = sha256_bytes(canonical_json(raw).encode("utf-8"))
                prompt_tokens, completion_tokens, cost = _usage_cost(raw, settings)
                total_cost += cost
                raw_record = {
                    "batch_index": batch_index,
                    "model": settings["model"],
                    "provider": raw.get("provider"),
                    "response_id": raw.get("id"),
                    "created": raw.get("created"),
                    "usage": raw.get("usage"),
                    "estimated_or_reported_cost_usd": cost,
                    "raw_response": raw,
                    "raw_response_hash": raw_hash,
                    **provenance,
                }
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False, sort_keys=True) + "\n")
                raw_handle.flush()
                for annotation_id, item_labels in sorted(labels.items()):
                    row = rows[annotation_id]
                    label_record = {
                        "annotation_id": annotation_id,
                        "prediction_id": str(row.prediction_id),
                        "rationale_hash": sha256_bytes(row.rationale_text.encode("utf-8")),
                        "model": settings["model"],
                        "prompt_hash": prompt_hash,
                        "base_url": settings["base_url"],
                        "temperature": settings["temperature"],
                        "seed": settings["seed"],
                        "labels": item_labels,
                        "raw_batch_response_hash": raw_hash,
                        "prompt_tokens_batch": prompt_tokens,
                        "completion_tokens_batch": completion_tokens,
                    }
                    label_handle.write(
                        json.dumps(label_record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    existing[annotation_id] = label_record
                label_handle.flush()
                if total_cost > float(settings["max_annotation_cost_usd"]):
                    raise RuntimeError(
                        f"Annotation calls cost ${total_cost:.2f}, exceeding the configured cap"
                    )

    label_cache_path.touch(exist_ok=True)
    raw_batch_path.touch(exist_ok=True)
    records: list[dict[str, Any]] = []
    for annotation_id, row in rows.items():
        cached = existing[annotation_id]
        _validate_cached_item(cached, row, config, prompt_hash)
        labels = cached["labels"]
        if set(labels) != set(categories) or any(
            not isinstance(labels[category], bool) for category in categories
        ):
            raise ValueError(f"Invalid cached label vector for {annotation_id}")
        keyword = _keyword_labels(row.rationale_text, settings["categories"])
        record: dict[str, Any] = {
            "annotation_id": annotation_id,
            "prediction_id": str(row.prediction_id),
            "rationale_text": row.rationale_text,
            "model_id": row.model_id,
            "access_condition": row.access_condition,
            "match_id": row.match_id,
        }
        record.update({f"keyword__{key}": keyword[key] for key in categories})
        record.update({f"llm__{key}": bool(labels[key]) for key in categories})
        records.append(record)
    return pd.DataFrame(records), label_cache_path, raw_batch_path


def _cohen_kappa(first: pd.Series, second: pd.Series) -> float | None:
    a = first.astype(bool).to_numpy()
    b = second.astype(bool).to_numpy()
    observed = float(np.mean(a == b))
    p_first = float(np.mean(a))
    p_second = float(np.mean(b))
    expected = p_first * p_second + (1.0 - p_first) * (1.0 - p_second)
    return None if np.isclose(expected, 1.0) else float((observed - expected) / (1.0 - expected))


def _immutable_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    serialized = (
        frame[columns].fillna("").astype(str).to_csv(index=False, lineterminator="\n").encode()
    )
    return hashlib.sha256(serialized).hexdigest()


def _read_human(value: object) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Human label is not a valid boolean: {value!r}")


def _agreement_table(
    frame: pd.DataFrame, categories: list[str], first_prefix: str, second_prefix: str
) -> pd.DataFrame:
    rows = []
    for category in categories:
        first = frame[f"{first_prefix}__{category}"].astype(bool)
        second = frame[f"{second_prefix}__{category}"].astype(bool)
        rows.append(
            {
                "category": category,
                "first_coder": first_prefix,
                "second_coder": second_prefix,
                "first_prevalence": float(first.mean()),
                "second_prevalence": float(second.mean()),
                "raw_agreement": float((first == second).mean()),
                "cohen_kappa": _cohen_kappa(first, second),
                "n_rationales": len(frame),
            }
        )
    return pd.DataFrame(rows)


def _audit_sample(automated: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    per_cell = int(config.section("annotation")["human_audit_per_model_access_cell"])
    pieces = []
    for (model_id, access), group in automated.groupby(["model_id", "access_condition"], sort=True):
        if len(group) < per_cell:
            raise ValueError(f"Audit cell {model_id}/{access} has fewer than {per_cell} rows")
        pieces.append(
            group.sample(
                n=per_cell,
                random_state=config.derived_seed(f"annotations.audit.{model_id}.{access}"),
            )
        )
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sample(
        frac=1.0, random_state=config.derived_seed("annotations.audit.shuffle")
    ).reset_index(drop=True)


def run(config: AnalysisConfig, manifest: Manifest) -> pd.DataFrame:
    settings = config.section("annotation")
    if not settings["enabled"]:
        raise AnnotationConfigurationRequired(
            "Annotation stage is mandatory and cannot be disabled for a complete run"
        )
    corpus_record = manifest.require("derived_rationale_corpus")
    full_corpus = pd.read_parquet(corpus_record.path)
    corpus = annotation_corpus(full_corpus, config)
    categories = list(settings["categories"])
    directory = config.resolve_path("annotations")
    automated, label_cache_path, raw_batch_path = _automated_labels(corpus, config, directory)
    manifest.add(
        "annotation_llm_label_cache",
        label_cache_path,
        "jsonl",
        "annotations",
        {"rationale_corpus": corpus_record.sha256},
        {"rows": len(automated), "model": settings["model"]},
    )
    manifest.add(
        "annotation_llm_raw_cache",
        raw_batch_path,
        "jsonl",
        "annotations",
        {"rationale_corpus": corpus_record.sha256},
        {"batches": len(_jsonl_records(raw_batch_path)), "model": settings["model"]},
    )

    keyword_columns = _label_columns("keyword", categories)
    llm_columns = _label_columns("llm", categories)
    automated["complete_vector_agreement"] = (
        automated[keyword_columns].to_numpy() == automated[llm_columns].to_numpy()
    ).all(axis=1)
    agreement = _agreement_table(automated, categories, "keyword", "llm")
    _, prompt_hash = annotator_prompt(config)
    scoped_ids = set(automated["annotation_id"])
    scoped_batches = [
        item
        for item in _jsonl_records(raw_batch_path)
        if item.get("model") == settings["model"]
        and item.get("prompt_hash") == prompt_hash
        and set(item.get("annotation_ids", [])).issubset(scoped_ids)
    ]
    run_summary = {
        "model": settings["model"],
        "prompt_hash": prompt_hash,
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "rationales": len(automated),
        "cached_batches": len(scoped_batches),
        "cached_item_labels": len(automated),
        "reported_or_estimated_cost_usd": sum(
            float(item.get("estimated_or_reported_cost_usd") or 0.0) for item in scoped_batches
        ),
        "complete_vector_keyword_glm_agreement": float(
            automated["complete_vector_agreement"].mean()
        ),
        "reproducibility_note": (
            "Raw provider outputs are frozen; temperature zero is not treated as deterministic regeneration."
        ),
    }
    run_summary_path = directory / "annotation_run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add(
        "annotation_run_summary",
        run_summary_path,
        "json",
        "annotations",
        {"llm_cache": sha256_file(label_cache_path), "raw_cache": sha256_file(raw_batch_path)},
        {"rows": len(automated), "batches": len(scoped_batches)},
    )
    agreement_path = directory / "automated_agreement.parquet"
    agreement.to_parquet(agreement_path, index=False)
    manifest.add(
        "annotation_automated_agreement",
        agreement_path,
        "parquet",
        "annotations",
        {"llm_cache": sha256_file(label_cache_path)},
        {"rows": len(agreement), "n_rationales": len(automated)},
    )

    index_path = directory / "annotation_index.parquet"
    automated.drop(columns="rationale_text").to_parquet(index_path, index=False)
    manifest.add(
        "annotation_index",
        index_path,
        "parquet",
        "annotations",
        {"rationale_corpus": corpus_record.sha256},
        {"rows": len(automated)},
    )

    audit = _audit_sample(automated, config)
    human_columns = _label_columns("human", categories)
    immutable_columns = ["annotation_id", "rationale_text"]
    export = audit[immutable_columns].copy()
    for column in human_columns:
        export[column] = ""
    checkpoint_path = directory / "human_audit.csv"
    sidecar_path = directory / "human_audit.checkpoint.json"
    audit_index_path = directory / "human_audit_index.parquet"
    audit.drop(columns="rationale_text").to_parquet(audit_index_path, index=False)
    manifest.add(
        "annotation_human_audit_index",
        audit_index_path,
        "parquet",
        "annotations",
        {"annotation_index": sha256_file(index_path)},
        {"rows": len(audit), "blinded_export": checkpoint_path.name},
    )

    if not checkpoint_path.exists():
        export.to_csv(checkpoint_path, index=False)
        sidecar = {
            "immutable_hash": _immutable_hash(export, immutable_columns),
            "row_count": len(export),
            "annotation_ids": sorted(export["annotation_id"].tolist()),
            "human_columns": human_columns,
            "sampling_rule": "fixed-seed model-by-access balanced audit",
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        manifest.add(
            "annotation_checkpoint",
            checkpoint_path,
            "csv",
            "annotations",
            {"llm_cache": sha256_file(label_cache_path)},
            {"status": "adjudication_required", "rows": len(export)},
        )
        manifest.add(
            "annotation_checkpoint_contract",
            sidecar_path,
            "json",
            "annotations",
            {"checkpoint": sha256_file(checkpoint_path)},
        )
        manifest.write()
        raise AdjudicationRequired(f"Complete every human label in {checkpoint_path}")

    if not sidecar_path.exists():
        raise AdjudicationRequired("Human-audit sidecar is missing; audit cannot be verified")
    submitted = pd.read_csv(checkpoint_path, dtype=str, keep_default_na=False)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sorted(export["annotation_id"].tolist()) != sidecar["annotation_ids"]
        or _immutable_hash(export, immutable_columns) != sidecar["immutable_hash"]
    ):
        raise AdjudicationRequired(
            "The scoped corpus or audit sample changed; archive the audit and regenerate it"
        )
    if (
        len(submitted) != sidecar["row_count"]
        or sorted(submitted["annotation_id"].tolist()) != sidecar["annotation_ids"]
    ):
        raise AdjudicationRequired("Human-audit row count or blinded IDs changed")
    if _immutable_hash(submitted, immutable_columns) != sidecar["immutable_hash"]:
        raise AdjudicationRequired("Immutable human-audit columns changed")
    if any(column not in submitted for column in human_columns) or any(
        (submitted[column].str.strip() == "").any() for column in human_columns
    ):
        raise AdjudicationRequired(f"Complete every human label in {checkpoint_path}")
    for column in human_columns:
        submitted[column] = submitted[column].map(_read_human)

    audited = audit.merge(
        submitted[["annotation_id", *human_columns]],
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )
    human_agreement = pd.concat(
        [
            _agreement_table(audited, categories, "llm", "human"),
            _agreement_table(audited, categories, "keyword", "human"),
        ],
        ignore_index=True,
    )
    human_agreement_path = directory / "human_audit_agreement.parquet"
    human_agreement.to_parquet(human_agreement_path, index=False)
    manifest.add(
        "annotation_human_agreement",
        human_agreement_path,
        "parquet",
        "annotations",
        {"checkpoint": sha256_file(checkpoint_path)},
        {"rows": len(human_agreement), "audited_rationales": len(audited)},
    )

    resolved = automated[
        [
            "annotation_id",
            "prediction_id",
            "complete_vector_agreement",
            *keyword_columns,
            *llm_columns,
        ]
    ].copy()
    human_by_id = submitted.set_index("annotation_id")
    audited_ids = set(human_by_id.index)
    for category in categories:
        resolved[f"resolved__{category}"] = [
            bool(human_by_id.loc[annotation_id, f"human__{category}"])
            if annotation_id in audited_ids
            else bool(llm_value)
            for annotation_id, llm_value in zip(
                resolved["annotation_id"], resolved[f"llm__{category}"]
            )
        ]
    resolved["resolved_source"] = np.where(
        resolved["annotation_id"].isin(audited_ids), "human_audit", "llm"
    )
    resolved_path = directory / "resolved_annotations.parquet"
    resolved.to_parquet(resolved_path, index=False)
    manifest.add(
        "resolved_annotations",
        resolved_path,
        "parquet",
        "annotations",
        {"checkpoint": sha256_file(checkpoint_path), "llm_cache": sha256_file(label_cache_path)},
        {"rows": len(resolved), "audited_rows": len(audited)},
    )
    manifest.add(
        "annotation_checkpoint",
        checkpoint_path,
        "csv",
        "annotations",
        {"llm_cache": sha256_file(label_cache_path)},
        {"status": "complete", "rows": len(submitted)},
    )
    manifest.write()
    return resolved
