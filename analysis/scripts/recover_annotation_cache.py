from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".recovering")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def recover(directory: Path, archive: Path, expected_items: int) -> dict[str, Any]:
    label_path = directory / "llm_label_cache.jsonl"
    raw_path = directory / "llm_raw_batches.jsonl"
    labels = _read_jsonl(label_path)
    raw_batches = _read_jsonl(raw_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labels:
        grouped[record["annotation_id"]].append(record)
    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    differing = {
        key: values
        for key, values in duplicates.items()
        if len({json.dumps(value["labels"], sort_keys=True) for value in values}) > 1
    }
    categories = list(labels[0]["labels"])
    per_category = {
        category: sum(
            len({value["labels"][category] for value in values}) > 1
            for values in duplicates.values()
        )
        for category in categories
    }

    first_labels: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in labels:
        if record["annotation_id"] not in seen_ids:
            first_labels.append(record)
            seen_ids.add(record["annotation_id"])
    selected_hashes = {record["raw_batch_response_hash"] for record in first_labels}
    raw_by_hash = {record["raw_response_hash"]: record for record in raw_batches}
    missing_raw = selected_hashes - set(raw_by_hash)
    if missing_raw:
        raise ValueError(f"Canonical item labels reference missing raw responses: {missing_raw}")
    canonical_raw = [
        record for record in raw_batches if record["raw_response_hash"] in selected_hashes
    ]
    if len(canonical_raw) != len(selected_hashes):
        raise ValueError("Raw-response hashes are duplicated in the incident cache")

    archive.mkdir(parents=True, exist_ok=False)
    archived_labels = archive / label_path.name
    archived_raw = archive / raw_path.name
    shutil.copy2(label_path, archived_labels)
    shutil.copy2(raw_path, archived_raw)
    _write_jsonl(label_path, first_labels)
    _write_jsonl(raw_path, canonical_raw)
    report = {
        "incident": "overlapping annotation processes after shell timeout",
        "selection_rule": "retain the first frozen response for every annotation_id",
        "expected_items": expected_items,
        "original_label_lines": len(labels),
        "original_raw_batches": len(raw_batches),
        "unique_annotation_ids": len(first_labels),
        "missing_annotation_ids": expected_items - len(first_labels),
        "duplicate_annotation_ids": len(duplicates),
        "duplicate_ids_with_differing_vectors": len(differing),
        "per_category_duplicate_label_differences": per_category,
        "canonical_raw_batches": len(canonical_raw),
        "superseded_raw_batches": len(raw_batches) - len(canonical_raw),
        "archive": {
            "label_cache": str(archived_labels.resolve()),
            "label_cache_sha256": _sha256(archived_labels),
            "raw_cache": str(archived_raw.resolve()),
            "raw_cache_sha256": _sha256(archived_raw),
        },
        "canonical": {
            "label_cache_sha256": _sha256(label_path),
            "raw_cache_sha256": _sha256(raw_path),
        },
    }
    report_path = archive / "recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-items", required=True, type=int)
    arguments = parser.parse_args()
    report = recover(arguments.directory, arguments.archive, arguments.expected_items)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
