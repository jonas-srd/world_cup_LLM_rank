from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_mode",
    "random",
    "paths",
    "validation",
    "statistics",
    "timing",
    "design",
    "metrics",
    "calibration",
    "diversity",
    "special_questions",
    "annotation",
    "external_baselines",
    "reporting",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    digest: str

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    def resolve_path(self, name: str) -> Path:
        configured = Path(self.raw["paths"][name])
        return configured if configured.is_absolute() else (self.root / configured).resolve()

    def derived_seed(self, analysis_id: str) -> int:
        master = int(self.raw["random"]["master_seed"])
        payload = f"{master}:{analysis_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    @property
    def master_seed(self) -> int:
        return int(self.raw["random"]["master_seed"])

    @property
    def analysis_digest(self) -> str:
        analysis_config = {key: value for key, value in self.raw.items() if key != "reporting"}
        return sha256_bytes(canonical_json(analysis_config).encode("utf-8"))

    @property
    def is_final(self) -> bool:
        return self.raw["run_mode"] == "final"


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Analysis configuration must be a YAML mapping")
    missing = REQUIRED_TOP_LEVEL - set(raw)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if raw["run_mode"] not in {"development", "final"}:
        raise ValueError("run_mode must be 'development' or 'final'")
    root = config_path.parent
    return AnalysisConfig(
        path=config_path,
        root=root,
        raw=raw,
        digest=sha256_bytes(canonical_json(raw).encode("utf-8")),
    )
