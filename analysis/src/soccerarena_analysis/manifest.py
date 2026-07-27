from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AnalysisConfig, sha256_file

REPORTING_INDEPENDENT_MODULES = {
    "freeze",
    "load_validate",
    "derive",
    "reconcile_public_export",
    "external_baselines",
}


@dataclass
class ArtifactRecord:
    artifact_id: str
    path: str
    sha256: str
    kind: str
    module: str
    config_hash: str
    source_hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    return {
        distribution.metadata["Name"]: distribution.version
        for distribution in sorted(
            importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].casefold()
        )
    }


class Manifest:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.path = config.resolve_path("manifest")
        self.records: dict[str, ArtifactRecord] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for item in payload.get("artifacts", []):
                record = ArtifactRecord(**item)
                self.records[record.artifact_id] = record

    def add(
        self,
        artifact_id: str,
        path: Path,
        kind: str,
        module: str,
        source_hashes: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            artifact_id=artifact_id,
            path=str(path.resolve()),
            sha256=sha256_file(path),
            kind=kind,
            module=module,
            config_hash=self.config.digest,
            source_hashes=source_hashes or {},
            metadata={**(metadata or {}), "analysis_config_hash": self.config.analysis_digest},
        )
        self.records[artifact_id] = record
        return record

    def require(self, artifact_id: str) -> ArtifactRecord:
        if artifact_id not in self.records:
            raise FileNotFoundError(f"Required manifest artifact is missing: {artifact_id}")
        record = self.records[artifact_id]
        if record.config_hash != self.config.digest:
            analysis_hash_matches = (
                record.module in REPORTING_INDEPENDENT_MODULES
                and record.metadata.get("analysis_config_hash") == self.config.analysis_digest
            )
            if not analysis_hash_matches:
                raise ValueError(f"Manifest configuration hash mismatch for {artifact_id}")
        path = Path(record.path)
        if not path.exists() or sha256_file(path) != record.sha256:
            raise ValueError(f"Manifest hash mismatch for {artifact_id}")
        return record

    def discard_generated(self, artifact_ids: list[str]) -> None:
        artifacts_root = self.config.resolve_path("artifacts")
        for artifact_id in artifact_ids:
            record = self.records.pop(artifact_id, None)
            if record is None:
                continue
            path = Path(record.path).resolve()
            if not path.is_relative_to(artifacts_root):
                raise ValueError(
                    f"Refusing to remove generated artifact outside artifacts root: {path}"
                )
            path.unlink(missing_ok=True)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.config.raw["schema_version"],
            "config_path": str(self.config.path),
            "config_hash": self.config.digest,
            "run_mode": self.config.raw["run_mode"],
            "repository_commit": git_commit(self.config.root.parent),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(),
                "dependency_lock_sha256": sha256_file(self.config.resolve_path("dependency_lock")),
            },
            "artifacts": [asdict(self.records[key]) for key in sorted(self.records)],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
