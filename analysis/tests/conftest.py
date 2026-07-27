from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from soccerarena_analysis.config import canonical_json, load_config, sha256_bytes


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config(project_root):
    return load_config(project_root / "analysis.yaml")


@pytest.fixture
def fast_config(config):
    raw = deepcopy(config.raw)
    raw["statistics"]["bootstrap_replicates"] = 200
    raw["statistics"]["permutation_replicates"] = 200
    digest = sha256_bytes(canonical_json(raw).encode("utf-8"))
    return replace(config, raw=raw, digest=digest)
