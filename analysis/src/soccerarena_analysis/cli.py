from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

from .config import AnalysisConfig, load_config
from .manifest import Manifest
from .stages import (
    annotations,
    derive,
    external_baselines,
    freeze,
    load_validate,
    reconcile_public_export,
)
from .stages.acceptance import verify_final_outputs, verify_pre_results
from .stages.annotations import AdjudicationRequired, AnnotationConfigurationRequired


def _prepare(config: AnalysisConfig, manifest: Manifest) -> None:
    freeze.freeze_database(config, manifest)
    load_validate.validate_frozen_database(config, manifest)
    derive.derive_tables(config, manifest)
    reconcile_public_export.reconcile_public_export(config, manifest)
    external_baselines.run(config, manifest)
    verify_pre_results(config, manifest)


def _run_all(config: AnalysisConfig, manifest: Manifest) -> None:
    from .analyses import (
        closing_odds,
        direct_odds,
        overall,
        rq1_access,
        rq2_elicitation,
        rq3_calibration,
        rq4_diversity,
        rq5_reasoning,
        rq6_tournament,
        t24_odds,
    )
    from .stages import rationale_embeddings

    _prepare(config, manifest)
    overall.run(config, manifest)
    closing_odds.run(config, manifest)
    t24_odds.run(config, manifest)
    direct_odds.run(config, manifest)
    rq1_access.run(config, manifest)
    rq2_elicitation.run(config, manifest)
    rq3_calibration.run(config, manifest)
    rq4_diversity.run(config, manifest)
    rq5_reasoning.run_objective(config, manifest)
    rationale_embeddings.run(config, manifest)
    annotations.run(config, manifest)
    rq5_reasoning.run_final(config, manifest)
    rq6_tournament.run(config, manifest)
    verify_final_outputs(config, manifest)


def _run_rq5(config: AnalysisConfig, manifest: Manifest) -> object:
    from .analyses import rq5_reasoning
    from .stages import rationale_embeddings

    rq5_reasoning.run_objective(config, manifest)
    rationale_embeddings.run(config, manifest)
    annotations.run(config, manifest)
    return rq5_reasoning.run_final(config, manifest)


def _run_closing_odds(config: AnalysisConfig, manifest: Manifest) -> object:
    from .analyses import closing_odds

    external_baselines.run(config, manifest)
    return closing_odds.run(config, manifest)


def _run_t24_odds(config: AnalysisConfig, manifest: Manifest) -> object:
    from .analyses import t24_odds

    external_baselines.run(config, manifest)
    return t24_odds.run(config, manifest)


def _run_direct_odds(config: AnalysisConfig, manifest: Manifest) -> object:
    from .analyses import direct_odds

    external_baselines.run(config, manifest)
    return direct_odds.run(config, manifest)


COMMAND_NAMES = (
    "run",
    "prepare",
    "overall",
    "closing-odds",
    "t24-odds",
    "direct-odds",
    "rq1",
    "rq2",
    "rq3",
    "rq4",
    "annotations",
    "rq5",
    "rq6",
)


def _commands() -> dict[str, Callable[[AnalysisConfig, Manifest], object]]:
    from .analyses import (
        overall,
        rq1_access,
        rq2_elicitation,
        rq3_calibration,
        rq4_diversity,
        rq6_tournament,
    )

    return {
        "run": _run_all,
        "prepare": _prepare,
        "overall": overall.run,
        "closing-odds": _run_closing_odds,
        "t24-odds": _run_t24_odds,
        "direct-odds": _run_direct_odds,
        "rq1": rq1_access.run,
        "rq2": rq2_elicitation.run,
        "rq3": rq3_calibration.run,
        "rq4": rq4_diversity.run,
        "annotations": annotations.run,
        "rq5": _run_rq5,
        "rq6": rq6_tournament.run,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soccerarena-analysis")
    parser.add_argument("command", choices=COMMAND_NAMES)
    parser.add_argument("--config", required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config)
    matplotlib_cache = config.resolve_path("artifacts") / "runtime" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    manifest = Manifest(config)
    try:
        _commands()[arguments.command](config, manifest)
    except AnnotationConfigurationRequired as error:
        print(json.dumps({"status": "configuration_required", "message": str(error)}, indent=2))
        return 2
    except AdjudicationRequired as error:
        print(json.dumps({"status": "adjudication_required", "message": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {"status": "complete", "command": arguments.command, "manifest": str(manifest.path)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
