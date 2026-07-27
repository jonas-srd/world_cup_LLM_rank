# LLM SoccerArena analysis

This directory is an isolated, installable Python project for the reproducible
World Cup 2026 analysis. SQLite is the sole source of truth. Each analysis reads
only manifest-verified Parquet tables derived from a consistent frozen database;
the public website CSV is read once for reconciliation and never supplies
analytical values.

The code was imported from `upstream/kdd_submission` at commit
`6547ea70bae02cd1734842cb03a5a049b7788e16`. Planning drafts and generated
snapshots from that branch were intentionally excluded from this source package.

## Project layout

```text
analysis/
├── src/soccerarena_analysis/
│   ├── analyses/           # Overall results and RQ1–RQ6
│   ├── reporting/          # Figures, tables, and headline JSON
│   ├── stages/             # Freeze, validate, derive, reconcile, audit
│   └── statistics/         # Bootstrap, metrics, and multiplicity control
├── tests/                  # Unit, statistical, and data-contract tests
├── scripts/                # Maintenance helpers
├── assets/                 # Provider icons and country flags
├── external/               # Versioned external-baseline runtime metadata
├── inputs/                 # Local/private inputs; ignored by Git
├── docs/                   # Protocol and human-audit instructions
├── analysis.yaml           # Locked analysis contract
├── pyproject.toml
└── uv.lock
```

## Setup

Use Python 3.11–3.13 and `uv`:

```bash
cd analysis
uv sync --extra dev
```

The lockfile pins the reproducible environment. Do not replace it with an
unversioned `requirements.txt`.

## Required inputs

The default paths are defined in `analysis.yaml`:

| Input | Default path | Purpose |
| --- | --- | --- |
| SQLite database | `../website/data/world-cup.db` | Sole analytical source of truth |
| Public website CSV | `inputs/worldcup2026-full-prediction-dataset.csv` | Reconciliation only |
| Markus archive | `inputs/llmsoccerarena-analysis-main.zip` | External-baseline audit |
| OpenRouter credentials | `../website/.env` | RQ5 annotation calls only |

Large/private inputs are ignored. See [inputs/README.md](inputs/README.md) for
the expected filenames.

## Run the analysis

From `analysis/`, run the complete pipeline:

```bash
uv run soccerarena-analysis run --config analysis.yaml
```

The pipeline:

1. freezes and validates the SQLite database;
2. derives typed Parquet analysis tables;
3. reconciles the public CSV without using it as an analytical input;
4. audits external baselines and applies the pre-results acceptance gate;
5. generates overall, bookmaker, and RQ1–RQ6 results;
6. writes figures, tables, headline JSON, verification reports, and a manifest.

After `prepare`, individual stages can be run independently:

```bash
uv run soccerarena-analysis prepare --config analysis.yaml
uv run soccerarena-analysis overall --config analysis.yaml
uv run soccerarena-analysis rq1 --config analysis.yaml
uv run soccerarena-analysis rq2 --config analysis.yaml
uv run soccerarena-analysis rq3 --config analysis.yaml
uv run soccerarena-analysis rq4 --config analysis.yaml
uv run soccerarena-analysis rq5 --config analysis.yaml
uv run soccerarena-analysis rq6 --config analysis.yaml
```

Bookmaker comparisons are available separately:

```bash
uv run soccerarena-analysis closing-odds --config analysis.yaml
uv run soccerarena-analysis t24-odds --config analysis.yaml
uv run soccerarena-analysis direct-odds --config analysis.yaml
```

All adjustable quantities live in `analysis.yaml`.

## RQ5 annotation checkpoint

The first annotation run creates the blinded, model-by-access-balanced human
audit at `artifacts/annotations/human_audit.csv` and exits with status
`adjudication_required`.

Fill every human category field without changing the blinded ID or rationale
text, then rerun the same command. The resumed run verifies the checkpoint,
combines the adjudicated rows with cached model labels, and completes RQ5, RQ6,
and the final claim-discipline gate. Detailed instructions are in
[docs/HUMAN_AUDIT_GUIDE.md](docs/HUMAN_AUDIT_GUIDE.md).

## Outputs

Generated files live below `artifacts/`:

- `frozen/` — immutable database snapshot;
- `derived/` — validated Parquet inputs;
- `verification/` — integrity, reconciliation, and acceptance reports;
- `results/` — analysis tables;
- `paper/figures/`, `paper/tables/`, and `paper/headlines/`;
- `manifest.json` — hashes, provenance, and stage metadata.

The entire directory is ignored because outputs must be regenerated from the
versioned code, configuration, and local inputs.

## Validation

Code-only checks:

```bash
uv run ruff check src tests scripts
uv run pytest \
  --ignore=tests/test_data_and_external_contracts.py \
  -k "not normalized_tournament_tables_match_registered_design"
```

The exclusions cover the source-database/external-baseline contract and the RQ6
test that reads `artifacts/derived/special_predictions.parquet`.

With all required inputs present and `prepare` artifacts generated:

```bash
uv run pytest
```

The scientific acceptance criteria are documented in
[docs/ACCEPTANCE_CHECKLIST.md](docs/ACCEPTANCE_CHECKLIST.md) and
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
