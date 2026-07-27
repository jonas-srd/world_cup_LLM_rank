# LLM SoccerArena

This repository contains the World Cup 2026 LLM forecasting service and its
reproducible scientific analysis. The two concerns are deliberately separated:
the production system lives in `website/`, while the Python analysis package
lives in `analysis/`.

## Repository layout

```text
.
├── website/                 # Next.js site, cron jobs, SQLite layer, LLM client
│   ├── apps/web/            # Public dashboard
│   ├── apps/cron/           # Data sync, prediction, evaluation, and backups
│   ├── packages/            # Shared database, LLM, and scoring packages
│   ├── data/                # Local runtime data (databases are ignored)
│   ├── deploy/seed/         # Compressed database seed for deployment
│   └── docs/                # Website and Railway operations
├── analysis/                # Installable Python analysis project
│   ├── src/                 # `soccerarena_analysis` package
│   ├── tests/               # Statistical and data-contract tests
│   ├── inputs/              # Local/private inputs (ignored)
│   ├── assets/              # Versioned plotting assets
│   └── docs/                # Analysis protocol and audit checklists
├── .python-version          # Recommended local Python version
├── LICENSE
└── README.md
```

The website was imported from `origin/internal-cron-staging` at
`b0e097ab360ae93a0873a96714b127d1e00ba5e4`. The Python package was imported
from `upstream/kdd_submission` at
`6547ea70bae02cd1734842cb03a5a049b7788e16`.

Paper drafts, IDE settings, local database backups, virtual environments, and
generated analysis snapshots are not part of the cleaned source tree.

## Website quick start

Requirements: Node.js 20 or newer and npm.

```bash
cd website
npm ci
cp .env.example .env
npm run db:init
npm run dev
```

On Windows Command Prompt, use `copy .env.example .env` instead of `cp`.
The dashboard is available at `http://localhost:3000`.

Common checks:

```bash
npm run typecheck
npm run test:analytics
npm run test:special
npm run test:benchmark-scheduling
npm run build
```

See [website/README.md](website/README.md) for environment variables, prediction
jobs, data exports, database backups, and Railway deployment.

## Analysis quick start

Requirements: Python 3.11–3.13 and `uv`.

```bash
cd analysis
uv sync --extra dev
uv run ruff check src tests scripts
uv run pytest \
  --ignore=tests/test_data_and_external_contracts.py \
  -k "not normalized_tournament_tables_match_registered_design"
```

The complete pipeline needs the following local inputs:

1. `website/data/world-cup.db` — the primary SQLite source of truth.
2. `analysis/inputs/worldcup2026-full-prediction-dataset.csv` — the consolidated
   website CSV used only for reconciliation.
3. `analysis/inputs/llmsoccerarena-analysis-main.zip` — the registered external
   baseline archive used by the audit.

Run the pipeline from `analysis/`:

```bash
uv run soccerarena-analysis run --config analysis.yaml
```

With all inputs prepared and `prepare` artifacts available, run the complete
test suite with `uv run pytest`.

Results, figures, tables, verification reports, and the manifest are written to
`analysis/artifacts/`, which is intentionally ignored by Git.

See [analysis/README.md](analysis/README.md) for the pipeline stages, annotation
checkpoint, configuration, and individual analysis commands.

## End-to-end data flow

```text
football-data.org + OpenRouter
              │
              ▼
website/data/world-cup.db
       │              │
       ▼              ▼
public dashboard   website exports
       │              │
       └──────┬───────┘
              ▼
      analysis/analysis.yaml
              │
              ▼
     analysis/artifacts/
```

SQLite remains the analytical source of truth. Public CSV data is reconciled
against the frozen database but never supplies analytical values.

## Secrets and generated data

Never commit `.env`, API keys, local databases, downloaded backups, website
exports, or `analysis/artifacts/`. The root and project-specific ignore files
cover these paths.

The compressed seed at `website/deploy/seed/world-cup.db.gz` is intentionally
versioned because it is part of the website deployment process.

## License

See [LICENSE](LICENSE).
