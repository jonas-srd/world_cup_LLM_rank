# Local analysis inputs

This directory contains large or private inputs and is intentionally ignored by
Git, except for this file.

Place these files here before running the complete analysis:

- `worldcup2026-full-prediction-dataset.csv`: the consolidated CSV downloaded
  from the website analytics view; it is used only for reconciliation.
- `llmsoccerarena-analysis-main.zip`: the registered external Markus baseline
  archive used by the provenance and leakage audit.

The primary SQLite input is shared directly with the website at
`website/data/world-cup.db` and remains the sole analytical source of truth.
