# Paper artifact generation

Reproducible generator for the LLM-SoccerArena paper. It reads the live
benchmark SQLite database and emits, into the Overleaf project, self-contained
LaTeX fragments that the paper `\input{}`s:

- `macros.tex` — numeric macros (counts, rates) used in prose
- `tab_models.tex`, `tab_coverage.tex`, `tab_reliability.tex`, `tab_cost.tex` — booktabs tables
- `fig_search_rate.tex`, `fig_consistency.tex`, `fig_champion.tex` — pgfplots figures

It also writes the underlying CSVs to `paper/artifacts/` for transparency.

`generated/fig_flow.tex` (the architecture diagram) is hand-authored TikZ in the
Overleaf project and is **not** produced by this script.

All outputs are **pre-outcome diagnostics**: they describe stored predictions
and operational behaviour, not forecast accuracy (computed only after matches
finish).

## Run

```bash
node paper/analysis/generate-paper-artifacts.cjs \
  --db=data/world-cup.db \
  --out="<overleaf-project>/generated" \
  --csv=paper/artifacts
```

Defaults: `--db=data/world-cup.db`, `--out=paper/generated`, `--csv=paper/artifacts`.

The paper preamble adds `\usepackage{pgfplots}` / `\usepackage{tikz}`; no
external plotting toolchain is required (figures render at LaTeX compile time).
Regenerate after each fresh prediction/evaluation run to refresh all numbers,
tables, and figures.
