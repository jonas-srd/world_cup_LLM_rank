# Evaluation protocol contract

This file records the analysis rules enforced by the code. The YAML configuration,
frozen-data manifest, and machine-readable outputs are authoritative if a prose
description and an executable detail ever diverge.

## Registration tiers

- **Prospective core:** the complete-panel T−24h performance display, access,
  prompting, calibration, diversity, reasoning/tool metadata, and the 15
  tournament-question units.
- **Final-data-locked extensions:** stage heterogeneity, operational snapshot
  contrasts, timing sensitivity, conditions-versus-models decomposition, and
  same-cell ensemble comparisons fixed before the remaining outcomes arrive.
- **Exploratory:** the elicited-confidence association, rationale length, and
  illustrative rationale excerpts. These cannot authorize primary claims.

No invalid forecast is imputed. No model or contrast is chosen after observing its
score. Contrasts cannot be redefined using outcomes, and no new forecast sample may
be collected after its outcome. The universal primary target is the home/draw/away
result after 90 minutes.

## Primary inference

Every paired contrast is reduced to one prespecified difference per match. Matches
are resampled as whole clusters within group and knockout strata. The pipeline uses
10,000 bootstrap-t replicates for the 95% interval and 10,000 sign flips for the
p-value of the same mean paired difference, with independent streams derived from
master seed 20260715. Effect tables also show the untested median difference. The
primary access contrast includes a leave-one-match-out minimum/maximum sweep.

The registered Holm families are read from `analysis.yaml`. The conditions-versus-
models intersection–union diagnostic is separate: every component must support
`A - |M_ij| > 0` at alpha 0.05, with no Holm layer. It remains an appendix
diagnostic. The locked sentence in the YAML is emitted only when both of its coded
predicates hold; otherwise access and model findings are stated separately.

## Data and timing

SQLite is frozen with its backup API and is the only analytical source. All modules
consume typed, hashed Parquet tables. The public CSV is read once for reconciliation
and can only create discrepancy records. Special-question values come from the
normalized SQLite tables. Final mode requires all 104 official 90-minute outcomes,
a complete registered panel, and all 15 tournament outcomes.

T−24h and T−2h are operational snapshots. Their actual lead-time distributions are
reported. The T−24h sensitivity retains only calls within 1,440±90 minutes. Closed-
book changes are the negative-control design check because no new external
information is available to that condition.

## Baselines and annotations

Opta, bookmaker odds, and each Markus candidate must pass their own provenance,
timing, leakage, target, reproduction, and redistribution gate. Bookmaker overround
is removed per bookmaker before median aggregation. The primary closing-odds benchmark
uses open-book probabilities-first T−2h forecasts, and the primary same-horizon market
sensitivity uses open-book probabilities-first T−24h forecasts against T−24h odds.
Secondary prompt-strategy sensitivities repeat both comparisons using only the H/D/A
probabilities reported in the matched open-book direct-score cells; these are labeled
as secondary rather than substituted for the registered primary comparisons. Frozen
Markus forecasts can only be paired with stage-opening forecasts. A failed gate never
consumes main-text space.

Every non-empty match rationale is coded by the released keyword lexicon and a separately
configured LLM annotator using locked operational category definitions. The coder prompt hash
covers the exact instruction, category order and definitions, endpoint, and temperature; raw
provider responses are cached and hashed. Per-category raw agreement and Cohen's kappa are
reported over the full corpus. All
complete-vector disagreements and 50 fixed-seed agreements are exported without
outcomes, scores, match IDs, or model identities. The run stops until every human
field is completed and the immutable columns pass their hash check. Reported rates
use the resolved labels over the full corpus; there is no extrapolation step.

The 15 tournament questions are equal-weight prospective analysis units drawn only from normalized
SQLite tables. Forecast rows describe within-question variation and are never treated as 480
independent observations. Single-choice vectors are scored only when they sum to one. Semifinal
marginals are evaluated as supplied and audited against the required sum of four; they are never
silently normalized after the coherence check.
