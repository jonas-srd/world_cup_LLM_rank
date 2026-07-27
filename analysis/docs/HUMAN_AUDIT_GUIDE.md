# RQ5 rationale audit guide

Audit file: `artifacts/annotations/human_audit.csv`

The file contains a fixed-seed, model-by-access-balanced sample of 196 rationales. It is blinded
to model, access condition, match, outcome, and forecast score. Code only what is stated in the
rationale. Do not look up external information or try to identify the underlying forecast.

## How to complete the file

- Fill every `human__*` cell with `true` or `false`.
- Treat the nine categories as independent multi-label fields; multiple categories may be true.
- Mark a category true only when the information is presented as relevant to the forecast.
- A negated factual observation can count when the absence is used as evidence. A statement that
  information was not checked or used does not count by itself.
- Do not evaluate whether a factual claim is correct, current, or useful.
- Do not edit, sort, or reorder rows. Do not change `annotation_id`, `rationale_text`, headers, or
  the number or order of columns. The pipeline verifies these immutable fields before resuming.

## Category definitions

- `markets_odds`: Betting odds, bookmaker or prediction-market probabilities, market prices, or
  an explicitly market-implied assessment. A player's transfer-market value alone does not count.
- `recent_form`: Recent match results, a recent performance run, streak, or current form used as
  evidence. A timeless statement about general team quality does not count.
- `injuries_lineups`: Injury, suspension, player availability, announced or expected lineup,
  rotation, or a named selection decision.
- `rankings_strength`: Ranking, rating, squad quality, player or squad market value, historical
  strength, or an explicit relative-strength comparison used as evidence.
- `tactics`: Tactical style, formation, pressing, possession, counterattacking, defensive block,
  matchup, or another on-field strategic mechanism.
- `venue_travel`: Venue, home or host advantage, crowd, travel, altitude, climate, weather, or
  another location-specific condition used as evidence.
- `tournament_context`: Group standings, qualification or elimination scenarios, must-win
  incentives, rest between matches, fixture congestion, or stage-specific tournament strategy.
  A bare reference to the tournament or match stage does not count.
- `explicit_sourcing`: A factual statement attributed to a named or linked source, a URL, or an
  explicit citation-like phrase. Merely saying that information is known does not count.
- `unsupported_generic`: A generic predictive assertion or football cliché without a concrete
  factual or mechanistic basis stated in the rationale. This concerns support in the text, not
  whether the assertion is true in the world.

After all nine category labels have been completed for each of the 196 audited rationales
(196 × 9 = 1,764 `true`/`false` cells in total), rerun:

```text
uv run soccerarena-analysis rq5 --config analysis.yaml
```

The resumed run verifies the immutable hash and valid booleans before it creates resolved labels,
category rates, excerpts, figures, tables, and headline JSON.
