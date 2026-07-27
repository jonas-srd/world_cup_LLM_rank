import assert from "node:assert/strict";
import test from "node:test";
import type { MatchRow } from "@llm-kicktipp/db";
import {
  canonicalTeamName,
  extractClosingOddsRows,
  findMatchingEvent
} from "./fetch-worldcup-closing-odds";
import type { OddsEvent } from "./fetch-worldcup-closing-odds";

const match: MatchRow = {
  id: "match-1",
  utc_date: "2026-06-12T01:00:00Z",
  competition: "FIFA World Cup - GROUP_STAGE",
  home_team: "United States",
  away_team: "South Korea",
  venue: null,
  status: "FINISHED",
  home_score: 1,
  away_score: 0,
  source_match_id: "123",
  tournament_edition: "FIFA World Cup 2026"
};

const event: OddsEvent = {
  id: "odds-event-1",
  sport_key: "soccer_fifa_world_cup",
  commence_time: "2026-06-12T01:00:00Z",
  home_team: "USA",
  away_team: "Korea Republic",
  bookmakers: [{
    key: "pinnacle",
    title: "Pinnacle",
    last_update: "2026-06-12T00:58:00Z",
    markets: [{
      key: "h2h",
      last_update: "2026-06-12T00:58:00Z",
      outcomes: [
        { name: "USA", price: 2 },
        { name: "Draw", price: 4 },
        { name: "Korea Republic", price: 4 }
      ]
    }]
  }]
};

test("canonicalTeamName handles common provider aliases", () => {
  assert.equal(canonicalTeamName("USA"), canonicalTeamName("United States"));
  assert.equal(canonicalTeamName("Korea Republic"), canonicalTeamName("South Korea"));
  assert.equal(canonicalTeamName("Côte d'Ivoire"), canonicalTeamName("Ivory Coast"));
  assert.equal(canonicalTeamName("Czech Republic"), canonicalTeamName("Czechia"));
  assert.equal(canonicalTeamName("Cape Verde"), canonicalTeamName("Cape Verde Islands"));
});

test("findMatchingEvent matches kickoff and aliased team names", () => {
  assert.equal(findMatchingEvent(match, [event])?.id, event.id);
});

test("extractClosingOddsRows calculates vig-free probabilities", () => {
  const rows = extractClosingOddsRows(
    match,
    event,
    "2026-06-12T00:59:59Z",
    "2026-06-12T00:55:00Z"
  );
  assert.equal(rows.length, 1);
  assert.equal(rows[0].home_odds_decimal, 2);
  assert.equal(rows[0].draw_odds_decimal, 4);
  assert.equal(rows[0].away_odds_decimal, 4);
  assert.equal(rows[0].overround, 1);
  assert.equal(rows[0].fair_home_probability, 0.5);
  assert.equal(rows[0].fair_draw_probability, 0.25);
  assert.equal(rows[0].fair_away_probability, 0.25);
});
