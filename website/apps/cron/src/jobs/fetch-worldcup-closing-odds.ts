/**
 * Purpose: Fetches the final pre-kickoff 1X2 odds snapshots for FIFA World Cup 2026 matches.
 * Historical responses are cached so interrupted or repeated runs do not spend credits twice.
 */
import "../load-env";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createSqliteDb, getDefaultDbPath, listMatches } from "@llm-kicktipp/db";
import type { MatchRow } from "@llm-kicktipp/db";

const API_BASE_URL = "https://api.the-odds-api.com";
const SPORT_KEY = "soccer_fifa_world_cup";
const MARKET_KEY = "h2h";
const TOURNAMENT_EDITION = "FIFA World Cup 2026";
const HISTORICAL_CREDITS_PER_CALL = 10;
const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");

type CliOptions = {
  dryRun: boolean;
  force: boolean;
  from: string | null;
  to: string | null;
  limit: number | null;
  matchId: string | null;
  outputDir: string;
  region: string;
  snapshotBeforeSeconds: number;
};

type OddsOutcome = {
  name: string;
  price: number;
  point?: number;
};

type OddsMarket = {
  key: string;
  last_update?: string;
  outcomes: OddsOutcome[];
};

type OddsBookmaker = {
  key: string;
  title: string;
  last_update?: string;
  markets: OddsMarket[];
};

export type OddsEvent = {
  id: string;
  sport_key: string;
  commence_time: string;
  home_team: string;
  away_team: string;
  bookmakers: OddsBookmaker[];
};

type HistoricalOddsResponse = {
  timestamp: string;
  previous_timestamp: string | null;
  next_timestamp: string | null;
  data: OddsEvent[];
};

type CachedSnapshot = {
  request: {
    commenceTimeFrom: string;
    commenceTimeTo: string;
    date: string;
    market: string;
    region: string;
    sport: string;
  };
  quota: QuotaHeaders;
  response: HistoricalOddsResponse;
};

type QuotaHeaders = {
  last: number | null;
  remaining: number | null;
  used: number | null;
};

export type ClosingOddsRow = {
  match_id: string;
  source_match_id: string | null;
  odds_event_id: string;
  kickoff_utc: string;
  requested_snapshot_utc: string;
  actual_snapshot_utc: string;
  home_team: string;
  away_team: string;
  bookmaker_key: string;
  bookmaker_title: string;
  bookmaker_last_update_utc: string | null;
  market_last_update_utc: string | null;
  home_odds_decimal: number | null;
  draw_odds_decimal: number | null;
  away_odds_decimal: number | null;
  overround: number | null;
  fair_home_probability: number | null;
  fair_draw_probability: number | null;
  fair_away_probability: number | null;
};

type MatchResult = {
  awayTeam: string;
  bookmakerCount: number;
  eventId: string | null;
  homeTeam: string;
  kickoffUtc: string;
  matchId: string;
  reason: string | null;
};

type ApiErrorBody = {
  error_code?: string;
  message?: string;
};

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const db = createSqliteDb();
  const allMatches = await listMatches(db);
  db.close();

  const selectedMatches = selectMatches(allMatches, options);
  if (selectedMatches.length === 0) {
    throw new Error("No World Cup 2026 matches matched the supplied filters.");
  }

  const kickoffGroups = groupByKickoff(selectedMatches);
  const estimatedCredits = kickoffGroups.length * HISTORICAL_CREDITS_PER_CALL;

  console.log(`SQLite DB: ${getDefaultDbPath()}`);
  console.log(`Matches: ${selectedMatches.length}`);
  console.log(`Distinct kickoff snapshots/API calls: ${kickoffGroups.length}`);
  console.log(`Estimated historical API credits: ${estimatedCredits}`);
  console.log(`Region/market: ${options.region}/${MARKET_KEY}`);

  if (options.dryRun) {
    console.log("Dry run complete. No API request was made.");
    return;
  }

  mkdirSync(options.outputDir, { recursive: true });
  const rawDir = resolve(options.outputDir, "raw");
  mkdirSync(rawDir, { recursive: true });

  const apiKey = process.env.THE_ODDS_API_KEY ?? process.env.ODDS_API_KEY;
  const networkSnapshotsNeeded = options.force
    ? kickoffGroups.length
    : kickoffGroups.filter((group) => {
      const kickoff = new Date(group.kickoffUtc);
      const snapshot = toApiIso(new Date(kickoff.getTime() - options.snapshotBeforeSeconds * 1_000));
      return !existsSync(resolve(rawDir, cacheFileName(options.region, snapshot)));
    }).length;
  console.log(`Network snapshots needed: ${networkSnapshotsNeeded}`);

  if (networkSnapshotsNeeded > 0) {
    if (!apiKey) {
      throw new Error("Missing THE_ODDS_API_KEY. Put it in the repository .env file or set it for this process.");
    }
    const preflightQuota = await validateApiKey(apiKey);
    if (preflightQuota.remaining !== null) {
      console.log(`API credits remaining before fetch: ${preflightQuota.remaining}`);
    }
  }

  const rows: ClosingOddsRow[] = [];
  const matchResults: MatchResult[] = [];
  let creditsSpentThisRun = 0;
  let networkCalls = 0;
  let cachedCalls = 0;

  for (let index = 0; index < kickoffGroups.length; index += 1) {
    const group = kickoffGroups[index];
    const kickoff = new Date(group.kickoffUtc);
    const requestedSnapshot = toApiIso(new Date(kickoff.getTime() - options.snapshotBeforeSeconds * 1_000));
    const cachePath = resolve(rawDir, cacheFileName(options.region, requestedSnapshot));
    const request = buildRequestMetadata(group.kickoffUtc, requestedSnapshot, options.region);

    let snapshot: CachedSnapshot;
    if (!options.force && existsSync(cachePath)) {
      snapshot = readCachedSnapshot(cachePath, request);
      cachedCalls += 1;
      console.log(`[${index + 1}/${kickoffGroups.length}] cache ${group.kickoffUtc}`);
    } else {
      if (!apiKey) {
        throw new Error("Missing THE_ODDS_API_KEY for an uncached historical snapshot.");
      }
      snapshot = await fetchHistoricalSnapshot(apiKey, request);
      writeFileSync(cachePath, `${JSON.stringify(snapshot, null, 2)}\n`, "utf8");
      networkCalls += 1;
      creditsSpentThisRun += snapshot.quota.last ?? HISTORICAL_CREDITS_PER_CALL;
      const remaining = snapshot.quota.remaining === null ? "unknown" : String(snapshot.quota.remaining);
      console.log(`[${index + 1}/${kickoffGroups.length}] fetched ${group.kickoffUtc} (remaining: ${remaining})`);
    }

    for (const match of group.matches) {
      const event = findMatchingEvent(match, snapshot.response.data);
      if (!event) {
        matchResults.push({
          awayTeam: match.away_team,
          bookmakerCount: 0,
          eventId: null,
          homeTeam: match.home_team,
          kickoffUtc: new Date(match.utc_date).toISOString(),
          matchId: match.id,
          reason: "No API event matched kickoff and teams"
        });
        continue;
      }

      const eventRows = extractClosingOddsRows(match, event, requestedSnapshot, snapshot.response.timestamp);
      rows.push(...eventRows);
      matchResults.push({
        awayTeam: match.away_team,
        bookmakerCount: eventRows.length,
        eventId: event.id,
        homeTeam: match.home_team,
        kickoffUtc: new Date(match.utc_date).toISOString(),
        matchId: match.id,
        reason: eventRows.length > 0 ? null : "Matched event had no h2h bookmaker odds"
      });
    }
  }

  const unmatchedMatches = matchResults.filter((result) => result.reason !== null);
  const metadata = {
    generated_at_utc: new Date().toISOString(),
    source: "The Odds API historical odds endpoint",
    sport_key: SPORT_KEY,
    market: MARKET_KEY,
    region: options.region,
    closing_definition: `closest historical API snapshot at or before kickoff minus ${options.snapshotBeforeSeconds} second(s)`,
    selected_matches: selectedMatches.length,
    matched_matches: matchResults.length - unmatchedMatches.length,
    unmatched_matches: unmatchedMatches.length,
    bookmaker_rows: rows.length,
    distinct_kickoffs: kickoffGroups.length,
    network_calls: networkCalls,
    cached_calls: cachedCalls,
    credits_spent_this_run: creditsSpentThisRun
  };

  const jsonPath = resolve(options.outputDir, "closing-odds.json");
  const csvPath = resolve(options.outputDir, "closing-odds.csv");
  const summaryPath = resolve(options.outputDir, "summary.json");

  writeFileSync(jsonPath, `${JSON.stringify({ metadata, rows }, null, 2)}\n`, "utf8");
  writeFileSync(csvPath, toCsv(rows), "utf8");
  writeFileSync(summaryPath, `${JSON.stringify({ metadata, matches: matchResults }, null, 2)}\n`, "utf8");

  console.log(`Wrote ${rows.length} bookmaker rows to ${csvPath}`);
  console.log(`Wrote normalized JSON to ${jsonPath}`);
  console.log(`Wrote match summary to ${summaryPath}`);
  if (unmatchedMatches.length > 0) {
    console.warn(`Warning: ${unmatchedMatches.length} match(es) had no closing h2h odds. See summary.json.`);
  }
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    dryRun: false,
    force: false,
    from: null,
    to: null,
    limit: null,
    matchId: null,
    outputDir: resolve(REPOSITORY_ROOT, "exports/worldcup2026_closing_odds"),
    region: "eu",
    snapshotBeforeSeconds: 1
  };

  for (const arg of argv) {
    if (arg === "--dry-run") {
      options.dryRun = true;
    } else if (arg === "--force") {
      options.force = true;
    } else if (arg === "--help") {
      printHelp();
      process.exit(0);
    } else if (arg.startsWith("--from=")) {
      options.from = parseDateArg(arg, "--from=");
    } else if (arg.startsWith("--to=")) {
      options.to = parseDateArg(arg, "--to=");
    } else if (arg.startsWith("--limit=")) {
      options.limit = parsePositiveInteger(arg, "--limit=");
    } else if (arg.startsWith("--match-id=")) {
      options.matchId = requiredArgValue(arg, "--match-id=");
    } else if (arg.startsWith("--output-dir=")) {
      options.outputDir = resolve(REPOSITORY_ROOT, requiredArgValue(arg, "--output-dir="));
    } else if (arg.startsWith("--region=")) {
      options.region = parseRegion(requiredArgValue(arg, "--region="));
    } else if (arg.startsWith("--snapshot-before-seconds=")) {
      options.snapshotBeforeSeconds = parsePositiveInteger(arg, "--snapshot-before-seconds=");
    } else {
      throw new Error(`Unknown argument: ${arg}. Run with --help for usage.`);
    }
  }

  return options;
}

function printHelp(): void {
  console.log(`Fetch FIFA World Cup 2026 closing 1X2 odds from The Odds API.

Usage:
  npm run odds:closing -- --dry-run
  npm run odds:closing -- --limit=1
  npm run odds:closing

Options:
  --dry-run                       Show calls and estimated credits without using the API
  --force                         Ignore cached raw snapshots and fetch them again
  --from=<ISO8601>                Include kickoffs at or after this time
  --to=<ISO8601>                  Include kickoffs at or before this time
  --limit=<n>                     Limit the number of local matches
  --match-id=<id>                 Fetch one local match ID
  --region=<eu|uk|us|us2|au>     Bookmaker region (default: eu)
  --snapshot-before-seconds=<n>  Requested pre-kickoff offset (default: 1)
  --output-dir=<path>             Output directory
  --help                          Show this help

The API key is read from THE_ODDS_API_KEY (ODDS_API_KEY is also accepted).`);
}

function selectMatches(matches: MatchRow[], options: CliOptions): MatchRow[] {
  const now = Date.now();
  const from = options.from ? Date.parse(options.from) : Number.NEGATIVE_INFINITY;
  const to = options.to ? Date.parse(options.to) : Number.POSITIVE_INFINITY;
  const selected = matches.filter((match) => {
    const kickoff = Date.parse(match.utc_date);
    return match.tournament_edition === TOURNAMENT_EDITION
      && Number.isFinite(kickoff)
      && kickoff < now
      && kickoff >= from
      && kickoff <= to
      && (options.matchId === null || match.id === options.matchId);
  });

  return options.limit === null ? selected : selected.slice(0, options.limit);
}

function groupByKickoff(matches: MatchRow[]): Array<{ kickoffUtc: string; matches: MatchRow[] }> {
  const groups = new Map<string, MatchRow[]>();
  for (const match of matches) {
    const kickoffUtc = new Date(match.utc_date).toISOString();
    const group = groups.get(kickoffUtc) ?? [];
    group.push(match);
    groups.set(kickoffUtc, group);
  }

  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([kickoffUtc, groupedMatches]) => ({ kickoffUtc, matches: groupedMatches }));
}

function buildRequestMetadata(kickoffUtc: string, date: string, region: string): CachedSnapshot["request"] {
  const kickoff = Date.parse(kickoffUtc);
  return {
    commenceTimeFrom: toApiIso(new Date(kickoff - 60_000)),
    commenceTimeTo: toApiIso(new Date(kickoff + 60_000)),
    date,
    market: MARKET_KEY,
    region,
    sport: SPORT_KEY
  };
}

async function validateApiKey(apiKey: string): Promise<QuotaHeaders> {
  const url = new URL("/v4/sports/", API_BASE_URL);
  url.searchParams.set("apiKey", apiKey);
  url.searchParams.set("all", "true");
  const response = await fetch(url);
  const body = await readJson(response);
  if (!response.ok) {
    throw apiError("API key validation failed", response.status, body);
  }
  if (!Array.isArray(body) || !body.some((sport) => isRecord(sport) && sport.key === SPORT_KEY)) {
    throw new Error(`The API key is valid, but ${SPORT_KEY} was not present in the all-sports response.`);
  }
  return readQuotaHeaders(response);
}

async function fetchHistoricalSnapshot(
  apiKey: string,
  request: CachedSnapshot["request"]
): Promise<CachedSnapshot> {
  const url = new URL(`/v4/historical/sports/${request.sport}/odds`, API_BASE_URL);
  url.searchParams.set("apiKey", apiKey);
  url.searchParams.set("regions", request.region);
  url.searchParams.set("markets", request.market);
  url.searchParams.set("oddsFormat", "decimal");
  url.searchParams.set("dateFormat", "iso");
  url.searchParams.set("date", request.date);
  url.searchParams.set("commenceTimeFrom", request.commenceTimeFrom);
  url.searchParams.set("commenceTimeTo", request.commenceTimeTo);

  let response: Response | null = null;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    response = await fetch(url);
    if (response.status !== 429 && response.status < 500) {
      break;
    }
    if (attempt < 3) {
      await delay(500 * 2 ** (attempt - 1));
    }
  }

  if (!response) {
    throw new Error("Historical odds request did not produce a response.");
  }
  const body = await readJson(response);
  if (!response.ok) {
    throw apiError("Historical odds request failed", response.status, body);
  }
  if (!isHistoricalOddsResponse(body)) {
    throw new Error("Historical odds response did not match the expected { timestamp, data } schema.");
  }

  return {
    request,
    quota: readQuotaHeaders(response),
    response: body
  };
}

function readCachedSnapshot(path: string, expectedRequest: CachedSnapshot["request"]): CachedSnapshot {
  const parsed = JSON.parse(readFileSync(path, "utf8")) as CachedSnapshot;
  if (JSON.stringify(parsed.request) !== JSON.stringify(expectedRequest) || !isHistoricalOddsResponse(parsed.response)) {
    throw new Error(`Cached snapshot is incompatible or invalid: ${path}. Use --force to replace it.`);
  }
  return parsed;
}

export function findMatchingEvent(match: MatchRow, events: OddsEvent[]): OddsEvent | null {
  const kickoff = Date.parse(match.utc_date);
  const candidates = events.filter((event) => Math.abs(Date.parse(event.commence_time) - kickoff) <= 60_000);
  const exact = candidates.find((event) =>
    canonicalTeamName(event.home_team) === canonicalTeamName(match.home_team)
    && canonicalTeamName(event.away_team) === canonicalTeamName(match.away_team)
  );
  if (exact) {
    return exact;
  }
  return candidates.length === 1 ? candidates[0] : null;
}

export function extractClosingOddsRows(
  match: MatchRow,
  event: OddsEvent,
  requestedSnapshotUtc: string,
  actualSnapshotUtc: string
): ClosingOddsRow[] {
  const homeName = canonicalTeamName(match.home_team);
  const awayName = canonicalTeamName(match.away_team);

  return event.bookmakers.flatMap((bookmaker) => {
    const market = bookmaker.markets.find((candidate) => candidate.key === MARKET_KEY);
    if (!market) {
      return [];
    }

    const homeOdds = findOutcomePrice(market.outcomes, (name) => canonicalTeamName(name) === homeName);
    const awayOdds = findOutcomePrice(market.outcomes, (name) => canonicalTeamName(name) === awayName);
    const drawOdds = findOutcomePrice(market.outcomes, (name) => canonicalTeamName(name) === "draw");
    const probabilities = fairProbabilities(homeOdds, drawOdds, awayOdds);

    return [{
      match_id: match.id,
      source_match_id: match.source_match_id ?? null,
      odds_event_id: event.id,
      kickoff_utc: new Date(match.utc_date).toISOString(),
      requested_snapshot_utc: requestedSnapshotUtc,
      actual_snapshot_utc: actualSnapshotUtc,
      home_team: match.home_team,
      away_team: match.away_team,
      bookmaker_key: bookmaker.key,
      bookmaker_title: bookmaker.title,
      bookmaker_last_update_utc: bookmaker.last_update ?? null,
      market_last_update_utc: market.last_update ?? null,
      home_odds_decimal: homeOdds,
      draw_odds_decimal: drawOdds,
      away_odds_decimal: awayOdds,
      overround: probabilities?.overround ?? null,
      fair_home_probability: probabilities?.home ?? null,
      fair_draw_probability: probabilities?.draw ?? null,
      fair_away_probability: probabilities?.away ?? null
    }];
  });
}

export function canonicalTeamName(value: string): string {
  const normalized = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
  const aliases: Record<string, string> = {
    caboverde: "capeverde",
    capeverdeislands: "capeverde",
    congodr: "drcongo",
    czechrepublic: "czechia",
    democraticrepublicofthecongo: "drcongo",
    iriran: "iran",
    ivorycoast: "cotedivoire",
    korearepublic: "southkorea",
    usa: "unitedstates",
    unitedstatesofamerica: "unitedstates"
  };
  return aliases[normalized] ?? normalized;
}

function findOutcomePrice(outcomes: OddsOutcome[], predicate: (name: string) => boolean): number | null {
  const price = outcomes.find((outcome) => predicate(outcome.name))?.price;
  return typeof price === "number" && Number.isFinite(price) && price > 1 ? price : null;
}

function fairProbabilities(homeOdds: number | null, drawOdds: number | null, awayOdds: number | null): {
  away: number;
  draw: number;
  home: number;
  overround: number;
} | null {
  if (homeOdds === null || drawOdds === null || awayOdds === null) {
    return null;
  }
  const rawHome = 1 / homeOdds;
  const rawDraw = 1 / drawOdds;
  const rawAway = 1 / awayOdds;
  const overround = rawHome + rawDraw + rawAway;
  return {
    home: rawHome / overround,
    draw: rawDraw / overround,
    away: rawAway / overround,
    overround
  };
}

function toCsv(rows: ClosingOddsRow[]): string {
  const headers: Array<keyof ClosingOddsRow> = [
    "match_id",
    "source_match_id",
    "odds_event_id",
    "kickoff_utc",
    "requested_snapshot_utc",
    "actual_snapshot_utc",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update_utc",
    "market_last_update_utc",
    "home_odds_decimal",
    "draw_odds_decimal",
    "away_odds_decimal",
    "overround",
    "fair_home_probability",
    "fair_draw_probability",
    "fair_away_probability"
  ];
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvCell(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function cacheFileName(region: string, requestedSnapshot: string): string {
  return `${region}-${MARKET_KEY}-${requestedSnapshot.replace(/[:.]/g, "-")}.json`;
}

function readQuotaHeaders(response: Response): QuotaHeaders {
  return {
    last: parseHeaderNumber(response.headers.get("x-requests-last")),
    remaining: parseHeaderNumber(response.headers.get("x-requests-remaining")),
    used: parseHeaderNumber(response.headers.get("x-requests-used"))
  };
}

function parseHeaderNumber(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { message: text.slice(0, 500) };
  }
}

function apiError(prefix: string, status: number, body: unknown): Error {
  const errorBody = isRecord(body) ? body as ApiErrorBody : {};
  const details = [errorBody.error_code, errorBody.message].filter(Boolean).join(": ");
  return new Error(`${prefix} (HTTP ${status})${details ? `: ${details}` : ""}`);
}

function isHistoricalOddsResponse(value: unknown): value is HistoricalOddsResponse {
  return isRecord(value) && typeof value.timestamp === "string" && Array.isArray(value.data);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseDateArg(arg: string, prefix: string): string {
  const value = requiredArgValue(arg, prefix);
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) {
    throw new Error(`Invalid ISO8601 date for ${prefix.slice(0, -1)}: ${value}`);
  }
  return date.toISOString();
}

function parsePositiveInteger(arg: string, prefix: string): number {
  const value = Number(requiredArgValue(arg, prefix));
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${prefix.slice(0, -1)} must be a positive integer.`);
  }
  return value;
}

function requiredArgValue(arg: string, prefix: string): string {
  const value = arg.slice(prefix.length).trim();
  if (!value) {
    throw new Error(`${prefix.slice(0, -1)} requires a value.`);
  }
  return value;
}

function parseRegion(value: string): string {
  const allowed = new Set(["eu", "uk", "us", "us2", "au"]);
  if (!allowed.has(value)) {
    throw new Error(`Unsupported region: ${value}. Use one of ${[...allowed].join(", ")}.`);
  }
  return value;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

function toApiIso(date: Date): string {
  return date.toISOString().replace(/\.\d{3}Z$/, "Z");
}

const isDirectExecution = process.argv[1]
  && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isDirectExecution) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
