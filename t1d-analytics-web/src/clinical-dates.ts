/**
 * @file clinical-dates.ts
 * @module clinical-dates
 * Clinical date/time parsing and observation-window filtering utilities for CGM datasets.
 */

/**
 * Primitive or Date cell value accepted in clinical CGM dataset rows.
 */
export type ClinicalRowValue =
  string | number | boolean | Date | null | undefined;

/**
 * Input value accepted by clinical date/time parsers.
 */
export type ClinicalDateInput = ClinicalRowValue | Record<string, never>;

/**
 * Parsed clinical date structure containing day identifier, minute of day, and epoch milliseconds.
 */
export interface ParsedClinicalDate {
  /** Calendar day key formatted with 1-indexed month as `YYYY-M-D`. */
  dayKey: string;
  /** Minute offset from midnight (0 to 1439). */
  minuteOfDay: number;
  /** Epoch timestamp in milliseconds. */
  epochMs: number;
}

/**
 * Lookup mapping 3-letter lowercase English month abbreviations to 1-indexed month numbers (1-12).
 */
const MONTH_NAMES: Record<string, number> = {
  jan: 1,
  feb: 2,
  mar: 3,
  apr: 4,
  may: 5,
  jun: 6,
  jul: 7,
  aug: 8,
  sep: 9,
  oct: 10,
  nov: 11,
  dec: 12,
};

/**
 * Formats a 1-indexed calendar `(year, month1Based, day)` tuple into a `YYYY-M-D` day key.
 * @param {number} year - Four-digit calendar year.
 * @param {number} month1Based - 1-indexed calendar month (1 = January, 12 = December).
 * @param {number} day - Day of month.
 * @returns {string} Canonical `YYYY-M-D` day key.
 */
function formatDayKey(year: number, month1Based: number, day: number): string {
  return `${year}-${month1Based}-${day}`;
}

/**
 * Builds a `ParsedClinicalDate` preserving wall-clock `dayKey` and `minuteOfDay` while resolving `epochMs` with optional timezone offset.
 * @param {number} year - Four-digit year.
 * @param {number} month1Based - 1-indexed month (1-12).
 * @param {number} day - Day of month (1-31).
 * @param {number} hours - Hours (0-23).
 * @param {number} minutes - Minutes (0-59).
 * @param {number} seconds - Seconds (0-59).
 * @param {string | undefined} tz - Optional ISO timezone suffix (`Z` or `±HH:MM`).
 * @returns {ParsedClinicalDate} Constructed clinical date record.
 */
function buildWallClockClinicalDate(
  year: number,
  month1Based: number,
  day: number,
  hours: number,
  minutes: number,
  seconds: number,
  tz: string | undefined,
): ParsedClinicalDate {
  const dayKey = formatDayKey(year, month1Based, day);
  const minuteOfDay = hours * 60 + minutes;
  if (tz && tz.toUpperCase() !== "Z") {
    const isoStr = `${year}-${String(month1Based).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}${tz}`;
    const parsedTzDate = new Date(isoStr);
    if (!isNaN(parsedTzDate.getTime())) {
      return { dayKey, minuteOfDay, epochMs: parsedTzDate.getTime() };
    }
  }
  return {
    dayKey,
    minuteOfDay,
    epochMs: Date.UTC(year, month1Based - 1, day, hours, minutes, seconds),
  };
}

/**
 * Extracts a UTC-based `ParsedClinicalDate` from a validated `Date` instance.
 * @param {Date} d - Validated JavaScript Date object.
 * @returns {ParsedClinicalDate} UTC-normalized clinical date record.
 */
function buildUtcClinicalDate(d: Date): ParsedClinicalDate {
  return {
    dayKey: formatDayKey(
      d.getUTCFullYear(),
      d.getUTCMonth() + 1,
      d.getUTCDate(),
    ),
    minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
    epochMs: d.getTime(),
  };
}

/**
 * Parses an ISO or CDISC clinical date string into `ParsedClinicalDate`, or returns null.
 * @param {string} trimmed - Trimmed raw date string.
 * @returns {ParsedClinicalDate | null} Parsed date or null if neither ISO nor CDISC matched.
 */
function parseIsoOrCdiscString(trimmed: string): ParsedClinicalDate | null {
  const isoMatch = trimmed.match(
    /^(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:[T\s](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/i,
  );
  if (isoMatch) {
    return buildWallClockClinicalDate(
      parseInt(String(isoMatch[1]), 10),
      parseInt(String(isoMatch[2]), 10),
      parseInt(String(isoMatch[3]), 10),
      isoMatch[4] ? parseInt(isoMatch[4], 10) : 0,
      isoMatch[5] ? parseInt(isoMatch[5], 10) : 0,
      isoMatch[6] ? parseInt(isoMatch[6], 10) : 0,
      isoMatch[7],
    );
  }

  const cdiscMatch = trimmed.match(
    /^(\d{1,2})[-/]([a-zA-Z]{3})[-/](\d{4})(?:[T\s](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/i,
  );
  if (cdiscMatch) {
    const month1Based = MONTH_NAMES[String(cdiscMatch[2]).toLowerCase()] ?? 1;
    return buildWallClockClinicalDate(
      parseInt(String(cdiscMatch[3]), 10),
      month1Based,
      parseInt(String(cdiscMatch[1]), 10),
      cdiscMatch[4] ? parseInt(cdiscMatch[4], 10) : 0,
      cdiscMatch[5] ? parseInt(cdiscMatch[5], 10) : 0,
      cdiscMatch[6] ? parseInt(cdiscMatch[6], 10) : 0,
      cdiscMatch[7],
    );
  }
  return null;
}

/**
 * Parses a slash-separated (`MM/DD/YYYY` or `DD/MM/YYYY`) or UTC fallback string into `ParsedClinicalDate`.
 * @param {string} trimmed - Trimmed raw date string.
 * @returns {ParsedClinicalDate | null} Parsed clinical date or null if unparseable.
 */
function parseSlashOrFallbackString(
  trimmed: string,
): ParsedClinicalDate | null {
  const slashMatch = trimmed.match(
    /^(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:[T\s](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/i,
  );
  if (slashMatch) {
    const part1 = parseInt(String(slashMatch[1]), 10);
    const part2 = parseInt(String(slashMatch[2]), 10);
    const isEuOrder = part1 > 12 && part2 <= 12;
    const month1Based = isEuOrder ? part2 : part1;
    const day = isEuOrder ? part1 : part2;
    return buildWallClockClinicalDate(
      parseInt(String(slashMatch[3]), 10),
      month1Based,
      day,
      slashMatch[4] ? parseInt(slashMatch[4], 10) : 0,
      slashMatch[5] ? parseInt(slashMatch[5], 10) : 0,
      slashMatch[6] ? parseInt(slashMatch[6], 10) : 0,
      slashMatch[7],
    );
  }

  const hasExplicitTz =
    /(?:GMT|UTC)?[+-]\d{2}:?\d{2}$/i.test(trimmed) ||
    /(?:GMT|UTC|Z)$/i.test(trimmed);
  const fallbackInput = hasExplicitTz ? trimmed : `${trimmed} UTC`;
  const dUtc = new Date(fallbackInput);
  return !isNaN(dUtc.getTime()) ? buildUtcClinicalDate(dUtc) : null;
}

/**
 * Parse a raw timestamp value into date day key and minute of the day without timezone shifts.
 * @param {ClinicalDateInput} rawTime - The raw date/time string, Date, or timestamp number.
 * @returns {ParsedClinicalDate | null} Parsed day key, minute of day, and epoch milliseconds.
 */
export function parseDateAndMinuteOfDay(
  rawTime: ClinicalDateInput,
): ParsedClinicalDate | null {
  if (rawTime instanceof Date) {
    return isNaN(rawTime.getTime()) ? null : buildUtcClinicalDate(rawTime);
  }

  if (typeof rawTime === "string") {
    const trimmed = rawTime.trim();
    return (
      parseIsoOrCdiscString(trimmed) ?? parseSlashOrFallbackString(trimmed)
    );
  }

  if (typeof rawTime === "number") {
    const epoch = rawTime < 1e11 ? rawTime * 1000 : rawTime;
    const d = new Date(epoch);
    return !isNaN(d.getTime()) ? buildUtcClinicalDate(d) : null;
  }

  return null;
}

/**
 * Filter CGM rows to a configurable observation window.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - Full CGM dataset rows.
 * @param {string} timeCol - Column name representing date/time.
 * @param {number} [days=14] - Duration of the observation window in days (0 for all data).
 * @returns {Array<Record<string, ClinicalRowValue>>} Filtered rows within the window of latest reading.
 */
export function filterObservationWindow(
  rows: Array<Record<string, ClinicalRowValue>>,
  timeCol: string,
  days: number = 14,
): Array<Record<string, ClinicalRowValue>> {
  if (days <= 0) return rows;
  let maxTime = -Infinity;
  const parsedRows: Array<{
    row: Record<string, ClinicalRowValue>;
    timeMs: number;
  }> = [];

  for (const r of rows) {
    const rawTime = r[timeCol];
    const parsed = parseDateAndMinuteOfDay(rawTime);
    if (!parsed) continue;

    const timeMs = parsed.epochMs;
    if (timeMs > 0) {
      parsedRows.push({ row: r, timeMs });
      if (timeMs > maxTime) maxTime = timeMs;
    }
  }

  if (maxTime === -Infinity) return rows;

  const cutoff = maxTime - days * 24 * 60 * 60 * 1000;
  return parsedRows
    .filter((item) => item.timeMs >= cutoff)
    .map((item) => item.row);
}

/**
 * Filter CGM rows to the most recent 14-day window.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - Full CGM dataset rows.
 * @param {string} timeCol - Column name representing date/time.
 * @returns {Array<Record<string, ClinicalRowValue>>} Filtered rows within 14 days of latest reading.
 */
export function filter14DayWindow(
  rows: Array<Record<string, ClinicalRowValue>>,
  timeCol: string,
): Array<Record<string, ClinicalRowValue>> {
  return filterObservationWindow(rows, timeCol, 14);
}
