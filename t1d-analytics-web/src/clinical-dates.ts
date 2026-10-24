/**
 * @file clinical-dates.ts
 * @module clinical-dates
 * Clinical date/time parsing and observation-window filtering utilities for CGM datasets.
 */

/**
 * Parsed clinical date structure containing day identifier, minute of day, and epoch milliseconds.
 */
export interface ParsedClinicalDate {
  /** Calendar day key formatted as YYYY-M-D. */
  dayKey: string;
  /** Minute offset from midnight (0 to 1439). */
  minuteOfDay: number;
  /** Epoch timestamp in milliseconds. */
  epochMs: number;
}

/**
 * Lookup mapping 3-letter lowercase English month abbreviations to 0-indexed month numbers (0-11).
 */
const MONTH_NAMES: Record<string, number> = {
  jan: 0,
  feb: 1,
  mar: 2,
  apr: 3,
  may: 4,
  jun: 5,
  jul: 6,
  aug: 7,
  sep: 8,
  oct: 9,
  nov: 10,
  dec: 11,
};

/**
 * Parse a raw timestamp value into date day key and minute of the day without timezone shifts.
 * @param {unknown} rawTime - The raw date/time string, Date, or timestamp number.
 * @returns {ParsedClinicalDate | null} Parsed day key, minute of day, and epoch milliseconds.
 */
export function parseDateAndMinuteOfDay(
  rawTime: unknown,
): ParsedClinicalDate | null {
  if (rawTime instanceof Date) {
    const epochMs = rawTime.getTime();
    if (isNaN(epochMs)) return null;
    const dayKey = `${rawTime.getUTCFullYear()}-${rawTime.getUTCMonth()}-${rawTime.getUTCDate()}`;
    const minuteOfDay = rawTime.getUTCHours() * 60 + rawTime.getUTCMinutes();
    return { dayKey, minuteOfDay, epochMs };
  }

  if (typeof rawTime === "string") {
    const trimmed = rawTime.trim();

    // 1. ISO format: YYYY-MM-DD or YYYY/MM/DD with optional HH:mm[:ss]
    const isoMatch = trimmed.match(
      /^(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:[T\s](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/i,
    );
    if (isoMatch) {
      const year = parseInt(isoMatch[1], 10);
      const month = parseInt(isoMatch[2], 10) - 1;
      const day = parseInt(isoMatch[3], 10);
      const hours = isoMatch[4] ? parseInt(isoMatch[4], 10) : 0;
      const minutes = isoMatch[5] ? parseInt(isoMatch[5], 10) : 0;
      const seconds = isoMatch[6] ? parseInt(isoMatch[6], 10) : 0;
      const tz = isoMatch[7];

      if (tz && tz.toUpperCase() === "Z") {
        return {
          dayKey: `${year}-${month}-${day}`,
          minuteOfDay: hours * 60 + minutes,
          epochMs: Date.UTC(year, month, day, hours, minutes, seconds),
        };
      } else if (tz) {
        const d = new Date(trimmed);
        if (!isNaN(d.getTime())) {
          return {
            dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
            minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
            epochMs: d.getTime(),
          };
        }
      }
      return {
        dayKey: `${year}-${month}-${day}`,
        minuteOfDay: hours * 60 + minutes,
        epochMs: Date.UTC(year, month, day, hours, minutes, seconds),
      };
    }

    // 2. CDISC / SAS clinical format: DD-Mon-YYYY with optional HH:mm[:ss]
    const cdiscMatch = trimmed.match(
      /^(\d{1,2})[-/]([a-zA-Z]{3})[-/](\d{4})(?:[T\s](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/i,
    );
    if (cdiscMatch) {
      const day = parseInt(cdiscMatch[1], 10);
      const monthStr = cdiscMatch[2].toLowerCase();
      const month = MONTH_NAMES[monthStr] ?? 0;
      const year = parseInt(cdiscMatch[3], 10);
      const hours = cdiscMatch[4] ? parseInt(cdiscMatch[4], 10) : 0;
      const minutes = cdiscMatch[5] ? parseInt(cdiscMatch[5], 10) : 0;
      const seconds = cdiscMatch[6] ? parseInt(cdiscMatch[6], 10) : 0;
      const tz = cdiscMatch[7];

      if (tz && tz.toUpperCase() !== "Z") {
        const isoStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}${tz}`;
        const d = new Date(isoStr);
        if (!isNaN(d.getTime())) {
          return {
            dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
            minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
            epochMs: d.getTime(),
          };
        }
      }
      return {
        dayKey: `${year}-${month}-${day}`,
        minuteOfDay: hours * 60 + minutes,
        epochMs: Date.UTC(year, month, day, hours, minutes, seconds),
      };
    }

    // 3. US (MM/DD/YYYY) or European (DD/MM/YYYY) format
    const slashMatch = trimmed.match(
      /^(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:[T\s](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/i,
    );
    if (slashMatch) {
      const part1 = parseInt(slashMatch[1], 10);
      const part2 = parseInt(slashMatch[2], 10);
      const year = parseInt(slashMatch[3], 10);
      const hours = slashMatch[4] ? parseInt(slashMatch[4], 10) : 0;
      const minutes = slashMatch[5] ? parseInt(slashMatch[5], 10) : 0;
      const seconds = slashMatch[6] ? parseInt(slashMatch[6], 10) : 0;
      const tz = slashMatch[7];

      let month: number;
      let day: number;
      if (part1 > 12 && part2 <= 12) {
        day = part1;
        month = part2 - 1;
      } else {
        month = part1 - 1;
        day = part2;
      }

      if (tz && tz.toUpperCase() !== "Z") {
        const isoStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}${tz}`;
        const d = new Date(isoStr);
        if (!isNaN(d.getTime())) {
          return {
            dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
            minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
            epochMs: d.getTime(),
          };
        }
      }
      return {
        dayKey: `${year}-${month}-${day}`,
        minuteOfDay: hours * 60 + minutes,
        epochMs: Date.UTC(year, month, day, hours, minutes, seconds),
      };
    }

    const d = new Date(trimmed);
    if (!isNaN(d.getTime())) {
      const hasTz = /[Z+-]\d{2}:?\d{2}$/i.test(trimmed) || /Z$/i.test(trimmed);
      return {
        dayKey: hasTz
          ? `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`
          : `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`,
        minuteOfDay: hasTz
          ? d.getUTCHours() * 60 + d.getUTCMinutes()
          : d.getHours() * 60 + d.getMinutes(),
        epochMs: d.getTime(),
      };
    }
  }

  if (typeof rawTime === "number") {
    const epoch = rawTime < 1e11 ? rawTime * 1000 : rawTime;
    const d = new Date(epoch);
    if (!isNaN(d.getTime())) {
      return {
        dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
        minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
        epochMs: d.getTime(),
      };
    }
  }

  return null;
}

/**
 * Filter CGM rows to a configurable observation window.
 * @param {Array<Record<string, unknown>>} rows - Full CGM dataset rows.
 * @param {string} timeCol - Column name representing date/time.
 * @param {number} [days=14] - Duration of the observation window in days (0 for all data).
 * @returns {Array<Record<string, unknown>>} Filtered rows within the window of latest reading.
 */
export function filterObservationWindow(
  rows: Array<Record<string, unknown>>,
  timeCol: string,
  days: number = 14,
): Array<Record<string, unknown>> {
  if (days <= 0) return rows;
  let maxTime = -Infinity;
  const parsedRows: Array<{ row: Record<string, unknown>; timeMs: number }> =
    [];

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
 * @param {Array<Record<string, unknown>>} rows - Full CGM dataset rows.
 * @param {string} timeCol - Column name representing date/time.
 * @returns {Array<Record<string, unknown>>} Filtered rows within 14 days of latest reading.
 */
export function filter14DayWindow(
  rows: Array<Record<string, unknown>>,
  timeCol: string,
): Array<Record<string, unknown>> {
  return filterObservationWindow(rows, timeCol, 14);
}
