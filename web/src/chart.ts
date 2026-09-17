/**
 * @file chart.ts
 * Clinical Continuous Glucose Monitoring (CGM) analytics and SVG visualization.
 * Computes Time-in-Range (TIR) metrics and renders clinical charts.
 */

/**
 * Clinical Time-in-Range (TIR) metrics percentages and statistics.
 */
export interface TIRMetrics {
  /** Percentage of readings below 54 mg/dL (<3.0 mmol/L). */
  veryLow: number;
  /** Percentage of readings between 54 and 69 mg/dL (3.0-3.8 mmol/L). */
  low: number;
  /** Percentage of readings in target range 70 to 180 mg/dL (3.9-10.0 mmol/L). */
  inRange: number;
  /** Percentage of readings between 181 and 250 mg/dL (10.1-13.9 mmol/L). */
  high: number;
  /** Percentage of readings above 250 mg/dL (>13.9 mmol/L). */
  veryHigh: number;
  /** Mean glucose value in mg/dL. */
  mean: number;
  /** Total number of valid sensor readings. */
  count: number;
  /** Glucose Management Indicator (estimated HbA1c equivalent percentage). */
  gmi: number;
  /** Coefficient of Variation (CV) percentage (standard deviation / mean * 100). */
  cv: number;
  /** Standard deviation of glucose readings in mg/dL. */
  sd: number;
  /** Low Blood Glucose Index (Kovatchev LBGI). */
  lbgi: number;
  /** High Blood Glucose Index (Kovatchev HBGI). */
  hbgi: number;
}

/**
 * Known column names representing CGM or blood glucose values.
 */
const GLUCOSE_COLUMNS = [
  "cgm",
  "glucose",
  "historic_glucose",
  "blood_glucose",
  "sensor_glucose",
  "bg",
  "value",
  "val",
];

/**
 * Detect whether a tabular dataset has continuous glucose monitoring columns.
 * @param {Array<Record<string, unknown>>} rows - The dataset rows.
 * @returns {{ glucoseCol?: string; timeCol?: string }} Detected column identifiers.
 */
export function detectCgmColumns(rows: Array<Record<string, unknown>>): {
  glucoseCol?: string;
  timeCol?: string;
} {
  if (!rows || rows.length === 0) return {};
  const sample = rows[0];
  const keys = Object.keys(sample);

  const glucoseCol = keys.find((k) =>
    GLUCOSE_COLUMNS.includes(k.toLowerCase().trim()),
  );

  const timeCol = keys.find((k) => {
    const lk = k.toLowerCase().trim();
    return (
      lk.includes("time") ||
      lk.includes("date") ||
      lk.includes("timestamp") ||
      lk === "ts"
    );
  });

  return { glucoseCol, timeCol };
}

/**
 * Calculate clinical Time-in-Range (TIR) percentages from an array of glucose readings.
 * @param {number[]} values - Sensor glucose values in mg/dL.
 * @returns {TIRMetrics} Calculated clinical TIR metrics.
 */
export function calculateTIR(values: number[]): TIRMetrics {
  const valid = values.filter(
    (v) => typeof v === "number" && !isNaN(v) && v > 0,
  );
  if (valid.length === 0) {
    return {
      veryLow: 0,
      low: 0,
      inRange: 0,
      high: 0,
      veryHigh: 0,
      mean: 0,
      count: 0,
      gmi: 0,
      cv: 0,
      sd: 0,
      lbgi: 0,
      hbgi: 0,
    };
  }

  // Check if readings might be in mmol/L (median reading < 25)
  // Scale to mg/dL for clinical standard formula (1 mmol/L = 18.0182 mg/dL)
  const sorted = [...valid].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const isMmol = median > 0 && median < 25 && valid.every((v) => v < 35);
  const scaledValues = isMmol ? valid.map((v) => v * 18.0182) : valid;

  let vl = 0;
  let l = 0;
  let ir = 0;
  let h = 0;
  let vh = 0;
  let sum = 0;

  for (const v of scaledValues) {
    sum += v;
    if (v < 54) vl++;
    else if (v < 70) l++;
    else if (v <= 180) ir++;
    else if (v <= 250) h++;
    else vh++;
  }

  const n = scaledValues.length;
  const meanVal = sum / n;

  // Standard deviation and Coefficient of Variation (CV)
  let varianceSum = 0;
  let lbgiSum = 0;
  let hbgiSum = 0;

  for (const v of scaledValues) {
    varianceSum += (v - meanVal) ** 2;

    // Guard Kovatchev transformation: clamp glucose value to >= 1.01 to ensure Math.log(v) > 0
    // and prevent Math.pow(negative, 1.084) from returning NaN.
    const clampedV = Math.max(1.01, v);
    const fG = 1.509 * (Math.pow(Math.log(clampedV), 1.084) - 5.381);
    const rG = 10 * (fG * fG);
    if (fG < 0) {
      lbgiSum += rG;
    } else {
      hbgiSum += rG;
    }
  }

  const sd = Math.round(Math.sqrt(varianceSum / (n > 1 ? n - 1 : 1)) * 10) / 10;
  const cv = Math.round((sd / meanVal) * 100 * 10) / 10;
  const gmi = Math.round((3.31 + 0.02392 * meanVal) * 10) / 10;
  const lbgi = Math.round((lbgiSum / n) * 10) / 10;
  const hbgi = Math.round((hbgiSum / n) * 10) / 10;

  return {
    veryLow: Math.round((vl / n) * 1000) / 10,
    low: Math.round((l / n) * 1000) / 10,
    inRange: Math.round((ir / n) * 1000) / 10,
    high: Math.round((h / n) * 1000) / 10,
    veryHigh: Math.round((vh / n) * 1000) / 10,
    mean: Math.round(meanVal * 10) / 10,
    count: n,
    gmi,
    cv,
    sd,
    lbgi,
    hbgi,
  };
}

/**
 * Calculate Day (06:00-22:00) vs Night (22:00-06:00) TIR metrics splits.
 * @param {Array<Record<string, unknown>>} rows - The dataset rows.
 * @param {string} timeCol - Column name representing timestamp.
 * @param {string} glucoseCol - Column name representing glucose.
 * @returns {{ day: TIRMetrics; night: TIRMetrics }} Day and night metrics.
 */
export function calculateDayNightTIR(
  rows: Array<Record<string, unknown>>,
  timeCol: string,
  glucoseCol: string,
): { day: TIRMetrics; night: TIRMetrics } {
  const dayValues: number[] = [];
  const nightValues: number[] = [];

  for (const r of rows) {
    const rawTime = r[timeCol];
    const rawG = r[glucoseCol];
    let val: number | null = null;
    if (typeof rawG === "number") val = rawG;
    else if (typeof rawG === "string") {
      const p = parseFloat(rawG);
      if (!isNaN(p)) val = p;
    }
    if (val === null || val <= 0) continue;

    const parsed = parseDateAndMinuteOfDay(rawTime);
    if (!parsed) continue;

    // 06:00 is minute 360, 22:00 is minute 1320
    if (parsed.minuteOfDay >= 360 && parsed.minuteOfDay < 1320) {
      dayValues.push(val);
    } else {
      nightValues.push(val);
    }
  }

  return {
    day: calculateTIR(dayValues),
    night: calculateTIR(nightValues),
  };
}

/**
 * Create an accessible SVG stacked bar chart representing clinical Time-in-Range.
 * @param {TIRMetrics} metrics - The TIR percentages to render.
 * @returns {SVGSVGElement} The generated SVG chart element.
 */
export function renderTIRBarSvg(metrics: TIRMetrics): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 500 70");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "70");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `Time in Range Chart: In Range ${metrics.inRange}%, Low ${metrics.low}%, Very Low ${metrics.veryLow}%, High ${metrics.high}%, Very High ${metrics.veryHigh}%`,
  );

  const segments = [
    { label: "Very Low (<54)", pct: metrics.veryLow, color: "#8b0000" },
    { label: "Low (54-69)", pct: metrics.low, color: "#e74c3c" },
    { label: "In Range (70-180)", pct: metrics.inRange, color: "#2ecc71" },
    { label: "High (181-250)", pct: metrics.high, color: "#f39c12" },
    { label: "Very High (>250)", pct: metrics.veryHigh, color: "#c0392b" },
  ];

  let currentX = 0;
  const barHeight = 35;

  for (const seg of segments) {
    if (seg.pct <= 0) continue;
    const width = (seg.pct / 100) * 500;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(currentX));
    rect.setAttribute("y", "10");
    rect.setAttribute("width", String(width));
    rect.setAttribute("height", String(barHeight));
    rect.setAttribute("fill", seg.color);

    const title = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "title",
    );
    title.textContent = `${seg.label}: ${seg.pct}%`;
    rect.appendChild(title);
    svg.appendChild(rect);

    if (width >= 35) {
      const text = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "text",
      );
      text.setAttribute("x", String(currentX + width / 2));
      text.setAttribute("y", "32");
      text.setAttribute("fill", "#ffffff");
      text.setAttribute("font-size", "11");
      text.setAttribute("font-family", "system-ui, sans-serif");
      text.setAttribute("text-anchor", "middle");
      text.textContent = `${seg.pct}%`;
      svg.appendChild(text);
    }

    currentX += width;
  }

  // Legend caption
  const desc = document.createElementNS("http://www.w3.org/2000/svg", "text");
  desc.setAttribute("x", "0");
  desc.setAttribute("y", "62");
  desc.setAttribute("font-size", "11");
  desc.setAttribute("font-family", "system-ui, sans-serif");
  desc.setAttribute("fill", "#888888");
  desc.textContent = `Mean: ${metrics.mean} mg/dL | Readings: ${metrics.count} | Target (70-180 mg/dL): ${metrics.inRange}%`;
  svg.appendChild(desc);

  return svg;
}

/**
 * Create an accessible SVG glucose distribution histogram.
 * @param {number[]} values - Array of raw glucose readings.
 * @returns {SVGSVGElement} The generated SVG histogram element.
 */
export function renderHistogramSvg(values: number[]): SVGSVGElement {
  const valid = values.filter(
    (v) => typeof v === "number" && !isNaN(v) && v > 0,
  );
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 500 120");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "120");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Glucose Distribution Histogram");

  if (valid.length === 0) return svg;

  const min = 40;
  const max = 300;
  const binSize = 20;
  const numBins = Math.ceil((max - min) / binSize);
  const bins = new Array(numBins).fill(0);

  for (const v of valid) {
    const clamped = Math.max(min, Math.min(max - 1, v));
    const idx = Math.floor((clamped - min) / binSize);
    bins[idx]++;
  }

  const maxCount = Math.max(...bins, 1);
  const barWidth = 500 / numBins;

  for (let i = 0; i < numBins; i++) {
    const count = bins[i];
    const barHeight = (count / maxCount) * 80;
    const x = i * barWidth;
    const y = 90 - barHeight;
    const binCenter = min + i * binSize + binSize / 2;

    let color = "#2ecc71"; // in-range
    if (binCenter < 70) color = "#e74c3c";
    else if (binCenter > 180) color = "#f39c12";

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x + 1));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(Math.max(1, barWidth - 2)));
    rect.setAttribute("height", String(barHeight));
    rect.setAttribute("fill", color);

    const title = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "title",
    );
    title.textContent = `${min + i * binSize}-${min + (i + 1) * binSize} mg/dL: ${count} readings`;
    rect.appendChild(title);
    svg.appendChild(rect);
  }

  return svg;
}

/**
 * Clinical 24-hour Ambulatory Glucose Profile (AGP) interval statistics.
 */
export interface AGPIntervalMetrics {
  /** Minutes from midnight (e.g. 0, 30, 60... 1410). */
  minuteOfDay: number;
  /** Formatted time label (e.g. "00:00", "01:30"). */
  timeLabel: string;
  /** 5th percentile glucose value (mg/dL). */
  p5: number;
  /** 25th percentile glucose value (mg/dL). */
  p25: number;
  /** 50th percentile / median glucose value (mg/dL). */
  p50: number;
  /** 75th percentile glucose value (mg/dL). */
  p75: number;
  /** 95th percentile glucose value (mg/dL). */
  p95: number;
  /** Number of readings in this interval. */
  count: number;
}

/**
 * 24-Hour Ambulatory Glucose Profile dataset across multi-day CGM traces.
 */
export interface AGPData {
  /** Array of metrics for each time-of-day interval. */
  intervals: AGPIntervalMetrics[];
  /** Estimated number of monitored days. */
  totalDays: number;
  /** Total number of valid CGM readings. */
  totalReadings: number;
}

/**
 * Compute an interpolated percentile from an array of sorted numbers.
 * @param {number[]} sortedValues - Array of sorted numbers.
 * @param {number} p - Percentile rank from 0 to 100.
 * @returns {number} The calculated percentile value.
 */
export function computePercentile(sortedValues: number[], p: number): number {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0];
  const index = (p / 100) * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  const val = sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
  return Math.round(val * 10) / 10;
}

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
 * @returns {{ dayKey: string; minuteOfDay: number } | null} Parsed day key and minute of day.
 */
export function parseDateAndMinuteOfDay(
  rawTime: unknown,
): { dayKey: string; minuteOfDay: number } | null {
  if (rawTime instanceof Date) {
    if (isNaN(rawTime.getTime())) return null;
    const dayKey = `${rawTime.getUTCFullYear()}-${rawTime.getUTCMonth()}-${rawTime.getUTCDate()}`;
    const minuteOfDay = rawTime.getUTCHours() * 60 + rawTime.getUTCMinutes();
    return { dayKey, minuteOfDay };
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
      const tz = isoMatch[7];

      if (tz && tz.toUpperCase() === "Z") {
        return {
          dayKey: `${year}-${month}-${day}`,
          minuteOfDay: hours * 60 + minutes,
        };
      } else if (tz) {
        const d = new Date(trimmed);
        if (!isNaN(d.getTime())) {
          return {
            dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
            minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
          };
        }
      }
      // No timezone offset specified: preserve literal hours/minutes
      return {
        dayKey: `${year}-${month}-${day}`,
        minuteOfDay: hours * 60 + minutes,
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
      const tz = cdiscMatch[7];

      if (tz && tz.toUpperCase() !== "Z") {
        const isoStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:00${tz}`;
        const d = new Date(isoStr);
        if (!isNaN(d.getTime())) {
          return {
            dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
            minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
          };
        }
      }
      return {
        dayKey: `${year}-${month}-${day}`,
        minuteOfDay: hours * 60 + minutes,
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
        const isoStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:00${tz}`;
        const d = new Date(isoStr);
        if (!isNaN(d.getTime())) {
          return {
            dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
            minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
          };
        }
      }
      return {
        dayKey: `${year}-${month}-${day}`,
        minuteOfDay: hours * 60 + minutes,
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
      };
    }
  }

  if (typeof rawTime === "number") {
    const d = new Date(rawTime);
    if (!isNaN(d.getTime())) {
      return {
        dayKey: `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`,
        minuteOfDay: d.getUTCHours() * 60 + d.getUTCMinutes(),
      };
    }
  }

  return null;
}

/**
 * Compute 24-Hour Ambulatory Glucose Profile (AGP) percentiles across time of day.
 * @param {Array<Record<string, unknown>>} rows - Tabular dataset rows.
 * @param {string} timeCol - Column name representing date/time.
 * @param {string} glucoseCol - Column name representing glucose values.
 * @param {number} [intervalMinutes=30] - Binning interval in minutes.
 * @returns {AGPData | null} Computed AGP profile or null if insufficient data.
 */
export function calculateAGP(
  rows: Array<Record<string, unknown>>,
  timeCol: string,
  glucoseCol: string,
  intervalMinutes: number = 30,
): AGPData | null {
  const numBuckets = Math.floor(1440 / intervalMinutes);
  const buckets: number[][] = Array.from({ length: numBuckets }, () => []);
  const days = new Set<string>();
  let totalValid = 0;

  for (const r of rows) {
    const rawTime = r[timeCol];
    const rawGlucose = r[glucoseCol];

    let glucoseVal: number | null = null;
    if (typeof rawGlucose === "number") glucoseVal = rawGlucose;
    else if (typeof rawGlucose === "string") {
      const p = parseFloat(rawGlucose);
      if (!isNaN(p)) glucoseVal = p;
    }

    if (glucoseVal === null || isNaN(glucoseVal) || glucoseVal <= 0) continue;

    const parsedTime = parseDateAndMinuteOfDay(rawTime);
    if (!parsedTime) continue;

    days.add(parsedTime.dayKey);
    const bucketIdx = Math.min(
      numBuckets - 1,
      Math.floor(parsedTime.minuteOfDay / intervalMinutes),
    );
    buckets[bucketIdx].push(glucoseVal);
    totalValid++;
  }

  if (totalValid === 0) return null;

  const intervals: AGPIntervalMetrics[] = [];
  for (let i = 0; i < numBuckets; i++) {
    const bVals = buckets[i].sort((a, b) => a - b);
    const minuteOfDay = i * intervalMinutes;
    const hours = Math.floor(minuteOfDay / 60);
    const mins = minuteOfDay % 60;
    const timeLabel = `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`;

    if (bVals.length === 0) {
      intervals.push({
        minuteOfDay,
        timeLabel,
        p5: 0,
        p25: 0,
        p50: 0,
        p75: 0,
        p95: 0,
        count: 0,
      });
    } else {
      intervals.push({
        minuteOfDay,
        timeLabel,
        p5: computePercentile(bVals, 5),
        p25: computePercentile(bVals, 25),
        p50: computePercentile(bVals, 50),
        p75: computePercentile(bVals, 75),
        p95: computePercentile(bVals, 95),
        count: bVals.length,
      });
    }
  }

  // Linear interpolation for empty intervals to prevent 0-drops in continuous profile
  for (let i = 0; i < numBuckets; i++) {
    if (intervals[i].count === 0) {
      let prevIdx = -1;
      for (let j = i - 1; j >= 0; j--) {
        if (intervals[j].count > 0) {
          prevIdx = j;
          break;
        }
      }
      let nextIdx = -1;
      for (let j = i + 1; j < numBuckets; j++) {
        if (intervals[j].count > 0) {
          nextIdx = j;
          break;
        }
      }

      if (prevIdx !== -1 && nextIdx !== -1) {
        const factor = (i - prevIdx) / (nextIdx - prevIdx);
        const prev = intervals[prevIdx];
        const next = intervals[nextIdx];
        intervals[i].p50 =
          Math.round((prev.p50 + (next.p50 - prev.p50) * factor) * 10) / 10;
        intervals[i].p25 =
          Math.round((prev.p25 + (next.p25 - prev.p25) * factor) * 10) / 10;
        intervals[i].p75 =
          Math.round((prev.p75 + (next.p75 - prev.p75) * factor) * 10) / 10;
        intervals[i].p5 =
          Math.round((prev.p5 + (next.p5 - prev.p5) * factor) * 10) / 10;
        intervals[i].p95 =
          Math.round((prev.p95 + (next.p95 - prev.p95) * factor) * 10) / 10;
      } else if (prevIdx !== -1) {
        const prev = intervals[prevIdx];
        intervals[i].p50 = prev.p50;
        intervals[i].p25 = prev.p25;
        intervals[i].p75 = prev.p75;
        intervals[i].p5 = prev.p5;
        intervals[i].p95 = prev.p95;
      } else {
        const next = intervals[nextIdx];
        intervals[i].p50 = next.p50;
        intervals[i].p25 = next.p25;
        intervals[i].p75 = next.p75;
        intervals[i].p5 = next.p5;
        intervals[i].p95 = next.p95;
      }
    }
  }

  return {
    intervals,
    totalDays: Math.max(1, days.size),
    totalReadings: totalValid,
  };
}

/**
 * Render an accessible SVG Ambulatory Glucose Profile (AGP) 24-hour curve chart.
 * @param {AGPData} agp - The calculated AGP profile metrics.
 * @returns {SVGSVGElement} Accessible SVG element displaying percentile ribbons and target band.
 */
export function renderAGPSvg(agp: AGPData): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 600 230");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "230");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `24-Hour Ambulatory Glucose Profile (AGP) Curve: median glucose across 24 hours with 5th to 95th percentile ranges and 70-180 mg/dL target band (${agp.totalReadings} readings across ${agp.totalDays} days).`,
  );

  const minG = 40;
  const maxG = 350;
  const topY = 25;
  const bottomY = 195;
  const leftX = 50;
  const rightX = 575;

  const mapY = (g: number): number => {
    const clamped = Math.max(minG, Math.min(maxG, g));
    return bottomY - ((clamped - minG) / (maxG - minG)) * (bottomY - topY);
  };

  const mapX = (minute: number): number => {
    return leftX + (minute / 1440) * (rightX - leftX);
  };

  // 1. Target range green band (70 - 180 mg/dL)
  const y70 = mapY(70);
  const y180 = mapY(180);
  const targetRect = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "rect",
  );
  targetRect.setAttribute("x", String(leftX));
  targetRect.setAttribute("y", String(y180));
  targetRect.setAttribute("width", String(rightX - leftX));
  targetRect.setAttribute("height", String(y70 - y180));
  targetRect.setAttribute("fill", "rgba(46, 204, 113, 0.15)");
  svg.appendChild(targetRect);

  // Target lines (dashed)
  for (const [targetVal, color] of [
    [70, "#2ecc71"],
    [180, "#2ecc71"],
    [250, "#f39c12"],
  ] as const) {
    const lineY = mapY(targetVal);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(leftX));
    line.setAttribute("y1", String(lineY));
    line.setAttribute("x2", String(rightX));
    line.setAttribute("y2", String(lineY));
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-dasharray", "3,3");
    line.setAttribute("stroke-width", "1");
    svg.appendChild(line);

    // Y-axis label
    const yLabel = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    yLabel.setAttribute("x", String(leftX - 8));
    yLabel.setAttribute("y", String(lineY + 4));
    yLabel.setAttribute("text-anchor", "end");
    yLabel.setAttribute("font-size", "10");
    yLabel.setAttribute("fill", "#666666");
    yLabel.textContent = `${targetVal}`;
    svg.appendChild(yLabel);
  }

  // Active intervals with readings or interpolated profile values
  const activeIntervals = agp.intervals.filter(
    (intv) => intv.count > 0 || intv.p50 > 0,
  );

  if (activeIntervals.length > 1) {
    // 2. Outer percentile ribbon (5th - 95th)
    const p95Points = activeIntervals.map(
      (intv) =>
        `${mapX(intv.minuteOfDay).toFixed(1)},${mapY(intv.p95).toFixed(1)}`,
    );
    const p5Points = [...activeIntervals]
      .reverse()
      .map(
        (intv) =>
          `${mapX(intv.minuteOfDay).toFixed(1)},${mapY(intv.p5).toFixed(1)}`,
      );
    const outerPolygon = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polygon",
    );
    outerPolygon.setAttribute(
      "points",
      `${p95Points.join(" ")} ${p5Points.join(" ")}`,
    );
    outerPolygon.setAttribute("fill", "rgba(52, 152, 219, 0.2)");
    svg.appendChild(outerPolygon);

    // 3. Inner percentile ribbon (25th - 75th / IQR)
    const p75Points = activeIntervals.map(
      (intv) =>
        `${mapX(intv.minuteOfDay).toFixed(1)},${mapY(intv.p75).toFixed(1)}`,
    );
    const p25Points = [...activeIntervals]
      .reverse()
      .map(
        (intv) =>
          `${mapX(intv.minuteOfDay).toFixed(1)},${mapY(intv.p25).toFixed(1)}`,
      );
    const innerPolygon = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polygon",
    );
    innerPolygon.setAttribute(
      "points",
      `${p75Points.join(" ")} ${p25Points.join(" ")}`,
    );
    innerPolygon.setAttribute("fill", "rgba(41, 128, 185, 0.35)");
    svg.appendChild(innerPolygon);

    // 4. Median curve (50th percentile)
    const p50Points = activeIntervals.map(
      (intv) =>
        `${mapX(intv.minuteOfDay).toFixed(1)},${mapY(intv.p50).toFixed(1)}`,
    );
    const medianPolyline = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polyline",
    );
    medianPolyline.setAttribute("points", p50Points.join(" "));
    medianPolyline.setAttribute("stroke", "#1b4f72");
    medianPolyline.setAttribute("stroke-width", "2.5");
    medianPolyline.setAttribute("fill", "none");
    svg.appendChild(medianPolyline);
  }

  // X-axis time points (00:00, 06:00, 12:00, 18:00, 24:00)
  for (let h = 0; h <= 24; h += 6) {
    const minVal = h * 60;
    const xPos = mapX(minVal);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", String(xPos));
    tick.setAttribute("y1", String(bottomY));
    tick.setAttribute("x2", String(xPos));
    tick.setAttribute("y2", String(bottomY + 5));
    tick.setAttribute("stroke", "#888888");
    svg.appendChild(tick);

    const xLabel = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    xLabel.setAttribute("x", String(xPos));
    xLabel.setAttribute("y", String(bottomY + 18));
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.setAttribute("font-size", "10");
    xLabel.setAttribute("fill", "#666666");
    xLabel.textContent = `${String(h % 24).padStart(2, "0")}:00`;
    svg.appendChild(xLabel);
  }

  // Header caption
  const header = document.createElementNS("http://www.w3.org/2000/svg", "text");
  header.setAttribute("x", String(leftX));
  header.setAttribute("y", "14");
  header.setAttribute("font-size", "11");
  header.setAttribute("font-weight", "600");
  header.setAttribute("fill", "#333333");
  header.textContent = `24-Hour Ambulatory Glucose Profile (AGP) - ${agp.totalDays} Days (${agp.totalReadings} Readings)`;
  svg.appendChild(header);

  return svg;
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
  let maxTime = -Infinity;
  const parsedRows: Array<{ row: Record<string, unknown>; timeMs: number }> =
    [];

  for (const r of rows) {
    const rawTime = r[timeCol];
    const parsed = parseDateAndMinuteOfDay(rawTime);
    if (!parsed) continue;

    let timeMs = 0;
    if (rawTime instanceof Date) {
      timeMs = rawTime.getTime();
    } else if (typeof rawTime === "string") {
      const normalized = rawTime.trim().replace(" ", "T");
      const d = new Date(normalized);
      if (!isNaN(d.getTime())) {
        timeMs = d.getTime();
      }
    } else if (typeof rawTime === "number" && rawTime > 0) {
      timeMs = rawTime;
    }

    if (timeMs > 0) {
      parsedRows.push({ row: r, timeMs });
      if (timeMs > maxTime) maxTime = timeMs;
    }
  }

  if (maxTime === -Infinity) return rows;

  const fourteenDaysMs = 14 * 24 * 60 * 60 * 1000;
  const cutoff = maxTime - fourteenDaysMs;
  return parsedRows
    .filter((item) => item.timeMs >= cutoff)
    .map((item) => item.row);
}

/**
 * Calculate active CGM sensor wear time percentage according to clinical consensus.
 * Clinical standard expects 288 readings/day for 5-minute sampling (>=70% active wear).
 * @param {Array<Record<string, unknown>>} rows - CGM dataset rows.
 * @param {string} timeCol - Column name representing timestamp.
 * @param {number} [expectedReadingsPerDay=288] - Expected readings per 24 hours.
 * @param {number} [monitoringDays] - Optional explicit duration of the monitoring/observation window in days.
 * @returns {{ activeDays: number; totalReadings: number; expectedReadings: number; wearPercentage: number; isValidWear: boolean; calendarDays?: number }} Active wear metrics.
 */
export function calculateActiveWear(
  rows: Array<Record<string, unknown>>,
  timeCol: string,
  expectedReadingsPerDay: number = 288,
  monitoringDays?: number,
): {
  activeDays: number;
  totalReadings: number;
  expectedReadings: number;
  wearPercentage: number;
  isValidWear: boolean;
  calendarDays?: number;
} {
  const days = new Set<string>();
  let totalReadings = 0;
  let minTime = Infinity;
  let maxTime = -Infinity;

  for (const r of rows) {
    const rawTime = r[timeCol];
    const parsed = parseDateAndMinuteOfDay(rawTime);
    if (parsed) {
      days.add(parsed.dayKey);
      totalReadings++;
    }
    let tMs = 0;
    if (rawTime instanceof Date) {
      if (!isNaN(rawTime.getTime())) tMs = rawTime.getTime();
    } else if (typeof rawTime === "string") {
      const d = new Date(rawTime.trim().replace(" ", "T"));
      if (!isNaN(d.getTime())) tMs = d.getTime();
    } else if (typeof rawTime === "number") {
      if (!isNaN(rawTime) && rawTime > 0) tMs = rawTime;
    }
    if (tMs > 0) {
      if (tMs < minTime) minTime = tMs;
      if (tMs > maxTime) maxTime = tMs;
    }
  }

  if (totalReadings === 0) {
    return {
      activeDays: 0,
      calendarDays: 0,
      totalReadings: 0,
      expectedReadings: 0,
      wearPercentage: 0,
      isValidWear: false,
    };
  }

  let calendarDays = monitoringDays;
  if (!calendarDays || calendarDays < 1) {
    if (minTime !== Infinity && maxTime !== -Infinity && maxTime >= minTime) {
      const minDate = new Date(minTime);
      const maxDate = new Date(maxTime);
      const minUtc = Date.UTC(
        minDate.getUTCFullYear(),
        minDate.getUTCMonth(),
        minDate.getUTCDate(),
      );
      const maxUtc = Date.UTC(
        maxDate.getUTCFullYear(),
        maxDate.getUTCMonth(),
        maxDate.getUTCDate(),
      );
      const dayDiff = Math.round((maxUtc - minUtc) / (24 * 3600 * 1000));
      calendarDays = Math.max(1, dayDiff + 1);
    } else {
      calendarDays = Math.max(1, days.size);
    }
  }

  const activeDays = days.size;
  const expectedReadings = calendarDays * expectedReadingsPerDay;
  const wearPercentage =
    expectedReadings > 0
      ? Math.min(
          100,
          Math.round((totalReadings / expectedReadings) * 1000) / 10,
        )
      : 0;
  const isValidWear = wearPercentage >= 70.0;

  return {
    activeDays,
    calendarDays,
    totalReadings,
    expectedReadings,
    wearPercentage,
    isValidWear,
  };
}

/**
 * Render a complete CGM analytics card element from table rows.
 * @param {Array<Record<string, unknown>>} rows - The data rows.
 * @returns {HTMLElement | null} The analytics card element or null if not CGM data.
 */
export function renderCgmCard(
  rows: Array<Record<string, unknown>>,
): HTMLElement | null {
  const { glucoseCol, timeCol } = detectCgmColumns(rows);
  if (!glucoseCol) return null;

  const values: number[] = [];
  for (const r of rows) {
    const raw = r[glucoseCol];
    if (typeof raw === "number") values.push(raw);
    else if (typeof raw === "string") {
      const parsed = parseFloat(raw);
      if (!isNaN(parsed)) values.push(parsed);
    }
  }

  if (values.length === 0) return null;
  const metrics = calculateTIR(values);

  const container = document.createElement("div");
  container.className = "cgm-analytics-card";
  container.style.padding = "1rem";
  container.style.margin = "0.75rem 0";
  container.style.border = "1px solid var(--border-color, #e0e0e0)";
  container.style.borderRadius = "8px";
  container.style.backgroundColor = "var(--bg-secondary, #fafafa)";

  const headerRow = document.createElement("div");
  headerRow.style.display = "flex";
  headerRow.style.justifyContent = "space-between";
  headerRow.style.alignItems = "center";
  headerRow.style.marginBottom = "0.5rem";

  const title = document.createElement("h4");
  title.textContent = "📊 CGM Analytics (Ambulatory Glucose Profile & TIR)";
  title.style.margin = "0";
  title.style.fontSize = "0.95rem";
  headerRow.appendChild(title);

  const printBtn = document.createElement("button");
  printBtn.className = "btn-secondary btn-sm print-report-btn";
  printBtn.textContent = "🖨️ Clinical AGP Report (PDF)";
  printBtn.style.padding = "0.25rem 0.6rem";
  printBtn.style.fontSize = "0.8rem";
  printBtn.onclick = () => window.print();
  headerRow.appendChild(printBtn);
  container.appendChild(headerRow);

  // Active wear badge if time column present
  if (timeCol) {
    const wear = calculateActiveWear(rows, timeCol);
    const wearBadge = document.createElement("div");
    wearBadge.style.fontSize = "0.8rem";
    wearBadge.style.marginBottom = "0.5rem";
    wearBadge.style.padding = "0.2rem 0.5rem";
    wearBadge.style.borderRadius = "4px";
    wearBadge.style.display = "inline-block";
    if (wear.isValidWear) {
      wearBadge.style.backgroundColor = "rgba(46, 204, 113, 0.2)";
      wearBadge.style.color = "#27ae60";
      wearBadge.textContent = `✓ Sensor Wear: ${wear.wearPercentage}% (${wear.activeDays} days, ${wear.totalReadings} readings) - Valid (>=70%)`;
    } else {
      wearBadge.style.backgroundColor = "rgba(243, 156, 18, 0.2)";
      wearBadge.style.color = "#d35400";
      wearBadge.textContent = `⚠️ Sensor Wear: ${wear.wearPercentage}% (${wear.activeDays} days) - Caution (<70% wear)`;
    }
    container.appendChild(wearBadge);
  }

  // Clinical Indices Badges (GMI, CV, LBGI, HBGI, Day/Night)
  const badgesRow = document.createElement("div");
  badgesRow.className = "cgm-clinical-badges";
  badgesRow.style.display = "flex";
  badgesRow.style.flexWrap = "wrap";
  badgesRow.style.gap = "0.5rem";
  badgesRow.style.margin = "0.5rem 0";

  const gmiBadge = document.createElement("div");
  gmiBadge.style.fontSize = "0.8rem";
  gmiBadge.style.padding = "0.2rem 0.5rem";
  gmiBadge.style.borderRadius = "4px";
  gmiBadge.style.backgroundColor = "rgba(52, 152, 219, 0.15)";
  gmiBadge.style.color = "#2980b9";
  gmiBadge.textContent = `GMI: ${metrics.gmi}% (Mean: ${metrics.mean} mg/dL)`;
  badgesRow.appendChild(gmiBadge);

  const cvBadge = document.createElement("div");
  cvBadge.style.fontSize = "0.8rem";
  cvBadge.style.padding = "0.2rem 0.5rem";
  cvBadge.style.borderRadius = "4px";
  const isCvOptimal = metrics.cv <= 36.0;
  cvBadge.style.backgroundColor = isCvOptimal
    ? "rgba(46, 204, 113, 0.15)"
    : "rgba(231, 76, 60, 0.15)";
  cvBadge.style.color = isCvOptimal ? "#27ae60" : "#c0392b";
  cvBadge.textContent = `CV: ${metrics.cv}% (Target ≤36% | SD: ${metrics.sd} mg/dL)`;
  badgesRow.appendChild(cvBadge);

  const lbgiBadge = document.createElement("div");
  lbgiBadge.style.fontSize = "0.8rem";
  lbgiBadge.style.padding = "0.2rem 0.5rem";
  lbgiBadge.style.borderRadius = "4px";
  lbgiBadge.style.backgroundColor = "rgba(155, 89, 182, 0.15)";
  lbgiBadge.style.color = "#8e44ad";
  lbgiBadge.textContent = `LBGI: ${metrics.lbgi} | HBGI: ${metrics.hbgi}`;
  badgesRow.appendChild(lbgiBadge);

  if (timeCol) {
    const dn = calculateDayNightTIR(rows, timeCol, glucoseCol);
    if (dn.day.count > 0 && dn.night.count > 0) {
      const dnBadge = document.createElement("div");
      dnBadge.style.fontSize = "0.8rem";
      dnBadge.style.padding = "0.2rem 0.5rem";
      dnBadge.style.borderRadius = "4px";
      dnBadge.style.backgroundColor = "rgba(241, 196, 15, 0.15)";
      dnBadge.style.color = "#d68910";
      dnBadge.textContent = `TIR Day: ${dn.day.inRange}% (n=${dn.day.count}) | Night: ${dn.night.inRange}% (n=${dn.night.count})`;
      badgesRow.appendChild(dnBadge);
    }
  }

  container.appendChild(badgesRow);

  // 1. Time-in-Range stacked bar
  const tirChart = renderTIRBarSvg(metrics);
  container.appendChild(tirChart);

  // 2. 24-Hour Ambulatory Glucose Profile (AGP) Curve
  if (timeCol) {
    const agpData = calculateAGP(rows, timeCol, glucoseCol);
    if (agpData && agpData.totalReadings > 0) {
      const agpChart = renderAGPSvg(agpData);
      container.appendChild(agpChart);
    }
  }

  // 3. Glucose distribution histogram
  const histChart = renderHistogramSvg(values);
  container.appendChild(histChart);

  return container;
}
