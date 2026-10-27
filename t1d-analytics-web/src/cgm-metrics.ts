/**
 * @file cgm-metrics.ts
 * @module cgm-metrics
 * Pure clinical Continuous Glucose Monitoring (CGM) mathematical and statistical calculations.
 * Computes Time-in-Range (TIR), Glucose Management Indicator (GMI), Coefficient of Variation (CV),
 * Kovatchev LBGI/HBGI risk indices, 24-hour Ambulatory Glucose Profile (AGP) percentiles, and active sensor wear.
 */

import {
  type ClinicalRowValue,
  parseDateAndMinuteOfDay,
} from "./clinical-dates";

/**
 * Strongly-typed theme and layout configuration for clinical CGM SVG visualizations.
 */
export interface ClinicalChartThemeConfig {
  /** Threshold for very low hypoglycemia in mg/dL (<54). */
  veryLowThresholdMgDl: number;
  /** Threshold for low hypoglycemia in mg/dL (<70). */
  lowThresholdMgDl: number;
  /** Upper bound for target euglycemic range in mg/dL (<=180). */
  inRangeUpperMgDl: number;
  /** Upper bound for level 1 hyperglycemia in mg/dL (<=250). */
  highUpperMgDl: number;
  /** Conversion factor from mmol/L to mg/dL. */
  mmolToMgDlFactor: number;
  /** Color palette for clinical TIR bands and AGP ribbons. */
  colors: {
    /** Color for very low (<54 mg/dL). */
    veryLow: string;
    /** Color for low (54-69 mg/dL). */
    low: string;
    /** Color for target in-range (70-180 mg/dL). */
    inRange: string;
    /** Color for high (181-250 mg/dL). */
    high: string;
    /** Color for very high (>250 mg/dL). */
    veryHigh: string;
    /** Fill color for AGP 5th-95th outer ribbon. */
    agpOuterRibbon: string;
    /** Fill color for AGP 25th-75th IQR inner ribbon. */
    agpInnerRibbon: string;
    /** Stroke color for AGP 50th percentile median curve. */
    agpMedianLine: string;
    /** Fill color for AGP target green band. */
    agpTargetBand: string;
  };
}

/**
 * Default clinical CGM chart layout and threshold constants.
 */
export const CLINICAL_CHART_THEME: ClinicalChartThemeConfig = {
  veryLowThresholdMgDl: 54,
  lowThresholdMgDl: 70,
  inRangeUpperMgDl: 180,
  highUpperMgDl: 250,
  mmolToMgDlFactor: 18.0182,
  colors: {
    veryLow: "#8b0000",
    low: "#e74c3c",
    inRange: "#2ecc71",
    high: "#f39c12",
    veryHigh: "#c0392b",
    agpOuterRibbon: "rgba(52, 152, 219, 0.2)",
    agpInnerRibbon: "rgba(41, 128, 185, 0.35)",
    agpMedianLine: "#1b4f72",
    agpTargetBand: "rgba(46, 204, 113, 0.15)",
  },
};

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
const SPECIFIC_GLUCOSE_COLUMNS = [
  "cgm",
  "glucose",
  "historic_glucose",
  "blood_glucose",
  "sensor_glucose",
  "bg",
  "sg",
  "egv",
  "glucose_value",
  "glucose_val",
  "cgm_value",
];

/**
 * Fallback generic column names inspected when a timestamp column is present.
 */
const GENERIC_VALUE_COLUMNS = ["value", "val"];

/**
 * Detected CGM glucose and timestamp column names.
 */
export interface DetectedCgmColumns {
  /** Detected glucose column identifier, if present. */
  glucoseCol?: string;
  /** Detected timestamp column identifier, if present. */
  timeCol?: string;
}

/**
 * Day (06:00-22:00) and Night (22:00-06:00) Time-in-Range metrics split.
 */
export interface DayNightTIRMetrics {
  /** Daytime TIR metrics (06:00-22:00). */
  day: TIRMetrics;
  /** Nighttime TIR metrics (22:00-06:00). */
  night: TIRMetrics;
}

/**
 * Active CGM sensor wear time metrics.
 */
export interface ActiveWearMetrics {
  /** Number of distinct calendar days with sensor readings. */
  activeDays: number;
  /** Total number of valid sensor readings. */
  totalReadings: number;
  /** Expected number of readings across the observation window. */
  expectedReadings: number;
  /** Percentage of expected sensor wear achieved (0-100). */
  wearPercentage: number;
  /** Whether active wear meets the >=70% clinical validity threshold. */
  isValidWear: boolean;
  /** Total calendar days spanned by the observation window. */
  calendarDays?: number;
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
 * Detect whether a tabular dataset has continuous glucose monitoring columns.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - The dataset rows.
 * @returns {DetectedCgmColumns} Detected column identifiers.
 */
export function detectCgmColumns(
  rows: Array<Record<string, ClinicalRowValue>>,
): DetectedCgmColumns {
  if (!rows || rows.length === 0) return {};
  const sample = rows[0];
  const keys = Object.keys(sample);

  let glucoseCol = keys.find((k) =>
    SPECIFIC_GLUCOSE_COLUMNS.includes(k.toLowerCase().trim()),
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

  if (!glucoseCol && timeCol) {
    const genericCol = keys.find((k) =>
      GENERIC_VALUE_COLUMNS.includes(k.toLowerCase().trim()),
    );
    if (genericCol) {
      const numSamples = rows
        .slice(0, 10)
        .map((r) => {
          const v = r[genericCol];
          return typeof v === "number" ? v : parseFloat(String(v));
        })
        .filter((v) => !isNaN(v) && v > 0);
      if (numSamples.length > 0) {
        const median = numSamples.sort((a, b) => a - b)[
          Math.floor(numSamples.length / 2)
        ];
        if (median >= 2 && median <= 500) {
          glucoseCol = genericCol;
        }
      }
    }
  }

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

  const sorted = [...valid].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  const isMmol = median > 0 && median < 25 && p95 < 55;
  const scaledValues = isMmol ? valid.map((v) => v * 18.0182) : valid;

  let vl = 0;
  let l = 0;
  let ir = 0;
  let h = 0;
  let vh = 0;
  let sum = 0;

  for (const v of scaledValues) {
    sum += v;
    if (v < CLINICAL_CHART_THEME.veryLowThresholdMgDl) vl++;
    else if (v < CLINICAL_CHART_THEME.lowThresholdMgDl) l++;
    else if (v <= CLINICAL_CHART_THEME.inRangeUpperMgDl) ir++;
    else if (v <= CLINICAL_CHART_THEME.highUpperMgDl) h++;
    else vh++;
  }

  const n = scaledValues.length;
  const meanVal = sum / n;

  let varianceSum = 0;
  let lbgiSum = 0;
  let hbgiSum = 0;

  for (const v of scaledValues) {
    varianceSum += (v - meanVal) ** 2;
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
 * @param {Array<Record<string, ClinicalRowValue>>} rows - The dataset rows.
 * @param {string} timeCol - Column name representing timestamp.
 * @param {string} glucoseCol - Column name representing glucose.
 * @returns {DayNightTIRMetrics} Day and night metrics.
 */
export function calculateDayNightTIR(
  rows: Array<Record<string, ClinicalRowValue>>,
  timeCol: string,
  glucoseCol: string,
): DayNightTIRMetrics {
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

    if (parsed.minuteOfDay >= 360 && parsed.minuteOfDay < 1320) {
      dayValues.push(val);
    } else {
      nightValues.push(val);
    }
  }

  const allValues = [...dayValues, ...nightValues].filter(
    (v) => !isNaN(v) && v > 0,
  );
  const sorted = [...allValues].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  const isMmol = median > 0 && median < 25 && p95 < 55;
  const scaledDay = isMmol ? dayValues.map((v) => v * 18.0182) : dayValues;
  const scaledNight = isMmol
    ? nightValues.map((v) => v * 18.0182)
    : nightValues;

  return {
    day: calculateTIR(scaledDay),
    night: calculateTIR(scaledNight),
  };
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

/**
 * Compute 24-Hour Ambulatory Glucose Profile (AGP) percentiles across time of day.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - Tabular dataset rows.
 * @param {string} timeCol - Column name representing date/time.
 * @param {string} glucoseCol - Column name representing glucose values.
 * @param {number} [intervalMinutes=30] - Binning interval in minutes.
 * @returns {AGPData | null} Computed AGP profile or null if insufficient data.
 */
export function calculateAGP(
  rows: Array<Record<string, ClinicalRowValue>>,
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
 * Calculate active CGM sensor wear time percentage according to clinical consensus.
 * Clinical standard expects 288 readings/day for 5-minute sampling (>=70% active wear).
 * @param {Array<Record<string, ClinicalRowValue>>} rows - CGM dataset rows.
 * @param {string} timeCol - Column name representing timestamp.
 * @param {number} [expectedReadingsPerDay=288] - Expected readings per 24 hours.
 * @param {number} [monitoringDays] - Optional explicit duration of the monitoring/observation window in days.
 * @returns {ActiveWearMetrics} Active wear metrics.
 */
export function calculateActiveWear(
  rows: Array<Record<string, ClinicalRowValue>>,
  timeCol: string,
  expectedReadingsPerDay: number = 288,
  monitoringDays?: number,
): ActiveWearMetrics {
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
      const tMs = parsed.epochMs;
      if (tMs > 0) {
        if (tMs < minTime) minTime = tMs;
        if (tMs > maxTime) maxTime = tMs;
      }
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
