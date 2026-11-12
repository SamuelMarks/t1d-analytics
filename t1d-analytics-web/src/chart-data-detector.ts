/**
 * @file chart-data-detector.ts
 * @module chart-data-detector
 * Responsive SVG viewport layout constants, typed SVG DOM builders, CGM row extraction,
 * and clinical summary metric badges (GMI, CV, LBGI/HBGI, Day/Night TIR).
 */

import i18next from "./i18n";
import { formatNumber } from "./formatters";
import {
  type ClinicalRowValue,
  filterObservationWindow,
} from "./clinical-dates";
import {
  CV_TARGET_PCT,
  type TIRMetrics,
  calculateDayNightTIR,
} from "./cgm-metrics";

/** XML namespace URI for SVG 1.1 / SVG 2 elements. */
export const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Responsive coordinate system layout for stacked Time-in-Range (TIR) SVG bars.
 */
export interface TirViewBoxLayout {
  /** SVG viewBox width. */
  readonly width: 500;
  /** SVG viewBox height. */
  readonly height: 70;
  /** Top Y coordinate of the stacked TIR bar. */
  readonly barTopY: 10;
  /** Height of the stacked TIR bar. */
  readonly barHeight: 35;
  /** Baseline Y coordinate for inline segment percentage labels. */
  readonly barLabelBaselineY: 32;
  /** Minimum segment width required to render an inline percentage label. */
  readonly minSegmentLabelWidth: 35;
  /** Baseline Y coordinate for the bottom summary caption. */
  readonly captionBaselineY: 62;
}

/**
 * Responsive coordinate system layout for stacked Time-in-Range (TIR) SVG bars.
 */
export const TIR_VIEWBOX: TirViewBoxLayout = {
  width: 500,
  height: 70,
  barTopY: 10,
  barHeight: 35,
  barLabelBaselineY: 32,
  minSegmentLabelWidth: 35,
  captionBaselineY: 62,
};

/**
 * Responsive coordinate system layout for glucose distribution SVG histograms.
 */
export interface HistogramViewBoxLayout {
  /** SVG viewBox width. */
  readonly width: 500;
  /** SVG viewBox height. */
  readonly height: 120;
  /** Minimum glucose bin threshold in mg/dL. */
  readonly minGlucoseMgDl: 40;
  /** Maximum glucose bin threshold in mg/dL. */
  readonly maxGlucoseMgDl: 300;
  /** Glucose bin width in mg/dL. */
  readonly binWidthMgDl: 20;
  /** Maximum rendered bar height in viewBox units. */
  readonly maxBarHeight: 80;
  /** Baseline Y coordinate of histogram bars. */
  readonly baselineY: 90;
  /** Hypoglycemia threshold in mg/dL. */
  readonly lowThresholdMgDl: 70;
  /** Hyperglycemia threshold in mg/dL. */
  readonly highThresholdMgDl: 180;
}

/**
 * Responsive coordinate system layout for glucose distribution SVG histograms.
 */
export const HISTOGRAM_VIEWBOX: HistogramViewBoxLayout = {
  width: 500,
  height: 120,
  minGlucoseMgDl: 40,
  maxGlucoseMgDl: 300,
  binWidthMgDl: 20,
  maxBarHeight: 80,
  baselineY: 90,
  lowThresholdMgDl: 70,
  highThresholdMgDl: 180,
};

/**
 * Responsive coordinate system layout for 24-hour Ambulatory Glucose Profile (AGP) SVG curves.
 */
export interface AgpViewBoxLayout {
  /** SVG viewBox width. */
  readonly width: 600;
  /** SVG viewBox height. */
  readonly height: 230;
  /** Minimum Y-axis glucose value in mg/dL. */
  readonly minGlucoseMgDl: 40;
  /** Maximum Y-axis glucose value in mg/dL. */
  readonly maxGlucoseMgDl: 350;
  /** Top plot margin Y coordinate. */
  readonly plotTopY: 25;
  /** Bottom plot margin Y coordinate. */
  readonly plotBottomY: 195;
  /** Left plot margin X coordinate. */
  readonly plotLeftX: 50;
  /** Right plot margin X coordinate. */
  readonly plotRightX: 575;
  /** Total minutes in a 24-hour cycle. */
  readonly minutesPerDay: 1440;
  /** Header title baseline Y coordinate. */
  readonly headerBaselineY: 14;
  /** Rotated Y-axis title X coordinate. */
  readonly yAxisTitleX: 12;
  /** Axis tick mark length. */
  readonly tickLength: 5;
  /** X-axis tick label vertical offset. */
  readonly tickLabelOffsetY: 18;
  /** X-axis title vertical offset. */
  readonly xAxisTitleOffsetY: 24;
  /** Y-axis tick label horizontal offset. */
  readonly yTickLabelOffsetX: 8;
  /** Y-axis tick label vertical offset. */
  readonly yTickLabelOffsetY: 4;
}

/**
 * Responsive coordinate system layout for 24-hour Ambulatory Glucose Profile (AGP) SVG curves.
 */
export const AGP_VIEWBOX: AgpViewBoxLayout = {
  width: 600,
  height: 230,
  minGlucoseMgDl: 40,
  maxGlucoseMgDl: 350,
  plotTopY: 25,
  plotBottomY: 195,
  plotLeftX: 50,
  plotRightX: 575,
  minutesPerDay: 1440,
  headerBaselineY: 14,
  yAxisTitleX: 12,
  tickLength: 5,
  tickLabelOffsetY: 18,
  xAxisTitleOffsetY: 24,
  yTickLabelOffsetX: 8,
  yTickLabelOffsetY: 4,
};

/**
 * Creates a typed SVG DOM element in the `http://www.w3.org/2000/svg` namespace with attributes and optional text content.
 * @template K - SVG tag name key.
 * @param {K} tag - SVG element tag name.
 * @param {Record<string, string>} attrs - Key-value map of SVG attributes to set.
 * @param {string} [textContent] - Optional textContent to assign to the element.
 * @returns {SVGElementTagNameMap[K]} Constructed SVG element.
 */
export function createSvgElement<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string>,
  textContent?: string,
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  if (textContent !== undefined) {
    el.textContent = textContent;
  }
  return el;
}

/**
 * Filtered observation window rows and extracted numeric glucose readings.
 */
export interface ObservationGlucoseReadings {
  /** Rows falling within the requested observation window. */
  displayRows: Array<Record<string, ClinicalRowValue>>;
  /** Parsed numeric glucose values extracted from `displayRows`. */
  values: number[];
}

/**
 * Filters CGM rows to the selected observation window and extracts finite numeric glucose values.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - Full CGM dataset rows.
 * @param {string} glucoseCol - Detected glucose column identifier.
 * @param {string | undefined} timeCol - Optional detected timestamp column identifier.
 * @param {number} observationDays - Duration in days for observation window (`0` for all data).
 * @returns {ObservationGlucoseReadings} Filtered rows and numeric readings.
 */
export function extractObservationGlucoseReadings(
  rows: Array<Record<string, ClinicalRowValue>>,
  glucoseCol: string,
  timeCol: string | undefined,
  observationDays: number,
): ObservationGlucoseReadings {
  const displayRows =
    timeCol && observationDays > 0
      ? filterObservationWindow(rows, timeCol, observationDays)
      : rows;

  const values: number[] = [];
  for (const r of displayRows) {
    const raw = r[glucoseCol];
    if (typeof raw === "number") {
      values.push(raw);
    } else if (typeof raw === "string") {
      const parsed = parseFloat(raw);
      if (!Number.isNaN(parsed)) {
        values.push(parsed);
      }
    }
  }
  return { displayRows, values };
}

/**
 * Builds the clinical summary metric badges (GMI, CV, LBGI/HBGI, and Day/Night TIR).
 * @param {TIRMetrics} metrics - Calculated TIR metrics.
 * @param {Array<Record<string, ClinicalRowValue>>} displayRows - Filtered rows for the observation window.
 * @param {string} glucoseCol - Glucose column identifier.
 * @param {string} [timeCol] - Optional timestamp column identifier.
 * @returns {HTMLDivElement} Container holding clinical metric badges.
 */
export function buildMetricSummaryTiles(
  metrics: TIRMetrics,
  displayRows: Array<Record<string, ClinicalRowValue>>,
  glucoseCol: string,
  timeCol?: string,
): HTMLDivElement {
  const badgesRow = document.createElement("div");
  badgesRow.className = "cgm-clinical-badges";

  const gmiBadge = document.createElement("div");
  gmiBadge.className = "cgm-badge cgm-badge-info";
  const formattedGmi = formatNumber(metrics.gmi);
  const formattedMean = formatNumber(metrics.mean);
  gmiBadge.textContent = i18next.t(
    "cgm.gmi",
    `GMI: ${formattedGmi}% (Mean: ${formattedMean} mg/dL)`,
    { gmi: formattedGmi, mean: formattedMean },
  );
  badgesRow.appendChild(gmiBadge);

  const cvBadge = document.createElement("div");
  const isCvOptimal = metrics.cv <= CV_TARGET_PCT;
  cvBadge.className = `cgm-badge ${isCvOptimal ? "cgm-badge-success" : "cgm-badge-danger"}`;
  const formattedCv = formatNumber(metrics.cv);
  const formattedSd = formatNumber(metrics.sd);
  cvBadge.textContent = i18next.t(
    "cgm.cv",
    `CV: ${formattedCv}% (Target ≤36% | SD: ${formattedSd} mg/dL)`,
    { cv: formattedCv, sd: formattedSd },
  );
  badgesRow.appendChild(cvBadge);

  const lbgiBadge = document.createElement("div");
  lbgiBadge.className = "cgm-badge cgm-badge-purple";
  const formattedLbgi = formatNumber(metrics.lbgi);
  const formattedHbgi = formatNumber(metrics.hbgi);
  lbgiBadge.textContent = i18next.t(
    "cgm.lbgiHbgi",
    `LBGI: ${formattedLbgi} | HBGI: ${formattedHbgi}`,
    { lbgi: formattedLbgi, hbgi: formattedHbgi },
  );
  badgesRow.appendChild(lbgiBadge);

  if (timeCol) {
    const dn = calculateDayNightTIR(displayRows, timeCol, glucoseCol);
    if (dn.day.count > 0 && dn.night.count > 0) {
      const dnBadge = document.createElement("div");
      dnBadge.className = "cgm-badge cgm-badge-warning";
      const dayPct = formatNumber(dn.day.inRange);
      const dayCount = formatNumber(dn.day.count);
      const nightPct = formatNumber(dn.night.inRange);
      const nightCount = formatNumber(dn.night.count);
      dnBadge.textContent = i18next.t(
        "cgm.dayNight",
        `TIR Day: ${dayPct}% (n=${dayCount}) | Night: ${nightPct}% (n=${nightCount})`,
        {
          day: dayPct,
          dayCount,
          night: nightPct,
          nightCount,
        },
      );
      badgesRow.appendChild(dnBadge);
    }
  }

  return badgesRow;
}
