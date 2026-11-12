/**
 * @file chart.ts
 * @module chart
 * Clinical Continuous Glucose Monitoring (CGM) analytics card coordinator re-exporting
 * `chart-data-detector`, `bar-line-svg-renderer`, and `agp-svg-renderer`.
 */

import i18next, { isRtlLanguage } from "./i18n";
import { formatNumber } from "./formatters";
import {
  type ClinicalDateInput,
  type ClinicalRowValue,
  type ParsedClinicalDate,
  parseDateAndMinuteOfDay,
  filterObservationWindow,
  filter14DayWindow,
} from "./clinical-dates";
import {
  type ClinicalChartThemeConfig,
  CLINICAL_CHART_THEME,
  CV_TARGET_PCT,
  type TIRMetrics,
  type DetectedCgmColumns,
  type DayNightTIRMetrics,
  type ActiveWearMetrics,
  type AGPIntervalMetrics,
  type AGPData,
  detectCgmColumns,
  normalizeGlucoseToMgDl,
  calculateTIR,
  calculateDayNightTIR,
  computePercentile,
  calculateAGP,
  detectSensorIntervalMinutes,
  calculateActiveWear,
} from "./cgm-metrics";
import {
  AGP_VIEWBOX,
  HISTOGRAM_VIEWBOX,
  SVG_NS,
  TIR_VIEWBOX,
  buildMetricSummaryTiles,
  createSvgElement,
  extractObservationGlucoseReadings,
} from "./chart-data-detector";
import {
  buildTirStackedBar,
  renderHistogramSvg,
  renderTIRBarSvg,
} from "./bar-line-svg-renderer";
import {
  buildAgpGridAndAxes,
  buildAgpPercentileBands,
  buildAgpTargetZone,
  renderAGPSvg,
} from "./agp-svg-renderer";

export {
  type ClinicalDateInput,
  type ClinicalRowValue,
  type ParsedClinicalDate,
  parseDateAndMinuteOfDay,
  filterObservationWindow,
  filter14DayWindow,
  type ClinicalChartThemeConfig,
  CLINICAL_CHART_THEME,
  CV_TARGET_PCT,
  type TIRMetrics,
  type DetectedCgmColumns,
  type DayNightTIRMetrics,
  type ActiveWearMetrics,
  type AGPIntervalMetrics,
  type AGPData,
  detectCgmColumns,
  normalizeGlucoseToMgDl,
  calculateTIR,
  calculateDayNightTIR,
  computePercentile,
  calculateAGP,
  detectSensorIntervalMinutes,
  calculateActiveWear,
  SVG_NS,
  TIR_VIEWBOX,
  HISTOGRAM_VIEWBOX,
  AGP_VIEWBOX,
  createSvgElement,
  extractObservationGlucoseReadings,
  buildMetricSummaryTiles,
  buildTirStackedBar,
  renderTIRBarSvg,
  renderHistogramSvg,
  buildAgpTargetZone,
  buildAgpPercentileBands,
  buildAgpGridAndAxes,
  renderAGPSvg,
};

/**
 * Populates the header, controls, wear badge, summary tiles, and SVG charts inside a CGM analytics card container.
 * @param {HTMLElement} container - Target `.cgm-analytics-card` container element.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - Full CGM dataset rows.
 * @param {string} glucoseCol - Detected glucose column identifier.
 * @param {string | undefined} timeCol - Optional detected timestamp column identifier.
 * @param {number} observationDays - Duration in days for observation window (0 for all data).
 * @param {string} idSuffix - Optional suffix for unique SVG `<title>` and `<desc>` element IDs.
 * @returns {boolean} True if valid numeric glucose readings were rendered, false otherwise.
 */
export function populateCgmCardBody(
  container: HTMLElement,
  rows: Array<Record<string, ClinicalRowValue>>,
  glucoseCol: string,
  timeCol: string | undefined,
  observationDays: number,
  idSuffix: string,
): boolean {
  const { displayRows, values } = extractObservationGlucoseReadings(
    rows,
    glucoseCol,
    timeCol,
    observationDays,
  );

  if (values.length === 0) return false;
  const metrics = calculateTIR(values);

  container.replaceChildren();
  if (isRtlLanguage()) {
    container.setAttribute("dir", "rtl");
  } else {
    container.removeAttribute("dir");
  }

  const headerRow = document.createElement("div");
  headerRow.className = "cgm-card-header";

  const title = document.createElement("h2");
  title.className = "cgm-card-title";
  title.textContent = i18next.t(
    "cgm.title",
    "📊 CGM Analytics (Ambulatory Glucose Profile & TIR)",
  );
  headerRow.appendChild(title);

  const controlsRow = document.createElement("div");
  controlsRow.className = "cgm-card-controls";

  if (timeCol) {
    const windowSelect = document.createElement("select");
    windowSelect.className = "agp-window-select form-select form-select-sm";
    windowSelect.setAttribute(
      "aria-label",
      i18next.t("cgm.obsWindow", "Observation Window:"),
    );
    windowSelect.setAttribute("data-i18n-aria-label", "cgm.obsWindow");

    const options = [
      { val: 7, label: i18next.t("cgm.days7", "7 Days") },
      { val: 14, label: i18next.t("cgm.days14", "14 Days (Standard)") },
      { val: 30, label: i18next.t("cgm.days30", "30 Days") },
      { val: 90, label: i18next.t("cgm.days90", "90 Days") },
      { val: 0, label: i18next.t("cgm.daysAll", "All Available Data") },
    ];
    for (const opt of options) {
      const el = document.createElement("option");
      el.value = String(opt.val);
      el.textContent = opt.label;
      if (opt.val === observationDays) el.selected = true;
      windowSelect.appendChild(el);
    }
    windowSelect.onchange = () => {
      const selectedDays = parseInt(windowSelect.value, 10);
      populateCgmCardBody(
        container,
        rows,
        glucoseCol,
        timeCol,
        selectedDays,
        idSuffix,
      );
    };
    controlsRow.appendChild(windowSelect);
  }

  const printBtn = document.createElement("button");
  printBtn.className = "btn-secondary btn-sm print-report-btn";
  printBtn.textContent = i18next.t(
    "cgm.reportPdf",
    "🖨️ Clinical AGP Report (PDF)",
  );
  printBtn.onclick = () => window.print();
  controlsRow.appendChild(printBtn);
  headerRow.appendChild(controlsRow);
  container.appendChild(headerRow);

  if (timeCol) {
    const wear = calculateActiveWear(displayRows, timeCol, 288);
    const wearBadge = document.createElement("div");
    wearBadge.className = `cgm-badge cgm-wear-badge ${wear.isValidWear ? "cgm-badge-success" : "cgm-badge-warning"}`;
    const formattedPct = formatNumber(wear.wearPercentage);
    const formattedDays = formatNumber(wear.activeDays);
    const formattedReadings = formatNumber(wear.totalReadings);
    if (wear.isValidWear) {
      wearBadge.textContent = i18next.t(
        "cgm.wearValid",
        `✓ Sensor Wear: ${formattedPct}% (${formattedDays} days, ${formattedReadings} readings) - Valid (>=70%)`,
        {
          pct: formattedPct,
          days: formattedDays,
          readings: formattedReadings,
        },
      );
    } else {
      wearBadge.textContent = i18next.t(
        "cgm.wearCaution",
        `⚠️ Sensor Wear: ${formattedPct}% (${formattedDays} days) - Caution (<70% wear)`,
        {
          pct: formattedPct,
          days: formattedDays,
          readings: formattedReadings,
        },
      );
    }
    container.appendChild(wearBadge);
  }

  container.appendChild(
    buildMetricSummaryTiles(metrics, displayRows, glucoseCol, timeCol),
  );
  container.appendChild(renderTIRBarSvg(metrics, idSuffix));

  if (timeCol) {
    const agpData = calculateAGP(displayRows, timeCol, glucoseCol);
    if (agpData && agpData.totalReadings > 0) {
      container.appendChild(renderAGPSvg(agpData, idSuffix));
    }
  }

  container.appendChild(renderHistogramSvg(values, idSuffix));
  return true;
}

/**
 * Renders a complete CGM analytics card element from table rows.
 * @param {Array<Record<string, ClinicalRowValue>>} rows - The data rows.
 * @param {number} [observationDays=14] - Duration in days for observation window (default: 14, 0 for all data).
 * @param {string} [idSuffix=""] - Optional suffix for unique SVG `<title>` and `<desc>` element IDs.
 * @returns {HTMLElement | null} The analytics card element or null if not CGM data.
 */
export function renderCgmCard(
  rows: Array<Record<string, ClinicalRowValue>>,
  observationDays: number = 14,
  idSuffix: string = "",
): HTMLElement | null {
  const { glucoseCol, timeCol } = detectCgmColumns(rows);
  if (!glucoseCol) return null;

  const container = document.createElement("div");
  container.className = "cgm-analytics-card";
  const populated = populateCgmCardBody(
    container,
    rows,
    glucoseCol,
    timeCol,
    observationDays,
    idSuffix,
  );
  return populated ? container : null;
}
