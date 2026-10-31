/**
 * @file chart.ts
 * @module chart
 * Clinical Continuous Glucose Monitoring (CGM) analytics and SVG visualization.
 * Computes Time-in-Range (TIR) metrics and renders clinical charts.
 */

import i18next, { isRtlLanguage } from "./i18n";
import { formatNumber, formatPercent } from "./formatters";
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

export {
  type ClinicalDateInput,
  type ClinicalRowValue,
  type ParsedClinicalDate,
  parseDateAndMinuteOfDay,
  filterObservationWindow,
  filter14DayWindow,
  type ClinicalChartThemeConfig,
  CLINICAL_CHART_THEME,
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
};

const SVG_NS = "http://www.w3.org/2000/svg";

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
 * Create an accessible SVG stacked bar chart representing clinical Time-in-Range.
 * @param {TIRMetrics} metrics - The TIR percentages to render.
 * @param {string} [idSuffix=""] - Optional suffix for unique SVG `<title>` and `<desc>` element IDs.
 * @returns {SVGSVGElement} The generated SVG chart element.
 */
export function renderTIRBarSvg(
  metrics: TIRMetrics,
  idSuffix: string = "",
): SVGSVGElement {
  const suffix = idSuffix ? `-${idSuffix}` : "";
  const titleId = `tir-svg-title${suffix}`;
  const descId = `tir-svg-desc${suffix}`;
  const svg = createSvgElement("svg", {
    viewBox: "0 0 500 70",
    width: "100%",
    height: "70",
    role: "img",
  });

  const svgTitle = createSvgElement(
    "title",
    { id: titleId },
    i18next.t(
      "cgm.title",
      "📊 CGM Analytics (Ambulatory Glucose Profile & TIR)",
    ),
  );
  svg.appendChild(svgTitle);

  const descText = i18next.t("cgm.tirChartAria", {
    inRange: formatNumber(metrics.inRange),
    low: formatNumber(metrics.low),
    veryLow: formatNumber(metrics.veryLow),
    high: formatNumber(metrics.high),
    veryHigh: formatNumber(metrics.veryHigh),
    defaultValue: `Time in Range Chart: In Range ${formatNumber(metrics.inRange)}%, Low ${formatNumber(metrics.low)}%, Very Low ${formatNumber(metrics.veryLow)}%, High ${formatNumber(metrics.high)}%, Very High ${formatNumber(metrics.veryHigh)}%`,
  });
  const svgDesc = createSvgElement("desc", { id: descId }, descText);
  svg.appendChild(svgDesc);
  svg.setAttribute("aria-label", descText);
  svg.setAttribute("aria-labelledby", `${titleId} ${descId}`);

  const segments = [
    {
      label: i18next.t("cgm.veryLowLabel", "Very Low (<54)"),
      pct: metrics.veryLow,
      color: CLINICAL_CHART_THEME.colors.veryLow,
    },
    {
      label: i18next.t("cgm.lowLabel", "Low (54-69)"),
      pct: metrics.low,
      color: CLINICAL_CHART_THEME.colors.low,
    },
    {
      label: i18next.t("cgm.inRangeLabel", "In Range (70-180)"),
      pct: metrics.inRange,
      color: CLINICAL_CHART_THEME.colors.inRange,
    },
    {
      label: i18next.t("cgm.highLabel", "High (181-250)"),
      pct: metrics.high,
      color: CLINICAL_CHART_THEME.colors.high,
    },
    {
      label: i18next.t("cgm.veryHighLabel", "Very High (>250)"),
      pct: metrics.veryHigh,
      color: CLINICAL_CHART_THEME.colors.veryHigh,
    },
  ];

  let currentX = 0;
  const barHeight = 35;

  for (const seg of segments) {
    if (seg.pct <= 0) continue;
    const width = (seg.pct / 100) * 500;
    const formattedSegPct = formatPercent(seg.pct);

    const rect = createSvgElement("rect", {
      x: String(currentX),
      y: "10",
      width: String(width),
      height: String(barHeight),
      fill: seg.color,
    });
    rect.appendChild(
      createSvgElement("title", {}, `${seg.label}: ${formattedSegPct}`),
    );
    svg.appendChild(rect);

    if (width >= 35) {
      svg.appendChild(
        createSvgElement(
          "text",
          {
            x: String(currentX + width / 2),
            y: "32",
            fill: "#ffffff",
            "font-size": "11",
            "font-family": "system-ui, sans-serif",
            "text-anchor": "middle",
          },
          formattedSegPct,
        ),
      );
    }

    currentX += width;
  }

  // Legend caption
  svg.appendChild(
    createSvgElement(
      "text",
      {
        x: "0",
        y: "62",
        "font-size": "11",
        "font-family": "system-ui, sans-serif",
        fill: "var(--text-muted, #bac2de)",
      },
      i18next.t("cgm.meanLabel", {
        mean: formatNumber(metrics.mean),
        readings: formatNumber(metrics.count),
        inRange: formatNumber(metrics.inRange),
        defaultValue: `Mean: ${formatNumber(metrics.mean)} mg/dL | Readings: ${formatNumber(metrics.count)} | Target (70-180 mg/dL): ${formatNumber(metrics.inRange)}%`,
      }),
    ),
  );

  return svg;
}

/**
 * Create an accessible SVG glucose distribution histogram.
 * @param {number[]} values - Array of raw glucose readings.
 * @param {string} [idSuffix=""] - Optional suffix for unique SVG `<title>` and `<desc>` element IDs.
 * @returns {SVGSVGElement} The generated SVG histogram element.
 */
export function renderHistogramSvg(
  values: number[],
  idSuffix: string = "",
): SVGSVGElement {
  const suffix = idSuffix ? `-${idSuffix}` : "";
  const titleId = `hist-svg-title${suffix}`;
  const descId = `hist-svg-desc${suffix}`;
  const valid = normalizeGlucoseToMgDl(values);
  const histTitle = i18next.t(
    "cgm.histChartAria",
    "Glucose Distribution Histogram",
  );
  const svg = createSvgElement("svg", {
    viewBox: "0 0 500 120",
    width: "100%",
    height: "120",
    role: "img",
    "aria-label": histTitle,
    "aria-labelledby": `${titleId} ${descId}`,
  });
  svg.appendChild(createSvgElement("title", { id: titleId }, histTitle));
  svg.appendChild(createSvgElement("desc", { id: descId }, histTitle));

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

    const binMin = formatNumber(min + i * binSize);
    const binMax = formatNumber(min + (i + 1) * binSize);
    const formattedCount = formatNumber(count);
    const tooltipText = i18next.t("cgm.histogramBinTooltip", {
      min: binMin,
      max: binMax,
      count: formattedCount,
      defaultValue: `${binMin}-${binMax} mg/dL: ${formattedCount} readings`,
    });
    const rect = createSvgElement("rect", {
      x: String(x + 1),
      y: String(y),
      width: String(Math.max(1, barWidth - 2)),
      height: String(barHeight),
      fill: color,
      tabindex: "0",
      "aria-label": tooltipText,
    });
    rect.appendChild(createSvgElement("title", {}, tooltipText));
    svg.appendChild(rect);
  }

  return svg;
}

/**
 * Render an accessible SVG Ambulatory Glucose Profile (AGP) 24-hour curve chart.
 * @param {AGPData} agp - The calculated AGP profile metrics.
 * @param {string} [idSuffix=""] - Optional suffix for unique SVG `<title>` and `<desc>` element IDs.
 * @returns {SVGSVGElement} Accessible SVG element displaying percentile ribbons and target band.
 */
export function renderAGPSvg(
  agp: AGPData,
  idSuffix: string = "",
): SVGSVGElement {
  const suffix = idSuffix ? `-${idSuffix}` : "";
  const titleId = `agp-svg-title${suffix}`;
  const descId = `agp-svg-desc${suffix}`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 600 230");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "230");
  svg.setAttribute("role", "img");

  const svgTitle = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "title",
  );
  svgTitle.id = titleId;
  svgTitle.textContent = i18next.t(
    "cgm.agpCurveTitle",
    "24-Hour Ambulatory Glucose Profile (Median & Percentiles)",
  );
  svg.appendChild(svgTitle);

  const formattedDays = formatNumber(agp.totalDays);
  const formattedReadings = formatNumber(agp.totalReadings);

  const svgDesc = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "desc",
  );
  svgDesc.id = descId;
  svgDesc.textContent = i18next.t("cgm.agpSummaryAria", {
    days: formattedDays,
    readings: formattedReadings,
    defaultValue: `24-Hour Ambulatory Glucose Profile (AGP) Curve: median glucose across 24 hours with 5th to 95th percentile ranges and 70-180 mg/dL target band (${formattedReadings} readings across ${formattedDays} days).`,
  });
  svg.appendChild(svgDesc);
  svg.setAttribute("aria-label", svgDesc.textContent);
  svg.setAttribute("aria-labelledby", `${titleId} ${descId}`);

  const minG = 40;
  const maxG = 350;
  const topY = 25;
  const bottomY = 195;
  const leftX = 50;
  const rightX = 575;

  /**
   * Maps a glucose reading in mg/dL to an SVG Y-coordinate.
   * @param {number} g - Glucose value in mg/dL.
   * @returns {number} SVG vertical coordinate.
   */
  const mapY = (g: number): number => {
    const clamped = Math.max(minG, Math.min(maxG, g));
    return bottomY - ((clamped - minG) / (maxG - minG)) * (bottomY - topY);
  };

  /**
   * Maps a minute-of-day offset (0-1440) to an SVG X-coordinate.
   * @param {number} minute - Minute of the day from midnight.
   * @returns {number} SVG horizontal coordinate.
   */
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
  const targetBandTitle = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "title",
  );
  targetBandTitle.textContent = i18next.t(
    "cgm.targetRange",
    "Target Range (70-180 mg/dL)",
  );
  targetRect.appendChild(targetBandTitle);
  svg.appendChild(targetRect);

  // Target lines (dashed)
  const targetLines: Array<[number, string]> = [
    [70, "#2ecc71"],
    [180, "#2ecc71"],
    [250, "#f39c12"],
  ];
  for (const [targetVal, color] of targetLines) {
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
    yLabel.setAttribute("fill", "var(--text-muted, #bac2de)");
    yLabel.textContent = formatNumber(targetVal);
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
    tick.setAttribute("stroke", "var(--text-muted, #bac2de)");
    svg.appendChild(tick);

    const xLabel = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    xLabel.setAttribute("x", String(xPos));
    xLabel.setAttribute("y", String(bottomY + 18));
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.setAttribute("font-size", "10");
    xLabel.setAttribute("fill", "var(--text-muted, #bac2de)");
    const formattedHour = formatNumber(h % 24, {
      minimumIntegerDigits: 2,
      useGrouping: false,
    });
    const formattedMin = formatNumber(0, {
      minimumIntegerDigits: 2,
      useGrouping: false,
    });
    xLabel.textContent = `${formattedHour}:${formattedMin}`;
    svg.appendChild(xLabel);
  }

  // Header caption
  const header = document.createElementNS("http://www.w3.org/2000/svg", "text");
  header.setAttribute("x", String(leftX));
  header.setAttribute("y", "14");
  header.setAttribute("font-size", "11");
  header.setAttribute("font-weight", "600");
  header.setAttribute("fill", "var(--text-light, #cdd6f4)");
  header.textContent = i18next.t("cgm.agpHeaderCaption", {
    days: formattedDays,
    readings: formattedReadings,
    defaultValue: `24-Hour Ambulatory Glucose Profile (AGP) - ${formattedDays} Days (${formattedReadings} Readings)`,
  });
  svg.appendChild(header);

  // Y-axis title (Glucose mg/dL)
  const yAxisTitle = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "text",
  );
  yAxisTitle.setAttribute("x", "12");
  yAxisTitle.setAttribute("y", String((topY + bottomY) / 2));
  yAxisTitle.setAttribute("text-anchor", "middle");
  yAxisTitle.setAttribute("font-size", "9");
  yAxisTitle.setAttribute("fill", "var(--text-muted, #bac2de)");
  yAxisTitle.setAttribute(
    "transform",
    `rotate(-90, 12, ${(topY + bottomY) / 2})`,
  );
  yAxisTitle.textContent = i18next.t("cgm.glucoseMgDl", "Glucose (mg/dL)");
  svg.appendChild(yAxisTitle);

  // X-axis title (Time of Day)
  const xAxisTitle = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "text",
  );
  xAxisTitle.setAttribute("x", String((leftX + rightX) / 2));
  xAxisTitle.setAttribute("y", String(bottomY + 24));
  xAxisTitle.setAttribute("text-anchor", "middle");
  xAxisTitle.setAttribute("font-size", "9");
  xAxisTitle.setAttribute("fill", "var(--text-muted, #bac2de)");
  xAxisTitle.textContent = i18next.t("cgm.timeOfDay", "Time of Day (Hour)");
  svg.appendChild(xAxisTitle);

  return svg;
}

/**
 * Render a complete CGM analytics card element from table rows.
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

  const displayRows =
    timeCol && observationDays > 0
      ? filterObservationWindow(rows, timeCol, observationDays)
      : rows;

  const values: number[] = [];
  for (const r of displayRows) {
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

  if (isRtlLanguage()) {
    container.setAttribute("dir", "rtl");
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

  // Window selector dropdown if time column is available
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
      const newCard = renderCgmCard(rows, selectedDays, idSuffix);
      if (newCard && container.parentNode) {
        container.parentNode.replaceChild(newCard, container);
      }
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

  // Active wear badge if time column present
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

  // Clinical Indices Badges (GMI, CV, LBGI, HBGI, Day/Night)
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
  const isCvOptimal = metrics.cv <= 36.0;
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

  container.appendChild(badgesRow);

  // 1. Time-in-Range stacked bar
  const tirChart = renderTIRBarSvg(metrics, idSuffix);
  container.appendChild(tirChart);

  // 2. 24-Hour Ambulatory Glucose Profile (AGP) Curve
  if (timeCol) {
    const agpData = calculateAGP(displayRows, timeCol, glucoseCol);
    if (agpData && agpData.totalReadings > 0) {
      const agpChart = renderAGPSvg(agpData, idSuffix);
      container.appendChild(agpChart);
    }
  }

  // 3. Glucose distribution histogram
  const histChart = renderHistogramSvg(values, idSuffix);
  container.appendChild(histChart);

  return container;
}
