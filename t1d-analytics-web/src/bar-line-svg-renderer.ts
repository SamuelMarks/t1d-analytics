/**
 * @file bar-line-svg-renderer.ts
 * @module bar-line-svg-renderer
 * Accessible, responsive viewBox SVG renderers for stacked Time-in-Range (TIR) bars
 * and glucose distribution histograms.
 */

import i18next from "./i18n";
import { formatNumber, formatPercent } from "./formatters";
import {
  CLINICAL_CHART_THEME,
  type TIRMetrics,
  normalizeGlucoseToMgDl,
} from "./cgm-metrics";
import {
  HISTOGRAM_VIEWBOX,
  TIR_VIEWBOX,
  createSvgElement,
} from "./chart-data-detector";

/**
 * Appends the stacked TIR segments and legend caption onto an SVG container.
 * @param {SVGSVGElement} svg - Target SVG element.
 * @param {TIRMetrics} metrics - Calculated TIR metrics.
 * @returns {void}
 */
export function buildTirStackedBar(
  svg: SVGSVGElement,
  metrics: TIRMetrics,
): void {
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

  for (const seg of segments) {
    if (seg.pct <= 0) continue;
    const width = (seg.pct / 100) * TIR_VIEWBOX.width;
    const formattedSegPct = formatPercent(seg.pct);

    const rect = createSvgElement("rect", {
      x: String(currentX),
      y: String(TIR_VIEWBOX.barTopY),
      width: String(width),
      height: String(TIR_VIEWBOX.barHeight),
      fill: seg.color,
    });
    rect.appendChild(
      createSvgElement("title", {}, `${seg.label}: ${formattedSegPct}`),
    );
    svg.appendChild(rect);

    if (width >= TIR_VIEWBOX.minSegmentLabelWidth) {
      svg.appendChild(
        createSvgElement(
          "text",
          {
            x: String(currentX + width / 2),
            y: String(TIR_VIEWBOX.barLabelBaselineY),
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

  svg.appendChild(
    createSvgElement(
      "text",
      {
        x: "0",
        y: String(TIR_VIEWBOX.captionBaselineY),
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
}

/**
 * Creates an accessible responsive SVG stacked bar chart representing clinical Time-in-Range.
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
    viewBox: `0 0 ${TIR_VIEWBOX.width} ${TIR_VIEWBOX.height}`,
    preserveAspectRatio: "xMidYMid meet",
    width: "100%",
    height: String(TIR_VIEWBOX.height),
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

  buildTirStackedBar(svg, metrics);
  return svg;
}

/**
 * Creates an accessible responsive SVG glucose distribution histogram.
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
    viewBox: `0 0 ${HISTOGRAM_VIEWBOX.width} ${HISTOGRAM_VIEWBOX.height}`,
    preserveAspectRatio: "xMidYMid meet",
    width: "100%",
    height: String(HISTOGRAM_VIEWBOX.height),
    role: "img",
    "aria-label": histTitle,
    "aria-labelledby": `${titleId} ${descId}`,
  });
  svg.appendChild(createSvgElement("title", { id: titleId }, histTitle));
  svg.appendChild(createSvgElement("desc", { id: descId }, histTitle));

  if (valid.length === 0) return svg;

  const min = HISTOGRAM_VIEWBOX.minGlucoseMgDl;
  const max = HISTOGRAM_VIEWBOX.maxGlucoseMgDl;
  const binSize = HISTOGRAM_VIEWBOX.binWidthMgDl;
  const numBins = Math.ceil((max - min) / binSize);
  const bins: number[] = new Array<number>(numBins).fill(0);

  for (const v of valid) {
    const clamped = Math.max(min, Math.min(max - 1, v));
    const idx = Math.floor((clamped - min) / binSize);
    bins[idx] = Number(bins[idx]) + 1;
  }

  const maxCount = Math.max(...bins, 1);
  const barWidth = HISTOGRAM_VIEWBOX.width / numBins;

  for (let i = 0; i < numBins; i++) {
    const count = Number(bins[i]);
    const barHeight = (count / maxCount) * HISTOGRAM_VIEWBOX.maxBarHeight;
    const x = i * barWidth;
    const y = HISTOGRAM_VIEWBOX.baselineY - barHeight;
    const binCenter = min + i * binSize + binSize / 2;

    let color: string = CLINICAL_CHART_THEME.colors.inRange;
    if (binCenter < HISTOGRAM_VIEWBOX.lowThresholdMgDl) {
      color = CLINICAL_CHART_THEME.colors.low;
    } else if (binCenter > HISTOGRAM_VIEWBOX.highThresholdMgDl) {
      color = CLINICAL_CHART_THEME.colors.high;
    }

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
