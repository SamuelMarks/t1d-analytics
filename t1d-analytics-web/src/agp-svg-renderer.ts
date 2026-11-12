/**
 * @file agp-svg-renderer.ts
 * @module agp-svg-renderer
 * Accessible, responsive viewBox SVG renderer for 24-hour Ambulatory Glucose Profile (AGP)
 * percentile ribbons, target glucose band, and diurnal axes.
 */

import i18next from "./i18n";
import { formatNumber } from "./formatters";
import {
  CLINICAL_CHART_THEME,
  type AGPData,
  type AGPIntervalMetrics,
} from "./cgm-metrics";
import { AGP_VIEWBOX, createSvgElement } from "./chart-data-detector";

/**
 * Appends the 70-180 mg/dL target band and dashed reference threshold lines onto an AGP SVG.
 * @param {SVGSVGElement} svg - Target AGP SVG element.
 * @param {number} leftX - Left chart margin X coordinate.
 * @param {number} rightX - Right chart margin X coordinate.
 * @param {(g: number) => number} mapY - Glucose-to-Y coordinate mapper.
 * @returns {void}
 */
export function buildAgpTargetZone(
  svg: SVGSVGElement,
  leftX: number,
  rightX: number,
  mapY: (g: number) => number,
): void {
  const y70 = mapY(70);
  const y180 = mapY(180);
  const targetRect = createSvgElement("rect", {
    x: String(leftX),
    y: String(y180),
    width: String(rightX - leftX),
    height: String(y70 - y180),
    fill: CLINICAL_CHART_THEME.colors.agpTargetBand,
  });
  targetRect.appendChild(
    createSvgElement(
      "title",
      {},
      i18next.t("cgm.targetRange", "Target Range (70-180 mg/dL)"),
    ),
  );
  svg.appendChild(targetRect);

  const targetLines: Array<[number, string]> = [
    [70, CLINICAL_CHART_THEME.colors.inRange],
    [180, CLINICAL_CHART_THEME.colors.inRange],
    [250, CLINICAL_CHART_THEME.colors.high],
  ];
  for (const [targetVal, color] of targetLines) {
    const lineY = mapY(targetVal);
    svg.appendChild(
      createSvgElement("line", {
        x1: String(leftX),
        y1: String(lineY),
        x2: String(rightX),
        y2: String(lineY),
        stroke: color,
        "stroke-dasharray": "3,3",
        "stroke-width": "1",
      }),
    );
    svg.appendChild(
      createSvgElement(
        "text",
        {
          x: String(leftX - AGP_VIEWBOX.yTickLabelOffsetX),
          y: String(lineY + AGP_VIEWBOX.yTickLabelOffsetY),
          "text-anchor": "end",
          "font-size": "10",
          fill: "var(--text-muted, #bac2de)",
        },
        formatNumber(targetVal),
      ),
    );
  }
}

/**
 * Appends the 5th-95th outer ribbon, 25th-75th IQR inner ribbon, and 50th median curve onto an AGP SVG.
 * @param {SVGSVGElement} svg - Target AGP SVG element.
 * @param {AGPIntervalMetrics[]} intervals - AGP intervals.
 * @param {(minute: number) => number} mapX - Minute-of-day to X coordinate mapper.
 * @param {(g: number) => number} mapY - Glucose to Y coordinate mapper.
 * @returns {void}
 */
export function buildAgpPercentileBands(
  svg: SVGSVGElement,
  intervals: AGPIntervalMetrics[],
  mapX: (minute: number) => number,
  mapY: (g: number) => number,
): void {
  const activeIntervals = intervals.filter(
    (intv) => intv.count > 0 || intv.p50 > 0,
  );
  if (activeIntervals.length <= 1) return;

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
  svg.appendChild(
    createSvgElement("polygon", {
      points: `${p95Points.join(" ")} ${p5Points.join(" ")}`,
      fill: CLINICAL_CHART_THEME.colors.agpOuterRibbon,
    }),
  );

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
  svg.appendChild(
    createSvgElement("polygon", {
      points: `${p75Points.join(" ")} ${p25Points.join(" ")}`,
      fill: CLINICAL_CHART_THEME.colors.agpInnerRibbon,
    }),
  );

  const p50Points = activeIntervals.map(
    (intv) =>
      `${mapX(intv.minuteOfDay).toFixed(1)},${mapY(intv.p50).toFixed(1)}`,
  );
  svg.appendChild(
    createSvgElement("polyline", {
      points: p50Points.join(" "),
      stroke: CLINICAL_CHART_THEME.colors.agpMedianLine,
      "stroke-width": "2.5",
      fill: "none",
    }),
  );
}

/**
 * Appends X-axis time ticks, header caption, and X/Y axis titles onto an AGP SVG.
 * @param {SVGSVGElement} svg - Target AGP SVG element.
 * @param {number} leftX - Left chart margin X coordinate.
 * @param {number} rightX - Right chart margin X coordinate.
 * @param {number} topY - Top chart margin Y coordinate.
 * @param {number} bottomY - Bottom chart margin Y coordinate.
 * @param {(minute: number) => number} mapX - Minute-of-day to X coordinate mapper.
 * @param {string} formattedDays - Formatted day count string.
 * @param {string} formattedReadings - Formatted reading count string.
 * @returns {void}
 */
export function buildAgpGridAndAxes(
  svg: SVGSVGElement,
  leftX: number,
  rightX: number,
  topY: number,
  bottomY: number,
  mapX: (minute: number) => number,
  formattedDays: string,
  formattedReadings: string,
): void {
  for (let h = 0; h <= 24; h += 6) {
    const minVal = h * 60;
    const xPos = mapX(minVal);
    svg.appendChild(
      createSvgElement("line", {
        x1: String(xPos),
        y1: String(bottomY),
        x2: String(xPos),
        y2: String(bottomY + AGP_VIEWBOX.tickLength),
        stroke: "var(--text-muted, #bac2de)",
      }),
    );

    const formattedHour = formatNumber(h % 24, {
      minimumIntegerDigits: 2,
      useGrouping: false,
    });
    const formattedMin = formatNumber(0, {
      minimumIntegerDigits: 2,
      useGrouping: false,
    });
    svg.appendChild(
      createSvgElement(
        "text",
        {
          x: String(xPos),
          y: String(bottomY + AGP_VIEWBOX.tickLabelOffsetY),
          "text-anchor": "middle",
          "font-size": "10",
          fill: "var(--text-muted, #bac2de)",
        },
        `${formattedHour}:${formattedMin}`,
      ),
    );
  }

  svg.appendChild(
    createSvgElement(
      "text",
      {
        x: String(leftX),
        y: String(AGP_VIEWBOX.headerBaselineY),
        "font-size": "11",
        "font-weight": "600",
        fill: "var(--text-light, #cdd6f4)",
      },
      i18next.t("cgm.agpHeaderCaption", {
        days: formattedDays,
        readings: formattedReadings,
        defaultValue: `24-Hour Ambulatory Glucose Profile (AGP) - ${formattedDays} Days (${formattedReadings} Readings)`,
      }),
    ),
  );

  const midY = (topY + bottomY) / 2;
  svg.appendChild(
    createSvgElement(
      "text",
      {
        x: String(AGP_VIEWBOX.yAxisTitleX),
        y: String(midY),
        "text-anchor": "middle",
        "font-size": "9",
        fill: "var(--text-muted, #bac2de)",
        transform: `rotate(-90, ${AGP_VIEWBOX.yAxisTitleX}, ${midY})`,
      },
      i18next.t("cgm.glucoseMgDl", "Glucose (mg/dL)"),
    ),
  );

  svg.appendChild(
    createSvgElement(
      "text",
      {
        x: String((leftX + rightX) / 2),
        y: String(bottomY + AGP_VIEWBOX.xAxisTitleOffsetY),
        "text-anchor": "middle",
        "font-size": "9",
        fill: "var(--text-muted, #bac2de)",
      },
      i18next.t("cgm.timeOfDay", "Time of Day (Hour)"),
    ),
  );
}

/**
 * Renders an accessible responsive SVG Ambulatory Glucose Profile (AGP) 24-hour curve chart.
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
  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${AGP_VIEWBOX.width} ${AGP_VIEWBOX.height}`,
    preserveAspectRatio: "xMidYMid meet",
    width: "100%",
    height: String(AGP_VIEWBOX.height),
    role: "img",
  });

  svg.appendChild(
    createSvgElement(
      "title",
      { id: titleId },
      i18next.t(
        "cgm.agpCurveTitle",
        "24-Hour Ambulatory Glucose Profile (Median & Percentiles)",
      ),
    ),
  );

  const formattedDays = formatNumber(agp.totalDays);
  const formattedReadings = formatNumber(agp.totalReadings);
  const descText = i18next.t("cgm.agpSummaryAria", {
    days: formattedDays,
    readings: formattedReadings,
    defaultValue: `24-Hour Ambulatory Glucose Profile (AGP) Curve: median glucose across 24 hours with 5th to 95th percentile ranges and 70-180 mg/dL target band (${formattedReadings} readings across ${formattedDays} days).`,
  });
  svg.appendChild(createSvgElement("desc", { id: descId }, descText));
  svg.setAttribute("aria-label", descText);
  svg.setAttribute("aria-labelledby", `${titleId} ${descId}`);

  const minG = AGP_VIEWBOX.minGlucoseMgDl;
  const maxG = AGP_VIEWBOX.maxGlucoseMgDl;
  const topY = AGP_VIEWBOX.plotTopY;
  const bottomY = AGP_VIEWBOX.plotBottomY;
  const leftX = AGP_VIEWBOX.plotLeftX;
  const rightX = AGP_VIEWBOX.plotRightX;

  /**
   * Maps a glucose reading in mg/dL to a vertical SVG Y coordinate.
   * @param {number} g - Glucose reading in mg/dL.
   * @returns {number} Vertical SVG coordinate.
   */
  const mapY = (g: number): number => {
    const clamped = Math.max(minG, Math.min(maxG, g));
    return bottomY - ((clamped - minG) / (maxG - minG)) * (bottomY - topY);
  };
  /**
   * Maps a minute-of-day offset (`0..1440`) to a horizontal SVG X coordinate.
   * @param {number} minute - Minute of day.
   * @returns {number} Horizontal SVG coordinate.
   */
  const mapX = (minute: number): number =>
    leftX + (minute / AGP_VIEWBOX.minutesPerDay) * (rightX - leftX);

  buildAgpTargetZone(svg, leftX, rightX, mapY);
  buildAgpPercentileBands(svg, agp.intervals, mapX, mapY);
  buildAgpGridAndAxes(
    svg,
    leftX,
    rightX,
    topY,
    bottomY,
    mapX,
    formattedDays,
    formattedReadings,
  );

  return svg;
}
