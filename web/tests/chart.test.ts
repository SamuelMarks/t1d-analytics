/**
 * @file chart.test.ts
 * Unit tests for clinical CGM analytics and SVG visualization.
 */

import { describe, it, expect } from "vitest";
import i18next from "i18next";
import {
  detectCgmColumns,
  calculateTIR,
  renderTIRBarSvg,
  renderHistogramSvg,
  renderCgmCard,
  computePercentile,
  calculateAGP,
  renderAGPSvg,
  filterObservationWindow,
  filter14DayWindow,
  calculateActiveWear,
  parseDateAndMinuteOfDay,
  calculateDayNightTIR,
} from "../src/chart";

describe("CGM Charting Module", () => {
  it("computePercentile calculates interpolated percentiles", () => {
    expect(computePercentile([], 50)).toBe(0);
    expect(computePercentile([100], 50)).toBe(100);

    const values = [10, 20, 30, 40, 50];
    expect(computePercentile(values, 0)).toBe(10);
    expect(computePercentile(values, 50)).toBe(30);
    expect(computePercentile(values, 100)).toBe(50);
  });

  it("calculateAGP computes 24-hour percentile intervals across multi-day CGM traces", () => {
    expect(calculateAGP([], "time", "glucose")).toBeNull();
    expect(
      calculateAGP([{ time: "invalid", glucose: "bad" }], "time", "glucose"),
    ).toBeNull();

    // Generate multi-day readings across different hours
    const rows = [
      { time: new Date("2024-01-01T00:15:00Z"), glucose: 100 },
      { time: "2024-01-02T00:20:00Z", glucose: "120" },
      { time: "2024-01-01T06:30:00Z", glucose: 150 },
      { time: "2024-01-02T06:45:00Z", glucose: 160 },
      { time: "2024-01-01T12:00:00Z", glucose: 210 },
      { time: "2024-01-02T12:15:00Z", glucose: 190 },
      { time: "2024-01-01T18:00:00Z", glucose: 80 },
      { time: "2024-01-02T18:30:00Z", glucose: 95 },
    ];

    const agp = calculateAGP(rows, "time", "glucose", 30);
    expect(agp).not.toBeNull();
    expect(agp?.totalDays).toBe(2);
    expect(agp?.totalReadings).toBe(8);
    expect(agp?.intervals.length).toBe(48);

    // Interval 0 (00:00 - 00:30)
    const intv0 = agp?.intervals[0];
    expect(intv0?.count).toBe(2);
    expect(intv0?.p50).toBe(110);
    expect(intv0?.p5).toBe(101);
    expect(intv0?.p95).toBe(119);
  });

  it("renderAGPSvg generates accessible SVG curve with ribbons and target band", () => {
    const rows = [
      { time: "2024-01-01T00:15:00Z", glucose: 100 },
      { time: "2024-01-02T00:20:00Z", glucose: 120 },
      { time: "2024-01-01T12:00:00Z", glucose: 140 },
      { time: "2024-01-02T12:15:00Z", glucose: 160 },
    ];
    const agp = calculateAGP(rows, "time", "glucose", 30);
    expect(agp).not.toBeNull();

    const svg = renderAGPSvg(agp!);
    expect(svg.tagName.toLowerCase()).toBe("svg");
    expect(svg.getAttribute("role")).toBe("img");
    expect(svg.getAttribute("aria-label")).toContain(
      "24-Hour Ambulatory Glucose Profile",
    );
    expect(svg.querySelectorAll("rect").length).toBeGreaterThan(0); // Target range rect
    expect(svg.querySelectorAll("polygon").length).toBe(2); // Outer and inner percentile ribbons
    expect(svg.querySelectorAll("polyline").length).toBe(1); // Median curve
  });

  it("detectCgmColumns detects glucose and time columns correctly", () => {
    expect(detectCgmColumns([])).toEqual({});

    const detected = detectCgmColumns([
      { patient_id: 1, cgm: 120, timestamp: "2023-01-01T12:00:00" },
    ]);
    expect(detected.glucoseCol).toBe("cgm");
    expect(detected.timeCol).toBe("timestamp");

    const nonCgm = detectCgmColumns([{ name: "Alice", age: 30 }]);
    expect(nonCgm.glucoseCol).toBeUndefined();
    expect(nonCgm.timeCol).toBeUndefined();
  });

  it("calculateTIR calculates metrics across all clinical glucose bands", () => {
    // Empty case
    expect(calculateTIR([])).toEqual({
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
    });

    // 5 values, 1 in each band:
    // <54: 50
    // 54-69: 60
    // 70-180: 120
    // 181-250: 200
    // >250: 300
    const tir = calculateTIR([50, 60, 120, 200, 300]);
    expect(tir.veryLow).toBe(20);
    expect(tir.low).toBe(20);
    expect(tir.inRange).toBe(20);
    expect(tir.high).toBe(20);
    expect(tir.veryHigh).toBe(20);
    expect(tir.mean).toBe(146);
    expect(tir.count).toBe(5);
  });

  it("renderTIRBarSvg generates an accessible SVG chart", () => {
    const tir = calculateTIR([50, 60, 120, 200, 300]);
    const svg = renderTIRBarSvg(tir);
    expect(svg.tagName.toLowerCase()).toBe("svg");
    expect(svg.getAttribute("role")).toBe("img");
    expect(svg.querySelectorAll("rect").length).toBe(5);
    expect(svg.textContent).toContain("Mean: 146 mg/dL");
  });

  it("renderHistogramSvg generates accessible histogram bars", () => {
    const emptySvg = renderHistogramSvg([]);
    expect(emptySvg.tagName.toLowerCase()).toBe("svg");

    const histSvg = renderHistogramSvg([50, 65, 80, 110, 150, 190, 260]);
    expect(histSvg.querySelectorAll("rect").length).toBeGreaterThan(0);
  });

  it("renderCgmCard returns a styled card or null based on column presence", () => {
    expect(renderCgmCard([])).toBeNull();
    expect(renderCgmCard([{ patient: "Bob", notes: "none" }])).toBeNull();

    const card = renderCgmCard([
      { time: "2023-01-01 10:00", glucose: 115 },
      { time: "2023-01-01 10:05", glucose: "130.5" },
      { time: "2023-01-01 14:00", glucose: 150 },
      { time: "2023-01-01 14:30", glucose: 175 },
    ]);
    expect(card).not.toBeNull();
    expect(card?.className).toBe("cgm-analytics-card");
    expect(card?.textContent).toContain("CGM Analytics");
    expect(card?.querySelectorAll("svg").length).toBe(3); // TIR bar, AGP curve, and Histogram

    // Test print button click
    const printBtn = card?.querySelector(
      ".print-report-btn",
    ) as HTMLButtonElement | null;
    expect(printBtn).not.toBeNull();
    let printed = false;
    window.print = () => {
      printed = true;
    };
    printBtn?.click();
    expect(printed).toBe(true);
  });

  it("parseDateAndMinuteOfDay handles various input formats and invalid dates", () => {
    // Invalid inputs
    expect(parseDateAndMinuteOfDay(null)).toBeNull();
    expect(parseDateAndMinuteOfDay("not-a-date")).toBeNull();
    expect(parseDateAndMinuteOfDay(new Date("invalid"))).toBeNull();
    expect(parseDateAndMinuteOfDay(NaN)).toBeNull();

    // Date object
    const dateObj = new Date("2024-03-15T08:30:00Z");
    const parsedDate = parseDateAndMinuteOfDay(dateObj);
    expect(parsedDate?.minuteOfDay).toBe(510);

    // Literal string without timezone
    const strNoTz = parseDateAndMinuteOfDay("2024-03-15 14:45:00");
    expect(strNoTz?.minuteOfDay).toBe(14 * 60 + 45);

    // Literal string with Z
    const strZ = parseDateAndMinuteOfDay("2024-03-15T14:45:00Z");
    expect(strZ?.minuteOfDay).toBe(14 * 60 + 45);

    // String with offset
    const strOffset = parseDateAndMinuteOfDay("2024-03-15T14:45:00+02:00");
    expect(strOffset).not.toBeNull();

    // Natural date string fallback
    const naturalDate = parseDateAndMinuteOfDay("March 15, 2024 10:00:00");
    expect(naturalDate).not.toBeNull();

    // Numeric timestamp
    const numTs = parseDateAndMinuteOfDay(Date.UTC(2024, 2, 15, 6, 15));
    expect(numTs?.minuteOfDay).toBe(375);
  });

  it("filter14DayWindow slices traces to the most recent 14 days", () => {
    // Empty rows or invalid dates
    expect(filter14DayWindow([], "time")).toEqual([]);
    expect(filter14DayWindow([{ time: "bad" }], "time")).toEqual([
      { time: "bad" },
    ]);

    const baseMs = new Date("2024-06-30T12:00:00Z").getTime();
    const rows = [
      { time: new Date(baseMs - 20 * 24 * 3600 * 1000), glucose: 100 }, // Date instance (dropped)
      { time: new Date(baseMs - 10 * 24 * 3600 * 1000), glucose: 120 }, // Date instance (kept)
      {
        time: new Date(baseMs - 2 * 24 * 3600 * 1000).toISOString(),
        glucose: 130,
      }, // 2 days ago (kept)
      { time: new Date(baseMs).toISOString(), glucose: 140 }, // Latest (kept)
    ];

    const filtered = filter14DayWindow(rows, "time");
    expect(filtered.length).toBe(3);
    expect(filtered[0].glucose).toBe(120);
    expect(filtered[2].glucose).toBe(140);
  });

  it("renderCgmCard displays valid wear badge when compliance >= 70%", () => {
    const validRows = Array.from({ length: 250 }, (_, i) => ({
      time: `2024-01-01 10:${String(i % 60).padStart(2, "0")}`,
      glucose: 120,
    }));
    const card = renderCgmCard(validRows);
    expect(card).not.toBeNull();
    expect(card?.textContent).toContain("Valid (>=70%)");
  });

  it("calculateActiveWear computes compliance against clinical standard (288 readings/day)", () => {
    // 1 day with 250 readings (>70% wear, 250/288 = 86.8%)
    const validRows = Array.from({ length: 250 }, (_, i) => ({
      time: `2024-01-01 10:${String(i % 60).padStart(2, "0")}`,
      glucose: 120,
    }));
    const validWear = calculateActiveWear(validRows, "time");
    expect(validWear.isValidWear).toBe(true);
    expect(validWear.wearPercentage).toBeGreaterThanOrEqual(70.0);

    // 1 day with 50 readings (<70% wear, 50/288 = 17.4%)
    const cautionRows = Array.from({ length: 50 }, (_, i) => ({
      time: `2024-01-01 10:${String(i % 60).padStart(2, "0")}`,
      glucose: 120,
    }));
    const cautionWear = calculateActiveWear(cautionRows, "time");
    expect(cautionWear.isValidWear).toBe(false);
    expect(cautionWear.wearPercentage).toBeLessThan(70.0);
  });

  it("renders CGM card with RTL and handles observation window select changes", async () => {
    const parent = document.createElement("div");
    document.body.appendChild(parent);

    await i18next.changeLanguage("ar");
    const sampleRows = [
      { time: "2024-01-01T12:00:00Z", glucose: 100 },
      { time: "2024-01-05T12:00:00Z", glucose: 110 },
      { time: "2024-01-20T12:00:00Z", glucose: 120 },
    ];
    const card = renderCgmCard(sampleRows);
    expect(card).not.toBeNull();
    expect(card?.getAttribute("dir")).toBe("rtl");
    parent.appendChild(card!);

    const select = card?.querySelector(
      ".agp-window-select",
    ) as HTMLSelectElement | null;
    expect(select).not.toBeNull();
    if (select && select.onchange) {
      select.value = "30";
      select.onchange(new Event("change") as any);
      expect(parent.querySelector(".cgm-analytics-card")).not.toBeNull();
    }

    // Test onchange when container is unattached (container.parentNode is null)
    const unattached = renderCgmCard(sampleRows);
    const unattachedSelect = unattached?.querySelector(
      ".agp-window-select",
    ) as HTMLSelectElement | null;
    if (unattachedSelect && unattachedSelect.onchange) {
      unattachedSelect.onchange(new Event("change") as any);
    }

    // Test filterObservationWindow with days <= 0 and with default days argument
    const allData = filterObservationWindow(sampleRows, "time", 0);
    expect(allData.length).toBe(sampleRows.length);
    const defaultData = filterObservationWindow(sampleRows, "time");
    expect(defaultData.length).toBeGreaterThan(0);

    await i18next.changeLanguage("en");
    document.body.removeChild(parent);
  });

  it("covers chart edge branches", () => {
    // 1. Glucose <= 0 or invalid in calculateAGP
    const zeroRows = [
      { time: "2024-01-01 12:00:00", glucose: 0 },
      { time: "2024-01-01 12:05:00", glucose: -10 },
      { time: "2024-01-01 12:10:00", glucose: "bad" },
    ];
    expect(calculateAGP(zeroRows, "time", "glucose")).toBeNull();

    // 2. Interpolation with nextIdx when interval 0 is empty
    const noonRows = [
      { time: "2024-01-01 12:00:00", glucose: 120 },
      { time: "2024-01-02 12:00:00", glucose: 130 },
    ];
    const noonAgp = calculateAGP(noonRows, "time", "glucose", 60);
    expect(noonAgp).not.toBeNull();
    expect(noonAgp?.intervals[0].p50).toBe(125); // Interpolated from nextIdx

    // 3. renderAGPSvg with activeIntervals.length <= 1
    const singleIntervalAgp = {
      intervals: [
        {
          minuteOfDay: 0,
          count: 1,
          p50: 100,
          p25: 90,
          p75: 110,
          p5: 80,
          p95: 120,
          mean: 100,
        },
      ],
      totalDays: 1,
      totalReadings: 1,
    };
    const singleSvg = renderAGPSvg(singleIntervalAgp as any);
    expect(singleSvg.querySelectorAll("polygon").length).toBe(0);

    // 4. renderTIRBarSvg with narrow segment (width < 35)
    const narrowTir = calculateTIR([120]); // 100% in range, others 0% (< 35 width)
    const narrowSvg = renderTIRBarSvg(narrowTir);
    expect(narrowSvg.querySelectorAll("text").length).toBeGreaterThan(0);

    // 5. renderCgmCard with empty glucose values returns null
    const noValidGlucose = [{ time: "2024-01-01 10:00", glucose: "invalid" }];
    expect(renderCgmCard(noValidGlucose)).toBeNull();

    // 6. renderCgmCard with timeCol but calculateAGP returning null
    const badTimeRows = [{ time: "invalid-time", glucose: 120 }];
    const badTimeCard = renderCgmCard(badTimeRows);
    expect(badTimeCard).not.toBeNull();
    expect(badTimeCard?.querySelectorAll("svg").length).toBe(2); // TIR and histogram, no AGP

    // 7. parseDateAndMinuteOfDay with invalid offset date string falling back to literal
    expect(parseDateAndMinuteOfDay("1000-00-00T00:00:00+00:00")).toMatchObject({
      dayKey: "1000--1-0",
      minuteOfDay: 0,
    });

    // 8. filter14DayWindow with numeric timestamps and unparseable date strings
    const mixedRows = [
      { time: Date.now(), glucose: 120 },
      { time: Date.now() - 30 * 24 * 3600 * 1000, glucose: 100 },
      { time: "bad-date", glucose: 110 },
    ];
    expect(filter14DayWindow(mixedRows, "time").length).toBe(1);

    // 9. calculateActiveWear with empty rows
    const emptyWear = calculateActiveWear([], "time");
    expect(emptyWear.wearPercentage).toBe(0);
    expect(emptyWear.isValidWear).toBe(false);
  });

  it("calculates advanced clinical indices (GMI, CV, LBGI, HBGI, Day/Night TIR)", () => {
    // 1. GMI, CV, LBGI, HBGI calculation
    const readings = [70, 90, 110, 130, 150, 170, 190, 210];
    const metrics = calculateTIR(readings);
    expect(metrics.mean).toBe(140);
    expect(metrics.gmi).toBe(6.7); // 3.31 + 0.02392 * 140 = 6.6588 -> 6.7
    expect(metrics.sd).toBeGreaterThan(0);
    expect(metrics.cv).toBeGreaterThan(0);
    expect(metrics.lbgi).toBeDefined();
    expect(metrics.hbgi).toBeDefined();

    // High CV (> 36%) test
    const volatileReadings = [40, 50, 250, 300, 350];
    const volatileMetrics = calculateTIR(volatileReadings);
    expect(volatileMetrics.cv).toBeGreaterThan(36.0);

    // 2. Day vs Night TIR splits
    const multiDayRows = [
      { timestamp: "2024-01-01 10:00:00", cgm: 120 }, // Day
      { timestamp: "2024-01-01 14:00:00", cgm: 140 }, // Day
      { timestamp: "2024-01-01 23:00:00", cgm: 80 }, // Night
      { timestamp: "2024-01-02 03:00:00", cgm: 95 }, // Night
      { timestamp: "bad-time", cgm: 100 }, // Invalid time
      { timestamp: "2024-01-02 04:00:00", cgm: -5 }, // Invalid glucose
    ];
    const dn = calculateDayNightTIR(multiDayRows, "timestamp", "cgm");
    expect(dn.day.count).toBe(2);
    expect(dn.night.count).toBe(2);
    expect(dn.day.mean).toBe(130);
    expect(dn.night.mean).toBe(87.5);

    // 3. Badges rendered in CGM card
    const card = renderCgmCard(multiDayRows);
    expect(card).not.toBeNull();
    const badges = card?.querySelector(".cgm-clinical-badges");
    expect(badges).not.toBeNull();
    expect(badges?.textContent).toContain("GMI:");
    expect(badges?.textContent).toContain("CV:");
    expect(badges?.textContent).toContain("LBGI:");
    expect(badges?.textContent).toContain("TIR Day:");
  });

  it("parses multi-format clinical dates and preserves time-of-day invariance", () => {
    // US Format: MM/DD/YYYY HH:mm:ss
    const usDate = parseDateAndMinuteOfDay("03/15/2024 14:30:00");
    expect(usDate).not.toBeNull();
    expect(usDate?.dayKey).toBe("2024-2-15");
    expect(usDate?.minuteOfDay).toBe(14 * 60 + 30);

    // European Format: DD/MM/YYYY HH:mm:ss
    const euDate = parseDateAndMinuteOfDay("25/03/2024 09:15:00");
    expect(euDate).not.toBeNull();
    expect(euDate?.dayKey).toBe("2024-2-25");
    expect(euDate?.minuteOfDay).toBe(9 * 60 + 15);

    // CDISC / SAS Format: DD-Mon-YYYY HH:mm:ss
    const cdiscDate = parseDateAndMinuteOfDay("15-Mar-2024 18:45:00");
    expect(cdiscDate).not.toBeNull();
    expect(cdiscDate?.dayKey).toBe("2024-2-15");
    expect(cdiscDate?.minuteOfDay).toBe(18 * 60 + 45);

    // CDISC with offset
    const cdiscOffset = parseDateAndMinuteOfDay("15-Mar-2024 18:45:00+02:00");
    expect(cdiscOffset).not.toBeNull();

    // US format without time
    const usDateOnly = parseDateAndMinuteOfDay("04/20/2024");
    expect(usDateOnly?.minuteOfDay).toBe(0);

    // European format with offset
    const euOffset = parseDateAndMinuteOfDay("25/03/2024 09:15:00-05:00");
    expect(euOffset).not.toBeNull();
  });

  it("guards Kovatchev LBGI and HBGI indices against NaN and handles mmol/L conversions", () => {
    // Extreme sub-1.0 values
    const lowReadings = [0.5, 0.8, 0.9, 1.0, 2.0];
    const lowMetrics = calculateTIR(lowReadings);
    expect(Number.isFinite(lowMetrics.lbgi)).toBe(true);
    expect(Number.isFinite(lowMetrics.hbgi)).toBe(true);
    expect(isNaN(lowMetrics.lbgi ?? NaN)).toBe(false);
    expect(isNaN(lowMetrics.hbgi ?? NaN)).toBe(false);

    // Standard mmol/L values (e.g. 4.5, 5.5, 6.0, 7.2, 8.5 mmol/L)
    const mmolReadings = [4.5, 5.5, 6.0, 7.2, 8.5];
    const mmolMetrics = calculateTIR(mmolReadings);
    expect(mmolMetrics.mean).toBeGreaterThan(70); // Scaled to mg/dL
    expect(mmolMetrics.inRange).toBe(100);
    expect(Number.isFinite(mmolMetrics.lbgi)).toBe(true);
    expect(Number.isFinite(mmolMetrics.hbgi)).toBe(true);
  });

  it("computes active wear percentage correctly over sparse multi-day monitoring periods", () => {
    // Readings on Day 1 and Day 14 (span of 14 days, with 288 readings per active day = 576 total readings)
    const sparseRows = [
      ...Array.from({ length: 288 }, (_, i) => ({
        time: `2024-01-01 10:${String(i % 60).padStart(2, "0")}`,
        glucose: 110,
      })),
      ...Array.from({ length: 288 }, (_, i) => ({
        time: `2024-01-14 10:${String(i % 60).padStart(2, "0")}`,
        glucose: 120,
      })),
    ];

    const sparseWear = calculateActiveWear(sparseRows, "time", 288);
    expect(sparseWear.activeDays).toBe(2);
    expect(sparseWear.calendarDays).toBe(14);
    expect(sparseWear.totalReadings).toBe(576);
    expect(sparseWear.expectedReadings).toBe(14 * 288);
    // 576 / 4032 = ~14.3% wear (NOT 100%)
    expect(sparseWear.wearPercentage).toBeLessThan(20.0);
    expect(sparseWear.isValidWear).toBe(false);

    // Explicit monitoringDays override
    const explicitWear = calculateActiveWear(sparseRows, "time", 288, 30);
    expect(explicitWear.calendarDays).toBe(30);
    expect(explicitWear.expectedReadings).toBe(30 * 288);
  });

  it("covers additional edge branches for 14-day window and active wear", () => {
    // 1. filter14DayWindow with invalid times returns original rows
    const invalidRows = [{ time: "not-a-date", glucose: 100 }];
    const res = filter14DayWindow(invalidRows, "time");
    expect(res).toEqual(invalidRows);

    // 2. calculateActiveWear with empty rows
    const emptyWear = calculateActiveWear([], "time");
    expect(emptyWear.totalReadings).toBe(0);
    expect(emptyWear.wearPercentage).toBe(0);
    expect(emptyWear.isValidWear).toBe(false);

    // 3. calculateActiveWear with numeric timestamp (epoch ms)
    const fixedTime = Date.UTC(2024, 0, 15, 12, 0, 0);
    const numRows = [
      { time: fixedTime, glucose: 100 },
      { time: fixedTime + 300000, glucose: 110 },
    ];
    const numWear = calculateActiveWear(numRows, "time", 288);
    expect(numWear.totalReadings).toBe(2);
    expect(numWear.activeDays).toBe(1);

    // 4. renderCgmCard returns null if detected glucose column has no valid numbers
    const noNumRows = [{ glucose: "not-a-number", time: "2024-01-01 10:00" }];
    expect(renderCgmCard(noNumRows)).toBeNull();

    // 5. calculateActiveWear with cdisc date string hitting days.size fallback
    const cdiscRows = [{ time: "15-Mar-2024 10:00", glucose: 120 }];
    const cdiscWear = calculateActiveWear(cdiscRows, "time");
    expect(cdiscWear.calendarDays).toBe(1);
    expect(cdiscWear.activeDays).toBe(1);

    // 6. calculateActiveWear with Date instance
    const dateRows = [{ time: new Date(), glucose: 130 }];
    const dateWear = calculateActiveWear(dateRows, "time");
    expect(dateWear.totalReadings).toBe(1);
    expect(dateWear.activeDays).toBe(1);
  });

  it("covers remaining chart branches and boundary conditions", () => {
    // 1. calculateDayNightTIR edge branches
    const dnEdge = calculateDayNightTIR(
      [
        { time: "2024-01-01T12:00:00Z", glucose: null }, // rawG not number/string
        { time: "2024-01-01T12:00:00Z", glucose: "bad_str" }, // rawG string but isNaN
        { time: "2024-01-01T12:00:00Z", glucose: 0 }, // val <= 0
        { time: "2024-01-01T12:00:00Z", glucose: -15 }, // val < 0
        { time: "bad_time", glucose: 100 }, // parsed is null
        { time: "2024-01-01T12:00:00Z", glucose: 100 }, // daytime
        { time: "2024-01-01T23:00:00Z", glucose: 120 }, // nighttime
      ],
      "time",
      "glucose",
    );
    expect(dnEdge.day.count).toBe(1);
    expect(dnEdge.night.count).toBe(1);

    // 2. renderTIRBarSvg with narrow segment (width < 35) to skip text rendering
    const narrowMetrics = {
      veryLow: 1.0, // width = 5 < 35
      low: 0,
      inRange: 99.0,
      high: 0,
      veryHigh: 0,
      mean: 100,
      count: 100,
      gmi: 5.7,
      cv: 25.0,
      sd: 25.0,
      lbgi: 1.0,
      hbgi: 1.0,
    };
    const narrowSvg = renderTIRBarSvg(narrowMetrics);
    expect(narrowSvg).toBeInstanceOf(SVGSVGElement);

    // 3. parseDateAndMinuteOfDay format edge cases
    // ISO with no time
    const isoDateOnly = parseDateAndMinuteOfDay("2024-01-15");
    expect(isoDateOnly?.minuteOfDay).toBe(0);

    // CDISC with unknown month and no time
    const cdiscUnknown = parseDateAndMinuteOfDay("15-xyz-2024");
    expect(cdiscUnknown?.minuteOfDay).toBe(0);

    // CDISC with tz that yields invalid Date
    const cdiscInvalidTz = parseDateAndMinuteOfDay(
      "15-Feb-2024 10:00:00+99:99",
    );
    expect(cdiscInvalidTz?.minuteOfDay).toBe(600);

    // Slash with tz that yields invalid Date
    const slashInvalidTz = parseDateAndMinuteOfDay("15/02/2024 10:00:00+99:99");
    expect(slashInvalidTz?.minuteOfDay).toBe(600);

    // Standard JS date format fallback
    const stdDateTz = parseDateAndMinuteOfDay(
      "Wed Sep 16 2026 12:00:00 GMT+0000",
    );
    expect(stdDateTz).not.toBeNull();
    const stdDateNoTz = parseDateAndMinuteOfDay("Wed Sep 16 2026 12:00:00");
    expect(stdDateNoTz).not.toBeNull();
    const stdInvalid = parseDateAndMinuteOfDay("unparseable string format");
    expect(stdInvalid).toBeNull();

    // Numeric rawTime as NaN
    expect(parseDateAndMinuteOfDay(NaN)).toBeNull();

    // Non-Date, non-string, non-numeric rawTime
    expect(parseDateAndMinuteOfDay(null)).toBeNull();
    expect(parseDateAndMinuteOfDay(true)).toBeNull();

    // 4. calculateAGP with non-number non-string rawGlucose and string isNaN
    const agpEdge = calculateAGP(
      [
        { time: "2024-01-01T12:00:00Z", glucose: null },
        { time: "2024-01-01T12:00:00Z", glucose: "bad_glucose" },
      ],
      "time",
      "glucose",
    );
    expect(agpEdge).toBeNull();

    // 5. filter14DayWindow with Date instance, negative timestamp, non-Date non-string, and cdisc string
    const now = Date.now();
    const windowRows = [
      { time: new Date(now), glucose: 100 },
      { time: -500, glucose: 100 }, // rawTime <= 0
      { time: false, glucose: 100 }, // not date/string/number
      { time: "01-Jan-2024 12:00:00", glucose: 100 }, // cdisc string where new Date is NaN
    ];
    const winResult = filter14DayWindow(windowRows, "time");
    expect(winResult.length).toBe(1);

    // 6. calculateActiveWear with invalid Date, negative number, non-primitive, and expectedReadingsPerDay = 0
    const wearEdgeRows = [
      { time: new Date("invalid"), glucose: 100 },
      { time: -100, glucose: 100 },
      { time: null, glucose: 100 },
      { time: "2024-01-01T12:00:00Z", glucose: 100 },
    ];
    const wearZero = calculateActiveWear(wearEdgeRows, "time", 0);
    expect(wearZero.wearPercentage).toBe(0);

    // 7. renderCgmCard:
    // - non-number non-string raw glucose in rows
    // - cv > 36.0 (suboptimal CV styling)
    // - no timeCol (only glucoseCol)
    // - all day readings (dn.night.count === 0)
    // - all night readings (dn.day.count === 0)
    // - timeCol with unparseable timestamps (agpData === null)
    const cardRowsSuboptimal = [
      { glucose: null, time: "2024-01-01T12:00:00Z" },
      { glucose: true, time: "2024-01-01T12:00:00Z" },
      { glucose: 50, time: "2024-01-01T12:00:00Z" }, // low
      { glucose: 250, time: "2024-01-01T12:30:00Z" }, // high -> high SD / CV > 36%
    ];
    const cardSuboptimal = renderCgmCard(cardRowsSuboptimal);
    expect(cardSuboptimal).not.toBeNull();

    // Card with no time column
    const cardNoTime = renderCgmCard([
      { sensor_glucose: 100 },
      { sensor_glucose: 110 },
    ]);
    expect(cardNoTime).not.toBeNull();

    // Card with all day readings (night count 0)
    const cardAllDay = renderCgmCard([
      { glucose: 100, time: "2024-01-01T12:00:00Z" },
      { glucose: 110, time: "2024-01-01T14:00:00Z" },
    ]);
    expect(cardAllDay).not.toBeNull();

    // Card with all night readings (day count 0)
    const cardAllNight = renderCgmCard([
      { glucose: 100, time: "2024-01-01T02:00:00Z" },
      { glucose: 110, time: "2024-01-01T04:00:00Z" },
    ]);
    expect(cardAllNight).not.toBeNull();

    // Card with unparseable time stamps (agpData is null)
    const cardUnparseableTime = renderCgmCard([
      { glucose: 100, time: "bad_timestamp_1" },
      { glucose: 110, time: "bad_timestamp_2" },
    ]);
    expect(cardUnparseableTime).not.toBeNull();
  });

  it("handles CDISC SAS and European dates in filter14DayWindow and calculateActiveWear", () => {
    const cdiscRows = [
      { time: "01-JAN-2024 10:00:00", glucose: 100 },
      { time: "10-JAN-2024 12:00:00", glucose: 110 },
      { time: "14-JAN-2024 16:30:00", glucose: 120 },
      { time: "15-JAN-2024 08:00:00", glucose: 130 },
    ];
    const filtered = filter14DayWindow(cdiscRows, "time");
    expect(filtered.length).toBe(4);

    const wear = calculateActiveWear(cdiscRows, "time");
    expect(wear.activeDays).toBe(4);
    expect(wear.calendarDays).toBe(15);
    expect(wear.totalReadings).toBe(4);

    // European date slash format
    const euroRows = [
      { time: "15/01/2024 09:00:00", glucose: 105 },
      { time: "16/01/2024 09:00:00", glucose: 115 },
    ];
    const euroFiltered = filter14DayWindow(euroRows, "time");
    expect(euroFiltered.length).toBe(2);
  });

  it("correctly identifies mmol/L units even with severe DKA hyperglycemia spikes >= 35 mmol/L", () => {
    // Normal readings 5-9 mmol/L with a spike to 38 mmol/L
    const mmolWithSpike = [5.5, 6.0, 7.2, 8.1, 6.5, 38.0, 5.8, 6.2, 7.0, 8.5];
    const tir = calculateTIR(mmolWithSpike);
    // Values scaled to mg/dL: 5.5 * 18.0182 = 99.1 mg/dL (in range 70-180)
    // 38 * 18.0182 = 684.7 mg/dL (very high > 250)
    expect(tir.veryLow).toBe(0);
    expect(tir.inRange).toBeGreaterThan(50);
    expect(tir.veryHigh).toBe(10); // Exactly 1 out of 10 is very high
  });

  it("calculates Day/Night TIR with mmol/L values scaled consistently", () => {
    const dayNightMmol = [
      { time: "2024-01-01T10:00:00Z", glucose: 6.5 },
      { time: "2024-01-01T11:00:00Z", glucose: 7.0 },
      { time: "2024-01-01T23:00:00Z", glucose: 6.0 },
      { time: "2024-01-02T01:00:00Z", glucose: 5.8 },
    ];
    const dn = calculateDayNightTIR(dayNightMmol, "time", "glucose");
    expect(dn.day.inRange).toBe(100);
    expect(dn.night.inRange).toBe(100);
    expect(dn.day.mean).toBeGreaterThan(100); // scaled by 18.0182

    // calculateActiveWear with expectedReadingsPerDay = 0
    const zeroExpectWear = calculateActiveWear(dayNightMmol, "time", 0);
    expect(zeroExpectWear.wearPercentage).toBe(0);

    // calculateActiveWear with zero timestamp (minTime stays Infinity, triggers fallback calendarDays = days.size)
    const zeroTimeWear = calculateActiveWear([{ time: 0 }], "time");
    expect(zeroTimeWear.calendarDays).toBe(1);
  });

  it("validates generic value/val columns contextually based on clinical ranges", () => {
    // Table with generic "val" and valid blood glucose numbers (median ~120)
    const validCgmTable = [
      { time: "2024-01-01T12:00:00Z", val: 110 },
      { time: "2024-01-01T12:05:00Z", val: "125" },
      { time: "2024-01-01T12:10:00Z", val: 130 },
    ];
    const detected1 = detectCgmColumns(validCgmTable);
    expect(detected1.glucoseCol).toBe("val");

    // Table with generic "value" containing non-glucose values (e.g. insulin doses 0.5 - 2.0 units)
    const insulinSettingsTable = [
      { time: "2024-01-01T12:00:00Z", value: 0.5 },
      { time: "2024-01-01T12:05:00Z", value: 1.0 },
      { time: "2024-01-01T12:10:00Z", value: 1.5 },
    ];
    const detected2 = detectCgmColumns(insulinSettingsTable);
    expect(detected2.glucoseCol).toBeUndefined();

    // Table with generic "value" with non-numeric samples (numSamples.length === 0)
    const textValueTable = [
      { time: "2024-01-01T12:00:00Z", value: "hello" },
      { time: "2024-01-01T12:05:00Z", value: null },
    ];
    const detected3 = detectCgmColumns(textValueTable);
    expect(detected3.glucoseCol).toBeUndefined();

    // Table with generic "value" with extreme median > 500
    const highMedianTable = [
      { time: "2024-01-01T12:00:00Z", value: 1000 },
      { time: "2024-01-01T12:05:00Z", value: 1200 },
    ];
    const detected4 = detectCgmColumns(highMedianTable);
    expect(detected4.glucoseCol).toBeUndefined();

    // Table without time column should not match generic "value"
    const noTimeTable = [{ value: 120 }, { value: 130 }];
    const detected5 = detectCgmColumns(noTimeTable);
    expect(detected5.glucoseCol).toBeUndefined();

    // Table with time column but no glucose column and no generic value column
    const timeOnlyTable = [{ timestamp: "2024-01-01", patient_name: "Alice" }];
    const detected6 = detectCgmColumns(timeOnlyTable);
    expect(detected6.glucoseCol).toBeUndefined();
    expect(detected6.timeCol).toBe("timestamp");
  });

  it("comprehensively tests clinical dates, daylight saving transitions, leap years, and day/night boundary crossing", () => {
    // 1. CDISC SAS format dates
    const cdisc1 = parseDateAndMinuteOfDay("01-JAN-2024 08:30:00");
    expect(cdisc1).not.toBeNull();
    expect(cdisc1?.minuteOfDay).toBe(8 * 60 + 30);

    const cdisc2 = parseDateAndMinuteOfDay("15-FEB-2024 23:45:00");
    expect(cdisc2).not.toBeNull();
    expect(cdisc2?.minuteOfDay).toBe(23 * 60 + 45);

    // 2. Leap year date (February 29, 2024)
    const leapYearCdisc = parseDateAndMinuteOfDay("29-FEB-2024 12:00:00");
    expect(leapYearCdisc).not.toBeNull();
    expect(leapYearCdisc?.dayKey).toBe("2024-1-29");

    const leapYearIso = parseDateAndMinuteOfDay("2024-02-29T15:30:00Z");
    expect(leapYearIso).not.toBeNull();
    expect(leapYearIso?.dayKey).toBe("2024-1-29");

    // 3. Day/Night boundary crossing (Day: 06:00:00 - 21:59:59; Night: 22:00:00 - 05:59:59)
    const boundaryRows = [
      { time: "2024-01-01 06:00:00", glucose: 100 }, // Exactly 06:00:00 -> minute 360 -> Day
      { time: "2024-01-01 21:59:00", glucose: 110 }, // 21:59 -> Day
      { time: "2024-01-01 22:00:00", glucose: 120 }, // Exactly 22:00:00 -> minute 1320 -> Night
      { time: "2024-01-01 05:59:00", glucose: 130 }, // 05:59 -> Night
    ];
    const dn = calculateDayNightTIR(boundaryRows, "time", "glucose");
    expect(dn.day.count).toBe(2);
    expect(dn.night.count).toBe(2);

    // 4. Daylight saving transition timezone strings
    const dst1 = parseDateAndMinuteOfDay("2024-03-10T01:59:00-05:00");
    expect(dst1).not.toBeNull();
    expect(dst1?.epochMs).toBeGreaterThan(0);

    const dst2 = parseDateAndMinuteOfDay("2024-03-10T03:00:00-04:00");
    expect(dst2).not.toBeNull();
    expect(dst2?.epochMs).toBeGreaterThan(0);
  });
});
