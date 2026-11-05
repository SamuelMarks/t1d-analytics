/**
 * @file table1-queries.ts
 * @module table1-queries
 * Curated Academic "Table 1" clinical baseline and stratified outcome examples across 7 complexity tiers.
 */

/**
 * Specification for an Academic Table 1 clinical starter query.
 */
export interface Table1ExampleSpec {
  /** Unique identifier for the Table 1 example tier. */
  id: string;
  /** Complexity rating label ('Low', 'Low-Medium', 'Medium', 'Medium-High', 'High', 'Very High'). */
  complexity:
    "Low" | "Low-Medium" | "Medium" | "Medium-High" | "High" | "Very High";
  /** i18n translation key for the button label. */
  i18nKey: string;
  /** Natural language clinical question corresponding to the Table 1 query. */
  question: string;
  /** Validated DuckDB SQL query producing the HTML Table 1 result. */
  sql: string;
}

/**
 * Curated Academic "Table 1" clinical baseline and stratified outcome examples across 7 complexity tiers.
 */
export const TABLE_1_EXAMPLE_QUERIES: ReadonlyArray<Table1ExampleSpec> = [
  {
    id: "table1-overall",
    complexity: "Low",
    i18nKey: "ui.exampleQueries.table1Overall",
    question:
      "Generate a Table 1 summary of overall participant count, mean age, percent female, mean HbA1c, and mean Time in Range from the patients table.",
    sql: `SELECT
    COUNT(*) AS "N Participants",
    ROUND(AVG(age), 1) AS "Mean Age (yr)",
    ROUND(STDDEV_SAMP(age), 1) AS "SD Age (yr)",
    SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS "Female (n)",
    ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) AS "Female (%)",
    ROUND(AVG(hba1c), 2) AS "Mean HbA1c (%)",
    ROUND(AVG(tir), 1) AS "Mean TIR 70-180 (%)",
    ROUND(AVG(tbr), 2) AS "Mean TBR <70 (%)",
    ROUND(AVG(tar), 1) AS "Mean TAR >180 (%)",
    ROUND(AVG(mean_glucose), 1) AS "Mean CGM (mg/dL)"
FROM patients;`,
  },
  {
    id: "table1-by-arm",
    complexity: "Low-Medium",
    i18nKey: "ui.exampleQueries.table1ByArm",
    question:
      "Create a Table 1 comparing Closed-Loop vs Control treatment arms in patients with N, Age (Mean ± SD), Female n (%), Baseline HbA1c (Mean ± SD), TIR, TBR, TAR, and total DKA events.",
    sql: `SELECT
    TxGroup AS "Treatment Arm",
    COUNT(*) AS "N",
    ROUND(AVG(age), 1) || ' ± ' || ROUND(STDDEV_SAMP(age), 1) AS "Age (yr, Mean ± SD)",
    SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) || '%)' AS "Female, n (%)",
    ROUND(AVG(hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(hba1c), 2) AS "HbA1c (%, Mean ± SD)",
    ROUND(AVG(tir), 1) || ' ± ' || ROUND(STDDEV_SAMP(tir), 1) AS "TIR 70-180 (%, Mean ± SD)",
    ROUND(AVG(tbr), 2) || ' ± ' || ROUND(STDDEV_SAMP(tbr), 2) AS "TBR <70 (%, Mean ± SD)",
    ROUND(AVG(tar), 1) || ' ± ' || ROUND(STDDEV_SAMP(tar), 1) AS "TAR >180 (%, Mean ± SD)",
    SUM(numhospdka) AS "Total DKA Events"
FROM patients
GROUP BY TxGroup
ORDER BY TxGroup;`,
  },
  {
    id: "table1-by-age",
    complexity: "Medium",
    i18nKey: "ui.exampleQueries.table1ByAge",
    question:
      "Build an age-stratified Table 1 across Pediatric (<18 yr), Adult (18-49 yr), and Older Adult (>=50 yr) cohorts showing sample size, % Closed-Loop, mean and median HbA1c, mean TIR, and % achieving ADA targets (TIR >= 70% and HbA1c < 7.0%).",
    sql: `SELECT
    CASE
        WHEN age < 18 THEN '1. Pediatric (<18 yr)'
        WHEN age < 50 THEN '2. Adult (18-49 yr)'
        ELSE '3. Older Adult (>=50 yr)'
    END AS "Age Stratum",
    COUNT(*) AS "N",
    ROUND(AVG(age), 1) AS "Mean Age (yr)",
    ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN 1.0 ELSE 0.0 END), 1) AS "Closed-Loop (%)",
    ROUND(AVG(hba1c), 2) AS "Mean HbA1c (%)",
    ROUND(MEDIAN(hba1c), 2) AS "Median HbA1c (%)",
    ROUND(AVG(tir), 1) AS "Mean TIR (%)",
    ROUND(100.0 * AVG(CASE WHEN tir >= 70.0 AND hba1c < 7.0 THEN 1.0 ELSE 0.0 END), 1) AS "ADA Goal Achieved (%)"
FROM patients
GROUP BY 1
ORDER BY 1;`,
  },
  {
    id: "table1-by-pump",
    complexity: "Medium",
    i18nKey: "ui.exampleQueries.table1ByPump",
    question:
      "Produce a device-stratified Table 1 joining subjects and patients by pump manufacturer (Insulet, Medtronic, Tandem) with N, Age (Mean ± SD), % Female, mean HbA1c, mean TIR, mean CGM glucose, and DKA count.",
    sql: `SELECT
    s.Pt_PumpManuf AS "Pump Manufacturer",
    COUNT(*) AS "N",
    ROUND(AVG(p.age), 1) || ' ± ' || ROUND(STDDEV_SAMP(p.age), 1) AS "Age (Mean ± SD)",
    ROUND(100.0 * AVG(CASE WHEN p.gender = 'F' THEN 1.0 ELSE 0.0 END), 1) AS "Female (%)",
    ROUND(AVG(p.hba1c), 2) AS "Mean HbA1c (%)",
    ROUND(AVG(p.tir), 1) AS "Mean TIR (%)",
    ROUND(AVG(p.mean_glucose), 1) AS "Mean Glucose (mg/dL)",
    SUM(p.numhospdka) AS "DKA Events"
FROM subjects s
JOIN patients p ON s.PtID = p.subject_id
GROUP BY s.Pt_PumpManuf
ORDER BY s.Pt_PumpManuf;`,
  },
  {
    id: "table1-longitudinal-lab",
    complexity: "Medium-High",
    i18nKey: "ui.exampleQueries.table1LongitudinalLab",
    question:
      "Generate a longitudinal laboratory Table 1 joining patients and hba1c_labs by treatment arm showing randomized N, lab completers N, baseline vs endpoint HbA1c, HbA1c change (delta), and baseline vs endpoint C-Peptide.",
    sql: `WITH lab_bounds AS (
    SELECT
        subject_id,
        FIRST(hba1c ORDER BY VisitDate ASC) AS baseline_hba1c,
        LAST(hba1c ORDER BY VisitDate ASC) AS endpoint_hba1c,
        FIRST(C_Peptide ORDER BY VisitDate ASC) AS baseline_c_peptide,
        LAST(C_Peptide ORDER BY VisitDate ASC) AS endpoint_c_peptide
    FROM hba1c_labs
    GROUP BY subject_id
)
SELECT
    p.TxGroup AS "Treatment Arm",
    COUNT(p.subject_id) AS "Randomized (N)",
    COUNT(l.subject_id) AS "Lab Completers (n)",
    ROUND(AVG(l.baseline_hba1c), 2) AS "Baseline HbA1c (%)",
    ROUND(AVG(l.endpoint_hba1c), 2) AS "Endpoint HbA1c (%)",
    ROUND(AVG(l.endpoint_hba1c - l.baseline_hba1c), 2) AS "Delta HbA1c (%)",
    ROUND(AVG(l.baseline_c_peptide), 3) AS "Baseline C-Peptide (nmol/L)",
    ROUND(AVG(l.endpoint_c_peptide), 3) AS "Endpoint C-Peptide (nmol/L)"
FROM patients p
LEFT JOIN lab_bounds l ON p.subject_id = l.subject_id
GROUP BY p.TxGroup
ORDER BY p.TxGroup;`,
  },
  {
    id: "table1-multi-domain-rollup",
    complexity: "High",
    i18nKey: "ui.exampleQueries.table1MultiDomainRollup",
    question:
      "Construct a multi-domain Table 1 with an Overall Cohort rollup joining patients, raw cgm_readings, and dka_adverse_events by treatment arm, reporting N, Age, HbA1c, raw sensor CGM Mean ± SD, sensor CV%, TIR%, DKA events, severe hypoglycemia events, and hospitalizations.",
    sql: `WITH cgm_agg AS (
    SELECT
        subject_id,
        AVG(glucose_value) AS sensor_mean_mgdl,
        STDDEV_SAMP(glucose_value) AS sensor_sd_mgdl,
        100.0 * STDDEV_SAMP(glucose_value) / NULLIF(AVG(glucose_value), 0) AS sensor_cv_pct
    FROM cgm_readings
    GROUP BY subject_id
),
ae_agg AS (
    SELECT
        subject_id,
        SUM(CASE WHEN EventType = 'DKA' THEN 1 ELSE 0 END) AS dka_ae,
        SUM(CASE WHEN EventType = 'Severe Hypoglycemia' THEN 1 ELSE 0 END) AS sev_hypo_ae,
        SUM(CASE WHEN Hospitalized THEN 1 ELSE 0 END) AS hosp_ae
    FROM dka_adverse_events
    GROUP BY subject_id
)
SELECT
    COALESCE(p.TxGroup, 'Overall Cohort') AS "Cohort Arm",
    COUNT(*) AS "N",
    ROUND(AVG(p.age), 1) || ' ± ' || ROUND(STDDEV_SAMP(p.age), 1) AS "Age (yr, Mean ± SD)",
    ROUND(AVG(p.hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(p.hba1c), 2) AS "HbA1c (%, Mean ± SD)",
    ROUND(AVG(c.sensor_mean_mgdl), 1) || ' ± ' || ROUND(AVG(c.sensor_sd_mgdl), 1) AS "Sensor CGM (mg/dL, Mean ± SD)",
    ROUND(AVG(c.sensor_cv_pct), 1) AS "Sensor CV (%)",
    ROUND(AVG(p.tir), 1) AS "TIR 70-180 (%)",
    COALESCE(SUM(a.dka_ae), 0) AS "DKA Events",
    COALESCE(SUM(a.sev_hypo_ae), 0) AS "Severe Hypo Events",
    COALESCE(SUM(a.hosp_ae), 0) AS "Hospitalizations"
FROM patients p
LEFT JOIN cgm_agg c ON p.subject_id = c.subject_id
LEFT JOIN ae_agg a ON p.subject_id = a.subject_id
GROUP BY ROLLUP(p.TxGroup)
ORDER BY CASE WHEN p.TxGroup IS NULL THEN 1 ELSE 0 END, p.TxGroup;`,
  },
  {
    id: "table1-transposed-nejm",
    complexity: "Very High",
    i18nKey: "ui.exampleQueries.table1TransposedNejm",
    question:
      "Generate a transposed NEJM/Lancet-style Table 1 where each row is a clinical characteristic (Sample Size N, Age Mean ± SD, Female Sex n (%), Baseline HbA1c Mean ± SD, Time in Range Mean ± SD, Time Below Range Median (IQR)) and columns are Closed-Loop Arm, Control Arm, Overall Cohort, and Arm Difference.",
    sql: `WITH base AS (
    SELECT p.*, s.Pt_PumpManuf
    FROM patients p
    LEFT JOIN subjects s ON p.subject_id = s.PtID
)
SELECT
    1 AS sort_ord,
    'Sample Size (N)' AS "Characteristic",
    CAST(SUM(CASE WHEN TxGroup = 'Closed-Loop' THEN 1 ELSE 0 END) AS VARCHAR) AS "Closed-Loop Arm",
    CAST(SUM(CASE WHEN TxGroup = 'Control' THEN 1 ELSE 0 END) AS VARCHAR) AS "Control Arm",
    CAST(COUNT(*) AS VARCHAR) AS "Overall Cohort",
    '—' AS "Arm Difference"
FROM base
UNION ALL
SELECT
    2,
    'Age (years), Mean ± SD',
    ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN age END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN age END), 1),
    ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN age END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN age END), 1),
    ROUND(AVG(age), 1) || ' ± ' || ROUND(STDDEV_SAMP(age), 1),
    CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN age END) - AVG(CASE WHEN TxGroup = 'Control' THEN age END), 2) AS VARCHAR)
FROM base
UNION ALL
SELECT
    3,
    'Female Sex, n (%)',
    SUM(CASE WHEN TxGroup = 'Closed-Loop' AND gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END END), 1) || '%)',
    SUM(CASE WHEN TxGroup = 'Control' AND gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Control' THEN CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END END), 1) || '%)',
    SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) || '%)',
    '—'
FROM base
UNION ALL
SELECT
    4,
    'Baseline HbA1c (%), Mean ± SD',
    ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END), 2) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END), 2),
    ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2),
    ROUND(AVG(hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(hba1c), 2),
    CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END) - AVG(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2) AS VARCHAR)
FROM base
UNION ALL
SELECT
    5,
    'Time in Range 70-180 mg/dL (%), Mean ± SD',
    ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END), 1),
    ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN tir END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN tir END), 1),
    ROUND(AVG(tir), 1) || ' ± ' || ROUND(STDDEV_SAMP(tir), 1),
    CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END) - AVG(CASE WHEN TxGroup = 'Control' THEN tir END), 2) AS VARCHAR)
FROM base
UNION ALL
SELECT
    6,
    'Time Below Range <70 mg/dL (%), Median (IQR)',
    ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.75), 2) || ')',
    ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.75), 2) || ')',
    ROUND(QUANTILE_CONT(tbr, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(tbr, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(tbr, 0.75), 2) || ')',
    '—'
FROM base
ORDER BY sort_ord;`,
  },
];
