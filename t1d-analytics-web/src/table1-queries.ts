/**
 * @file table1-queries.ts
 * @module table1-queries
 * Curated Academic "Table 1" clinical baseline and stratified outcome examples across 7 complexity tiers.
 */

import { fetchWithBackendError } from "./api-client";

/**
 * Complexity tier label shared across starter and Academic Table 1 templates.
 */
export type TemplateComplexityTier =
  "Low" | "Low-Medium" | "Medium" | "Medium-High" | "High" | "Very High";

/**
 * Specification for a starter clinical SQL template synchronized with `GET /api/templates`.
 */
export interface StarterTemplateSpec {
  /** Stable identifier ('starter-1' through 'starter-5'). */
  id: string;
  /** Complexity rating label. */
  complexity: TemplateComplexityTier;
  /** i18n key for the localized chip/card label. */
  i18nKey: string;
  /** Optional i18n key for leading SQL comment header. */
  commentI18nKey?: string;
  /** Default SQL comment fallback when `commentI18nKey` is present. */
  defaultComment?: string;
  /** Natural-language clinical question. */
  question: string;
  /** Canonical Gold-Standard DuckDB SQL query. */
  goldSql: string;
  /** Default execution mode ('sql' or 'gemma:2b'). */
  defaultMode: "sql" | "gemma:2b";
}

/**
 * Specification for an Academic Table 1 clinical starter query.
 */
export interface Table1ExampleSpec {
  /** Unique identifier for the Table 1 example tier. */
  id: string;
  /** Complexity rating label ('Low', 'Low-Medium', 'Medium', 'Medium-High', 'High', 'Very High'). */
  complexity: TemplateComplexityTier;
  /** i18n translation key for the button label. */
  i18nKey: string;
  /** Natural language clinical question corresponding to the Table 1 query. */
  question: string;
  /** Validated DuckDB SQL query producing the HTML Table 1 result. */
  sql: string;
}

/**
 * Wire item returned by backend `GET /api/templates`.
 */
export interface ClinicalTemplateWireItem {
  /** Stable template identifier. */
  id: string;
  /** Template category ('starter' or 'table1'). */
  category: "starter" | "table1";
  /** Complexity tier string. */
  complexity: TemplateComplexityTier;
  /** Localization key for display title. */
  i18nKey?: string;
  /** Snake-case localization key alias. */
  i18n_key?: string;
  /** Natural-language clinical question. */
  question: string;
  /** Camel-case canonical Gold SQL string. */
  goldSql?: string;
  /** Snake-case canonical Gold SQL string alias. */
  gold_sql?: string;
  /** Camel-case default execution mode. */
  defaultMode?: "sql" | "gemma:2b";
  /** Snake-case default execution mode alias. */
  default_mode?: "sql" | "gemma:2b";
}

/**
 * Wire payload returned by backend `GET /api/templates`.
 */
export interface TemplatesCatalogWirePayload {
  /** Ordered list of clinical starter and Table 1 templates. */
  templates?: ClinicalTemplateWireItem[];
}

/**
 * Default starter clinical SQL templates (hydrated at runtime from `GET /api/templates`).
 */
export const STARTER_TEMPLATE_EXAMPLES: ReadonlyArray<StarterTemplateSpec> = [
  {
    id: "starter-1",
    complexity: "Low",
    i18nKey: "ui.exampleQueries.patientDemographics",
    commentI18nKey: "ui.exampleQueries.patientDemographicsComment",
    defaultComment: "-- Patient Demographics & Treatment Breakdown",
    question:
      "Show patient count, average age, and total severe hypoglycemia events grouped by treatment group.",
    goldSql: `SELECT 
    TxGroup, 
    COUNT(*) as total_patients, 
    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,
    SUM(NumSevHypo) as total_severe_hypo_events
FROM tblaptsummary 
GROUP BY TxGroup;`,
    defaultMode: "sql",
  },
  {
    id: "starter-2",
    complexity: "Low",
    i18nKey: "ui.exampleQueries.adverseEvents",
    commentI18nKey: "ui.exampleQueries.adverseEventsComment",
    defaultComment: "-- Frequency of Adverse Events",
    question:
      "Show the frequency of each adverse event type ordered from highest to lowest occurrence count.",
    goldSql: `SELECT 
    event as adverse_event_type, 
    COUNT(*) as occurrence_count 
FROM adverseevents 
GROUP BY event 
ORDER BY occurrence_count DESC;`,
    defaultMode: "sql",
  },
  {
    id: "starter-3",
    complexity: "Low",
    i18nKey: "ui.exampleQueries.pumpManufacturers",
    commentI18nKey: "ui.exampleQueries.pumpManufacturersComment",
    defaultComment: "-- Patient Pump Manufacturers",
    question:
      "Count the number of subjects using each insulin pump manufacturer.",
    goldSql: `SELECT 
    Pt_PumpManuf as pump_manufacturer, 
    COUNT(*) as user_count 
FROM subjects 
WHERE Pt_PumpManuf IS NOT NULL 
GROUP BY Pt_PumpManuf 
ORDER BY user_count DESC;`,
    defaultMode: "sql",
  },
  {
    id: "starter-4",
    complexity: "Low-Medium",
    i18nKey: "ui.exampleQueries.hba1cDemographics",
    commentI18nKey: "ui.exampleQueries.hba1cDemographicsComment",
    defaultComment: "-- Joining Subject Summaries with HbA1c Lab Results",
    question:
      "Join subject summaries with HbA1c lab results and show average HbA1c and total tests by treatment group and gender.",
    goldSql: `SELECT 
    t.TxGroup, 
    t.Gender,
    ROUND(AVG(h.HbA1c), 2) as average_hba1c,
    COUNT(h.HbA1c) as total_tests_run
FROM tblaptsummary t
JOIN hba1c h ON t.PtID = h.PtID
WHERE h.HbA1c IS NOT NULL
GROUP BY t.TxGroup, t.Gender
ORDER BY t.TxGroup, average_hba1c DESC;`,
    defaultMode: "sql",
  },
  {
    id: "starter-5",
    complexity: "Low",
    i18nKey: "ui.exampleQueries.nlpFirst5",
    question: "Show me the first 5 patients in the demographics table.",
    goldSql: "SELECT * FROM demographics LIMIT 5;",
    defaultMode: "gemma:2b",
  },
];

/**
 * Curated Academic "Table 1" clinical baseline and stratified outcome examples across 7 complexity tiers.
 */
export const TABLE_1_EXAMPLE_QUERIES: ReadonlyArray<Table1ExampleSpec> = [
  {
    id: "table1-overall",
    complexity: "Low",
    i18nKey: "ui.exampleQueries.table1Overall",
    question:
      "What are the average age, sex breakdown, HbA1c, and time in range for everyone?",
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
      "Compare the Closed-Loop and Control groups on age, sex, HbA1c, TIR, and DKA events.",
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
      "Break down baseline stats by age group (under 18, 18 to 49, and 50 or older).",
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
      "Compare baseline patient stats across different insulin pump brands.",
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
      "How did HbA1c and C-peptide change from baseline to the end of the study by arm?",
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
      "Combine patient demographics, raw CGM sensor readings, and DKA events by treatment group.",
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
      "Create a publication Table 1 comparing Closed-Loop and Control with group differences.",
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
  {
    id: "table1-beck-2023",
    complexity: "Very High",
    i18nKey: "ui.exampleQueries.table1Beck2023",
    question: "Reproduce Table 1 from the Beck 2023 Control-IQ meta-analysis.",
    sql: `WITH base AS (
    SELECT
        age,
        COALESCE(TRY_CAST(age_exact AS DOUBLE), TRY_CAST(age AS DOUBLE) + 0.5) AS age_ex,
        gender,
        race_ethnicity,
        annual_income,
        t1d_duration_years AS dur,
        prestudy_insulin_delivery AS ins_deliv,
        prestudy_cgm_use AS cgm_use,
        hba1c,
        tir,
        tbr,
        CASE WHEN TxGroup IN ('Control-IQ', 'Closed-Loop', 'CLC') THEN 1 ELSE 0 END AS is_ciq,
        CASE WHEN TxGroup IN ('Control', 'SAP', 'SC') THEN 1 ELSE 0 END AS is_ctrl
    FROM patients
),
totals AS (
    SELECT
        COUNT(*) AS n_all,
        SUM(is_ciq) AS n_ciq,
        SUM(is_ctrl) AS n_ctrl
    FROM base
),
raw_rows AS (
    SELECT 1 AS ord, 'Participants, N' AS Characteristic, 0 AS is_cat, 0 AS c_all, 0 AS c_ciq, 0 AS c_ctrl, CAST(n_all AS VARCHAR) AS s_all, CAST(n_ciq AS VARCHAR) AS s_ciq, CAST(n_ctrl AS VARCHAR) AS s_ctrl FROM totals
    UNION ALL
    SELECT 2, 'Age at enrollment', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 3, ' Mean ± SD', 0, 0, 0, 0,
        CAST(ROUND(AVG(age_ex), 0) AS BIGINT) || ' ± ' || CAST(ROUND(STDDEV_SAMP(age_ex), 0) AS BIGINT),
        CAST(ROUND(AVG(CASE WHEN is_ciq = 1 THEN age_ex END), 0) AS BIGINT) || ' ± ' || CAST(ROUND(STDDEV_SAMP(CASE WHEN is_ciq = 1 THEN age_ex END), 0) AS BIGINT),
        CAST(ROUND(AVG(CASE WHEN is_ctrl = 1 THEN age_ex END), 0) AS BIGINT) || ' ± ' || CAST(ROUND(STDDEV_SAMP(CASE WHEN is_ctrl = 1 THEN age_ex END), 0) AS BIGINT)
    FROM base
    UNION ALL
    SELECT 4, ' Range', 0, 0, 0, 0,
        CAST(MIN(age) AS VARCHAR) || '–' || CAST(MAX(age) AS VARCHAR),
        CAST(MIN(CASE WHEN is_ciq = 1 THEN age END) AS VARCHAR) || '–' || CAST(MAX(CASE WHEN is_ciq = 1 THEN age END) AS VARCHAR),
        CAST(MIN(CASE WHEN is_ctrl = 1 THEN age END) AS VARCHAR) || '–' || CAST(MAX(CASE WHEN is_ctrl = 1 THEN age END) AS VARCHAR)
    FROM base
    UNION ALL
    SELECT 5, ' <6 years, n (%)', 1, SUM(CASE WHEN age < 6 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND age < 6 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND age < 6 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 6, ' 6 to <14 years, n (%)', 1, SUM(CASE WHEN age >= 6 AND age < 14 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND age >= 6 AND age < 14 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND age >= 6 AND age < 14 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 7, ' 14 to <25 years, n (%)', 1, SUM(CASE WHEN age >= 14 AND age < 25 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND age >= 14 AND age < 25 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND age >= 14 AND age < 25 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 8, ' 25 to <50 years, n (%)', 1, SUM(CASE WHEN age >= 25 AND age < 50 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND age >= 25 AND age < 50 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND age >= 25 AND age < 50 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 9, ' ≥50 years, n (%)', 1, SUM(CASE WHEN age >= 50 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND age >= 50 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND age >= 50 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 10, 'Sex, n (%)', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 11, ' Female', 1, SUM(CASE WHEN gender IN ('Female', 'F') THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND gender IN ('Female', 'F') THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND gender IN ('Female', 'F') THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 12, ' Male', 1, SUM(CASE WHEN gender IN ('Male', 'M') THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND gender IN ('Male', 'M') THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND gender IN ('Male', 'M') THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 13, 'Race/Ethnicity, n (%)', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 14, ' White non-Hispanic', 1, SUM(CASE WHEN race_ethnicity = 'White non-Hispanic' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND race_ethnicity = 'White non-Hispanic' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND race_ethnicity = 'White non-Hispanic' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 15, ' Black non-Hispanic', 1, SUM(CASE WHEN race_ethnicity = 'Black non-Hispanic' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND race_ethnicity = 'Black non-Hispanic' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND race_ethnicity = 'Black non-Hispanic' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 16, ' Hispanic or Latino', 1, SUM(CASE WHEN race_ethnicity = 'Hispanic or Latino' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND race_ethnicity = 'Hispanic or Latino' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND race_ethnicity = 'Hispanic or Latino' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 17, ' Asian', 1, SUM(CASE WHEN race_ethnicity = 'Asian' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND race_ethnicity = 'Asian' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND race_ethnicity = 'Asian' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 18, ' Native Hawaiian/other Pacific Islander', 1, SUM(CASE WHEN race_ethnicity = 'Native Hawaiian/other Pacific Islander' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND race_ethnicity = 'Native Hawaiian/other Pacific Islander' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND race_ethnicity = 'Native Hawaiian/other Pacific Islander' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 19, ' More than one race', 1, SUM(CASE WHEN race_ethnicity = 'More than one race' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND race_ethnicity = 'More than one race' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND race_ethnicity = 'More than one race' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 20, 'Annual household income, n (%)', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 21, ' < $75,000', 1, SUM(CASE WHEN annual_income = '< $75,000' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND annual_income = '< $75,000' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND annual_income = '< $75,000' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 22, ' ≥ $75,000', 1, SUM(CASE WHEN annual_income IN ('>= $75,000', '≥ $75,000') THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND annual_income IN ('>= $75,000', '≥ $75,000') THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND annual_income IN ('>= $75,000', '≥ $75,000') THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 23, ' Missing', 1, SUM(CASE WHEN annual_income = 'Missing' OR annual_income IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND (annual_income = 'Missing' OR annual_income IS NULL) THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND (annual_income = 'Missing' OR annual_income IS NULL) THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 24, 'Duration of T1D at enrollment', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 25, ' Mean ± SD', 0, 0, 0, 0,
        printf('%.1f', ROUND(AVG(dur), 1)) || ' ± ' || printf('%.1f', ROUND(STDDEV_SAMP(dur), 1)),
        printf('%.1f', ROUND(AVG(CASE WHEN is_ciq = 1 THEN dur END), 1)) || ' ± ' || printf('%.1f', ROUND(STDDEV_SAMP(CASE WHEN is_ciq = 1 THEN dur END), 1)),
        printf('%.1f', ROUND(AVG(CASE WHEN is_ctrl = 1 THEN dur END), 1)) || ' ± ' || printf('%.1f', ROUND(STDDEV_SAMP(CASE WHEN is_ctrl = 1 THEN dur END), 1))
    FROM base
    UNION ALL
    SELECT 26, ' Range', 0, 0, 0, 0,
        printf('%.1f', ROUND(MIN(dur), 1)) || '–' || printf('%.1f', ROUND(MAX(dur), 1)),
        printf('%.1f', ROUND(MIN(CASE WHEN is_ciq = 1 THEN dur END), 1)) || '–' || printf('%.1f', ROUND(MAX(CASE WHEN is_ciq = 1 THEN dur END), 1)),
        printf('%.1f', ROUND(MIN(CASE WHEN is_ctrl = 1 THEN dur END), 1)) || '–' || printf('%.1f', ROUND(MAX(CASE WHEN is_ctrl = 1 THEN dur END), 1))
    FROM base
    UNION ALL
    SELECT 27, ' <1 years, n (%)', 1, SUM(CASE WHEN dur < 1.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND dur < 1.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND dur < 1.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 28, ' 1 to <5 years, n (%)', 1, SUM(CASE WHEN dur >= 1.0 AND dur < 5.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND dur >= 1.0 AND dur < 5.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND dur >= 1.0 AND dur < 5.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 29, ' 5 to <10 years, n (%)', 1, SUM(CASE WHEN dur >= 5.0 AND dur < 10.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND dur >= 5.0 AND dur < 10.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND dur >= 5.0 AND dur < 10.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 30, ' ≥10 years, n (%)', 1, SUM(CASE WHEN dur >= 10.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND dur >= 10.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND dur >= 10.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 31, 'Prestudy insulin delivery, n (%)', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 32, ' Pump', 1, SUM(CASE WHEN ins_deliv = 'Pump' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND ins_deliv = 'Pump' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND ins_deliv = 'Pump' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 33, ' MDI', 1, SUM(CASE WHEN ins_deliv = 'MDI' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND ins_deliv = 'MDI' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND ins_deliv = 'MDI' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 34, 'Prestudy CGM use, n (%)', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 35, ' Yes', 1, SUM(CASE WHEN cgm_use = 'Yes' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND cgm_use = 'Yes' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND cgm_use = 'Yes' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 36, ' No', 1, SUM(CASE WHEN cgm_use = 'No' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND cgm_use = 'No' THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND cgm_use = 'No' THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 37, 'Baseline central lab HbA1c, %', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 38, ' Mean ± SD', 0, 0, 0, 0,
        printf('%.1f', ROUND(AVG(hba1c), 1)) || ' ± ' || printf('%.1f', ROUND(STDDEV_SAMP(hba1c), 1)),
        printf('%.1f', ROUND(AVG(CASE WHEN is_ciq = 1 THEN hba1c END), 1)) || ' ± ' || printf('%.1f', ROUND(STDDEV_SAMP(CASE WHEN is_ciq = 1 THEN hba1c END), 1)),
        printf('%.1f', ROUND(AVG(CASE WHEN is_ctrl = 1 THEN hba1c END), 1)) || ' ± ' || printf('%.1f', ROUND(STDDEV_SAMP(CASE WHEN is_ctrl = 1 THEN hba1c END), 1))
    FROM base
    UNION ALL
    SELECT 39, ' Range', 0, 0, 0, 0,
        printf('%.1f', ROUND(MIN(hba1c), 1)) || '–' || printf('%.1f', ROUND(MAX(hba1c), 1)),
        printf('%.1f', ROUND(MIN(CASE WHEN is_ciq = 1 THEN hba1c END), 1)) || '–' || printf('%.1f', ROUND(MAX(CASE WHEN is_ciq = 1 THEN hba1c END), 1)),
        printf('%.1f', ROUND(MIN(CASE WHEN is_ctrl = 1 THEN hba1c END), 1)) || '–' || printf('%.1f', ROUND(MAX(CASE WHEN is_ctrl = 1 THEN hba1c END), 1))
    FROM base
    UNION ALL
    SELECT 40, ' <7.0%, n (%)', 1, SUM(CASE WHEN hba1c < 7.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND hba1c < 7.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND hba1c < 7.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 41, ' 7.0%–7.4%, n (%)', 1, SUM(CASE WHEN hba1c >= 7.0 AND hba1c < 7.5 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND hba1c >= 7.0 AND hba1c < 7.5 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND hba1c >= 7.0 AND hba1c < 7.5 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 42, ' 7.5%–7.9%, n (%)', 1, SUM(CASE WHEN hba1c >= 7.5 AND hba1c < 8.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND hba1c >= 7.5 AND hba1c < 8.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND hba1c >= 7.5 AND hba1c < 8.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 43, ' 8.0%–8.4%, n (%)', 1, SUM(CASE WHEN hba1c >= 8.0 AND hba1c < 8.5 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND hba1c >= 8.0 AND hba1c < 8.5 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND hba1c >= 8.0 AND hba1c < 8.5 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 44, ' ≥8.5%, n (%)', 1, SUM(CASE WHEN hba1c >= 8.5 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND hba1c >= 8.5 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND hba1c >= 8.5 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 45, ' Missing, n (%)', 1, SUM(CASE WHEN hba1c IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND hba1c IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND hba1c IS NULL THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 46, 'Baseline time in range 70–180 mg/dL', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 47, ' Mean ± SD', 0, 0, 0, 0,
        CAST(ROUND(AVG(tir), 0) AS BIGINT) || '% ± ' || CAST(ROUND(STDDEV_SAMP(tir), 0) AS BIGINT) || '%',
        CAST(ROUND(AVG(CASE WHEN is_ciq = 1 THEN tir END), 0) AS BIGINT) || '% ± ' || CAST(ROUND(STDDEV_SAMP(CASE WHEN is_ciq = 1 THEN tir END), 0) AS BIGINT) || '%',
        CAST(ROUND(AVG(CASE WHEN is_ctrl = 1 THEN tir END), 0) AS BIGINT) || '% ± ' || CAST(ROUND(STDDEV_SAMP(CASE WHEN is_ctrl = 1 THEN tir END), 0) AS BIGINT) || '%'
    FROM base
    UNION ALL
    SELECT 48, ' Range', 0, 0, 0, 0,
        CAST(ROUND(MIN(tir), 0) AS BIGINT) || '%–' || CAST(ROUND(MAX(tir), 0) AS BIGINT) || '%',
        CAST(ROUND(MIN(CASE WHEN is_ciq = 1 THEN tir END), 0) AS BIGINT) || '%–' || CAST(ROUND(MAX(CASE WHEN is_ciq = 1 THEN tir END), 0) AS BIGINT) || '%',
        CAST(ROUND(MIN(CASE WHEN is_ctrl = 1 THEN tir END), 0) AS BIGINT) || '%–' || CAST(ROUND(MAX(CASE WHEN is_ctrl = 1 THEN tir END), 0) AS BIGINT) || '%'
    FROM base
    UNION ALL
    SELECT 49, ' ≥70%, n (%)', 1, SUM(CASE WHEN tir >= 70.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tir >= 70.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tir >= 70.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 50, ' 60% to <70%, n (%)', 1, SUM(CASE WHEN tir >= 60.0 AND tir < 70.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tir >= 60.0 AND tir < 70.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tir >= 60.0 AND tir < 70.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 51, ' 50% to <60%, n (%)', 1, SUM(CASE WHEN tir >= 50.0 AND tir < 60.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tir >= 50.0 AND tir < 60.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tir >= 50.0 AND tir < 60.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 52, ' 40% to <50%, n (%)', 1, SUM(CASE WHEN tir >= 40.0 AND tir < 50.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tir >= 40.0 AND tir < 50.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tir >= 40.0 AND tir < 50.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 53, ' <40%, n (%)', 1, SUM(CASE WHEN tir < 40.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tir < 40.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tir < 40.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 54, ' Missing, n (%)', 1, SUM(CASE WHEN tir IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tir IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tir IS NULL THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 55, 'Baseline time <70 mg/dL', 0, 0, 0, 0, '', '', ''
    UNION ALL
    SELECT 56, ' Mean ± SD', 0, 0, 0, 0,
        printf('%.1f', ROUND(AVG(tbr), 1)) || '% ± ' || printf('%.1f', ROUND(STDDEV_SAMP(tbr), 1)) || '%',
        printf('%.1f', ROUND(AVG(CASE WHEN is_ciq = 1 THEN tbr END), 1)) || '% ± ' || printf('%.1f', ROUND(STDDEV_SAMP(CASE WHEN is_ciq = 1 THEN tbr END), 1)) || '%',
        printf('%.1f', ROUND(AVG(CASE WHEN is_ctrl = 1 THEN tbr END), 1)) || '% ± ' || printf('%.1f', ROUND(STDDEV_SAMP(CASE WHEN is_ctrl = 1 THEN tbr END), 1)) || '%'
    FROM base
    UNION ALL
    SELECT 57, ' Range', 0, 0, 0, 0,
        printf('%.1f', ROUND(MIN(tbr), 1)) || '%–' || printf('%.1f', ROUND(MAX(tbr), 1)) || '%',
        printf('%.1f', ROUND(MIN(CASE WHEN is_ciq = 1 THEN tbr END), 1)) || '%–' || printf('%.1f', ROUND(MAX(CASE WHEN is_ciq = 1 THEN tbr END), 1)) || '%',
        printf('%.1f', ROUND(MIN(CASE WHEN is_ctrl = 1 THEN tbr END), 1)) || '%–' || printf('%.1f', ROUND(MAX(CASE WHEN is_ctrl = 1 THEN tbr END), 1)) || '%'
    FROM base
    UNION ALL
    SELECT 58, ' <1%, n (%)', 1, SUM(CASE WHEN tbr < 1.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tbr < 1.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tbr < 1.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 59, ' 1% to <4% n (%)', 1, SUM(CASE WHEN tbr >= 1.0 AND tbr < 4.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tbr >= 1.0 AND tbr < 4.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tbr >= 1.0 AND tbr < 4.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 60, ' ≥4%, n (%)', 1, SUM(CASE WHEN tbr >= 4.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tbr >= 4.0 THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tbr >= 4.0 THEN 1 ELSE 0 END), '', '', '' FROM base
    UNION ALL
    SELECT 61, ' Missing, n (%)', 1, SUM(CASE WHEN tbr IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ciq = 1 AND tbr IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN is_ctrl = 1 AND tbr IS NULL THEN 1 ELSE 0 END), '', '', '' FROM base
)
SELECT
    r.Characteristic AS "Characteristic",
    CASE
        WHEN r.is_cat = 1 THEN CAST(r.c_all AS VARCHAR) || ' (' || CASE WHEN r.c_all > 0 AND 100.0 * r.c_all / NULLIF(t.n_all, 0) < 1.0 THEN '<1' ELSE printf('%.0f', 100.0 * r.c_all / NULLIF(t.n_all, 0)) END || ')'
        ELSE r.s_all
    END AS "Overall",
    CASE
        WHEN r.is_cat = 1 THEN CAST(r.c_ciq AS VARCHAR) || ' (' || CASE WHEN r.c_ciq > 0 AND 100.0 * r.c_ciq / NULLIF(t.n_ciq, 0) < 1.0 THEN '<1' ELSE printf('%.0f', 100.0 * r.c_ciq / NULLIF(t.n_ciq, 0)) END || ')'
        ELSE r.s_ciq
    END AS "Control-IQ",
    CASE
        WHEN r.is_cat = 1 THEN CAST(r.c_ctrl AS VARCHAR) || ' (' || CASE WHEN r.c_ctrl > 0 AND 100.0 * r.c_ctrl / NULLIF(t.n_ctrl, 0) < 1.0 THEN '<1' ELSE printf('%.0f', 100.0 * r.c_ctrl / NULLIF(t.n_ctrl, 0)) END || ')'
        ELSE r.s_ctrl
    END AS "Control"
FROM raw_rows r
CROSS JOIN totals t
ORDER BY r.ord;`,
  },
];

let activeStarterCatalog: ReadonlyArray<StarterTemplateSpec> =
  STARTER_TEMPLATE_EXAMPLES;
let activeTable1Catalog: ReadonlyArray<Table1ExampleSpec> =
  TABLE_1_EXAMPLE_QUERIES;

/**
 * Returns the active starter clinical query templates (hydrated from `/api/templates` when available).
 * @returns {ReadonlyArray<StarterTemplateSpec>} Active starter template specifications.
 */
export function getStarterTemplateExamples(): ReadonlyArray<StarterTemplateSpec> {
  return activeStarterCatalog;
}

/**
 * Returns the active Academic Table 1 templates (hydrated from `/api/templates` when available).
 * @returns {ReadonlyArray<Table1ExampleSpec>} Active Table 1 template specifications.
 */
export function getTable1ExampleQueries(): ReadonlyArray<Table1ExampleSpec> {
  return activeTable1Catalog;
}

/**
 * Resets the in-memory clinical query catalog back to built-in offline defaults.
 * @returns {void}
 */
export function resetClinicalTemplateCatalog(): void {
  activeStarterCatalog = STARTER_TEMPLATE_EXAMPLES;
  activeTable1Catalog = TABLE_1_EXAMPLE_QUERIES;
}

/**
 * Appends a single validated wire template entry into `nextStarters` or `nextTable1`.
 * @param {NonNullable<TemplatesCatalogWirePayload["templates"]>[number]} item - Wire template item.
 * @param {Map<string, StarterTemplateSpec>} starterDefaultsById - Built-in starter metadata map.
 * @param {StarterTemplateSpec[]} nextStarters - Target starter catalog accumulator.
 * @param {Table1ExampleSpec[]} nextTable1 - Target Table 1 catalog accumulator.
 * @returns {void}
 */
function appendHydratedCatalogEntry(
  item: NonNullable<TemplatesCatalogWirePayload["templates"]>[number],
  starterDefaultsById: Map<string, StarterTemplateSpec>,
  nextStarters: StarterTemplateSpec[],
  nextTable1: Table1ExampleSpec[],
): void {
  const resolvedSql = item.goldSql ?? item.gold_sql ?? "";
  if (!item.id || !item.question || !resolvedSql) {
    return;
  }
  const resolvedKey = item.i18nKey ?? item.i18n_key ?? item.id;
  if (item.category === "starter") {
    const existing = starterDefaultsById.get(item.id);
    nextStarters.push({
      id: item.id,
      complexity: item.complexity,
      i18nKey: resolvedKey,
      ...(existing?.commentI18nKey
        ? { commentI18nKey: existing.commentI18nKey }
        : {}),
      ...(existing?.defaultComment
        ? { defaultComment: existing.defaultComment }
        : {}),
      question: item.question,
      goldSql: resolvedSql,
      defaultMode: item.defaultMode ?? item.default_mode ?? "sql",
    });
  } else {
    nextTable1.push({
      id: item.id,
      complexity: item.complexity,
      i18nKey: resolvedKey,
      question: item.question,
      sql: resolvedSql,
    });
  }
}

/**
 * Hydrates the in-memory starter and Table 1 query catalogs from a `/api/templates` payload.
 * @param {TemplatesCatalogWirePayload} payload - Parsed `/api/templates` JSON response.
 * @returns {boolean} True when at least one valid template item was applied.
 */
export function hydrateClinicalTemplateCatalog(
  payload: TemplatesCatalogWirePayload,
): boolean {
  if (!Array.isArray(payload.templates) || payload.templates.length === 0) {
    return false;
  }
  const starterDefaultsById = new Map(
    STARTER_TEMPLATE_EXAMPLES.map((s) => [s.id, s]),
  );
  const nextStarters: StarterTemplateSpec[] = [];
  const nextTable1: Table1ExampleSpec[] = [];

  for (const item of payload.templates) {
    appendHydratedCatalogEntry(
      item,
      starterDefaultsById,
      nextStarters,
      nextTable1,
    );
  }

  if (nextStarters.length === 0 && nextTable1.length === 0) {
    return false;
  }
  if (nextStarters.length > 0) {
    activeStarterCatalog = nextStarters;
  }
  if (nextTable1.length > 0) {
    activeTable1Catalog = nextTable1;
  }
  return true;
}

/**
 * Fetches the canonical clinical template catalog from `GET /api/templates` and updates in-memory caches.
 * Falls back cleanly to built-in defaults if the backend is offline.
 * @returns {Promise<boolean>} True if remote templates were fetched and applied, false on offline fallback.
 */
export async function fetchClinicalTemplateCatalog(): Promise<boolean> {
  try {
    const resp = await fetchWithBackendError("/api/templates");
    const data: TemplatesCatalogWirePayload = await resp.json();
    return hydrateClinicalTemplateCatalog(data);
  } catch {
    return false;
  }
}
