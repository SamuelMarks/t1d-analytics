"""Canonical catalog of clinical starter SQL templates and Academic Table 1 benchmarks."""

from typing import Literal

from typing_extensions import TypedDict


class Table1BenchmarkSpecDict(TypedDict):
    """Specification for an Academic Table 1 Text-to-SQL benchmark question and gold SQL."""

    id: str
    complexity: str
    question: str
    gold_sql: str


class ClinicalQueryTemplateDict(TypedDict):
    """Complete catalog item served by `GET /api/templates` for starter and Table 1 templates."""

    id: str
    category: Literal["starter", "table1"]
    complexity: str
    i18n_key: str
    question: str
    gold_sql: str
    default_mode: Literal["sql", "gemma:2b"]


STARTER_TEMPLATE_SUITE: tuple[Table1BenchmarkSpecDict, ...] = (
    {
        "id": "starter-1",
        "complexity": "Low",
        "question": (
            "Show patient count, average age, and total severe hypoglycemia events grouped by treatment group."
        ),
        "gold_sql": (
            "SELECT \n"
            "    TxGroup, \n"
            "    COUNT(*) as total_patients, \n"
            "    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,\n"
            "    SUM(NumSevHypo) as total_severe_hypo_events\n"
            "FROM tblaptsummary \n"
            "GROUP BY TxGroup;"
        ),
    },
    {
        "id": "starter-2",
        "complexity": "Low",
        "question": (
            "Show the frequency of each adverse event type ordered from highest to lowest occurrence count."
        ),
        "gold_sql": (
            "SELECT \n"
            "    event as adverse_event_type, \n"
            "    COUNT(*) as occurrence_count \n"
            "FROM adverseevents \n"
            "GROUP BY event \n"
            "ORDER BY occurrence_count DESC;"
        ),
    },
    {
        "id": "starter-3",
        "complexity": "Low",
        "question": (
            "Count the number of subjects using each insulin pump manufacturer."
        ),
        "gold_sql": (
            "SELECT \n"
            "    Pt_PumpManuf as pump_manufacturer, \n"
            "    COUNT(*) as user_count \n"
            "FROM subjects \n"
            "WHERE Pt_PumpManuf IS NOT NULL \n"
            "GROUP BY Pt_PumpManuf \n"
            "ORDER BY user_count DESC;"
        ),
    },
    {
        "id": "starter-4",
        "complexity": "Low-Medium",
        "question": (
            "Join subject summaries with HbA1c lab results and show average HbA1c and total tests by treatment group and gender."
        ),
        "gold_sql": (
            "SELECT \n"
            "    t.TxGroup, \n"
            "    t.Gender,\n"
            "    ROUND(AVG(h.HbA1c), 2) as average_hba1c,\n"
            "    COUNT(h.HbA1c) as total_tests_run\n"
            "FROM tblaptsummary t\n"
            "JOIN hba1c h ON t.PtID = h.PtID\n"
            "WHERE h.HbA1c IS NOT NULL\n"
            "GROUP BY t.TxGroup, t.Gender\n"
            "ORDER BY t.TxGroup, average_hba1c DESC;"
        ),
    },
    {
        "id": "starter-5",
        "complexity": "Low",
        "question": "Show me the first 5 patients in the demographics table.",
        "gold_sql": "SELECT * FROM demographics LIMIT 5;",
    },
)


BECK_2023_TABLE_1_SQL: str = """WITH base AS (
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
ORDER BY r.ord;"""


TABLE_1_BENCHMARK_SUITE: tuple[Table1BenchmarkSpecDict, ...] = (
    {
        "id": "table1-overall",
        "complexity": "Low",
        "question": "What are the average age, sex breakdown, HbA1c, and time in range for everyone?",
        "gold_sql": (
            "SELECT COUNT(*) AS n_participants, ROUND(AVG(age), 1) AS mean_age_yr, "
            "ROUND(STDDEV_SAMP(age), 1) AS sd_age_yr, "
            "SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_n, "
            "ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) AS female_pct, "
            "ROUND(AVG(hba1c), 2) AS mean_hba1c_pct, ROUND(AVG(tir), 1) AS mean_tir_pct, "
            "ROUND(AVG(tbr), 2) AS mean_tbr_pct, ROUND(AVG(tar), 1) AS mean_tar_pct, "
            "ROUND(AVG(mean_glucose), 1) AS mean_cgm_mgdl FROM patients;"
        ),
    },
    {
        "id": "table1-by-arm",
        "complexity": "Low-Medium",
        "question": "Compare the Closed-Loop and Control groups on age, sex, HbA1c, TIR, and DKA events.",
        "gold_sql": (
            "SELECT TxGroup AS treatment_arm, COUNT(*) AS n, "
            "ROUND(AVG(age), 1) || ' ± ' || ROUND(STDDEV_SAMP(age), 1) AS age_mean_sd, "
            "SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) || '%)' AS female_n_pct, "
            "ROUND(AVG(hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(hba1c), 2) AS hba1c_mean_sd, "
            "ROUND(AVG(tir), 1) || ' ± ' || ROUND(STDDEV_SAMP(tir), 1) AS tir_mean_sd, "
            "ROUND(AVG(tbr), 2) || ' ± ' || ROUND(STDDEV_SAMP(tbr), 2) AS tbr_mean_sd, "
            "ROUND(AVG(tar), 1) || ' ± ' || ROUND(STDDEV_SAMP(tar), 1) AS tar_mean_sd, "
            "SUM(numhospdka) AS total_dka_events FROM patients GROUP BY TxGroup ORDER BY TxGroup;"
        ),
    },
    {
        "id": "table1-by-age",
        "complexity": "Medium",
        "question": "Break down baseline stats by age group (under 18, 18 to 49, and 50 or older).",
        "gold_sql": (
            "SELECT CASE WHEN age < 18 THEN '1. Pediatric (<18 yr)' "
            "WHEN age < 50 THEN '2. Adult (18-49 yr)' ELSE '3. Older Adult (>=50 yr)' END AS age_stratum, "
            "COUNT(*) AS n, ROUND(AVG(age), 1) AS mean_age, "
            "ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN 1.0 ELSE 0.0 END), 1) AS closed_loop_pct, "
            "ROUND(AVG(hba1c), 2) AS mean_hba1c, ROUND(MEDIAN(hba1c), 2) AS median_hba1c, "
            "ROUND(AVG(tir), 1) AS mean_tir, "
            "ROUND(100.0 * AVG(CASE WHEN tir >= 70.0 AND hba1c < 7.0 THEN 1.0 ELSE 0.0 END), 1) AS ada_goal_achieved_pct "
            "FROM patients GROUP BY 1 ORDER BY 1;"
        ),
    },
    {
        "id": "table1-by-pump",
        "complexity": "Medium",
        "question": "Compare baseline patient stats across different insulin pump brands.",
        "gold_sql": (
            "SELECT s.Pt_PumpManuf AS pump_manufacturer, COUNT(*) AS n, "
            "ROUND(AVG(p.age), 1) || ' ± ' || ROUND(STDDEV_SAMP(p.age), 1) AS age_mean_sd, "
            "ROUND(100.0 * AVG(CASE WHEN p.gender = 'F' THEN 1.0 ELSE 0.0 END), 1) AS female_pct, "
            "ROUND(AVG(p.hba1c), 2) AS mean_hba1c, ROUND(AVG(p.tir), 1) AS mean_tir, "
            "ROUND(AVG(p.mean_glucose), 1) AS mean_glucose_mgdl, SUM(p.numhospdka) AS dka_events "
            "FROM subjects s JOIN patients p ON s.PtID = p.subject_id GROUP BY s.Pt_PumpManuf ORDER BY s.Pt_PumpManuf;"
        ),
    },
    {
        "id": "table1-longitudinal-lab",
        "complexity": "Medium-High",
        "question": "How did HbA1c and C-peptide change from baseline to the end of the study by arm?",
        "gold_sql": (
            "WITH lab_bounds AS (SELECT subject_id, FIRST(hba1c ORDER BY VisitDate ASC) AS baseline_hba1c, "
            "LAST(hba1c ORDER BY VisitDate ASC) AS endpoint_hba1c, "
            "FIRST(C_Peptide ORDER BY VisitDate ASC) AS baseline_c_peptide, "
            "LAST(C_Peptide ORDER BY VisitDate ASC) AS endpoint_c_peptide "
            "FROM hba1c_labs GROUP BY subject_id) "
            "SELECT p.TxGroup AS treatment_arm, COUNT(p.subject_id) AS randomized_n, "
            "COUNT(l.subject_id) AS lab_completers_n, ROUND(AVG(l.baseline_hba1c), 2) AS baseline_a1c, "
            "ROUND(AVG(l.endpoint_hba1c), 2) AS endpoint_a1c, "
            "ROUND(AVG(l.endpoint_hba1c - l.baseline_hba1c), 2) AS delta_a1c, "
            "ROUND(AVG(l.baseline_c_peptide), 3) AS baseline_c_pep, "
            "ROUND(AVG(l.endpoint_c_peptide), 3) AS endpoint_c_pep "
            "FROM patients p LEFT JOIN lab_bounds l ON p.subject_id = l.subject_id GROUP BY p.TxGroup ORDER BY p.TxGroup;"
        ),
    },
    {
        "id": "table1-multi-domain-rollup",
        "complexity": "High",
        "question": "Combine patient demographics, raw CGM sensor readings, and DKA events by treatment group.",
        "gold_sql": (
            "WITH cgm_agg AS (SELECT subject_id, AVG(glucose_value) AS sensor_mean_mgdl, "
            "STDDEV_SAMP(glucose_value) AS sensor_sd_mgdl, "
            "100.0 * STDDEV_SAMP(glucose_value) / NULLIF(AVG(glucose_value), 0) AS sensor_cv_pct "
            "FROM cgm_readings GROUP BY subject_id), "
            "ae_agg AS (SELECT subject_id, SUM(CASE WHEN EventType = 'DKA' THEN 1 ELSE 0 END) AS dka_ae, "
            "SUM(CASE WHEN EventType = 'Severe Hypoglycemia' THEN 1 ELSE 0 END) AS sev_hypo_ae, "
            "SUM(CASE WHEN Hospitalized THEN 1 ELSE 0 END) AS hosp_ae FROM dka_adverse_events GROUP BY subject_id) "
            "SELECT COALESCE(p.TxGroup, 'Overall Cohort') AS cohort_column, COUNT(*) AS n_subjects, "
            "ROUND(AVG(p.age), 1) || ' ± ' || ROUND(STDDEV_SAMP(p.age), 1) AS age_yr_mean_sd, "
            "ROUND(AVG(p.hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(p.hba1c), 2) AS hba1c_pct_mean_sd, "
            "ROUND(AVG(c.sensor_mean_mgdl), 1) || ' ± ' || ROUND(AVG(c.sensor_sd_mgdl), 1) AS cgm_mgdl_mean_sd, "
            "ROUND(AVG(c.sensor_cv_pct), 1) AS cgm_cv_pct, ROUND(AVG(p.tir), 1) AS tir_70_180_pct, "
            "COALESCE(SUM(a.dka_ae), 0) AS dka_events, COALESCE(SUM(a.sev_hypo_ae), 0) AS sev_hypo_events, "
            "COALESCE(SUM(a.hosp_ae), 0) AS hospitalizations "
            "FROM patients p LEFT JOIN cgm_agg c ON p.subject_id = c.subject_id "
            "LEFT JOIN ae_agg a ON p.subject_id = a.subject_id "
            "GROUP BY ROLLUP(p.TxGroup) ORDER BY CASE WHEN p.TxGroup IS NULL THEN 1 ELSE 0 END, p.TxGroup;"
        ),
    },
    {
        "id": "table1-transposed-nejm",
        "complexity": "Very High",
        "question": "Create a publication Table 1 comparing Closed-Loop and Control with group differences.",
        "gold_sql": (
            "WITH base AS (SELECT p.*, s.Pt_PumpManuf FROM patients p LEFT JOIN subjects s ON p.subject_id = s.PtID) "
            "SELECT 1 AS sort_ord, 'Sample Size (N)' AS characteristic, "
            "CAST(SUM(CASE WHEN TxGroup = 'Closed-Loop' THEN 1 ELSE 0 END) AS VARCHAR) AS closed_loop_arm, "
            "CAST(SUM(CASE WHEN TxGroup = 'Control' THEN 1 ELSE 0 END) AS VARCHAR) AS control_arm, "
            "CAST(COUNT(*) AS VARCHAR) AS overall_cohort, '—' AS arm_difference FROM base "
            "UNION ALL SELECT 2, 'Age (years), Mean ± SD', "
            "ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN age END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN age END), 1), "
            "ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN age END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN age END), 1), "
            "ROUND(AVG(age), 1) || ' ± ' || ROUND(STDDEV_SAMP(age), 1), "
            "CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN age END) - AVG(CASE WHEN TxGroup = 'Control' THEN age END), 2) AS VARCHAR) FROM base "
            "UNION ALL SELECT 3, 'Female Sex, n (%)', "
            "SUM(CASE WHEN TxGroup = 'Closed-Loop' AND gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END END), 1) || '%)', "
            "SUM(CASE WHEN TxGroup = 'Control' AND gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Control' THEN CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END END), 1) || '%)', "
            "SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) || '%)', '—' FROM base "
            "UNION ALL SELECT 4, 'Baseline HbA1c (%), Mean ± SD', "
            "ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END), 2) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END), 2), "
            "ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2), "
            "ROUND(AVG(hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(hba1c), 2), "
            "CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END) - AVG(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2) AS VARCHAR) FROM base "
            "UNION ALL SELECT 5, 'Time in Range 70-180 mg/dL (%), Mean ± SD', "
            "ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END), 1), "
            "ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN tir END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN tir END), 1), "
            "ROUND(AVG(tir), 1) || ' ± ' || ROUND(STDDEV_SAMP(tir), 1), "
            "CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END) - AVG(CASE WHEN TxGroup = 'Control' THEN tir END), 2) AS VARCHAR) FROM base "
            "UNION ALL SELECT 6, 'Time Below Range <70 mg/dL (%), Median (IQR)', "
            "ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.75), 2) || ')', "
            "ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.75), 2) || ')', "
            "ROUND(QUANTILE_CONT(tbr, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(tbr, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(tbr, 0.75), 2) || ')', '—' FROM base ORDER BY sort_ord;"
        ),
    },
    {
        "id": "table1-beck-2023",
        "complexity": "Very High",
        "question": "Reproduce Table 1 from the Beck 2023 Control-IQ meta-analysis.",
        "gold_sql": BECK_2023_TABLE_1_SQL,
    },
)


ALL_GOLD_TEMPLATE_SUITE: tuple[Table1BenchmarkSpecDict, ...] = (
    STARTER_TEMPLATE_SUITE + TABLE_1_BENCHMARK_SUITE
)

_TEMPLATE_I18N_KEYS: dict[str, str] = {
    "starter-1": "ui.exampleQueries.patientDemographics",
    "starter-2": "ui.exampleQueries.adverseEvents",
    "starter-3": "ui.exampleQueries.pumpManufacturers",
    "starter-4": "ui.exampleQueries.hba1cDemographics",
    "starter-5": "ui.exampleQueries.nlpFirst5",
    "table1-overall": "ui.exampleQueries.table1Overall",
    "table1-by-arm": "ui.exampleQueries.table1ByArm",
    "table1-by-age": "ui.exampleQueries.table1ByAge",
    "table1-by-pump": "ui.exampleQueries.table1ByPump",
    "table1-longitudinal-lab": "ui.exampleQueries.table1LongitudinalLab",
    "table1-multi-domain-rollup": "ui.exampleQueries.table1MultiDomainRollup",
    "table1-transposed-nejm": "ui.exampleQueries.table1TransposedNejm",
    "table1-beck-2023": "ui.exampleQueries.table1Beck2023",
}


def get_table_1_benchmark_suite() -> list[dict[str, str]]:
    """
    Return the Academic Table 1 Text-to-SQL benchmark dataset for `evaluate_text_to_sql`.

    Returns:
        list[dict[str, str]]: List of benchmark dicts with `id`, `complexity`, `question`, and `gold_sql`.

    """
    return [
        {
            "id": item["id"],
            "complexity": item["complexity"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
        }
        for item in TABLE_1_BENCHMARK_SUITE
    ]


def get_all_gold_templates() -> list[dict[str, str]]:
    """
    Return all Gold Templates (Starter Clinical Templates + Academic Table 1 Templates).

    Returns:
        list[dict[str, str]]: List of template dicts with `id`, `complexity`, `question`, and `gold_sql`.

    """
    return [
        {
            "id": item["id"],
            "complexity": item["complexity"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
        }
        for item in ALL_GOLD_TEMPLATE_SUITE
    ]


def get_clinical_query_catalog() -> list[ClinicalQueryTemplateDict]:
    """
    Build the complete clinical query catalog served by `GET /api/templates`.

    Returns:
        list[ClinicalQueryTemplateDict]: Ordered starter and Academic Table 1 template specifications.

    """
    catalog: list[ClinicalQueryTemplateDict] = []
    for starter in STARTER_TEMPLATE_SUITE:
        tid = starter["id"]
        catalog.append(
            {
                "id": tid,
                "category": "starter",
                "complexity": starter["complexity"],
                "i18n_key": _TEMPLATE_I18N_KEYS[tid],
                "question": starter["question"],
                "gold_sql": starter["gold_sql"],
                "default_mode": "gemma:2b" if tid == "starter-5" else "sql",
            }
        )
    for t1 in TABLE_1_BENCHMARK_SUITE:
        tid = t1["id"]
        catalog.append(
            {
                "id": tid,
                "category": "table1",
                "complexity": t1["complexity"],
                "i18n_key": _TEMPLATE_I18N_KEYS[tid],
                "question": t1["question"],
                "gold_sql": t1["gold_sql"],
                "default_mode": "sql",
            }
        )
    return catalog


__all__ = [
    "ALL_GOLD_TEMPLATE_SUITE",
    "BECK_2023_TABLE_1_SQL",
    "STARTER_TEMPLATE_SUITE",
    "TABLE_1_BENCHMARK_SUITE",
    "ClinicalQueryTemplateDict",
    "Table1BenchmarkSpecDict",
    "get_all_gold_templates",
    "get_clinical_query_catalog",
    "get_table_1_benchmark_suite",
]
