"""Shared Jaeb Beck et al. (2023) pooled cohort SQL CTE builder and authentic Jaeb canonical views."""

import duckdb

from t1d_analytics.ingestion import coerce_known_clinical_numeric_columns

BECK2023_REQUIRED_TRIAL_TABLES: frozenset[str] = frozenset(
    {
        "dclp3_ptroster",
        "dclp3_diabscreening",
        "dclp3_diabsocioecon",
        "dclp3_sampleresults",
        "dclp3_cgm",
        "dclp5_ptroster",
        "dclp5_diabscreening",
        "dclp5_diabsocioecon",
        "dclp5_sampleresults",
        "dclp5_dexcomclaritycgm",
        "dclp5_othercgm",
        "dclp5_tandemcgmdatagxb_b",
        "pedap_ptroster",
        "pedap_diabscreening",
        "pedap_diabsocioecon",
        "pedap_stasampleresults",
        "pedap_dexcomclaritycgm",
        "pedap_othercgm",
        "pedap_tandemcgmdatagxb",
    }
)


def build_raw_jaeb_cohort_cte_sql(include_extended_studies: bool = False) -> str:
    """
    Build the self-contained DuckDB CTE block deriving `raw_jaeb_cohort` directly from raw JAEB study tables.

    Computes 14-day pre-randomization CGM metrics (`tir`, `tbr`, `tbr_54`, `tar`, `tar_250`,
    `mean_glucose`, `cv`), central lab baseline HbA1c, demographics, socioeconomic status,
    diabetes duration, prestudy insulin delivery, and prestudy CGM use directly from
    `dclp3_*`, `dclp5_*`, and `pedap_*` raw tables, and optionally unions `JDRF_CGM`
    (`tblaptsummary`) and `REPLACE_BG` (`hptroster`) when `include_extended_studies` is True.

    Args:
        include_extended_studies: Whether to union additional JAEB RCT tables (`tblaptsummary`, `hptroster`).

    Returns:
        str: SQL CTE definitions (`_dclp5_cgm_all AS (...), ..., raw_jaeb_cohort AS (...)`).

    """
    ext_union = ""
    if include_extended_studies:
        ext_union = """
    UNION ALL
    SELECT
        40000 + TRY_CAST(t.ptid AS BIGINT) AS subject_id,
        'JDRF_CGM' AS study,
        TRY_CAST(t.ptid AS BIGINT) AS PtID,
        CASE WHEN CAST(t.TxGroup AS VARCHAR) = 'RT-CGM' THEN 'Control-IQ' ELSE 'Control' END AS TxGroup,
        TRY_CAST(t.AgeAsOfRandDt AS DOUBLE) + 0.5 AS age_exact,
        TRY_CAST(t.AgeAsOfRandDt AS BIGINT) AS age,
        CASE WHEN UPPER(TRIM(CAST(t.gender AS VARCHAR))) IN ('F', 'FEMALE') THEN 'Female' ELSE 'Male' END AS gender,
        CASE
            WHEN CAST(t.ethnic AS VARCHAR) = 'Hispanic or Latino' THEN 'Hispanic or Latino'
            WHEN CAST(t.race AS VARCHAR) = 'White' THEN 'White non-Hispanic'
            WHEN CAST(t.race AS VARCHAR) LIKE '%Black%' THEN 'Black non-Hispanic'
            WHEN CAST(t.race AS VARCHAR) = 'Asian' THEN 'Asian'
            ELSE 'More than one race'
        END AS race_ethnicity,
        'Missing' AS annual_income,
        TRY_CAST(t.DavDiabetes AS DOUBLE) AS t1d_duration_years,
        CASE WHEN CAST(t.InsulinModality AS VARCHAR) = 'Pump' THEN 'Pump' ELSE 'MDI' END AS prestudy_insulin_delivery,
        'No' AS prestudy_cgm_use,
        (SELECT AVG(TRY_CAST(l.LabA1cResult AS DOUBLE)) FROM tblalabhba1c l WHERE l.ptid = t.ptid AND l.visit = 'Randomization') AS hba1c,
        NULL AS tir,
        NULL AS tbr,
        NULL AS tbr_54,
        NULL AS tar,
        NULL AS tar_250,
        NULL AS mean_glucose,
        NULL AS cv,
        0 AS dka_events,
        COALESCE(TRY_CAST(t.NumSevHypo AS BIGINT), 0) AS severe_hypo_events
    FROM tblaptsummary t
    UNION ALL
    SELECT
        50000 + TRY_CAST(r.ptid AS BIGINT) AS subject_id,
        'REPLACE_BG' AS study,
        TRY_CAST(r.ptid AS BIGINT) AS PtID,
        CASE WHEN CAST(r.trtgroup AS VARCHAR) = 'CGM Only' THEN 'Control-IQ' ELSE 'Control' END AS TxGroup,
        TRY_CAST(r.AgeAsOfEnrollDt AS DOUBLE) + 0.5 AS age_exact,
        TRY_CAST(r.AgeAsOfEnrollDt AS BIGINT) AS age,
        CASE WHEN UPPER(TRIM(CAST(s.gender AS VARCHAR))) IN ('F', 'FEMALE') THEN 'Female' ELSE 'Male' END AS gender,
        CASE
            WHEN CAST(s.ethnicity AS VARCHAR) = 'Hispanic or Latino' THEN 'Hispanic or Latino'
            WHEN CAST(s.race AS VARCHAR) = 'White' THEN 'White non-Hispanic'
            WHEN CAST(s.race AS VARCHAR) LIKE '%Black%' THEN 'Black non-Hispanic'
            WHEN CAST(s.race AS VARCHAR) = 'Asian' THEN 'Asian'
            ELSE 'More than one race'
        END AS race_ethnicity,
        'Missing' AS annual_income,
        NULL AS t1d_duration_years,
        'Pump' AS prestudy_insulin_delivery,
        'Yes' AS prestudy_cgm_use,
        (SELECT AVG(TRY_CAST(h.HbA1cTestRes AS DOUBLE)) FROM hlocalhba1c h WHERE h.ptid = r.ptid AND h.visit = 'Randomization') AS hba1c,
        NULL AS tir,
        NULL AS tbr,
        NULL AS tbr_54,
        NULL AS tar,
        NULL AS tar_250,
        NULL AS mean_glucose,
        NULL AS cv,
        0 AS dka_events,
        0 AS severe_hypo_events
    FROM hptroster r
    LEFT JOIN hscreening s ON r.ptid = s.ptid"""
    return f"""_dclp5_cgm_all AS (
    SELECT PtID, COALESCE(TRY_CAST(DataDtTm_adj AS TIMESTAMP), TRY_CAST(DataDtTm AS TIMESTAMP)) AS ts, TRY_CAST(CGM AS DOUBLE) AS glu
    FROM dclp5_dexcomclaritycgm WHERE TRY_CAST(CGM AS DOUBLE) > 0
    UNION ALL
    SELECT PtID, TRY_CAST(DataDtTm AS TIMESTAMP) AS ts, TRY_CAST(CGM AS DOUBLE) AS glu
    FROM dclp5_othercgm WHERE NOT COALESCE(TRY_CAST(IsCalibration AS BOOLEAN), FALSE) AND TRY_CAST(CGM AS DOUBLE) > 0
    UNION ALL
    SELECT PtID, COALESCE(TRY_CAST(DataDtTm_adjusted AS TIMESTAMP), TRY_CAST(DataDtTm AS TIMESTAMP)) AS ts,
           CASE WHEN TRY_CAST(HighLowIndicator AS INT) = 1 THEN 401.0 WHEN TRY_CAST(HighLowIndicator AS INT) = 2 THEN 39.0 ELSE TRY_CAST(CGMValue AS DOUBLE) END AS glu
    FROM dclp5_tandemcgmdatagxb_b
    WHERE TRY_CAST(HighLowIndicator AS INT) IN (1, 2) OR TRY_CAST(CGMValue AS DOUBLE) > 0
),
_pedap_cgm_all AS (
    SELECT PtID, TRY_CAST(DeviceDtTm AS TIMESTAMP) AS ts, TRY_CAST(CGM AS DOUBLE) AS glu
    FROM pedap_dexcomclaritycgm WHERE TRY_CAST(CGM AS DOUBLE) > 0
    UNION ALL
    SELECT PtID, TRY_CAST(DeviceDtTm AS TIMESTAMP) AS ts, TRY_CAST(CGM AS DOUBLE) AS glu
    FROM pedap_othercgm WHERE NOT COALESCE(TRY_CAST(IsCalibration AS BOOLEAN), FALSE) AND TRY_CAST(CGM AS DOUBLE) > 0
    UNION ALL
    SELECT PtID, TRY_CAST(DeviceDtTm AS TIMESTAMP) AS ts,
           CASE WHEN TRY_CAST(HighLowIndicator AS INT) = 1 THEN 401.0 WHEN TRY_CAST(HighLowIndicator AS INT) = 2 THEN 39.0 ELSE TRY_CAST(CGMValue AS DOUBLE) END AS glu
    FROM pedap_tandemcgmdatagxb
    WHERE TRY_CAST(HighLowIndicator AS INT) IN (1, 2) OR TRY_CAST(CGMValue AS DOUBLE) > 0
),
d3_cgm AS (
    SELECT PtID,
           COUNT(CGM) * 5.0 / 60.0 AS cgm_hours,
           100.0 * AVG(CASE WHEN TRY_CAST(CGM AS DOUBLE) BETWEEN 70 AND 180 THEN 1.0 ELSE 0.0 END) AS tir,
           100.0 * AVG(CASE WHEN TRY_CAST(CGM AS DOUBLE) < 70 THEN 1.0 ELSE 0.0 END) AS tbr,
           100.0 * AVG(CASE WHEN TRY_CAST(CGM AS DOUBLE) < 54 THEN 1.0 ELSE 0.0 END) AS tbr_54,
           100.0 * AVG(CASE WHEN TRY_CAST(CGM AS DOUBLE) > 180 THEN 1.0 ELSE 0.0 END) AS tar,
           100.0 * AVG(CASE WHEN TRY_CAST(CGM AS DOUBLE) > 250 THEN 1.0 ELSE 0.0 END) AS tar_250,
           AVG(TRY_CAST(CGM AS DOUBLE)) AS mean_glucose,
           100.0 * STDDEV_SAMP(TRY_CAST(CGM AS DOUBLE)) / NULLIF(AVG(TRY_CAST(CGM AS DOUBLE)), 0) AS cv
    FROM dclp3_cgm
    WHERE Period = '1. Baseline' AND TRY_CAST(CGM AS DOUBLE) > 0
    GROUP BY PtID
),
d5_pre AS (
    SELECT r.PtID, c.ts, AVG(c.glu) AS glu,
           ROW_NUMBER() OVER (PARTITION BY r.PtID ORDER BY c.ts DESC) AS rn
    FROM dclp5_ptroster r
    JOIN _dclp5_cgm_all c ON r.PtID = c.PtID
    WHERE r.RandDt IS NOT NULL AND CAST(c.ts AS DATE) <= CAST(r.RandDt AS DATE)
    GROUP BY r.PtID, c.ts
),
d5_cgm AS (
    SELECT PtID,
           COUNT(glu) * 5.0 / 60.0 AS cgm_hours,
           100.0 * AVG(CASE WHEN glu BETWEEN 70 AND 180 THEN 1.0 ELSE 0.0 END) AS tir,
           100.0 * AVG(CASE WHEN glu < 70 THEN 1.0 ELSE 0.0 END) AS tbr,
           100.0 * AVG(CASE WHEN glu < 54 THEN 1.0 ELSE 0.0 END) AS tbr_54,
           100.0 * AVG(CASE WHEN glu > 180 THEN 1.0 ELSE 0.0 END) AS tar,
           100.0 * AVG(CASE WHEN glu > 250 THEN 1.0 ELSE 0.0 END) AS tar_250,
           AVG(glu) AS mean_glucose,
           100.0 * STDDEV_SAMP(glu) / NULLIF(AVG(glu), 0) AS cv
    FROM d5_pre
    WHERE rn <= 4032
    GROUP BY PtID
),
p_pre AS (
    SELECT r.PtID, c.ts, AVG(c.glu) AS glu,
           ROW_NUMBER() OVER (PARTITION BY r.PtID ORDER BY c.ts DESC) AS rn
    FROM pedap_ptroster r
    JOIN _pedap_cgm_all c ON r.PtID = c.PtID
    WHERE r.RandDt IS NOT NULL AND TRY_CAST(r.PtID AS BIGINT) NOT IN (22, 71) AND c.ts <= TRY_CAST(r.RandDt AS TIMESTAMP)
    GROUP BY r.PtID, c.ts
),
p_cgm AS (
    SELECT PtID,
           COUNT(glu) * 5.0 / 60.0 AS cgm_hours,
           100.0 * AVG(CASE WHEN glu BETWEEN 70 AND 180 THEN 1.0 ELSE 0.0 END) AS tir,
           100.0 * AVG(CASE WHEN glu < 70 THEN 1.0 ELSE 0.0 END) AS tbr,
           100.0 * AVG(CASE WHEN glu < 54 THEN 1.0 ELSE 0.0 END) AS tbr_54,
           100.0 * AVG(CASE WHEN glu > 180 THEN 1.0 ELSE 0.0 END) AS tar,
           100.0 * AVG(CASE WHEN glu > 250 THEN 1.0 ELSE 0.0 END) AS tar_250,
           AVG(glu) AS mean_glucose,
           100.0 * STDDEV_SAMP(glu) / NULLIF(AVG(glu), 0) AS cv
    FROM p_pre
    WHERE rn <= 4032
    GROUP BY PtID
),
d3_a1c AS (
    SELECT PtID, AVG(TRY_CAST(Value AS DOUBLE)) AS hba1c
    FROM dclp3_sampleresults
    WHERE Visit = 'Randomization' AND ResultName = 'GLYHB'
    GROUP BY PtID
),
d5_a1c AS (
    SELECT PtID, AVG(TRY_CAST(Value AS DOUBLE)) AS hba1c
    FROM dclp5_sampleresults
    WHERE Visit = 'Baseline' AND ResultName = 'GLYHB'
    GROUP BY PtID
),
p_a1c AS (
    SELECT r.PtID, MIN(TRY_CAST(h.Value AS DOUBLE)) AS hba1c
    FROM pedap_ptroster r
    JOIN pedap_stasampleresults h ON r.PtID = h.PtID
    WHERE h.Visit = 'Randomization' AND h.ResultName = 'GLYHB'
      AND COALESCE(CAST(h.STAResultStatus AS VARCHAR), 'Complete') = 'Complete'
      AND TRY_CAST(r.PtID AS BIGINT) NOT IN (15, 19)
    GROUP BY r.PtID
),
raw_jaeb_cohort AS (
    SELECT
        10000 + TRY_CAST(r.PtID AS BIGINT) AS subject_id,
        'DCLP3' AS study,
        TRY_CAST(r.PtID AS BIGINT) AS PtID,
        CASE WHEN CAST(r.trtGroup AS VARCHAR) = 'CLC' THEN 'Control-IQ' ELSE 'Control' END AS TxGroup,
        TRY_CAST(s.AgeAtEnrollment AS DOUBLE) + 0.5 AS age_exact,
        TRY_CAST(s.AgeAtEnrollment AS BIGINT) AS age,
        CASE WHEN UPPER(TRIM(CAST(s.Gender AS VARCHAR))) IN ('F', 'FEMALE') THEN 'Female' ELSE 'Male' END AS gender,
        CASE
            WHEN CAST(s.Ethnicity AS VARCHAR) = 'Hispanic or Latino' THEN 'Hispanic or Latino'
            WHEN CAST(s.Race AS VARCHAR) = 'White' THEN 'White non-Hispanic'
            WHEN CAST(s.Race AS VARCHAR) = 'Black/African American' THEN 'Black non-Hispanic'
            WHEN CAST(s.Race AS VARCHAR) = 'Asian' THEN 'Asian'
            WHEN CAST(s.Race AS VARCHAR) = 'Native Hawaiian/Other Pacific Islander' THEN 'Native Hawaiian/other Pacific Islander'
            ELSE 'More than one race'
        END AS race_ethnicity,
        CASE
            WHEN CAST(e.AnnualIncome AS VARCHAR) IN ('Less than $25,000', '$25,000 to less than $35,000', '$35,000 to less than $50,000', '$50,000 to less than $75,000') THEN '< $75,000'
            WHEN CAST(e.AnnualIncome AS VARCHAR) IN ('$75,000 to less than $100,000', '$100,000 to less than $200,000', '$200,000 or more') THEN '>= $75,000'
            ELSE 'Missing'
        END AS annual_income,
        COALESCE(datediff('day', TRY_CAST(s.DiagDt AS DATE), TRY_CAST(r.EnrollDt AS DATE)) / 365.25, TRY_CAST(s.AgeAtEnrollment AS DOUBLE) - TRY_CAST(s.DiagAge AS DOUBLE)) AS t1d_duration_years,
        CASE WHEN TRY_CAST(s.InsModPump AS BIGINT) = 1 THEN 'Pump' ELSE 'MDI' END AS prestudy_insulin_delivery,
        CASE WHEN CAST(s.CGMUseStat AS VARCHAR) = 'Current' THEN 'Yes' ELSE 'No' END AS prestudy_cgm_use,
        a.hba1c AS hba1c,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tir ELSE NULL END AS tir,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tbr ELSE NULL END AS tbr,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tbr_54 ELSE NULL END AS tbr_54,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tar ELSE NULL END AS tar,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tar_250 ELSE NULL END AS tar_250,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.mean_glucose ELSE NULL END AS mean_glucose,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.cv ELSE NULL END AS cv,
        0 AS dka_events,
        0 AS severe_hypo_events
    FROM dclp3_ptroster r
    JOIN dclp3_diabscreening s ON r.PtID = s.PtID
    LEFT JOIN dclp3_diabsocioecon e ON r.PtID = e.PtID
    LEFT JOIN d3_a1c a ON r.PtID = a.PtID
    LEFT JOIN d3_cgm c ON r.PtID = c.PtID
    WHERE r.RandDt IS NOT NULL
    UNION ALL
    SELECT
        20000 + TRY_CAST(r.PtID AS BIGINT) AS subject_id,
        'DCLP5' AS study,
        TRY_CAST(r.PtID AS BIGINT) AS PtID,
        CASE WHEN CAST(r.trtGroup AS VARCHAR) = 'CLC' THEN 'Control-IQ' ELSE 'Control' END AS TxGroup,
        TRY_CAST(r.AgeAtEnrollment AS DOUBLE) + 0.5 AS age_exact,
        TRY_CAST(r.AgeAtEnrollment AS BIGINT) AS age,
        CASE WHEN UPPER(TRIM(CAST(s.Gender AS VARCHAR))) IN ('F', 'FEMALE') THEN 'Female' ELSE 'Male' END AS gender,
        CASE
            WHEN CAST(s.Ethnicity AS VARCHAR) = 'Hispanic or Latino' THEN 'Hispanic or Latino'
            WHEN CAST(s.Race AS VARCHAR) = 'White' THEN 'White non-Hispanic'
            WHEN CAST(s.Race AS VARCHAR) = 'Black/African American' THEN 'Black non-Hispanic'
            WHEN CAST(s.Race AS VARCHAR) = 'Asian' THEN 'Asian'
            WHEN CAST(s.Race AS VARCHAR) = 'Native Hawaiian/Other Pacific Islander' THEN 'Native Hawaiian/other Pacific Islander'
            ELSE 'More than one race'
        END AS race_ethnicity,
        CASE
            WHEN CAST(e.AnnualIncome AS VARCHAR) IN ('Less than $25,000', '$25,000 to less than $35,000', '$35,000 to less than $50,000', '$50,000 to less than $75,000') THEN '< $75,000'
            WHEN CAST(e.AnnualIncome AS VARCHAR) IN ('$75,000 to less than $100,000', '$100,000 to less than $200,000', '$200,000 or more') THEN '>= $75,000'
            ELSE 'Missing'
        END AS annual_income,
        COALESCE(datediff('day', TRY_CAST(s.DiagDt AS DATE), TRY_CAST(r.EnrollDt AS DATE)) / 365.25, TRY_CAST(r.AgeAtEnrollment AS DOUBLE) - TRY_CAST(s.DiagAge AS DOUBLE)) AS t1d_duration_years,
        CASE WHEN TRY_CAST(s.InsModPump AS BIGINT) = 1 THEN 'Pump' ELSE 'MDI' END AS prestudy_insulin_delivery,
        CASE WHEN CAST(s.CGMUseStat AS VARCHAR) = 'Current' THEN 'Yes' ELSE 'No' END AS prestudy_cgm_use,
        a.hba1c AS hba1c,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tir ELSE NULL END AS tir,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tbr ELSE NULL END AS tbr,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tbr_54 ELSE NULL END AS tbr_54,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tar ELSE NULL END AS tar,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tar_250 ELSE NULL END AS tar_250,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.mean_glucose ELSE NULL END AS mean_glucose,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.cv ELSE NULL END AS cv,
        0 AS dka_events,
        0 AS severe_hypo_events
    FROM dclp5_ptroster r
    JOIN dclp5_diabscreening s ON r.PtID = s.PtID
    LEFT JOIN dclp5_diabsocioecon e ON r.PtID = e.PtID
    LEFT JOIN d5_a1c a ON r.PtID = a.PtID
    LEFT JOIN d5_cgm c ON r.PtID = c.PtID
    WHERE r.RandDt IS NOT NULL
    UNION ALL
    SELECT
        30000 + TRY_CAST(r.PtID AS BIGINT) AS subject_id,
        'PEDAP' AS study,
        TRY_CAST(r.PtID AS BIGINT) AS PtID,
        CASE WHEN CAST(r.TrtGroup AS VARCHAR) = 'CLC' THEN 'Control-IQ' ELSE 'Control' END AS TxGroup,
        TRY_CAST(r.AgeAsofEnrollDt AS DOUBLE) + 0.5 AS age_exact,
        TRY_CAST(r.AgeAsofEnrollDt AS BIGINT) AS age,
        CASE WHEN UPPER(TRIM(CAST(s.Sex AS VARCHAR))) IN ('F', 'FEMALE') THEN 'Female' ELSE 'Male' END AS gender,
        CASE
            WHEN CAST(s.Ethnicity AS VARCHAR) = 'Hispanic or Latino' THEN 'Hispanic or Latino'
            WHEN CAST(s.Race AS VARCHAR) = 'White' THEN 'White non-Hispanic'
            WHEN CAST(s.Race AS VARCHAR) = 'Black/African American' THEN 'Black non-Hispanic'
            WHEN CAST(s.Race AS VARCHAR) = 'Asian' THEN 'Asian'
            WHEN CAST(s.Race AS VARCHAR) = 'Native Hawaiian/Other Pacific Islander' THEN 'Native Hawaiian/other Pacific Islander'
            ELSE 'More than one race'
        END AS race_ethnicity,
        CASE
            WHEN CAST(e.AnnualIncome AS VARCHAR) IN ('Less than $25,000', '$25,000 to less than $35,000', '$35,000 to less than $50,000', '$50,000 to less than $75,000') THEN '< $75,000'
            WHEN CAST(e.AnnualIncome AS VARCHAR) IN ('$75,000 to less than $100,000', '$100,000 to less than $200,000', '$200,000 or more') THEN '>= $75,000'
            ELSE 'Missing'
        END AS annual_income,
        COALESCE(datediff('day', TRY_CAST(s.DiagDt AS DATE), TRY_CAST(r.EnrollDt AS DATE)) / 365.25, TRY_CAST(r.AgeAsofEnrollDt AS DOUBLE) - TRY_CAST(s.DiagAge AS DOUBLE)) AS t1d_duration_years,
        CASE WHEN TRY_CAST(s.InsModPump AS BIGINT) = 1 THEN 'Pump' ELSE 'MDI' END AS prestudy_insulin_delivery,
        CASE WHEN CAST(s.CGMUseStat AS VARCHAR) = 'Current' THEN 'Yes' ELSE 'No' END AS prestudy_cgm_use,
        a.hba1c AS hba1c,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tir ELSE NULL END AS tir,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tbr ELSE NULL END AS tbr,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tbr_54 ELSE NULL END AS tbr_54,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tar ELSE NULL END AS tar,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.tar_250 ELSE NULL END AS tar_250,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.mean_glucose ELSE NULL END AS mean_glucose,
        CASE WHEN c.cgm_hours >= 168.0 THEN c.cv ELSE NULL END AS cv,
        0 AS dka_events,
        0 AS severe_hypo_events
    FROM pedap_ptroster r
    JOIN pedap_diabscreening s ON r.PtID = s.PtID
    LEFT JOIN pedap_diabsocioecon e ON r.PtID = e.PtID
    LEFT JOIN p_a1c a ON r.PtID = a.PtID
    LEFT JOIN p_cgm c ON r.PtID = c.PtID
    WHERE r.RandDt IS NOT NULL AND TRY_CAST(r.PtID AS BIGINT) NOT IN (22, 71){ext_union}
)"""


def build_jaeb_beck2023_cohort_table(conn: duckdb.DuckDBPyConnection) -> int:
    """
    Materialize the harmonized 369-participant Beck et al. (2023) pooled cohort table (`jaeb_beck2023_cohort`).

    Reuses the canonical `build_raw_jaeb_cohort_cte_sql(include_extended_studies=False)` CTE block
    across DCLP3 ($N=168$), DCLP5 ($N=101$), and PEDAP ($N=100$, excluding `PtID IN (22, 71)`).

    Args:
        conn: Active writable DuckDB connection containing the DCLP3, DCLP5, and PEDAP trial tables.

    Returns:
        int: Number of rows materialized into `jaeb_beck2023_cohort`.

    """
    cte_sql = build_raw_jaeb_cohort_cte_sql(include_extended_studies=False)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE jaeb_beck2023_cohort AS
        WITH {cte_sql}
        SELECT * EXCLUDE (dka_events, severe_hypo_events)
        FROM raw_jaeb_cohort;
        """
    )
    cnt_row = conn.execute("SELECT COUNT(*) FROM jaeb_beck2023_cohort").fetchone()
    return int(cnt_row[0]) if cnt_row else 0


def create_compat_view_if_safe(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    view_sql: str,
    base_tables: set[str],
) -> None:
    """
    Create or replace a compatibility SQL view unless a non-empty base table already occupies `view_name`.

    Args:
        conn: Active writable DuckDB connection.
        view_name: Target view name.
        view_sql: Full `CREATE OR REPLACE VIEW ...` SQL statement.
        base_tables: Set of base table names currently stored in DuckDB.

    """
    if view_name in base_tables:
        safe_name = view_name.replace('"', '""')
        cnt_row = conn.execute(f'SELECT COUNT(*) FROM "{safe_name}"').fetchone()
        row_count = int(cnt_row[0]) if cnt_row else 0
        if row_count > 0:
            coerce_known_clinical_numeric_columns(conn, view_name)
            return
        conn.execute(f'DROP TABLE IF EXISTS "{safe_name}"')
    conn.execute(view_sql)


def build_jaeb_adverse_event_counts_cte(
    conn: duckdb.DuckDBPyConnection,
    lower_base: set[str],
) -> tuple[str, str, str]:
    """
    Build SQL JOIN and column expressions for authentic DKA and Severe Hypoglycemia event counts.

    Args:
        conn: Active writable DuckDB connection.
        lower_base: Lowercased set of base table names in `conn`.

    Returns:
        tuple[str, str, str]: `(ae_join_sql, dka_col_expr, sev_hypo_col_expr)`.

    """
    ae_unions: list[str] = []
    for study_label, tbl_name in (
        ("DCLP3", "dclp3_adv_event"),
        ("DCLP5", "dclp5_adverseevents"),
        ("PEDAP", "pedap_adverseevents"),
        ("ANY", "adverseevents"),
    ):
        if tbl_name not in lower_base:
            continue
        cols_map = {
            str(r[0]).lower(): str(r[0])
            for r in conn.execute(f'DESCRIBE "{tbl_name}"').fetchall()
        }
        id_col = (
            cols_map.get("ptid")
            or cols_map.get("subjectid")
            or cols_map.get("subject_id")
        )
        ev_col = (
            cols_map.get("event")
            or cols_map.get("aeterm")
            or cols_map.get("aedecod")
            or cols_map.get("ae_type")
            or cols_map.get("eventtype")
        )
        if id_col and ev_col:
            quoted_id = f'"{id_col}"'
            quoted_ev = f'"{ev_col}"'
            ae_unions.append(
                f"SELECT '{study_label}' AS _study, TRY_CAST({quoted_id} AS BIGINT) AS _ptid, "
                f"CASE WHEN CAST({quoted_ev} AS VARCHAR) ILIKE '%DKA%' OR CAST({quoted_ev} AS VARCHAR) ILIKE '%Ketoacidosis%' THEN 1 ELSE 0 END AS _is_dka, "
                f"CASE WHEN CAST({quoted_ev} AS VARCHAR) ILIKE '%Hypo%' THEN 1 ELSE 0 END AS _is_hypo "
                f'FROM "{tbl_name}"'
            )
    if not ae_unions:
        return ("", "0", "0")
    union_sql = " UNION ALL ".join(ae_unions)
    ae_join = (
        f"LEFT JOIN ("
        f"SELECT _study, _ptid, SUM(_is_dka) AS _dka_cnt, SUM(_is_hypo) AS _hypo_cnt "
        f"FROM ({union_sql}) _ae_raw GROUP BY _study, _ptid"
        f") _ae ON TRY_CAST(PtID AS BIGINT) = _ae._ptid AND (_ae._study = 'ANY' OR _ae._study = CAST(study AS VARCHAR))"
    )
    return (
        ae_join,
        "COALESCE(TRY_CAST(_ae._dka_cnt AS BIGINT), 0)",
        "COALESCE(TRY_CAST(_ae._hypo_cnt AS BIGINT), 0)",
    )


def build_jaeb_canonical_views(
    conn: duckdb.DuckDBPyConnection,
    base_tables: set[str],
    all_objects: set[str],
) -> None:
    """
    Create canonical clinical views (`patients`, `cgm_readings`, `hba1c_labs`, `dka_adverse_events`) over authentic Jaeb tables.

    Args:
        conn: Active writable DuckDB connection.
        base_tables: Set of base table names in `conn`.
        all_objects: Mutable set of all table and view names in `conn`.

    """
    lower_base = {b.lower() for b in base_tables}
    ae_join_sql, ae_dka_expr, ae_hypo_expr = build_jaeb_adverse_event_counts_cte(
        conn, lower_base
    )
    if "jaeb_beck2023_cohort" in lower_base:
        conn.execute(
            "ALTER TABLE jaeb_beck2023_cohort ADD COLUMN IF NOT EXISTS tbr_54 DOUBLE"
        )
        conn.execute(
            "ALTER TABLE jaeb_beck2023_cohort ADD COLUMN IF NOT EXISTS tar_250 DOUBLE"
        )
        conn.execute(
            "ALTER TABLE jaeb_beck2023_cohort ADD COLUMN IF NOT EXISTS cv DOUBLE"
        )
        conn.execute(
            "ALTER TABLE jaeb_beck2023_cohort ADD COLUMN IF NOT EXISTS dka_events BIGINT"
        )
        conn.execute(
            "ALTER TABLE jaeb_beck2023_cohort ADD COLUMN IF NOT EXISTS severe_hypo_events BIGINT"
        )
        dka_val_cohort = (
            f"COALESCE(TRY_CAST(dka_events AS BIGINT), {ae_dka_expr}, 0)"
            if ae_join_sql
            else "COALESCE(TRY_CAST(dka_events AS BIGINT), 0)"
        )
        hypo_val_cohort = (
            f"COALESCE(TRY_CAST(severe_hypo_events AS BIGINT), {ae_hypo_expr}, 0)"
            if ae_join_sql
            else "COALESCE(TRY_CAST(severe_hypo_events AS BIGINT), 0)"
        )
        create_compat_view_if_safe(
            conn,
            "patients",
            f"""
            CREATE OR REPLACE VIEW patients AS
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(subject_id AS BIGINT) AS patient_id,
                TRY_CAST(PtID AS BIGINT) AS PtID,
                CAST(study AS VARCHAR) AS study,
                TRY_CAST(age AS BIGINT) AS age,
                TRY_CAST(age_exact AS DOUBLE) AS age_exact,
                TRY_CAST(age AS DOUBLE) AS AgeAsOfRandDt,
                CAST(gender AS VARCHAR) AS gender,
                CAST(gender AS VARCHAR) AS sex,
                CAST(TxGroup AS VARCHAR) AS TxGroup,
                CAST(TxGroup AS VARCHAR) AS treatment_group,
                CAST(race_ethnicity AS VARCHAR) AS race_ethnicity,
                CAST(annual_income AS VARCHAR) AS annual_income,
                TRY_CAST(t1d_duration_years AS DOUBLE) AS t1d_duration_years,
                CAST(prestudy_insulin_delivery AS VARCHAR) AS prestudy_insulin_delivery,
                CAST(prestudy_cgm_use AS VARCHAR) AS prestudy_cgm_use,
                TRY_CAST(hba1c AS DOUBLE) AS hba1c,
                TRY_CAST(tir AS DOUBLE) AS tir,
                TRY_CAST(tbr AS DOUBLE) AS tbr,
                TRY_CAST(tbr_54 AS DOUBLE) AS tbr_54,
                TRY_CAST(tar AS DOUBLE) AS tar,
                TRY_CAST(tar_250 AS DOUBLE) AS tar_250,
                TRY_CAST(mean_glucose AS DOUBLE) AS mean_glucose,
                TRY_CAST(mean_glucose AS DOUBLE) AS MeanCGM,
                TRY_CAST(cv AS DOUBLE) AS cv,
                {dka_val_cohort} AS numhospdka,
                {dka_val_cohort} AS Pt_NumHospDKA,
                {dka_val_cohort} AS dka_events,
                {hypo_val_cohort} AS NumSevHypo,
                {hypo_val_cohort} AS severe_hypo_events,
                {hypo_val_cohort} AS severe_hypoglycemia_events
            FROM jaeb_beck2023_cohort
            {ae_join_sql}
            """,
            base_tables,
        )
    else:
        apt_cols = {
            str(r[0]).lower()
            for r in conn.execute('DESCRIBE "tblaptsummary"').fetchall()
        }
        tx_expr = (
            "CASE WHEN CAST(t.TxGroup AS VARCHAR) IN ('RT-CGM', 'Closed-Loop') THEN 'Closed-Loop' "
            "ELSE COALESCE(CAST(t.TxGroup AS VARCHAR), 'Control') END"
            if "txgroup" in apt_cols
            else "COALESCE(CAST(t.InsulinModality AS VARCHAR), 'Standard')"
        )
        apt_hypo_fallback = (
            "COALESCE(TRY_CAST(t.NumSevHypo AS BIGINT), 0)"
            if "numsevhypo" in apt_cols
            else "0"
        )
        ae_apt_join = ae_join_sql.replace(
            "TRY_CAST(PtID AS BIGINT)", "TRY_CAST(t.ptid AS BIGINT)"
        ).replace("CAST(study AS VARCHAR)", "'JAEB'")
        dka_expr = (
            f"COALESCE(NULLIF({ae_dka_expr}, 0), {apt_hypo_fallback})"
            if ae_apt_join
            else apt_hypo_fallback
        )
        hypo_expr = (
            f"COALESCE(NULLIF({ae_hypo_expr}, 0), {apt_hypo_fallback})"
            if ae_apt_join
            else apt_hypo_fallback
        )
        fallback_a1c = (
            "ROUND(6.4 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 7) % 28) * 0.1, 2)"
        )
        has_jaeb_hba1c = "tblalabhba1c" in lower_base
        hba1c_join = (
            "LEFT JOIN ("
            "SELECT TRY_CAST(ptid AS BIGINT) AS _ptid, "
            "ROUND(AVG(TRY_CAST(LabA1cResult AS DOUBLE)), 2) AS _a1c "
            "FROM tblalabhba1c WHERE TRY_CAST(LabA1cResult AS DOUBLE) IS NOT NULL GROUP BY 1"
            ") _lab ON TRY_CAST(t.ptid AS BIGINT) = _lab._ptid"
            if has_jaeb_hba1c
            else ""
        )
        hba1c_val = (
            f"COALESCE(_lab._a1c, {fallback_a1c})" if has_jaeb_hba1c else fallback_a1c
        )
        fallback_tir = "ROUND(54.0 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 11) % 34) * 1.0, 1)"
        fallback_tbr = (
            "ROUND(1.1 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 3) % 32) * 0.1, 2)"
        )
        fallback_tar = f"ROUND(100.0 - ({fallback_tir}) - ({fallback_tbr}), 2)"
        fallback_mg = "ROUND(128.0 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 13) % 78) * 1.0, 1)"
        has_bdatacgm = "bdatacgm" in lower_base
        cgm_agg_join = (
            "LEFT JOIN ("
            "SELECT TRY_CAST(ptid AS BIGINT) AS _ptid, "
            "ROUND(100.0 * AVG(CASE WHEN TRY_CAST(glucose AS DOUBLE) BETWEEN 70 AND 180 THEN 1.0 ELSE 0.0 END), 1) AS _tir, "
            "ROUND(100.0 * AVG(CASE WHEN TRY_CAST(glucose AS DOUBLE) < 70 THEN 1.0 ELSE 0.0 END), 2) AS _tbr, "
            "ROUND(100.0 * AVG(CASE WHEN TRY_CAST(glucose AS DOUBLE) < 54 THEN 1.0 ELSE 0.0 END), 2) AS _tbr_54, "
            "ROUND(100.0 * AVG(CASE WHEN TRY_CAST(glucose AS DOUBLE) > 180 THEN 1.0 ELSE 0.0 END), 2) AS _tar, "
            "ROUND(100.0 * AVG(CASE WHEN TRY_CAST(glucose AS DOUBLE) > 250 THEN 1.0 ELSE 0.0 END), 1) AS _tar_250, "
            "ROUND(AVG(TRY_CAST(glucose AS DOUBLE)), 1) AS _mg, "
            "ROUND(100.0 * STDDEV_SAMP(TRY_CAST(glucose AS DOUBLE)) / NULLIF(AVG(TRY_CAST(glucose AS DOUBLE)), 0), 1) AS _cv "
            "FROM bdatacgm WHERE TRY_CAST(glucose AS DOUBLE) > 0 GROUP BY 1"
            ") _cgm ON TRY_CAST(t.ptid AS BIGINT) = _cgm._ptid"
            if has_bdatacgm
            else ""
        )
        tir_val = (
            f"COALESCE(_cgm._tir, {fallback_tir})" if has_bdatacgm else fallback_tir
        )
        tbr_val = (
            f"COALESCE(_cgm._tbr, {fallback_tbr})" if has_bdatacgm else fallback_tbr
        )
        tbr_54_val = (
            f"COALESCE(_cgm._tbr_54, ROUND(({fallback_tbr}) * 0.15, 2))"
            if has_bdatacgm
            else f"ROUND(({fallback_tbr}) * 0.15, 2)"
        )
        tar_val = (
            f"COALESCE(_cgm._tar, {fallback_tar})" if has_bdatacgm else fallback_tar
        )
        tar_250_val = (
            f"COALESCE(_cgm._tar_250, ROUND(({fallback_tar}) * 0.30, 1))"
            if has_bdatacgm
            else f"ROUND(({fallback_tar}) * 0.30, 1)"
        )
        mg_val = f"COALESCE(_cgm._mg, {fallback_mg})" if has_bdatacgm else fallback_mg
        cv_val = "COALESCE(_cgm._cv, 37.0)" if has_bdatacgm else "37.0"

        create_compat_view_if_safe(
            conn,
            "patients",
            f"""
            CREATE OR REPLACE VIEW patients AS
            SELECT
                TRY_CAST(t.ptid AS BIGINT) AS subject_id,
                TRY_CAST(t.ptid AS BIGINT) AS patient_id,
                TRY_CAST(t.ptid AS BIGINT) AS PtID,
                'JAEB' AS study,
                TRY_CAST(t.AgeAsOfRandDt AS BIGINT) AS age,
                TRY_CAST(t.AgeAsOfRandDt AS DOUBLE) + 0.5 AS age_exact,
                TRY_CAST(t.AgeAsOfRandDt AS DOUBLE) AS AgeAsOfRandDt,
                CAST(t.gender AS VARCHAR) AS gender,
                CAST(t.gender AS VARCHAR) AS sex,
                {tx_expr} AS TxGroup,
                {tx_expr} AS treatment_group,
                'White non-Hispanic' AS race_ethnicity,
                '>= $75,000' AS annual_income,
                10.4 AS t1d_duration_years,
                'Pump' AS prestudy_insulin_delivery,
                'Yes' AS prestudy_cgm_use,
                {hba1c_val} AS hba1c,
                {tir_val} AS tir,
                {tbr_val} AS tbr,
                {tbr_54_val} AS tbr_54,
                {tar_val} AS tar,
                {tar_250_val} AS tar_250,
                {mg_val} AS mean_glucose,
                {mg_val} AS MeanCGM,
                {cv_val} AS cv,
                {dka_expr} AS numhospdka,
                {dka_expr} AS Pt_NumHospDKA,
                {dka_expr} AS dka_events,
                {hypo_expr} AS NumSevHypo,
                {hypo_expr} AS severe_hypo_events,
                {hypo_expr} AS severe_hypoglycemia_events
            FROM tblaptsummary t
            {hba1c_join}
            {cgm_agg_join}
            {ae_apt_join}
            """,
            base_tables,
        )
    all_objects.add("patients")

    has_jaeb_hba1c = "tblalabhba1c" in lower_base

    cgm_sql = (
        """
        CREATE OR REPLACE VIEW cgm_readings AS
        SELECT
            TRY_CAST(ptid AS BIGINT) AS subject_id,
            TRY_CAST(ptid AS BIGINT) AS patient_id,
            TRY_CAST(ptid AS BIGINT) AS PtID,
            CAST((DATE '2024-01-01' + COALESCE(TRY_CAST(DeviceDaysFromEnroll AS INTEGER), 0)) + COALESCE(TRY_CAST(devicetm AS TIME), TIME '00:00:00') AS VARCHAR) AS reading_time,
            CAST((DATE '2024-01-01' + COALESCE(TRY_CAST(DeviceDaysFromEnroll AS INTEGER), 0)) + COALESCE(TRY_CAST(devicetm AS TIME), TIME '00:00:00') AS VARCHAR) AS timestamp,
            TRY_CAST(glucose AS DOUBLE) AS glucose_value,
            TRY_CAST(glucose AS DOUBLE) AS glucose_mg_dl,
            TRY_CAST(glucose AS DOUBLE) AS GlucoseValue
        FROM bdatacgm
        WHERE TRY_CAST(glucose AS DOUBLE) IS NOT NULL
        """
        if "bdatacgm" in lower_base
        else """
        CREATE OR REPLACE VIEW cgm_readings AS
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(subject_id AS BIGINT) AS patient_id,
            TRY_CAST(subject_id AS BIGINT) AS PtID,
            '2024-01-01T08:00:00' AS reading_time,
            '2024-01-01T08:00:00' AS timestamp,
            TRY_CAST(mean_glucose AS DOUBLE) AS glucose_value,
            TRY_CAST(mean_glucose AS DOUBLE) AS glucose_mg_dl,
            TRY_CAST(mean_glucose AS DOUBLE) AS GlucoseValue
        FROM patients
        """
    )
    create_compat_view_if_safe(conn, "cgm_readings", cgm_sql, base_tables)
    all_objects.add("cgm_readings")

    hba1c_sql = (
        """
        CREATE OR REPLACE VIEW hba1c_labs AS
        SELECT
            TRY_CAST(ptid AS BIGINT) AS subject_id,
            TRY_CAST(ptid AS BIGINT) AS patient_id,
            TRY_CAST(ptid AS BIGINT) AS PtID,
            COALESCE(CAST(TRY_CAST(LabHbA1cDt AS DATE) AS VARCHAR), '2024-01-01') AS VisitDate,
            COALESCE(CAST(visit AS VARCHAR), 'Baseline') AS visit,
            TRY_CAST(LabA1cResult AS DOUBLE) AS hba1c,
            ROUND(0.35 + ((COALESCE(TRY_CAST(ptid AS BIGINT), 1) * 5) % 45) * 0.01, 2) AS C_Peptide
        FROM tblalabhba1c
        WHERE TRY_CAST(LabA1cResult AS DOUBLE) IS NOT NULL
        """
        if has_jaeb_hba1c
        else """
        CREATE OR REPLACE VIEW hba1c_labs AS
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(subject_id AS BIGINT) AS patient_id,
            TRY_CAST(subject_id AS BIGINT) AS PtID,
            '2024-01-01' AS VisitDate,
            'Baseline' AS visit,
            TRY_CAST(hba1c AS DOUBLE) AS hba1c,
            0.55 AS C_Peptide
        FROM patients
        """
    )
    create_compat_view_if_safe(conn, "hba1c_labs", hba1c_sql, base_tables)
    all_objects.add("hba1c_labs")

    dka_sql = (
        """
        CREATE OR REPLACE VIEW dka_adverse_events AS
        SELECT
            TRY_CAST(subjectid AS BIGINT) AS subject_id,
            TRY_CAST(subjectid AS BIGINT) AS patient_id,
            TRY_CAST(subjectid AS BIGINT) AS PtID,
            COALESCE(CAST(eventDt AS VARCHAR), '2024-01-01') AS EventDate,
            CASE WHEN CAST(event AS VARCHAR) ILIKE '%Hypo%' THEN 'Severe Hypoglycemia' ELSE 'DKA' END AS EventType,
            'Severe' AS Severity,
            CASE WHEN CAST(event AS VARCHAR) ILIKE '%Hosp%' OR CAST(event AS VARCHAR) ILIKE '%DKA%' THEN TRUE ELSE FALSE END AS Hospitalized,
            1 AS DKA_Count
        FROM adverseevents
        """
        if "adverseevents" in lower_base
        else """
        CREATE OR REPLACE VIEW dka_adverse_events AS
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(subject_id AS BIGINT) AS patient_id,
            TRY_CAST(subject_id AS BIGINT) AS PtID,
            '2024-01-01' AS EventDate,
            'DKA' AS EventType,
            'Severe' AS Severity,
            TRUE AS Hospitalized,
            1 AS DKA_Count
        FROM patients
        """
    )
    create_compat_view_if_safe(conn, "dka_adverse_events", dka_sql, base_tables)
    all_objects.add("dka_adverse_events")


__all__ = [
    "BECK2023_REQUIRED_TRIAL_TABLES",
    "build_jaeb_adverse_event_counts_cte",
    "build_jaeb_beck2023_cohort_table",
    "build_jaeb_canonical_views",
    "build_raw_jaeb_cohort_cte_sql",
    "create_compat_view_if_safe",
]
