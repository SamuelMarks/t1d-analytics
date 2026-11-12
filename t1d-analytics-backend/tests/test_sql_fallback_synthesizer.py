"""Unit tests for sanitized LLM failure diagnostics, schema-guided SQL fallback synthesis, and compositional Table 1 compilation."""

from pathlib import Path
from unittest.mock import patch

import duckdb

from t1d_analytics.seed import seed_database
from t1d_analytics.sql_fallback_synthesizer import (
    _is_beck_2023_table_1_query,
    format_llm_failure_reason,
    lookup_table_1_curated_sql,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import extract_sql_from_response
from t1d_analytics.table1_builder import (
    Table1QuerySpec,
    Table1StratumSpec,
    build_dynamic_table1_sql,
    has_custom_table1_modifiers,
    parse_nl_table1_spec,
)
from t1d_analytics.training_data import BECK_2023_TABLE_1_SQL


def test_format_llm_failure_reason_sanitizes_segfault_messages() -> None:
    """Verify format_llm_failure_reason sanitizes raw llama-server segmentation fault messages into a concise CPU runner diagnostic."""
    raw_exc = RuntimeError(
        "llama-server process has terminated: signal: segmentation fault (core dumped) (status code: 500)"
    )
    msg, code = format_llm_failure_reason(raw_exc, model="gemma:2b")
    assert msg == "Local model 'gemma:2b' was unavailable on CPU runner"
    assert code == "backend.llmMemoryCrash"
    assert "segmentation fault" not in (msg or "")
    assert "core dumped" not in (msg or "")

    msg_default, code_default = format_llm_failure_reason(raw_exc)
    assert msg_default == "Local model 'ollama' was unavailable on CPU runner"
    assert code_default == "backend.llmMemoryCrash"


def test_synthesize_fallback_clinical_sql_comment_header_and_extraction() -> None:
    """Verify synthesize_fallback_clinical_sql includes the clean SQL comment header inside the sql fence and extract_sql_from_response strips it cleanly."""
    schema = """Table: patients
Columns: subject_id (INT), age (DOUBLE), TxGroup (VARCHAR), hba1c (DOUBLE), tir (DOUBLE), tbr (DOUBLE), tar (DOUBLE), mean_glucose (DOUBLE), numhospdka (INT)
"""
    res = synthesize_fallback_clinical_sql(
        schema=schema,
        nl_query="How many patients are in the dataset?",
        include_diagnostic_note=True,
        llm_error="Local model 'gemma:2b' was unavailable on CPU runner",
        llm_error_code="backend.llmMemoryCrash",
        model="gemma:2b",
    )
    assert res.used_fallback is True
    assert res.generation_mode == "rule_based_fallback"
    assert (
        "-- Synthesized schema-guided fallback SQL for patients (gemma:2b)"
        in res.full_response
    )
    assert "segmentation fault" not in res.full_response
    assert res.sql_query == 'SELECT COUNT(*) AS count FROM "patients";'
    assert (
        extract_sql_from_response(res.full_response)
        == 'SELECT COUNT(*) AS count FROM "patients";'
    )


def test_compositional_table1_builder_beck2023_and_arbitrary_subsets(
    tmp_path: Path,
) -> None:
    """Verify table1_builder reproduces Beck et al. (2023) Table 1 exactly and compiles dynamic SQL for arbitrary subsets, columns, and rows."""
    db_file = tmp_path / "t1d_test.duckdb"
    seed_database(
        data_dir=str(Path(__file__).resolve().parent / "test_data"),
        db_path=str(db_file),
        sessions_db_path=str(tmp_path / "sessions.duckdb"),
    )
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        # 1. Exact reproduction of Beck et al. (2023) Table 1 (61 rows including 'Participants, N', 4 columns)
        beck_prompt = "Generate Beck et al. 2023 Table 1 participant characteristics at baseline from jaeb"
        assert has_custom_table1_modifiers(beck_prompt) is False
        assert _is_beck_2023_table_1_query(beck_prompt.lower()) is True
        assert lookup_table_1_curated_sql(beck_prompt) == BECK_2023_TABLE_1_SQL
        assert _is_beck_2023_table_1_query("how many patients are in jaeb") is False

        compiled_beck = build_dynamic_table1_sql(beck_prompt, db_path=str(db_file))
        rows_compiled = conn.execute(compiled_beck).fetchall()
        rows_gold = conn.execute(BECK_2023_TABLE_1_SQL).fetchall()
        assert len(rows_compiled) == 61
        assert rows_compiled[0] == ("Participants, N", "100", "50", "50")
        assert rows_compiled == rows_gold

        # Also verify omit N row option ('without n row')
        q_no_n = "Generate Beck et al. 2023 Table 1 without n row"
        assert has_custom_table1_modifiers(q_no_n) is True
        sql_no_n = build_dynamic_table1_sql(q_no_n, db_path=str(db_file))
        rows_no_n = conn.execute(sql_no_n).fetchall()
        assert len(rows_no_n) == 60
        assert rows_no_n[0][0] == "Age at enrollment"

        # Also verify default fallback when db_path is None (clean column aliases, no hardcoded N)
        spec_no_db = parse_nl_table1_sql_spec_helper(beck_prompt)
        assert [s.column_alias for s in spec_no_db.strata] == [
            "Overall",
            "Control-IQ",
            "Control",
        ]

        # 2. Single-study and multi-study subsets
        q_dclp3 = "Generate a Table 1 for DCLP3 participants only"
        assert has_custom_table1_modifiers(q_dclp3) is True
        assert _is_beck_2023_table_1_query(q_dclp3.lower()) is False
        assert lookup_table_1_curated_sql(q_dclp3) is None
        sql_dclp3 = build_dynamic_table1_sql(q_dclp3, db_path=str(db_file))
        assert "UPPER(study) = 'DCLP3'" in sql_dclp3
        res_dclp3 = conn.execute(sql_dclp3).fetchall()
        assert res_dclp3[0][0] == "Participants, N"

        q_two_studies = "Generate a Table 1 for DCLP5 and PEDAP children under 14 years"
        sql_two = build_dynamic_table1_sql(q_two_studies, db_path=str(db_file))
        assert "UPPER(study) IN ('DCLP5', 'PEDAP')" in sql_two
        assert "age < 14" in sql_two
        assert len(conn.execute(sql_two).fetchall()) > 10

        # 3. Age, glycemic, therapy, gender, treatment arm, and limit subsets
        q_sub_1 = (
            "Make a Table 1 for participants aged between 14 and 50 with HbA1c >= 7.5 and TIR < 60 "
            "who are MDI users and female participants in the Control-IQ group limit 50 participants"
        )
        sql_sub_1 = build_dynamic_table1_sql(q_sub_1, db_path=str(db_file))
        assert "age >= 14 AND age <= 50" in sql_sub_1
        assert "hba1c >= 7.5" in sql_sub_1
        assert "tir < 60.0" in sql_sub_1
        assert "prestudy_insulin_delivery = 'MDI'" in sql_sub_1
        assert "gender IN ('Female', 'F')" in sql_sub_1
        assert "TxGroup IN ('Control-IQ', 'Closed-Loop', 'CLC')" in sql_sub_1
        assert "LIMIT 50" in sql_sub_1
        assert len(conn.execute(sql_sub_1).fetchall()) > 10

        q_sub_2 = (
            "Make a Table 1 for participants over 25 years old with HbA1c < 8.0 and TIR >= 50 "
            "who are pump users and male participants in the Control group"
        )
        sql_sub_2 = build_dynamic_table1_sql(q_sub_2)
        assert "age >= 25" in sql_sub_2
        assert "hba1c < 8.0" in sql_sub_2
        assert "tir >= 50.0" in sql_sub_2
        assert "prestudy_insulin_delivery = 'Pump'" in sql_sub_2
        assert "gender IN ('Male', 'M')" in sql_sub_2
        assert "TxGroup IN ('Control', 'SAP', 'SC')" in sql_sub_2
        assert len(conn.execute(sql_sub_2).fetchall()) > 10

        for phrase, expected_clause in (
            ("Table 1 for pediatric youth", "age < 18"),
            ("Table 1 for toddlers", "age < 6"),
            ("Table 1 for adolescents", "age >= 14 AND age < 18"),
            ("Table 1 for adults only", "age >= 18"),
            ("Table 1 for older adults", "age >= 50"),
            ("Table 1 for participants over 100 years old", "age >= 100"),
            ("Table 1 for school-age participants", "age >= 6 AND age < 14"),
            ("Table 1 for young adults", "age >= 18 AND age < 25"),
            (
                "Table 1 for participants excluding DCLP3 and excluding PEDAP",
                "UPPER(study) NOT IN ('DCLP3', 'PEDAP')",
            ),
            (
                "Table 1 for participants excluding PEDAP",
                "UPPER(study) != 'PEDAP'",
            ),
            (
                "Table 1 for participants with HbA1c between 7.0 and 8.5%",
                "hba1c >= 7.0 AND hba1c <= 8.5",
            ),
            (
                "Table 1 for participants with TIR between 50 and 70%",
                "tir >= 50.0 AND tir <= 70.0",
            ),
            ("Table 1 for participants with TBR >= 4%", "tbr >= 4.0"),
            ("Table 1 for participants with TBR < 1%", "tbr < 1.0"),
            ("Table 1 for participants with TAR >= 30%", "tar >= 30.0"),
            (
                "Table 1 for participants with mean glucose >= 180",
                "mean_glucose >= 180.0",
            ),
            (
                "Table 1 for participants with mean glucose < 150",
                "mean_glucose < 150.0",
            ),
            (
                "Table 1 for participants with duration >= 5 years",
                "t1d_duration_years >= 5.0",
            ),
            (
                "Table 1 for participants with duration < 2 years",
                "t1d_duration_years < 2.0",
            ),
            ("Table 1 for newly diagnosed participants", "t1d_duration_years < 1.0"),
            ("Table 1 for CGM naive participants", "prestudy_cgm_use = 'No'"),
            ("Table 1 for prior CGM users", "prestudy_cgm_use = 'Yes'"),
            ("Table 1 for low-income participants", "annual_income = '< $75,000'"),
            (
                "Table 1 for high-income participants",
                "annual_income IN ('>= $75,000', '≥ $75,000')",
            ),
            (
                "Table 1 for White non-Hispanic participants",
                "race_ethnicity = 'White non-Hispanic'",
            ),
            ("Table 1 for Black participants", "race_ethnicity = 'Black non-Hispanic'"),
            (
                "Table 1 for Hispanic participants",
                "race_ethnicity = 'Hispanic or Latino'",
            ),
            ("Table 1 for Asian participants", "race_ethnicity = 'Asian'"),
            (
                "Table 1 for minority participants",
                "race_ethnicity != 'White non-Hispanic' AND race_ethnicity IS NOT NULL",
            ),
            (
                "Table 1 for complete cases",
                "annual_income IS NOT NULL AND annual_income != 'Missing' AND hba1c IS NOT NULL AND tir IS NOT NULL",
            ),
        ):
            s_sql = build_dynamic_table1_sql(phrase, db_path=str(db_file))
            assert expected_clause in s_sql
            assert len(conn.execute(s_sql).fetchall()) >= 5
    finally:
        conn.close()


def parse_nl_table1_sql_spec_helper(prompt: str) -> Table1QuerySpec:
    """
    Call `parse_nl_table1_spec` without a `db_path` for unit assertions.

    Args:
        prompt: Natural-language Table 1 query.

    Returns:
        Table1QuerySpec: Parsed specification.

    """
    return parse_nl_table1_spec(prompt, schema_map=None, db_path=None)


def test_compositional_table1_custom_strata_rows_and_schemas(
    tmp_path: Path,
) -> None:
    """Verify all custom stratification columns, custom row selections, Median (IQR), and schema column fallbacks."""
    db_file = tmp_path / "t1d_strata.duckdb"
    seed_database(
        data_dir=str(Path(__file__).resolve().parent / "test_data"),
        db_path=str(db_file),
        sessions_db_path=str(tmp_path / "sessions2.duckdb"),
    )
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        strata_queries = (
            (
                "Table 1 unstratified overall only",
                ["Characteristic", "Overall"],
            ),
            (
                "Table 1 stratified by study",
                ["Characteristic", "Overall", "DCLP3", "DCLP5", "PEDAP"],
            ),
            (
                "Table 1 stratified by sex showing age and sex",
                ["Characteristic", "Overall", "Female", "Male"],
            ),
            (
                "Table 1 stratified by sex",
                ["Characteristic", "Overall", "Female", "Male"],
            ),
            (
                "Table 1 stratified by prestudy insulin delivery showing age and insulin delivery",
                ["Characteristic", "Overall", "Pump", "MDI"],
            ),
            (
                "Table 1 stratified by prestudy insulin delivery",
                ["Characteristic", "Overall", "Pump", "MDI"],
            ),
            (
                "Table 1 stratified by prestudy CGM use",
                [
                    "Characteristic",
                    "Overall",
                    "Prestudy CGM",
                    "No Prestudy CGM",
                ],
            ),
            (
                "Table 1 stratified by annual household income",
                ["Characteristic", "Overall", "< $75,000", "≥ $75,000"],
            ),
            (
                "Table 1 stratified by age group",
                ["Characteristic", "Overall", "<18 years", "≥18 years"],
            ),
            (
                "Table 1 stratified by 5 age groups without overall",
                [
                    "Characteristic",
                    "<6 yr",
                    "6–<14 yr",
                    "14–<25 yr",
                    "25–<50 yr",
                    "≥50 yr",
                ],
            ),
            (
                "Table 1 stratified by 3 age groups",
                ["Characteristic", "Overall", "<18 yr", "18–49 yr", "≥50 yr"],
            ),
            (
                "Table 1 stratified by race",
                [
                    "Characteristic",
                    "Overall",
                    "White non-Hispanic",
                    "Racial/Ethnic Minority",
                ],
            ),
            (
                "Table 1 stratified by baseline HbA1c tier",
                [
                    "Characteristic",
                    "Overall",
                    "HbA1c <7.0%",
                    "HbA1c 7.0–<8.0%",
                    "HbA1c ≥8.0%",
                ],
            ),
            (
                "Table 1 stratified by baseline TIR tier",
                [
                    "Characteristic",
                    "Overall",
                    "TIR <50%",
                    "TIR 50–<70%",
                    "TIR ≥70%",
                ],
            ),
        )
        for q_text, expected_prefixes in strata_queries:
            assert has_custom_table1_modifiers(q_text) is True
            sql = build_dynamic_table1_sql(q_text, db_path=str(db_file))
            cur = conn.execute(sql)
            col_names = [d[0] for d in cur.description]
            for idx, pref in enumerate(expected_prefixes):
                assert col_names[idx].startswith(pref)

        # Test compact mode and has_custom_table1_modifiers with schema_map and median_iqr
        assert (
            has_custom_table1_modifiers(
                "Make Table 1 in compact mode",
                schema_map={"patients": ["age", "hba1c"]},
            )
            is True
        )
        assert (
            has_custom_table1_modifiers(
                "Make Table 1 with median IQR",
                schema_map={},
            )
            is True
        )
        sql_compact = build_dynamic_table1_sql(
            "Make Table 1 in compact mode",
            db_path=None,
        )
        assert "<6 yr" not in sql_compact

        # Custom row selections, extended metrics, and Median (IQR)
        q_only = "Table 1 showing only Age, Baseline HbA1c, and Median IQR"
        assert has_custom_table1_modifiers(q_only) is True
        sql_only = build_dynamic_table1_sql(q_only, db_path=str(db_file))
        rows_only = conn.execute(sql_only).fetchall()
        labels_only = [r[0] for r in rows_only]
        assert " Median (IQR)" in labels_only
        assert "Age at enrollment" in labels_only
        assert "Baseline central lab HbA1c, %" in labels_only
        assert "Race/Ethnicity, n (%)" not in labels_only

        q_short_subset = "Create a Table 1 showing Age, Sex, and HbA1c"
        sql_short = build_dynamic_table1_sql(q_short_subset, db_path=str(db_file))
        labels_short = [r[0] for r in conn.execute(sql_short).fetchall()]
        assert "Age at enrollment" in labels_short
        assert "Sex, n (%)" in labels_short
        assert "Baseline central lab HbA1c, %" in labels_short
        assert "Annual household income, n (%)" not in labels_short

        q_extended = (
            "Generate a Table 1 with time <54 mg/dL, time >180, time >250 mg/dL, "
            "mean glucose, coefficient of variation, DKA events, severe hypo, consensus targets, and study breakdown"
        )
        assert has_custom_table1_modifiers(q_extended) is True
        sql_ext = build_dynamic_table1_sql(q_extended, db_path=str(db_file))
        labels_ext = [r[0] for r in conn.execute(sql_ext).fetchall()]
        assert "Baseline time <54\u2009mg/dL" in labels_ext
        assert "Baseline time >180\u2009mg/dL" in labels_ext
        assert "Baseline time >250\u2009mg/dL" in labels_ext
        assert "Baseline mean glucose, mg/dL" in labels_ext
        assert "Baseline coefficient of variation" in labels_ext
        assert "DKA events, n (%)" in labels_ext
        assert "Severe hypoglycemia events, n (%)" in labels_ext
        assert "Glycemic consensus targets, n (%)" in labels_ext
        assert "Clinical trial, n (%)" in labels_ext

        # Test row exclusions, missing-row suppression, without range, and median only
        q_excl = (
            "Generate a Table 1 without race, without income, without prestudy cgm, "
            "without missing rows, and without range"
        )
        assert has_custom_table1_modifiers(q_excl) is True
        sql_excl = build_dynamic_table1_sql(q_excl, db_path=str(db_file))
        labels_excl = [r[0] for r in conn.execute(sql_excl).fetchall()]
        assert "Race/Ethnicity, n (%)" not in labels_excl
        assert "Annual household income, n (%)" not in labels_excl
        assert "Prestudy CGM use, n (%)" not in labels_excl
        assert " Missing, n (%)" not in labels_excl
        assert " Range" not in labels_excl

        q_med_only = "Generate a Table 1 showing only Age and HbA1c with median only"
        assert has_custom_table1_modifiers(q_med_only) is True
        sql_med_only = build_dynamic_table1_sql(q_med_only, db_path=str(db_file))
        labels_med_only = [r[0] for r in conn.execute(sql_med_only).fetchall()]
        assert " Median (IQR)" in labels_med_only
        assert " Mean\u2009±\u2009SD" not in labels_med_only

        # Test schema_map routing to jaeb_beck2023_cohort and missing optional columns fallback
        alt_schema = {
            "jaeb_beck2023_cohort": ["age", "gender", "TxGroup", "hba1c", "tir", "tbr"]
        }
        sql_alt = build_dynamic_table1_sql(
            "Make a Table 1 showing only Age and HbA1c",
            schema_map=alt_schema,
            db_path=str(db_file),
        )
        assert '"jaeb_beck2023_cohort"' in sql_alt
        assert "TRY_CAST(NULL AS DOUBLE)" in sql_alt

        # Test db_path connection error fallback in _is_categorical_column_in_db
        with patch("duckdb.connect", side_effect=duckdb.Error("simulated failure")):
            spec_err = parse_nl_table1_spec(
                "Table 1 stratified by sex showing bmi",
                schema_map={"patients": ["age", "bmi"]},
                db_path=str(db_file),
            )
            assert spec_err.strata[0].column_alias == "Overall"
            assert Table1StratumSpec("x", "Custom", "1=1").column_alias == "Custom"

        # Test dynamic schema column introspection for both categorical string and numeric columns on live DB
        cat_db = tmp_path / "cat_schema.duckdb"
        c_cat = duckdb.connect(str(cat_db))
        c_cat.execute(
            "CREATE TABLE patients AS SELECT 1 AS subject_id, 25 AS age, 25.0 AS age_exact, "
            "'F' AS gender, 'Control-IQ' AS TxGroup, 'DCLP3' AS study, 'White non-Hispanic' AS race_ethnicity, "
            "'>= $75,000' AS annual_income, 10.0 AS t1d_duration_years, 'Pump' AS prestudy_insulin_delivery, "
            "'Yes' AS prestudy_cgm_use, 7.2 AS hba1c, 70.0 AS tir, 1.5 AS tbr, 22.4 AS bmi, 'Tandem' AS pump_brand"
        )
        c_cat.close()
        cat_schema_map = {
            "patients": [
                "subject_id",
                "age",
                "gender",
                "TxGroup",
                "hba1c",
                "tir",
                "tbr",
                "bmi",
                "pump_brand",
            ]
        }
        sql_cat_num = build_dynamic_table1_sql(
            "Make a Table 1 showing only age, bmi, and pump_brand",
            schema_map=cat_schema_map,
            db_path=str(cat_db),
        )
        c_cat_ro = duckdb.connect(str(cat_db), read_only=True)
        cat_rows = [r[0] for r in c_cat_ro.execute(sql_cat_num).fetchall()]
        c_cat_ro.close()
        assert "Bmi" in cat_rows
        assert "Pump Brand, n (%)" in cat_rows
        assert " Tandem" in cat_rows
        assert 'GROUP BY CAST("pump_brand" AS VARCHAR)' in sql_cat_num
        sql_cat_no_db = build_dynamic_table1_sql(
            "Make a Table 1 showing only age and bmi",
            schema_map=cat_schema_map,
            db_path=None,
        )
        assert 'TRY_CAST("bmi" AS DOUBLE)' in sql_cat_no_db

        # Test custom numeric cut-point bins (age, hba1c, tir, tbr, duration)
        q_cutpoints = (
            "Make a Table 1 with age bins 10, 20, 40 and hba1c cutpoints 6.5, 7.5, 9.0 "
            "and tir brackets 50, 70 and tbr tiers 1.5, 3.0 and duration bins 2, 5, 15"
        )
        assert has_custom_table1_modifiers(q_cutpoints) is True
        sql_cutpoints = build_dynamic_table1_sql(q_cutpoints, db_path=str(db_file))
        rows_cut = [r[0] for r in conn.execute(sql_cutpoints).fetchall()]
        assert " <10 years, n (%)" in rows_cut
        assert " 10 to <20 years, n (%)" in rows_cut
        assert " ≥40 years, n (%)" in rows_cut
        assert " <6.5%, n (%)" in rows_cut
        assert " 6.5% to <7.5%, n (%)" in rows_cut
        assert " ≥9%, n (%)" in rows_cut
        assert " <50%, n (%)" in rows_cut
        assert " ≥70%, n (%)" in rows_cut
        assert " <1.5%, n (%)" in rows_cut
        assert " <2 years, n (%)" in rows_cut

        # Test extract_custom_numeric_cutpoints edge cases (empty numbers)
        from t1d_analytics.table1_builder import (
            db_has_extended_jaeb_tables,
            db_has_raw_jaeb_tables,
            extract_custom_numeric_cutpoints,
            format_table1_rows_markdown,
        )

        assert (
            extract_custom_numeric_cutpoints("age bins ...", "age", "age", "") is None
        )
        assert format_table1_rows_markdown([]) == ""
        md_tbl = format_table1_rows_markdown(
            [{"Characteristic": "Participants, N", "Overall": "100|A", "Empty": None}]
        )
        assert "| Characteristic | Overall | Empty |" in md_tbl
        assert "100\\|A" in md_tbl

        # Test raw JAEB study tables & extended JAEB tables CTE compilation
        raw_db = tmp_path / "raw_jaeb.duckdb"
        c_raw = duckdb.connect(str(raw_db))
        c_raw.execute(
            "CREATE TABLE dclp3_ptroster AS SELECT 1 AS PtID, '2020-01-01' AS EnrollDt, '2020-01-15' AS RandDt, 'CLC' AS trtGroup"
        )
        c_raw.execute(
            "CREATE TABLE dclp3_diabscreening AS SELECT 1 AS PtID, 25 AS AgeAtEnrollment, 'F' AS Gender, 'Not Hispanic' AS Ethnicity, 'White' AS Race, '2010-01-01' AS DiagDt, 15 AS DiagAge, 1 AS InsModPump, 'Current' AS CGMUseStat"
        )
        c_raw.execute(
            "CREATE TABLE dclp3_diabsocioecon AS SELECT 1 AS PtID, '$100,000 to less than $200,000' AS AnnualIncome"
        )
        c_raw.execute(
            "CREATE TABLE dclp3_sampleresults AS SELECT 1 AS PtID, 'Randomization' AS Visit, 'GLYHB' AS ResultName, 7.2 AS Value"
        )
        c_raw.execute(
            "CREATE TABLE dclp3_cgm AS SELECT 1 AS PtID, '1. Baseline' AS Period, '2020-01-10' AS DataDtTm, 140 AS CGM FROM range(2100)"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_ptroster AS SELECT 2 AS PtID, '2020-01-01' AS EnrollDt, '2020-01-15' AS RandDt, 'SAP' AS trtGroup, 10 AS AgeAtEnrollment"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_diabscreening AS SELECT 2 AS PtID, 'M' AS Gender, 'Hispanic or Latino' AS Ethnicity, 'White' AS Race, '2015-01-01' AS DiagDt, 5 AS DiagAge, 0 AS InsModPump, 'Past' AS CGMUseStat"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_diabsocioecon AS SELECT 2 AS PtID, 'Less than $25,000' AS AnnualIncome"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_sampleresults AS SELECT 2 AS PtID, 'Baseline' AS Visit, 'GLYHB' AS ResultName, 8.1 AS Value"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_dexcomclaritycgm AS SELECT 2 AS PtID, TIMESTAMP '2020-01-01 00:00:00' + INTERVAL (i * 5) MINUTE AS DataDtTm, NULL AS DataDtTm_adj, 160 AS CGM FROM range(2100) t(i)"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_othercgm AS SELECT 2 AS PtID, TIMESTAMP '2020-01-01 00:00:00' AS DataDtTm, 150 AS CGM, FALSE AS IsCalibration"
        )
        c_raw.execute(
            "CREATE TABLE dclp5_tandemcgmdatagxb_b AS SELECT 2 AS PtID, TIMESTAMP '2020-01-01 00:00:00' AS DataDtTm, NULL AS DataDtTm_adjusted, 155 AS CGMValue, 0 AS HighLowIndicator"
        )
        c_raw.execute(
            "CREATE TABLE pedap_ptroster AS SELECT 3 AS PtID, '2020-01-01' AS EnrollDt, '2020-01-15 12:00:00' AS RandDt, 'CLC' AS TrtGroup, 4 AS AgeAsofEnrollDt"
        )
        c_raw.execute(
            "CREATE TABLE pedap_diabscreening AS SELECT 3 AS PtID, 'F' AS Sex, 'Not Hispanic' AS Ethnicity, 'Asian' AS Race, '2018-01-01' AS DiagDt, 2 AS DiagAge, 1 AS InsModPump, 'Current' AS CGMUseStat"
        )
        c_raw.execute(
            "CREATE TABLE pedap_diabsocioecon AS SELECT 3 AS PtID, 'Do not wish to provide' AS AnnualIncome"
        )
        c_raw.execute(
            "CREATE TABLE pedap_stasampleresults AS SELECT 3 AS PtID, 'Randomization' AS Visit, 'GLYHB' AS ResultName, 7.5 AS Value, 'Complete' AS STAResultStatus"
        )
        c_raw.execute(
            "CREATE TABLE pedap_dexcomclaritycgm AS SELECT 3 AS PtID, TIMESTAMP '2020-01-01 00:00:00' + INTERVAL (i * 5) MINUTE AS DeviceDtTm, 130 AS CGM FROM range(2100) t(i)"
        )
        c_raw.execute(
            "CREATE TABLE pedap_othercgm AS SELECT 3 AS PtID, TIMESTAMP '2020-01-01 00:00:00' AS DeviceDtTm, 135 AS CGM, FALSE AS IsCalibration"
        )
        c_raw.execute(
            "CREATE TABLE pedap_tandemcgmdatagxb AS SELECT 3 AS PtID, TIMESTAMP '2020-01-01 00:00:00' AS DeviceDtTm, 140 AS CGMValue, 0 AS HighLowIndicator"
        )
        c_raw.execute(
            "CREATE TABLE tblaptsummary AS SELECT 10 AS ptid, 'F' AS gender, 30 AS AgeAsOfRandDt, 'White' AS race, 'Not Hispanic or Latino' AS ethnic, 12.0 AS DavDiabetes, 'Pump' AS InsulinModality, 0 AS NumSevHypo, 'RT-CGM' AS TxGroup"
        )
        c_raw.execute(
            "CREATE TABLE tblalabhba1c AS SELECT 10 AS ptid, 'Randomization' AS visit, 7.4 AS LabA1cResult"
        )
        c_raw.execute(
            "CREATE TABLE hptroster AS SELECT 20 AS ptid, 'CGM Only' AS trtgroup, 45 AS AgeAsOfEnrollDt"
        )
        c_raw.execute(
            "CREATE TABLE hscreening AS SELECT 20 AS ptid, 'M' AS gender, 'Not Hispanic or Latino' AS ethnicity, 'White' AS race"
        )
        c_raw.execute(
            "CREATE TABLE hlocalhba1c AS SELECT 20 AS ptid, 'Randomization' AS visit, 7.1 AS HbA1cTestRes"
        )
        c_raw.close()

        raw_schema_map = {
            t: ["PtID"]
            for t in (
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
                "tblaptsummary",
                "tblalabhba1c",
                "hptroster",
                "hscreening",
                "hlocalhba1c",
            )
        }
        assert db_has_raw_jaeb_tables(None, schema_map=raw_schema_map) is True
        assert db_has_raw_jaeb_tables(str(raw_db), schema_map=None) is True
        assert db_has_raw_jaeb_tables(None, schema_map={"patients": ["age"]}) is False
        assert db_has_extended_jaeb_tables(None, schema_map=raw_schema_map) is True
        assert db_has_extended_jaeb_tables(str(raw_db), schema_map=None) is True
        assert (
            db_has_extended_jaeb_tables(None, schema_map={"patients": ["age"]}) is False
        )
        with patch("duckdb.connect", side_effect=duckdb.Error("fail")):
            assert db_has_raw_jaeb_tables(str(raw_db)) is False
            assert db_has_extended_jaeb_tables(str(raw_db)) is False

        assert (
            has_custom_table1_modifiers("Reproduce Table 1 from raw jaeb tables")
            is True
        )
        sql_raw = build_dynamic_table1_sql(
            "Reproduce Table 1 from the Beck 2023 Control-IQ meta-analysis.",
            db_path=str(raw_db),
        )
        assert "raw_jaeb_cohort AS (" in sql_raw
        c_raw_ro = duckdb.connect(str(raw_db), read_only=True)
        raw_res = c_raw_ro.execute(sql_raw).fetchall()
        assert raw_res[0] == ("Participants, N", "3", "2", "1")

        # Also test extended studies across all JAEB studies stratified by study
        sql_ext_studies = build_dynamic_table1_sql(
            "Make a Table 1 across all jaeb studies stratified by study showing only age and hba1c",
            schema_map=raw_schema_map,
            db_path=str(raw_db),
        )
        assert "JDRF_CGM" in sql_ext_studies
        assert "REPLACE_BG" in sql_ext_studies
        ext_res = c_raw_ro.execute(sql_ext_studies).fetchall()
        assert ext_res[0] == ("Participants, N", "5", "1", "1", "1", "1", "1")
        c_raw_ro.close()

        # Also verify lookup_table_1_curated_sql with db_path pointing to raw_db
        curated_raw = lookup_table_1_curated_sql(
            "Reproduce Table 1 from the Beck 2023 Control-IQ meta-analysis.",
            db_path=str(raw_db),
        )
        assert curated_raw is not None and "raw_jaeb_cohort AS (" in curated_raw
        curated_raw_general = lookup_table_1_curated_sql(
            "Make Table 1 for the JAEB Control-IQ meta-analysis",
            db_path=str(raw_db),
        )
        assert (
            curated_raw_general is not None
            and "raw_jaeb_cohort AS (" in curated_raw_general
        )

        # Test legacy Table 1 template prompt aliases
        nl = "\n"
        fb_legacy = synthesize_fallback_clinical_sql(
            f"Table: patients{nl}Columns: age (INT), TxGroup (VARCHAR), hba1c (DOUBLE), tir (DOUBLE){nl}",
            "Build an age-stratified Table 1 across Pediatric (<18 yr), Adult (18-49 yr), and Older Adult (>=50 yr) cohorts.",
        )
        assert "age_stratum" in fb_legacy.sql_query

        # Test synthesize_fallback_clinical_sql with rich patients schema and custom Table 1 query
        rich_schema = (
            f"Table: patients{nl}"
            "Columns: subject_id (BIGINT), study (VARCHAR), age (BIGINT), age_exact (DOUBLE), "
            "gender (VARCHAR), TxGroup (VARCHAR), race_ethnicity (VARCHAR), annual_income (VARCHAR), "
            "t1d_duration_years (DOUBLE), prestudy_insulin_delivery (VARCHAR), prestudy_cgm_use (VARCHAR), "
            f"hba1c (DOUBLE), tir (DOUBLE), tbr (DOUBLE){nl}"
        )
        fb_dyn = synthesize_fallback_clinical_sql(
            rich_schema,
            "Generate a Table 1 for DCLP3 participants only showing only Age and HbA1c",
            db_path=str(db_file),
        )
        assert "UPPER(study) = 'DCLP3'" in fb_dyn.sql_query
        assert len(conn.execute(fb_dyn.sql_query).fetchall()) > 5

        # Also test non-patients table fallback Table 1 (with and without group column)
        simple_schema = (
            f"Table: custom_cohort{nl}"
            f"Columns: id (INT), age (DOUBLE), hba1c (DOUBLE), tir (DOUBLE){nl}"
        )
        fb_simple = synthesize_fallback_clinical_sql(
            simple_schema, "Generate a cohort summary table 1"
        )
        assert 'FROM "custom_cohort"' in fb_simple.sql_query

        grp_schema = (
            f"Table: custom_cohort{nl}"
            f"Columns: id (INT), TxGroup (VARCHAR), age (DOUBLE), hba1c (DOUBLE), tir (DOUBLE){nl}"
        )
        fb_grp = synthesize_fallback_clinical_sql(
            grp_schema, "Generate a cohort summary table 1"
        )
        assert 'GROUP BY "TxGroup"' in fb_grp.sql_query
    finally:
        conn.close()


def test_clinical_templates_catalog_endpoint_and_parity() -> None:
    """Verify GET /api/templates returns all 13 canonical starter and Table 1 templates matching clinical_query_catalog."""
    from fastapi.testclient import TestClient

    from t1d_analytics.api import create_app
    from t1d_analytics.clinical_query_catalog import get_clinical_query_catalog

    catalog = get_clinical_query_catalog()
    assert len(catalog) == 13
    client = TestClient(create_app())
    resp = client.get("/api/templates")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["templates"]) == 13
    ids = [item["id"] for item in payload["templates"]]
    assert ids[:5] == [f"starter-{i}" for i in range(1, 6)]
    assert "table1-beck-2023" in ids
