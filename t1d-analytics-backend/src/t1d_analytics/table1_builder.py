"""Natural-language clinical Table 1 parser and orchestrator."""

import itertools
import re
from collections.abc import Mapping, Sequence

from t1d_analytics.jaeb_cohort_sql import (
    build_raw_jaeb_cohort_cte_sql as build_raw_jaeb_cohort_cte_sql,
)
from t1d_analytics.sql_utils import escape_sql_identifier
from t1d_analytics.table1_formatter import (
    format_table1_rows_markdown as format_table1_rows_markdown,
)
from t1d_analytics.table1_specs import (
    BECK_2023_DEFAULT_ROW_KEYS as BECK_2023_DEFAULT_ROW_KEYS,
)
from t1d_analytics.table1_specs import (
    CategoryBinSpec as CategoryBinSpec,
)
from t1d_analytics.table1_specs import (
    ContinuousSummaryFormat as ContinuousSummaryFormat,
)
from t1d_analytics.table1_specs import (
    RowVariableKind as RowVariableKind,
)
from t1d_analytics.table1_specs import (
    Table1QuerySpec as Table1QuerySpec,
)
from t1d_analytics.table1_specs import (
    Table1RowSpec as Table1RowSpec,
)
from t1d_analytics.table1_specs import (
    Table1StratumSpec as Table1StratumSpec,
)
from t1d_analytics.table1_specs import (
    build_beck_2023_row_specs as build_beck_2023_row_specs,
)
from t1d_analytics.table1_specs import (
    db_has_extended_jaeb_tables as db_has_extended_jaeb_tables,
)
from t1d_analytics.table1_specs import (
    db_has_raw_jaeb_tables as db_has_raw_jaeb_tables,
)
from t1d_analytics.table1_specs import (
    is_categorical_column_in_db as is_categorical_column_in_db,
)
from t1d_analytics.table1_sql_compiler import (
    compile_table1_sql as compile_table1_sql,
)

_KNOWN_JAEB_STUDY_CODES: tuple[str, ...] = (
    "dclp3",
    "dclp5",
    "pedap",
    "jdrf_cgm",
    "replace_bg",
    "flair",
    "loop",
    "wisdm",
    "city",
    "t1dexi",
    "sence",
)


def _extract_study_filters(lower_q: str) -> list[str]:
    """
    Extract clinical trial inclusion and exclusion SQL predicates from `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        List[str]: SQL `WHERE` predicates for `study` inclusion or exclusion.

    """
    clauses: list[str] = []
    study_alt = "|".join(_KNOWN_JAEB_STUDY_CODES)
    excluded_studies = [
        f"'{m.group(1).upper()}'"
        for m in re.finditer(
            rf"\b(?:excluding|except|without)\s+({study_alt})\b",
            lower_q,
        )
    ]
    if excluded_studies:
        clauses.append(
            f"UPPER(study) != {excluded_studies[0]}"
            if len(excluded_studies) == 1
            else f"UPPER(study) NOT IN ({', '.join(excluded_studies)})"
        )

    excluded_set = set(excluded_studies)
    mentioned_studies = [
        f"'{code.upper()}'"
        for code in _KNOWN_JAEB_STUDY_CODES
        if re.search(rf"\b{code}\b", lower_q)
        and f"'{code.upper()}'" not in excluded_set
    ]
    is_full_beck_triad = set(mentioned_studies) == {
        "'DCLP3'",
        "'DCLP5'",
        "'PEDAP'",
    } and ("beck" in lower_q or "meta-analysis" in lower_q or "369" in lower_q)
    is_study_stratified = bool(
        re.search(
            r"\b(?:stratified|grouped|columns?)\s+by\s+(?:clinical\s+)?(?:study|trial)\b",
            lower_q,
        )
    )
    if mentioned_studies and not is_full_beck_triad and not is_study_stratified:
        if len(mentioned_studies) == 1:
            clauses.append(f"UPPER(study) = {mentioned_studies[0]}")
        else:
            clauses.append(f"UPPER(study) IN ({', '.join(mentioned_studies)})")
    return clauses


def _extract_study_and_age_filters(lower_q: str) -> list[str]:
    """
    Extract study/trial and age-range SQL filter predicates from `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        List[str]: SQL `WHERE` predicates for study and age subsets.

    """
    clauses: list[str] = _extract_study_filters(lower_q)

    between_age = re.search(
        r"\b(?:age[ds]?\s+between|aged|from\s+age)\s+(\d+)\s*(?:to|and|-|–)\s*(\d+)\b|\b(\d+)\s*[-–]\s*(\d+)\s*(?:years|yr|yo)\s+old\b",
        lower_q,
    )
    under_age = re.search(
        r"\b(?:age[ds]?\s+)?(?:under|below|less\s+than|younger\s+than)\s+(\d+)\s*(?:years|yr|yo|y/o)?\b|\bage[ds]?\s*(?:under|below|less\s+than|younger\s+than|<\s*)(\d+)\b",
        lower_q,
    )
    over_age = re.search(
        r"\b(?:age[ds]?\s+)?(?:over|above|at\s+least|older\s+than)\s+(\d+)\s*(?:years|yr|yo|y/o)?(?:\s+old|\s+and\s+older)?\b|(?:>=\s*|≥\s*|>\s*)(\d+)\s*(?:years|yr|yo|y/o)\b|\b(\d+)\+\s*(?:years|yr|yo)\b|\b(\d+)\s+and\s+older\b",
        lower_q,
    )
    is_age_stratified = bool(
        re.search(
            r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:\d+\s+)?(?:detailed\s+|beck\s+|pediatric\s+vs\.?\s+adult\s+)?age|\bpediatric\s+vs\.?\s+adult\s+columns\b",
            lower_q,
        )
    )
    if between_age and not is_age_stratified:
        lo_val = int(between_age.group(1) or between_age.group(3))
        hi_val = int(between_age.group(2) or between_age.group(4))
        clauses.append(f"age >= {lo_val} AND age <= {hi_val}")
    elif under_age and not is_age_stratified:
        age_val = int(under_age.group(1) or under_age.group(2))
        clauses.append(f"age < {age_val}")
    elif over_age and not is_age_stratified:
        age_over_val = int(
            over_age.group(1)
            or over_age.group(2)
            or over_age.group(3)
            or over_age.group(4)
        )
        clauses.append(f"age >= {age_over_val}")
    elif re.search(r"\b(?:toddlers?|young\s+children)\b", lower_q):
        clauses.append("age < 6")
    elif re.search(r"\bschool[- ]age(?:d)?\b", lower_q):
        clauses.append("age >= 6 AND age < 14")
    elif re.search(
        r"\b(?:adolescents?|teens?|teenagers?)\b", lower_q
    ) and not re.search(r"\badults?\b", lower_q):
        clauses.append("age >= 14 AND age < 18")
    elif re.search(r"\byoung\s+adults?\b", lower_q):
        clauses.append("age >= 18 AND age < 25")
    elif re.search(
        r"\b(?:pediatric|children|child|youth|kids)\b", lower_q
    ) and not re.search(r"\badults?\b", lower_q):
        clauses.append("age < 18")
    elif re.search(r"\bolder\s+adults?\b|\belderly\b|\bseniors?\b", lower_q):
        clauses.append("age >= 50")
    elif re.search(
        r"\badults?\s+only\b|\bonly\s+adults?\b|\badult\s+participants\b|\bfor\s+adults\b",
        lower_q,
    ):
        clauses.append("age >= 18")
    return clauses


def _extract_glycemic_threshold_clauses(lower_q: str) -> list[str]:
    """
    Extract continuous glycemic and clinical threshold predicates (`hba1c`, `tir`, `tbr`, `tar`, `mean_glucose`, `t1d_duration_years`).

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        List[str]: SQL `WHERE` predicates for clinical numeric bounds.

    """
    clauses: list[str] = []
    for col_name, between_pat in (
        (
            "hba1c",
            r"\b(?:hba1c|a1c)\s+between\s+(\d+(?:\.\d+)?)\s*(?:%?\s*(?:and|to|-|–)\s*)(\d+(?:\.\d+)?)",
        ),
        (
            "tir",
            r"\b(?:tir|time\s+in\s+range)\s+between\s+(\d+(?:\.\d+)?)\s*(?:%?\s*(?:and|to|-|–)\s*)(\d+(?:\.\d+)?)",
        ),
    ):
        mb = re.search(between_pat, lower_q)
        if mb:
            clauses.append(
                f"{col_name} >= {float(mb.group(1))} AND {col_name} <= {float(mb.group(2))}"
            )

    for col_name, pat, op in (
        (
            "hba1c",
            r"\b(?:hba1c|a1c)\s*(?:>=|≥|>|at\s+least|above|over)\s*(\d+(?:\.\d+)?)",
            ">=",
        ),
        (
            "hba1c",
            r"\b(?:hba1c|a1c)\s*(?:<|<=|≤|under|below|less\s+than)\s*(\d+(?:\.\d+)?)",
            "<",
        ),
        (
            "tir",
            r"\b(?:tir|time\s+in\s+range)\s*(?:<|<=|≤|under|below|less\s+than)\s*(\d+(?:\.\d+)?)",
            "<",
        ),
        (
            "tir",
            r"\b(?:tir|time\s+in\s+range)\s*(?:>=|≥|>|at\s+least|above|over)\s*(\d+(?:\.\d+)?)",
            ">=",
        ),
        (
            "tbr",
            r"\b(?:tbr|time\s*<\s*70|time\s+below\s+range)\s*(?:>=|≥|>|at\s+least|above|over)\s*(\d+(?:\.\d+)?)",
            ">=",
        ),
        (
            "tbr",
            r"\b(?:tbr|time\s*<\s*70|time\s+below\s+range)\s*(?:<|<=|≤|under|below|less\s+than)\s*(\d+(?:\.\d+)?)",
            "<",
        ),
        (
            "tar",
            r"\b(?:tar|time\s*>\s*180|time\s+above\s+range)\s*(?:>=|≥|>|at\s+least|above|over)\s*(\d+(?:\.\d+)?)",
            ">=",
        ),
        (
            "mean_glucose",
            r"\b(?:mean\s+glucose|average\s+glucose)\s*(?:>=|≥|>|at\s+least|above|over)\s*(\d+(?:\.\d+)?)",
            ">=",
        ),
        (
            "mean_glucose",
            r"\b(?:mean\s+glucose|average\s+glucose)\s*(?:<|<=|≤|under|below|less\s+than)\s*(\d+(?:\.\d+)?)",
            "<",
        ),
        (
            "t1d_duration_years",
            r"\bduration\s*(?:>=|≥|>|at\s+least|above|over)\s*(\d+(?:\.\d+)?)",
            ">=",
        ),
        (
            "t1d_duration_years",
            r"\bduration\s*(?:<|<=|≤|under|below|less\s+than)\s*(\d+(?:\.\d+)?)",
            "<",
        ),
    ):
        m = re.search(pat, lower_q)
        if m and not any(
            c.startswith(f"{col_name} >=") and "AND" in c for c in clauses
        ):
            clauses.append(f"{col_name} {op} {float(m.group(1))}")

    if re.search(r"\bnewly\s+diagnosed\b", lower_q) and not any(
        "t1d_duration_years" in c for c in clauses
    ):
        clauses.append("t1d_duration_years < 1.0")
    return clauses


def _extract_glycemic_and_therapy_filters(lower_q: str) -> list[str]:
    """
    Extract glycemic (`hba1c`, `tir`, `tbr`, `tar`, `mean_glucose`, `t1d_duration_years`) and therapy SQL predicates from `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        List[str]: SQL `WHERE` predicates for glycemic and therapy subsets.

    """
    is_gly_strat = bool(
        re.search(
            r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:baseline\s+)?(?:hba1c|a1c|tir|time\s+in\s+range)",
            lower_q,
        )
    )
    clauses: list[str] = (
        [] if is_gly_strat else _extract_glycemic_threshold_clauses(lower_q)
    )

    if not re.search(
        r"\b(?:stratified|grouped|columns?)\s+by\s+(?:prestudy\s+)?(?:insulin|delivery|pump|mdi)\b|\bcomparing\s+(?:prestudy\s+)?(?:insulin\s+)?pump\s+(?:vs\.?|and)\s+mdi\b",
        lower_q,
    ):
        if re.search(
            r"\b(?:mdi\s+users|on\s+mdi|only\s+mdi|prestudy\s+mdi\s+use|using\s+injections)\b",
            lower_q,
        ):
            clauses.append("prestudy_insulin_delivery = 'MDI'")
        elif re.search(r"\b(?:pump\s+users|on\s+pump|only\s+pump)\b", lower_q):
            clauses.append("prestudy_insulin_delivery = 'Pump'")

    if not re.search(
        r"\b(?:stratified|grouped|columns?)\s+by\s+(?:sex|gender)\b", lower_q
    ):
        if re.search(
            r"\b(?:female\s+participants|females\s+only|only\s+females|women\s+only|for\s+females|for\s+women|for\s+girls)\b",
            lower_q,
        ):
            clauses.append("gender IN ('Female', 'F')")
        elif re.search(
            r"\b(?:male\s+participants|males\s+only|only\s+males|men\s+only|for\s+males|for\s+men|for\s+boys)\b",
            lower_q,
        ):
            clauses.append("gender IN ('Male', 'M')")

    if re.search(
        r"\b(?:only\s+control-iq|in\s+the\s+control-iq\s+group|closed[- ]loop\s+users)\b",
        lower_q,
    ):
        clauses.append("TxGroup IN ('Control-IQ', 'Closed-Loop', 'CLC')")
    elif re.search(
        r"\b(?:only\s+control\s+group|in\s+the\s+control\s+group)\b", lower_q
    ):
        clauses.append("TxGroup IN ('Control', 'SAP', 'SC')")
    return clauses


def _extract_race_ethnicity_filters(lower_q: str) -> list[str]:
    """
    Extract race and ethnicity subset SQL predicates from `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        List[str]: SQL `WHERE` predicates for race/ethnicity subsets.

    """
    for pat, sql_pred in (
        (
            r"\b(?:white\s+non-hispanic\s+participants|only\s+white\s+non-hispanic)\b",
            "race_ethnicity = 'White non-Hispanic'",
        ),
        (
            r"\b(?:black\s+(?:non-hispanic\s+)?participants|only\s+black)\b",
            "race_ethnicity = 'Black non-Hispanic'",
        ),
        (
            r"\b(?:hispanic(?:\s+or\s+latino)?\s+participants|only\s+hispanic)\b",
            "race_ethnicity = 'Hispanic or Latino'",
        ),
        (r"\b(?:asian\s+participants|only\s+asian)\b", "race_ethnicity = 'Asian'"),
        (
            r"\b(?:minority(?:\s+participants)?|minoritized|non-white(?:\s+participants)?)\b",
            "race_ethnicity != 'White non-Hispanic' AND race_ethnicity IS NOT NULL",
        ),
    ):
        if re.search(pat, lower_q):
            return [sql_pred]
    return []


def _extract_demographic_and_ses_filters(lower_q: str) -> list[str]:
    """
    Extract socioeconomic, prestudy CGM, race/ethnicity, and non-missingness SQL predicates from `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        List[str]: SQL `WHERE` predicates for demographic and SES subsets.

    """
    clauses: list[str] = []
    if re.search(
        r"\b(?:cgm[- ]naive|no\s+prestudy\s+cgm|without\s+prior\s+cgm)\b", lower_q
    ):
        clauses.append("prestudy_cgm_use = 'No'")
    elif re.search(
        r"\b(?:prestudy\s+cgm\s+users|prior\s+cgm(?:\s+users)?|cgm\s+users)\b",
        lower_q,
    ):
        clauses.append("prestudy_cgm_use = 'Yes'")

    if re.search(
        r"\b(?:income\s+(?:under|below|<)\s*\$?75,?000|low[- ]income)\b", lower_q
    ):
        clauses.append("annual_income = '< $75,000'")
    elif re.search(
        r"\b(?:income\s+(?:>=|≥|at\s+least|over)\s*\$?75,?000|high[- ]income)\b",
        lower_q,
    ):
        clauses.append("annual_income IN ('>= $75,000', '≥ $75,000')")

    clauses.extend(_extract_race_ethnicity_filters(lower_q))

    if re.search(
        r"\b(?:complete\s+cases|excluding\s+missing|non-missing\s+income)\b", lower_q
    ):
        clauses.append(
            "annual_income IS NOT NULL AND annual_income != 'Missing' AND hba1c IS NOT NULL AND tir IS NOT NULL"
        )
    return clauses


def _extract_subset_where_clauses(lower_q: str) -> tuple[list[str], int | None]:
    """
    Parse natural-language cohort filter predicates (`WHERE` clauses) and optional `LIMIT` from `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        Tuple[List[str], Optional[int]]: `(where_clauses, limit_rows)`.

    """
    where_clauses = [
        *_extract_study_and_age_filters(lower_q),
        *_extract_glycemic_and_therapy_filters(lower_q),
        *_extract_demographic_and_ses_filters(lower_q),
    ]
    limit_match = re.search(
        r"\b(?:first|top|limit|sample\s+of|subset\s+of)\s+(\d+)\s*(?:patients|participants|subjects|rows)?\b",
        lower_q,
    )
    limit_rows = int(limit_match.group(1)) if limit_match else None
    return where_clauses, limit_rows


def _extract_clinical_tier_strata(
    lower_q: str,
    prefix: tuple[Table1StratumSpec, ...],
) -> tuple[Table1StratumSpec, ...] | None:
    """
    Check for multi-tier age, race/ethnicity, HbA1c, or TIR column stratifications in `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.
        prefix: Leading `Overall` stratum tuple (`(overall,)` or `()` when omitted).

    Returns:
        Optional[Tuple[Table1StratumSpec, ...]]: Matched strata tuple or `None`.

    """
    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:5|five|detailed|beck)\s+age(?:\s+groups?|\s+brackets?)?\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("age_u6", "<6 yr", "age < 6"),
            Table1StratumSpec("age_6_13", "6–<14 yr", "age >= 6 AND age < 14"),
            Table1StratumSpec("age_14_24", "14–<25 yr", "age >= 14 AND age < 25"),
            Table1StratumSpec("age_25_49", "25–<50 yr", "age >= 25 AND age < 50"),
            Table1StratumSpec("age_50p", "≥50 yr", "age >= 50"),
        )
    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:3|three)\s+age\b|\b18\s*[-–to]+\s*49\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("age_ped", "<18 yr", "age < 18"),
            Table1StratumSpec("age_mid", "18–49 yr", "age >= 18 AND age < 50"),
            Table1StratumSpec("age_sen", "≥50 yr", "age >= 50"),
        )
    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:race|ethnicity)\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec(
                "race_wnh",
                "White non-Hispanic",
                "race_ethnicity = 'White non-Hispanic'",
            ),
            Table1StratumSpec(
                "race_min",
                "Racial/Ethnic Minority",
                "race_ethnicity != 'White non-Hispanic' AND race_ethnicity IS NOT NULL",
            ),
        )
    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:baseline\s+)?(?:hba1c|a1c)(?:\s+tier|\s+group|\s+category)?\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("a1c_low", "HbA1c <7.0%", "hba1c < 7.0"),
            Table1StratumSpec(
                "a1c_mid", "HbA1c 7.0–<8.0%", "hba1c >= 7.0 AND hba1c < 8.0"
            ),
            Table1StratumSpec("a1c_high", "HbA1c ≥8.0%", "hba1c >= 8.0"),
        )
    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:baseline\s+)?(?:tir|time\s+in\s+range)(?:\s+tier|\s+group|\s+category)?\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("tir_low", "TIR <50%", "tir < 50.0"),
            Table1StratumSpec("tir_mid", "TIR 50–<70%", "tir >= 50.0 AND tir < 70.0"),
            Table1StratumSpec("tir_high", "TIR ≥70%", "tir >= 70.0"),
        )
    return None


def _extract_strata_specs(lower_q: str) -> tuple[Table1StratumSpec, ...]:
    """
    Parse column stratification specifications (`strata`) from a natural-language query.

    Args:
        lower_q: Lowercased natural-language query string.

    Returns:
        Tuple[Table1StratumSpec, ...]: Ordered tuple of `Table1StratumSpec`.

    """
    overall = Table1StratumSpec(key_id="all", label="Overall", predicate_sql="1=1")
    if re.search(r"\b(?:unstratified|overall\s+only|single\s+column)\b", lower_q):
        return (overall,)

    omit_overall = bool(
        re.search(
            r"\b(?:without\s+overall|no\s+overall\s+column|arms\s+only)\b", lower_q
        )
    )
    prefix: tuple[Table1StratumSpec, ...] = () if omit_overall else (overall,)

    tier_strata = _extract_clinical_tier_strata(lower_q, prefix)
    if tier_strata is not None:
        return tier_strata

    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:clinical\s+)?(?:study|trial|protocol)\b",
        lower_q,
    ) or re.search(r"\bacross\s+studies\b|\bby\s+study\s+columns\b", lower_q):
        ext_strata: tuple[Table1StratumSpec, ...] = (
            (
                Table1StratumSpec("jdrf_cgm", "JDRF_CGM", "UPPER(study) = 'JDRF_CGM'"),
                Table1StratumSpec(
                    "replace_bg", "REPLACE_BG", "UPPER(study) = 'REPLACE_BG'"
                ),
            )
            if re.search(
                r"\b(?:all\s+jaeb\s+studies|expanded\s+jaeb|jdrf_cgm|replace_bg)\b",
                lower_q,
            )
            else ()
        )
        return (
            *prefix,
            Table1StratumSpec("dclp3", "DCLP3", "UPPER(study) = 'DCLP3'"),
            Table1StratumSpec("dclp5", "DCLP5", "UPPER(study) = 'DCLP5'"),
            Table1StratumSpec("pedap", "PEDAP", "UPPER(study) = 'PEDAP'"),
            *ext_strata,
        )

    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:sex|gender)\b|\bcomparing\s+female\s+(?:vs\.?|and)\s+male\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("female", "Female", "gender IN ('Female', 'F')"),
            Table1StratumSpec("male", "Male", "gender IN ('Male', 'M')"),
        )

    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:prestudy\s+)?(?:insulin\s+delivery|delivery\s+modality|pump\s+vs\.?\s+mdi)\b|\bcomparing\s+(?:prestudy\s+)?(?:insulin\s+)?pump\s+(?:vs\.?|and)\s+mdi\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("pump", "Pump", "ins_deliv = 'Pump'"),
            Table1StratumSpec("mdi", "MDI", "ins_deliv = 'MDI'"),
        )

    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:prestudy\s+)?cgm(?:\s+use)?\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("cgm_yes", "Prestudy CGM", "cgm_use = 'Yes'"),
            Table1StratumSpec("cgm_no", "No Prestudy CGM", "cgm_use = 'No'"),
        )

    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+(?:annual\s+)?(?:household\s+)?income\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("inc_low", "< $75,000", "annual_income = '< $75,000'"),
            Table1StratumSpec(
                "inc_high",
                "≥ $75,000",
                "annual_income IN ('>= $75,000', '≥ $75,000')",
            ),
        )

    if re.search(
        r"\b(?:stratified|grouped|columns?|split|broken\s+down)\s+by\s+age(?:\s+group|\s+stratum|\s+bracket)?\b|\bpediatric\s+vs\.?\s+adult(?:\s+columns|\s+groups)?\b",
        lower_q,
    ):
        return (
            *prefix,
            Table1StratumSpec("ped", "<18 years", "age < 18"),
            Table1StratumSpec("adult", "≥18 years", "age >= 18"),
        )

    return (
        *prefix,
        Table1StratumSpec(
            "ciq",
            "Control-IQ",
            "TxGroup IN ('Control-IQ', 'Closed-Loop', 'CLC')",
        ),
        Table1StratumSpec(
            "ctrl",
            "Control",
            "TxGroup IN ('Control', 'SAP', 'SC')",
        ),
    )


_ROW_KEYWORD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("age", r"\bage\b"),
    ("gender", r"\b(?:sex|gender|female|male)\b"),
    ("race_ethnicity", r"\b(?:race|ethnicity|racial)\b"),
    ("annual_income", r"\b(?:income|socioeconomic|ses)\b"),
    ("t1d_duration", r"\b(?:duration|diabetes\s+duration)\b"),
    (
        "prestudy_insulin_delivery",
        r"\b(?:insulin\s+delivery|pump\s+or\s+mdi|prestudy\s+insulin)\b",
    ),
    ("prestudy_cgm_use", r"\b(?:prestudy\s+cgm|cgm\s+use)\b"),
    ("hba1c", r"\b(?:hba1c|a1c|glycated\s+hemoglobin)\b"),
    ("tir", r"\b(?:time\s+in\s+range|tir|70[–-]180)\b"),
    ("tbr", r"\b(?:time\s*<\s*70|tbr|below\s+range\s*<\s*70|hypoglycemia\s*<\s*70)\b"),
    ("tbr_54", r"\b(?:time\s*<\s*54|tbr_54|<\s*54\s*mg/dl|level\s+2\s+hypo)\b"),
    ("tar", r"\b(?:time\s*>\s*180|tar|above\s+range|hyperglycemia\s*>\s*180)\b"),
    ("tar_250", r"\b(?:time\s*>\s*250|tar_250|>\s*250\s*mg/dl)\b"),
    ("mean_glucose", r"\b(?:mean\s+glucose|average\s+glucose|meancgm)\b"),
    ("cv", r"\b(?:coefficient\s+of\s+variation|\bcv\b|glycemic\s+variability)\b"),
    ("dka_events", r"\b(?:dka|ketoacidosis|numhospdka)\b"),
    ("severe_hypo_events", r"\b(?:severe\s+hypo|numsevhypo)\b"),
    ("study", r"\b(?:study\s+breakdown|trial\s+breakdown|clinical\s+trial\s+row)\b"),
    (
        "consensus_targets",
        r"\b(?:consensus\s+targets?|ada\s+goals?|target\s+achievement)\b",
    ),
)

_BUILTIN_SCHEMA_COLUMNS: set[str] = {
    "ptid",
    "subject_id",
    "patient_id",
    "study",
    "age",
    "age_exact",
    "ageasofranddt",
    "gender",
    "sex",
    "txgroup",
    "treatment_group",
    "race_ethnicity",
    "annual_income",
    "t1d_duration_years",
    "prestudy_insulin_delivery",
    "prestudy_cgm_use",
    "hba1c",
    "tir",
    "tbr",
    "tbr_54",
    "tar",
    "tar_250",
    "mean_glucose",
    "meancgm",
    "cv",
    "numhospdka",
    "pt_numhospdka",
    "dka_events",
    "numsevhypo",
    "severe_hypo_events",
    "severe_hypoglycemia_events",
}


def _apply_compact_row_mode(
    rows: Sequence[Table1RowSpec],
    lower_q: str,
) -> tuple[Table1RowSpec, ...]:
    """
    Apply row exclusions, missing-row suppression, range/median toggles, and compact mode to `rows`.

    Args:
        rows: Selected `Table1RowSpec` sequence.
        lower_q: Lowercased natural-language query string.

    Returns:
        Tuple[Table1RowSpec, ...]: Filtered and formatted row specifications.

    """
    excluded_row_keys: set[str] = set()
    for row_key, excl_pat in (
        ("race_ethnicity", r"\b(?:without|excluding|no|omit)\s+(?:race|ethnicity)"),
        (
            "annual_income",
            r"\b(?:without|excluding|no|omit)\s+(?:annual\s+)?(?:household\s+)?income",
        ),
        ("prestudy_cgm_use", r"\b(?:without|excluding|no|omit)\s+(?:prestudy\s+)?cgm"),
    ):
        if re.search(excl_pat, lower_q):
            excluded_row_keys.add(row_key)

    filtered_rows = [r for r in rows if r.key not in excluded_row_keys]
    is_compact = bool(
        re.search(
            r"\b(?:compact|without\s+bins|no\s+bins|summary\s+only|concise|means?\s+only)\b",
            lower_q,
        )
    )
    no_missing = bool(
        re.search(r"\b(?:without\s+missing\s+rows|no\s+missing\s+rows)\b", lower_q)
    )
    no_range = bool(re.search(r"\b(?:without\s+range|no\s+range)\b", lower_q))
    median_only = bool(re.search(r"\b(?:median\s+only|only\s+median)\b", lower_q))

    if not (is_compact or no_missing or no_range or median_only):
        return tuple(filtered_rows)

    result: list[Table1RowSpec] = []
    for r in filtered_rows:
        fmts = r.summary_formats
        if median_only:
            fmts = ("median_iqr",)
        elif no_range:
            fmts = tuple(f for f in fmts if f != "range") or ("mean_sd",)
        result.append(
            Table1RowSpec(
                key=r.key,
                header_label=r.header_label,
                value_expr=r.value_expr,
                range_expr=r.range_expr,
                null_check_expr=r.null_check_expr,
                var_type="continuous"
                if (is_compact and r.var_type == "hybrid")
                else r.var_type,
                decimals=r.decimals,
                unit_suffix=r.unit_suffix,
                summary_formats=fmts,
                bins=() if (is_compact and r.var_type == "hybrid") else r.bins,
                include_missing_row=False
                if (is_compact or no_missing)
                else r.include_missing_row,
                dynamic_category_expr=r.dynamic_category_expr,
            )
        )
    return tuple(result)


def extract_custom_numeric_cutpoints(
    lower_q: str,
    var_key: str,
    col_expr: str,
    unit_label: str,
) -> tuple[CategoryBinSpec, ...] | None:
    """
    Parse custom numeric cut-point thresholds for `var_key` from `lower_q` (e.g. `'age bins 10, 20, 40'`).

    Args:
        lower_q: Lowercased natural-language query string.
        var_key: Variable identifier (`'age'`, `'hba1c'`, `'tir'`, `'tbr'`, `'t1d_duration'`).
        col_expr: SQL column expression (`'age'`, `'hba1c'`, `'tir'`, `'tbr'`, `'dur'`).
        unit_label: Unit suffix for bin labels (`' years'`, `'%'`, etc.).

    Returns:
        Optional[Tuple[CategoryBinSpec, ...]]: Custom `CategoryBinSpec` tuple if specified, else `None`.

    """
    key_patterns = {
        "age": r"age",
        "hba1c": r"(?:hba1c|a1c)",
        "tir": r"(?:tir|time\s+in\s+range)",
        "tbr": r"(?:tbr|time\s*<\s*70)",
        "t1d_duration": r"(?:duration|diabetes\s+duration)",
    }
    kw = key_patterns[var_key]
    m = re.search(
        rf"\b{kw}\s+(?:bins?|brackets?|cut[- ]?points?|thresholds?|tiers?)\s*(?:of|at|:)?\s*([0-9.,\s/|-]+)",
        lower_q,
    )
    if not m:
        return None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1))]
    cutpoints = sorted(set(nums))
    if not cutpoints:
        return None

    def _fmt_num(v: float) -> str:
        """
        Format a float cut-point as an integer string if whole, else 1-decimal string.

        Args:
            v: Numeric cut-point value.

        Returns:
            str: Formatted numeric label string.

        """
        return str(int(v)) if v.is_integer() else f"{v:.1f}"

    lo_unit = "%" if unit_label == "%" else ""
    bins: list[CategoryBinSpec] = []
    first = cutpoints[0]
    bins.append(
        CategoryBinSpec(
            label=f" <{_fmt_num(first)}{unit_label}, n (%)",
            sql_condition=f"{col_expr} < {first}",
        )
    )
    for lo, hi in itertools.pairwise(cutpoints):
        bins.append(
            CategoryBinSpec(
                label=f" {_fmt_num(lo)}{lo_unit} to <{_fmt_num(hi)}{unit_label}, n (%)",
                sql_condition=f"{col_expr} >= {lo} AND {col_expr} < {hi}",
            )
        )
    last = cutpoints[-1]
    bins.append(
        CategoryBinSpec(
            label=f" ≥{_fmt_num(last)}{unit_label}, n (%)",
            sql_condition=f"{col_expr} >= {last}",
        )
    )
    return tuple(bins)


def _extract_dynamic_schema_row_specs(
    lower_q: str,
    available_cols: Sequence[str] | None,
    summary_formats: tuple[ContinuousSummaryFormat, ...],
    db_path: str | None = None,
    target_table: str = "patients",
) -> tuple[Table1RowSpec, ...]:
    """
    Create dynamic `Table1RowSpec` objects for custom schema columns mentioned in `lower_q`.

    Uses pure SQL `GROUP BY` (`dynamic_category_expr`) for categorical string columns so
    distinct category labels and counts are discovered directly by DuckDB at query execution
    time without hardcoding category strings into the `SELECT` query.

    Args:
        lower_q: Lowercased natural-language query string.
        available_cols: Optional sequence of column names from `schema_map[target_table]`.
        summary_formats: Active continuous summary formats.
        db_path: Optional path to the active DuckDB database file for categorical value inspection.
        target_table: Target table name.

    Returns:
        Tuple[Table1RowSpec, ...]: Dynamic row specs for matched custom columns.

    """
    if not available_cols:
        return ()
    extra_specs: list[Table1RowSpec] = []
    for raw_col in available_cols:
        col_low = raw_col.lower()
        if col_low in _BUILTIN_SCHEMA_COLUMNS:
            continue
        col_phrase = col_low.replace("_", r"[\s_]+")
        if re.search(rf"\b{col_phrase}\b", lower_q):
            safe_col = escape_sql_identifier(raw_col)
            title_label = raw_col.replace("_", " ").title()
            is_cat = is_categorical_column_in_db(db_path, target_table, raw_col)
            if is_cat:
                extra_specs.append(
                    Table1RowSpec(
                        key=col_low,
                        header_label=f"{title_label}, n (%)",
                        value_expr="NULL",
                        range_expr="NULL",
                        null_check_expr=f'"{safe_col}"',
                        var_type="categorical",
                        bins=(),
                        dynamic_category_expr=f'"{safe_col}"',
                    )
                )
            else:
                extra_specs.append(
                    Table1RowSpec(
                        key=col_low,
                        header_label=title_label,
                        value_expr=f'TRY_CAST("{safe_col}" AS DOUBLE)',
                        range_expr=f'TRY_CAST("{safe_col}" AS DOUBLE)',
                        null_check_expr=f'"{safe_col}"',
                        var_type="continuous",
                        decimals=1,
                        summary_formats=summary_formats,
                    )
                )
    return tuple(extra_specs)


def _apply_custom_cutpoints_to_row_specs(
    rows: Sequence[Table1RowSpec],
    lower_q: str,
) -> tuple[Table1RowSpec, ...]:
    """
    Override standard cut-point bins on `rows` when custom numeric cut-points are requested in `lower_q`.

    Args:
        rows: Selected `Table1RowSpec` sequence.
        lower_q: Lowercased natural-language query string.

    Returns:
        Tuple[Table1RowSpec, ...]: Row specifications with any custom cut-point bins applied.

    """
    unit_map = {
        "age": ("age", " years"),
        "hba1c": ("hba1c", "%"),
        "tir": ("tir", "%"),
        "tbr": ("tbr", "%"),
        "t1d_duration": ("dur", " years"),
    }
    updated: list[Table1RowSpec] = []
    for r in rows:
        if r.key in unit_map:
            col_expr, unit_lbl = unit_map[r.key]
            custom_bins = extract_custom_numeric_cutpoints(
                lower_q, r.key, col_expr, unit_lbl
            )
            if custom_bins is not None:
                updated.append(
                    Table1RowSpec(
                        key=r.key,
                        header_label=r.header_label,
                        value_expr=r.value_expr,
                        range_expr=r.range_expr,
                        null_check_expr=r.null_check_expr,
                        var_type=r.var_type,
                        decimals=r.decimals,
                        unit_suffix=r.unit_suffix,
                        summary_formats=r.summary_formats,
                        bins=custom_bins,
                        include_missing_row=r.include_missing_row,
                        dynamic_category_expr=r.dynamic_category_expr,
                    )
                )
                continue
        updated.append(r)
    return tuple(updated)


def _extract_row_specs(
    lower_q: str,
    available_cols: Sequence[str] | None = None,
    db_path: str | None = None,
    target_table: str = "patients",
) -> tuple[Table1RowSpec, ...]:
    """
    Determine the ordered tuple of `Table1RowSpec` sections to include for `lower_q`.

    Args:
        lower_q: Lowercased natural-language query string.
        available_cols: Optional sequence of column names on the target table.
        db_path: Optional path to the active DuckDB database file.
        target_table: Target table name.

    Returns:
        Tuple[Table1RowSpec, ...]: Selected row specifications.

    """
    summary_formats: tuple[ContinuousSummaryFormat, ...] = (
        ("mean_sd", "median_iqr", "range")
        if re.search(r"\b(?:median|iqr|interquartile|non[- ]parametric)\b", lower_q)
        else ("mean_sd", "range")
    )
    catalog = build_beck_2023_row_specs(summary_formats=summary_formats)

    explicit_only = bool(
        re.search(
            r"\b(?:only|just|specifically|selected\s+rows|with\s+rows\s+for|showing\s+only)\b",
            lower_q,
        )
    )
    matched_keys: list[str] = []
    for key, pat in _ROW_KEYWORD_PATTERNS:
        if re.search(pat, lower_q):
            if (
                key == "gender"
                and re.search(
                    r"\b(?:stratified|grouped|columns?)\s+by\s+(?:sex|gender)\b",
                    lower_q,
                )
                and not re.search(
                    r"\b(?:showing|including|with|show|include)\b.*\b(?:sex|gender)\b",
                    lower_q,
                )
            ):
                continue
            if (
                key == "prestudy_insulin_delivery"
                and re.search(
                    r"\b(?:stratified|grouped|columns?)\s+by\s+(?:prestudy\s+)?insulin\s+delivery\b|\bcomparing\s+(?:prestudy\s+)?(?:insulin\s+)?pump\s+(?:vs\.?|and)\s+mdi\b",
                    lower_q,
                )
                and not re.search(
                    r"\b(?:showing|including|with|show|include)\b.*\b(?:insulin\s+delivery|pump|mdi)\b",
                    lower_q,
                )
            ):
                continue
            matched_keys.append(key)

    extra_keys = [k for k in matched_keys if k not in BECK_2023_DEFAULT_ROW_KEYS]
    dynamic_schema_rows = _extract_dynamic_schema_row_specs(
        lower_q,
        available_cols,
        summary_formats,
        db_path=db_path,
        target_table=target_table,
    )

    if explicit_only and (matched_keys or dynamic_schema_rows):
        base_sel = [catalog[k] for k in matched_keys] + list(dynamic_schema_rows)
        return _apply_custom_cutpoints_to_row_specs(
            _apply_compact_row_mode(base_sel, lower_q), lower_q
        )

    if (
        matched_keys
        and len(matched_keys) < 8
        and not (
            "beck" in lower_q
            or "all characteristics" in lower_q
            or "full table 1" in lower_q
        )
        and re.search(r"\b(?:showing|including|with|for|summarize)\b", lower_q)
        and not extra_keys
        and "participant characteristics at baseline" not in lower_q
    ):
        base_sel = [catalog[k] for k in matched_keys] + list(dynamic_schema_rows)
        return _apply_custom_cutpoints_to_row_specs(
            _apply_compact_row_mode(base_sel, lower_q), lower_q
        )

    selected_keys = list(BECK_2023_DEFAULT_ROW_KEYS)
    for ek in extra_keys:
        selected_keys.append(ek)
    base_sel = [catalog[k] for k in selected_keys] + list(dynamic_schema_rows)
    return _apply_custom_cutpoints_to_row_specs(
        _apply_compact_row_mode(base_sel, lower_q), lower_q
    )


def has_custom_table1_modifiers(
    nl_query: str,
    schema_map: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """
    Return True when a natural-language Table 1 query specifies custom subsets, columns, or rows.

    Used to distinguish standard full-cohort Beck et al. (2023) Table 1 requests from
    custom research Table 1 queries over smaller/larger datasets, different columns,
    or custom row variables.

    Args:
        nl_query: User's natural language question.
        schema_map: Optional mapping from table names to column names.

    Returns:
        bool: True if custom subset filters, stratification columns, or row selections are present.

    """
    lower_q = " ".join(nl_query.strip().lower().split())
    if re.search(
        r"\b(?:raw\s+jaeb|from\s+raw\s+tables|raw\s+study\s+tables|all\s+jaeb\s+studies|expanded\s+jaeb)\b",
        lower_q,
    ):
        return True
    where_clauses, limit_rows = _extract_subset_where_clauses(lower_q)
    if where_clauses or limit_rows is not None:
        return True
    strata = _extract_strata_specs(lower_q)
    if [st.key_id for st in strata] != ["all", "ciq", "ctrl"]:
        return True
    avail_cols: Sequence[str] | None = (
        next(iter(schema_map.values()), None) if schema_map else None
    )
    rows = _extract_row_specs(lower_q, available_cols=avail_cols)
    if tuple(r.key for r in rows) != BECK_2023_DEFAULT_ROW_KEYS:
        return True
    if re.search(
        r"\b(?:without\s+n\s+row|no\s+n\s+row|omit\s+participants\s+row|without\s+missing\s+rows|no\s+missing\s+rows|without\s+range|no\s+range)\b",
        lower_q,
    ):
        return True
    default_catalog = build_beck_2023_row_specs()
    if any(
        r.bins != default_catalog[r.key].bins for r in rows if r.key in default_catalog
    ):
        return True
    return any("median_iqr" in r.summary_formats for r in rows)


def parse_nl_table1_spec(
    nl_query: str,
    schema_map: Mapping[str, Sequence[str]] | None = None,
    db_path: str | None = None,
) -> Table1QuerySpec:
    """
    Parse a natural-language Table 1 request into a structured `Table1QuerySpec`.

    Supports arbitrary cohort subset filters (`WHERE`), arbitrary column stratifications,
    and arbitrary row variable selections. When raw JAEB study tables (`dclp3_*`, `dclp5_*`,
    `pedap_*`) exist in `db_path` or `schema_map` (or when `'raw tables'` / `'from raw'` is
    requested), compiles self-contained raw JAEB CTEs (`raw_jaeb_cohort`).

    Args:
        nl_query: Natural language user question.
        schema_map: Optional mapping from table names to column names.
        db_path: Optional path to the active DuckDB database for schema introspection.

    Returns:
        Table1QuerySpec: Structured specification ready for `compile_table1_sql()`.

    """
    lower_q = " ".join(nl_query.strip().lower().split())
    wants_extended = bool(
        re.search(
            r"\b(?:all\s+jaeb\s+studies|expanded\s+jaeb|jdrf_cgm|replace_bg)\b",
            lower_q,
        )
        and db_has_extended_jaeb_tables(db_path, schema_map=schema_map)
    )
    use_raw_jaeb = bool(
        re.search(
            r"\b(?:raw\s+jaeb|from\s+raw\s+tables|raw\s+study\s+tables)\b", lower_q
        )
        or db_has_raw_jaeb_tables(db_path, schema_map=schema_map)
    )
    raw_source_cte: str | None = (
        build_raw_jaeb_cohort_cte_sql(include_extended_studies=wants_extended)
        if use_raw_jaeb
        else None
    )
    target_table = "raw_jaeb_cohort" if use_raw_jaeb else "patients"
    if not use_raw_jaeb and schema_map:
        lower_to_actual = {t.lower(): t for t in schema_map}
        if (
            "jaeb_beck2023_cohort" in lower_to_actual
            and "patients" not in lower_to_actual
        ):
            target_table = lower_to_actual["jaeb_beck2023_cohort"]

    where_clauses, limit_rows = _extract_subset_where_clauses(lower_q)
    strata = _extract_strata_specs(lower_q)
    table_cols = (
        schema_map.get("patients") or schema_map.get(target_table)
        if schema_map
        else None
    )
    inspect_table = "patients" if target_table == "raw_jaeb_cohort" else target_table
    rows = _extract_row_specs(
        lower_q,
        available_cols=table_cols,
        db_path=db_path,
        target_table=inspect_table,
    )
    include_n_row = not bool(
        re.search(
            r"\b(?:without\s+n\s+row|no\s+n\s+row|omit\s+participants\s+row)\b", lower_q
        )
    )

    return Table1QuerySpec(
        target_table=target_table,
        where_clauses=tuple(where_clauses),
        strata=strata,
        rows=rows,
        include_n_row=include_n_row,
        limit_rows=limit_rows,
        raw_source_cte_sql=raw_source_cte,
    )


def build_dynamic_table1_sql(
    nl_query: str,
    schema_map: Mapping[str, Sequence[str]] | None = None,
    db_path: str | None = None,
) -> str:
    """
    Parse `nl_query` and compile a dynamic DuckDB Table 1 SQL statement.

    Args:
        nl_query: Natural language user question.
        schema_map: Optional mapping of table names to column names.
        db_path: Optional path to the active DuckDB database file.

    Returns:
        str: Executable DuckDB SQL statement generating the requested Table 1.

    """
    spec = parse_nl_table1_spec(nl_query, schema_map=schema_map, db_path=db_path)
    avail_cols: set[str] | None = None
    if schema_map and spec.target_table in schema_map:
        avail_cols = {c.lower() for c in schema_map[spec.target_table]}
    return compile_table1_sql(spec, available_cols=avail_cols)


__all__ = [
    "BECK_2023_DEFAULT_ROW_KEYS",
    "CategoryBinSpec",
    "ContinuousSummaryFormat",
    "RowVariableKind",
    "Table1QuerySpec",
    "Table1RowSpec",
    "Table1StratumSpec",
    "build_beck_2023_row_specs",
    "build_dynamic_table1_sql",
    "build_raw_jaeb_cohort_cte_sql",
    "compile_table1_sql",
    "db_has_extended_jaeb_tables",
    "db_has_raw_jaeb_tables",
    "extract_custom_numeric_cutpoints",
    "format_table1_rows_markdown",
    "has_custom_table1_modifiers",
    "is_categorical_column_in_db",
    "parse_nl_table1_spec",
]
