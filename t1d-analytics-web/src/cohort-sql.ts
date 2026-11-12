/**
 * @file cohort-sql.ts
 * @module cohort-sql
 * Clinical relational ontology, foreign-key join resolver, and DuckDB cohort SQL generator.
 */

/**
 * Relationship definition between two clinical tables.
 */
export interface TableRelationship {
  /** The source primary table name. */
  fromTable: string;
  /** The target secondary table name. */
  toTable: string;
  /** Primary key column in the source table. */
  fromColumn: string;
  /** Foreign key column in the target table. */
  toColumn: string;
  /** Suggested default join type. */
  defaultJoinType: "INNER JOIN" | "LEFT JOIN";
}

/**
 * Standard clinical relational ontology mapping primary and foreign keys.
 */
export const CLINICAL_RELATIONSHIPS: TableRelationship[] = [
  {
    fromTable: "patients",
    toTable: "cgms",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "patients",
    toTable: "dclp3",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "patients",
    toTable: "visits",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "patients",
    toTable: "devices",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "patients",
    toTable: "labs",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "cgms",
    toTable: "dclp3",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "patients",
    toTable: "cgm_readings",
    fromColumn: "subject_id",
    toColumn: "subject_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "patients",
    toTable: "hba1c_labs",
    fromColumn: "subject_id",
    toColumn: "subject_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "patients",
    toTable: "dka_adverse_events",
    fromColumn: "subject_id",
    toColumn: "subject_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "demographics",
    toTable: "cgm_data",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "demographics",
    toTable: "cgm_readings",
    fromColumn: "subject_id",
    toColumn: "subject_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "demographics",
    toTable: "hba1c_labs",
    fromColumn: "subject_id",
    toColumn: "subject_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "tblaptsummary",
    toTable: "hba1c",
    fromColumn: "PtID",
    toColumn: "PtID",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "tblaptsummary",
    toTable: "adverseevents",
    fromColumn: "PtID",
    toColumn: "PtID",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "tblaptsummary",
    toTable: "subjects",
    fromColumn: "PtID",
    toColumn: "PtID",
    defaultJoinType: "INNER JOIN",
  },
];

/**
 * Resolve foreign key join matching between two tables using known ontology or shared ID columns.
 * @param {string} table1 - The primary table name.
 * @param {string} table2 - The secondary table name.
 * @param {string[]} [table1Cols] - Optional known columns of table 1.
 * @param {string[]} [table2Cols] - Optional known columns of table 2.
 * @returns {TableRelationship | null} The resolved relationship or null if none discovered.
 */
/**
 * Checks whether an ontology `TableRelationship` is compatible with optional column lists.
 * @param {TableRelationship} found - Matched ontology relationship.
 * @param {string} table1 - Primary table name.
 * @param {string} table2 - Secondary table name.
 * @param {string[]} [table1Cols] - Known primary table columns.
 * @param {string[]} [table2Cols] - Known secondary table columns.
 * @returns {TableRelationship | null} Validated relationship or null.
 */
function resolveOntologyRelationship(
  found: TableRelationship,
  table1: string,
  table2: string,
  table1Cols?: string[],
  table2Cols?: string[],
): TableRelationship | null {
  const isForward = found.fromTable.toLowerCase() === table1.toLowerCase();
  const candidateFrom = isForward ? found.fromColumn : found.toColumn;
  const candidateTo = isForward ? found.toColumn : found.fromColumn;
  const fromOk =
    !table1Cols ||
    table1Cols.length === 0 ||
    table1Cols.some((c) => c.toLowerCase() === candidateFrom.toLowerCase());
  const toOk =
    !table2Cols ||
    table2Cols.length === 0 ||
    table2Cols.some((c) => c.toLowerCase() === candidateTo.toLowerCase());
  return fromOk && toOk
    ? {
        fromTable: table1,
        toTable: table2,
        fromColumn: candidateFrom,
        toColumn: candidateTo,
        defaultJoinType: found.defaultJoinType,
      }
    : null;
}

/**
 * Resolves foreign-key relationship join columns between two clinical tables.
 * @param {string} table1 - Primary table name.
 * @param {string} table2 - Secondary table name.
 * @param {string[]} [table1Cols] - Optional known columns of table 1.
 * @param {string[]} [table2Cols] - Optional known columns of table 2.
 * @returns {TableRelationship | null} Resolved relationship or null if none discovered.
 */
export function resolveTableJoin(
  table1: string,
  table2: string,
  table1Cols?: string[],
  table2Cols?: string[],
): TableRelationship | null {
  const t1 = table1.toLowerCase();
  const t2 = table2.toLowerCase();

  const found = CLINICAL_RELATIONSHIPS.find(
    (r) =>
      (r.fromTable.toLowerCase() === t1 && r.toTable.toLowerCase() === t2) ||
      (r.fromTable.toLowerCase() === t2 && r.toTable.toLowerCase() === t1),
  );
  if (found) {
    const ontologyMatch = resolveOntologyRelationship(
      found,
      table1,
      table2,
      table1Cols,
      table2Cols,
    );
    if (ontologyMatch) {
      return ontologyMatch;
    }
  }

  const candidateKeys = [
    "patient_id",
    "subject_id",
    "pt_id",
    "id",
    "study_id",
    "device_id",
  ];
  for (const key of candidateKeys) {
    const match1 = table1Cols?.find((c) => c.toLowerCase() === key);
    const match2 = table2Cols?.find((c) => c.toLowerCase() === key);
    if (match1 && match2) {
      return {
        fromTable: table1,
        toTable: table2,
        fromColumn: match1,
        toColumn: match2,
        defaultJoinType: "INNER JOIN",
      };
    }
  }

  if (t1.includes("patient") || t2.includes("patient")) {
    return {
      fromTable: table1,
      toTable: table2,
      fromColumn: "patient_id",
      toColumn: "patient_id",
      defaultJoinType: "INNER JOIN",
    };
  }

  return null;
}

/**
 * Structured equality join specification between primary and secondary tables.
 */
export interface CohortJoinSpec {
  /** Column identifier on the primary table (`tableName`). */
  leftColumn: string;
  /** Equality join operator (`"="`). */
  operator?: "=";
  /** Column identifier on the secondary joined table (`joinTableName`). */
  rightColumn: string;
}

/**
 * Clinical cohort filter parameters for query generation.
 */
export interface CohortFilterParams {
  /** Target primary table name. */
  tableName: string;
  /** Optional secondary table name for relational join. */
  joinTableName?: string;
  /** Relational join condition type ('INNER JOIN' or 'LEFT JOIN'). */
  joinType?: "INNER JOIN" | "LEFT JOIN";
  /** Structured equality join column specification. */
  joinSpec?: CohortJoinSpec;
  /** Optional `ON` equality expression (`"t1"."col1" = "t2"."col2"`), strictly validated before quoting. */
  joinCondition?: string;
  /** Minimum patient age filter. */
  minAge?: number;
  /** Maximum patient age filter. */
  maxAge?: number;
  /** Biological sex or gender. */
  gender?: string;
  /** Clinical treatment group (e.g. Closed-Loop, MDI). */
  txGroup?: string;
  /** Time in Range maximum threshold percentage. */
  maxTIR?: number;
  /** Minimum glycated hemoglobin (HbA1c) threshold. */
  minHbA1c?: number;
  /** Optional schema column names for the primary table (`tableName`). */
  tableColumns?: string[];
  /** Optional schema column names for the secondary joined table (`joinTableName`). */
  joinTableColumns?: string[];
}

/**
 * Quotes a SQL table or column identifier using ANSI double-quote escaping.
 * @param {string} identifier - Raw table or column identifier.
 * @returns {string} Double-quoted SQL identifier with internal quotes doubled.
 */
export function escapeSqlIdentifier(identifier: string): string {
  return `"${identifier.replace(/"/g, '""')}"`;
}

/**
 * Parses a single simple or table-qualified SQL identifier (`col`, `"col"`, `"tbl".col`, or `"tbl"."col"`).
 * Rejects SQL keywords, comments, semicolons, parens, spaces in unquoted tokens, and operators.
 * @param {string} rawOperand - Raw operand side of an equality join expression.
 * @returns {{ table: string | null; column: string } | null} Parsed table/column pair or null if unsafe.
 */
function parseJoinOperand(
  rawOperand: string,
): { table: string | null; column: string } | null {
  const trimmed = rawOperand.trim();
  const identTokenPattern = '(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_]*)';
  const fullRe = new RegExp(
    `^(?:(${identTokenPattern})\\.)?(${identTokenPattern})$`,
  );
  const match = fullRe.exec(trimmed);
  if (!match) {
    return null;
  }
  /**
   * Unescapes a double-quoted SQL identifier token.
   * @param {string} tok - Raw SQL identifier token.
   * @returns {string} Unquoted SQL identifier.
   */
  const unquote = (tok: string): string =>
    tok.startsWith('"') && tok.endsWith('"')
      ? tok.slice(1, -1).replace(/""/g, '"')
      : tok;
  const col = unquote(String(match[2]));
  if (!col.trim()) {
    return null;
  }
  return {
    table: match[1] ? unquote(match[1]) : null,
    column: col,
  };
}

/**
 * Validates a column identifier against an optional schema column list, returning canonical casing when matched.
 * @param {string} candidateCol - Candidate column name.
 * @param {string[] | undefined} schemaCols - Known table column names, if available.
 * @returns {string | null} Validated column name or null when rejected by schema.
 */
function validateColumnAgainstSchema(
  candidateCol: string,
  schemaCols: string[] | undefined,
): string | null {
  const cleaned = candidateCol.trim();
  if (!cleaned) {
    return null;
  }
  if (!schemaCols || schemaCols.length === 0) {
    return cleaned;
  }
  const found = schemaCols.find(
    (col) => col.toLowerCase() === cleaned.toLowerCase(),
  );
  return found ?? null;
}

/**
 * Strictly parses and validates a join equality expression into a safe `CohortJoinSpec`,
 * rejecting arbitrary SQL predicates, multi-expressions, comments, or unknown schema columns.
 * @param {string} rawJoinCondition - Raw `ON` input expression.
 * @param {string} table1 - Primary table name.
 * @param {string} table2 - Secondary joined table name.
 * @param {string[]} [table1Cols] - Known columns on `table1`.
 * @param {string[]} [table2Cols] - Known columns on `table2`.
 * @returns {CohortJoinSpec | null} Validated join specification, or null if invalid/unsafe.
 */
export function parseStructuredJoinCondition(
  rawJoinCondition: string,
  table1: string,
  table2: string,
  table1Cols?: string[],
  table2Cols?: string[],
): CohortJoinSpec | null {
  const parts = rawJoinCondition.split("=");
  if (parts.length !== 2) {
    return null;
  }
  const left = parseJoinOperand(String(parts[0]));
  const right = parseJoinOperand(String(parts[1]));
  if (!left || !right) {
    return null;
  }
  let leftColRaw = left.column;
  let rightColRaw = right.column;
  if (
    left.table &&
    right.table &&
    left.table.toLowerCase() === table2.toLowerCase() &&
    right.table.toLowerCase() === table1.toLowerCase()
  ) {
    leftColRaw = right.column;
    rightColRaw = left.column;
  } else if (
    (left.table && left.table.toLowerCase() !== table1.toLowerCase()) ||
    (right.table && right.table.toLowerCase() !== table2.toLowerCase())
  ) {
    return null;
  }
  const validLeft = validateColumnAgainstSchema(leftColRaw, table1Cols);
  const validRight = validateColumnAgainstSchema(rightColRaw, table2Cols);
  if (!validLeft || !validRight) {
    return null;
  }
  return {
    leftColumn: validLeft,
    operator: "=",
    rightColumn: validRight,
  };
}

/**
 * Resolves the best matching column name from `tableColumns` for a candidate list, or returns `fallback`.
 * @param {string[] | undefined} tableColumns - Available column names on the table.
 * @param {string[]} candidates - Prioritized lowercase candidate column names.
 * @param {string} fallback - Default column name when no schema match is found.
 * @returns {string} Resolved column identifier.
 */
export function resolveCohortColumnName(
  tableColumns: string[] | undefined,
  candidates: string[],
  fallback: string,
): string {
  if (!tableColumns || tableColumns.length === 0) {
    return fallback;
  }
  for (const cand of candidates) {
    const found = tableColumns.find((col) => col.toLowerCase() === cand);
    if (found) {
      return found;
    }
  }
  return fallback;
}

/**
 * Normalizes a string into a valid SQL join type ('INNER JOIN' or 'LEFT JOIN').
 * @param {string | undefined} rawJoinType - Raw join type value from a select element.
 * @returns {"INNER JOIN" | "LEFT JOIN"} Normalized join type.
 */
export function normalizeJoinType(
  rawJoinType: string | undefined,
): "INNER JOIN" | "LEFT JOIN" {
  return rawJoinType === "LEFT JOIN" ? "LEFT JOIN" : "INNER JOIN";
}

/**
 * Resolves a qualified (`"table"."col"` or `"col"`) filter expression target across primary and join tables.
 * @param {CohortFilterParams} params - Cohort filter criteria.
 * @param {boolean} isJoin - Whether a join table is active.
 * @param {string | undefined} table2 - Optional joined table name.
 * @param {string} defaultPrefix - Default primary table prefix.
 * @param {string[]} candidates - Candidate column names.
 * @param {string} fallback - Fallback column identifier.
 * @returns {string} Escaped table-qualified or single-table column SQL identifier.
 */
function resolveQualifiedCohortColumn(
  params: CohortFilterParams,
  isJoin: boolean,
  table2: string | undefined,
  defaultPrefix: string,
  candidates: string[],
  fallback: string,
): string {
  const onJoinTable =
    isJoin &&
    Boolean(table2) &&
    params.tableColumns?.some((c) => candidates.includes(c.toLowerCase())) !==
      true &&
    params.joinTableColumns?.some((c) =>
      candidates.includes(c.toLowerCase()),
    ) === true;
  const col = resolveCohortColumnName(
    onJoinTable ? params.joinTableColumns : params.tableColumns,
    candidates,
    fallback,
  );
  const colPrefix =
    onJoinTable && table2 ? `${escapeSqlIdentifier(table2)}.` : defaultPrefix;
  return `${colPrefix}${escapeSqlIdentifier(col)}`;
}

/**
 * Builds validated SQL `WHERE` predicates from `CohortFilterParams`.
 * @param {CohortFilterParams} params - Cohort filter criteria.
 * @param {boolean} isJoin - Whether a join table is active.
 * @param {string | undefined} table2 - Optional joined table name.
 * @param {string} prefix - Default primary table prefix.
 * @returns {string[]} Array of escaped SQL predicate clauses.
 */
function buildCohortWherePredicates(
  params: CohortFilterParams,
  isJoin: boolean,
  table2: string | undefined,
  prefix: string,
): string[] {
  const predicates: string[] = [];
  const ageCol = `${prefix}${escapeSqlIdentifier(
    resolveCohortColumnName(
      params.tableColumns,
      ["age", "ageasofranddt", "agebaseline"],
      "age",
    ),
  )}`;
  const genderCol = `${prefix}${escapeSqlIdentifier(
    resolveCohortColumnName(params.tableColumns, ["gender", "sex"], "gender"),
  )}`;
  const txCol = `${prefix}${escapeSqlIdentifier(
    resolveCohortColumnName(
      params.tableColumns,
      ["txgroup", "tx_group", "treatment_group", "studyarm", "arm"],
      "txgroup",
    ),
  )}`;
  const tirExpr = resolveQualifiedCohortColumn(
    params,
    isJoin,
    table2,
    prefix,
    ["tir", "time_in_range"],
    "tir",
  );
  const hba1cExpr = resolveQualifiedCohortColumn(
    params,
    isJoin,
    table2,
    prefix,
    ["hba1c", "baseline_hba1c", "a1c"],
    "hba1c",
  );

  if (params.minAge !== undefined && Number.isFinite(params.minAge)) {
    predicates.push(`${ageCol} >= ${params.minAge}`);
  }
  if (params.maxAge !== undefined && Number.isFinite(params.maxAge)) {
    predicates.push(`${ageCol} <= ${params.maxAge}`);
  }
  if (params.gender) {
    predicates.push(`${genderCol} = '${params.gender.replace(/'/g, "''")}'`);
  }
  if (params.txGroup) {
    predicates.push(`${txCol} ILIKE '%${params.txGroup.replace(/'/g, "''")}%'`);
  }
  if (params.maxTIR !== undefined && Number.isFinite(params.maxTIR)) {
    predicates.push(`${tirExpr} <= ${params.maxTIR}`);
  }
  if (params.minHbA1c !== undefined && Number.isFinite(params.minHbA1c)) {
    predicates.push(`${hba1cExpr} >= ${params.minHbA1c}`);
  }
  return predicates;
}

/**
 * Resolves the validated SQL `JOIN ... ON ...` clause for `buildCohortSql`.
 * @param {CohortFilterParams} params - Cohort filter criteria.
 * @param {string} table1 - Primary table name.
 * @param {string} table2 - Secondary joined table name.
 * @param {"INNER JOIN" | "LEFT JOIN"} joinType - Normalized SQL join type.
 * @returns {string} Formatted SQL JOIN clause.
 */
function buildCohortJoinClause(
  params: CohortFilterParams,
  table1: string,
  table2: string,
  joinType: "INNER JOIN" | "LEFT JOIN",
): string {
  const quotedT1 = escapeSqlIdentifier(table1);
  const quotedT2 = escapeSqlIdentifier(table2);
  const resolvedJoin = resolveTableJoin(
    table1,
    table2,
    params.tableColumns,
    params.joinTableColumns,
  );
  let activeSpec: CohortJoinSpec | null = null;
  if (params.joinSpec) {
    const validSpecLeft = validateColumnAgainstSchema(
      params.joinSpec.leftColumn,
      params.tableColumns,
    );
    const validSpecRight = validateColumnAgainstSchema(
      params.joinSpec.rightColumn,
      params.joinTableColumns,
    );
    if (validSpecLeft && validSpecRight) {
      activeSpec = {
        leftColumn: validSpecLeft,
        operator: "=",
        rightColumn: validSpecRight,
      };
    }
  } else if (params.joinCondition?.trim()) {
    activeSpec = parseStructuredJoinCondition(
      params.joinCondition,
      table1,
      table2,
      params.tableColumns,
      params.joinTableColumns,
    );
  }
  const leftJoinCol =
    activeSpec?.leftColumn ?? resolvedJoin?.fromColumn ?? "patient_id";
  const rightJoinCol =
    activeSpec?.rightColumn ?? resolvedJoin?.toColumn ?? "patient_id";
  const safeOn = `${quotedT1}.${escapeSqlIdentifier(leftJoinCol)} = ${quotedT2}.${escapeSqlIdentifier(rightJoinCol)}`;
  return `\n${joinType} ${quotedT2}\n  ON ${safeOn}`;
}

/**
 * Construct an optimized DuckDB SQL query from clinician cohort filter parameters, supporting relational joins.
 * All table and column identifiers are double-quote escaped, and join conditions are strictly validated.
 * @param {CohortFilterParams} params - The selected filter criteria.
 * @returns {string} The generated DuckDB SQL query.
 */
export function buildCohortSql(params: CohortFilterParams): string {
  const table1 = params.tableName.trim() || "patients";
  const table2 = params.joinTableName?.trim();
  const joinType = normalizeJoinType(params.joinType);

  const isJoin = Boolean(table2 && table2 !== table1);
  const quotedT1 = escapeSqlIdentifier(table1);
  const prefix = isJoin ? `${quotedT1}.` : "";
  const predicates = buildCohortWherePredicates(params, isJoin, table2, prefix);

  let fromClause = `FROM ${quotedT1}`;
  if (isJoin && table2) {
    fromClause += buildCohortJoinClause(params, table1, table2, joinType);
  }

  const whereClause =
    predicates.length > 0 ? `\nWHERE ${predicates.join("\n  AND ")}` : "";
  return `SELECT *\n${fromClause}${whereClause};`;
}
