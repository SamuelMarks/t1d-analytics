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
];

/**
 * Resolve foreign key join matching between two tables using known ontology or shared ID columns.
 * @param {string} table1 - The primary table name.
 * @param {string} table2 - The secondary table name.
 * @param {string[]} [table1Cols] - Optional known columns of table 1.
 * @param {string[]} [table2Cols] - Optional known columns of table 2.
 * @returns {TableRelationship | null} The resolved relationship or null if none discovered.
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
    if (found.fromTable.toLowerCase() === t1) {
      return found;
    }
    return {
      fromTable: table1,
      toTable: table2,
      fromColumn: found.toColumn,
      toColumn: found.fromColumn,
      defaultJoinType: found.defaultJoinType,
    };
  }

  if (table1Cols && table2Cols) {
    const candidateKeys = [
      "patient_id",
      "subject_id",
      "pt_id",
      "id",
      "study_id",
      "device_id",
    ];
    for (const key of candidateKeys) {
      const match1 = table1Cols.find((c) => c.toLowerCase() === key);
      const match2 = table2Cols.find((c) => c.toLowerCase() === key);
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
 * Clinical cohort filter parameters for query generation.
 */
export interface CohortFilterParams {
  /** Target primary table name. */
  tableName: string;
  /** Optional secondary table name for relational join. */
  joinTableName?: string;
  /** Relational join condition type ('INNER JOIN' or 'LEFT JOIN'). */
  joinType?: "INNER JOIN" | "LEFT JOIN";
  /** Explicit ON condition for table joining. */
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
 * Construct an optimized DuckDB SQL query from clinician cohort filter parameters, supporting relational joins.
 * @param {CohortFilterParams} params - The selected filter criteria.
 * @returns {string} The generated DuckDB SQL query.
 */
export function buildCohortSql(params: CohortFilterParams): string {
  const table1 = params.tableName || "patients";
  const table2 = params.joinTableName?.trim();
  const joinType = params.joinType || "INNER JOIN";
  const joinOn = params.joinCondition?.trim();

  const isJoin = Boolean(table2 && table2 !== table1);
  const prefix = isJoin ? `"${table1.replace(/"/g, '""')}".` : "";
  const predicates: string[] = [];

  if (params.minAge !== undefined && Number.isFinite(params.minAge)) {
    predicates.push(`${prefix}age >= ${params.minAge}`);
  }
  if (params.maxAge !== undefined && Number.isFinite(params.maxAge)) {
    predicates.push(`${prefix}age <= ${params.maxAge}`);
  }
  if (params.gender) {
    const safeGender = params.gender.replace(/'/g, "''");
    predicates.push(`${prefix}gender = '${safeGender}'`);
  }
  if (params.txGroup) {
    const safeTx = params.txGroup.replace(/'/g, "''");
    predicates.push(`${prefix}txgroup ILIKE '%${safeTx}%'`);
  }
  if (params.maxTIR !== undefined && Number.isFinite(params.maxTIR)) {
    predicates.push(`${prefix}tir <= ${params.maxTIR}`);
  }
  if (params.minHbA1c !== undefined && Number.isFinite(params.minHbA1c)) {
    predicates.push(`${prefix}hba1c >= ${params.minHbA1c}`);
  }

  let fromClause = `FROM "${table1.replace(/"/g, '""')}"`;
  if (isJoin && table2) {
    const defaultOn = `"${table1.replace(/"/g, '""')}".patient_id = "${table2.replace(/"/g, '""')}".patient_id`;
    fromClause += `
${joinType} "${table2.replace(/"/g, '""')}"
  ON ${joinOn || defaultOn}`;
  }

  const whereClause =
    predicates.length > 0
      ? `
WHERE ${predicates.join(
          `
  AND `,
        )}`
      : "";
  return `SELECT *
${fromClause}${whereClause};`;
}
