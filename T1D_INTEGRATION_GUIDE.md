# T1D Integration Guide

This guide details the end-to-end pipeline for generating and loading Type 1 Diabetes (T1D) Text-to-SQL training data for Gemma-4-SQL.

## Pipeline Steps

### 1. Download Data
First, download the T1D datasets using the `t1d-analytics` CLI tool.
```bash
t1d-analytics download
```

### 2. Load Data
Next, parse and load the downloaded CSV data into a DuckDB file.
```bash
t1d-analytics load
```

### 3. Generate Training Data
Generate synthetic Text-to-SQL training pairs from the DuckDB schema using a local LLM.
```bash
t1d-analytics generate-training-data --num-pairs 1000
```

### 4. Extract, Transform, Load (ETL) for Gemma-4-SQL
Finally, ingest the generated training pairs natively into the AI-Hypercomputer training framework using `gemma-4-sql`.
```bash
gemma-4-sql etl sft --duckdb-path t1d.duckdb --duckdb-table sft_data
```
