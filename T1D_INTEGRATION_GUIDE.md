# T1D Integration Guide

This guide details the end-to-end pipeline for preparing Type 1 Diabetes (T1D) clinical trial datasets, generating synthetic Text-to-SQL training pairs (`pretrain_data`, `sft_data`, and `dpo_data`), and executing model training and evaluation via both the `t1d-analytics` CLI and `gemma-4-sql`.

## Pipeline Steps

### 1. Download, Extract, or Seed Clinical Data

Fetch and extract the public T1D datasets (or seed baseline clinical trial cohorts locally) using the `t1d-analytics` CLI tool:

```bash
# Download datasets from the T1D repository
t1d-analytics download -o ./data

# Extract downloaded zip archives
t1d-analytics extract -d ./data

# Or seed deterministic baseline T1D clinical cohorts (patients, cgm_readings, hba1c_labs, dka_adverse_events)
t1d-analytics seed -d ./data --db t1d_analytics.duckdb
```

### 2. Load Data into DuckDB

Parse and load the extracted CSV, TXT, SAS (`.sas7bdat`, `.xpt`), or SPSS (`.sav`) files into `t1d_analytics.duckdb`:

```bash
t1d-analytics load -d ./data --db t1d_analytics.duckdb
```

### 3. Generate Training Data

Generate synthetic Text-to-SQL training pairs (`pretrain_data`, `sft_data`, and `dpo_data`, plus optional stratified `_train`, `_val`, and `_test` splits) from the DuckDB schema using a local Ollama model or configured `any-llm` provider:

```bash
t1d-analytics generate-training-data \
    --db t1d_analytics.duckdb \
    --num-pairs 1000 \
    --model gemma4 \
    --split-ratios "0.8,0.1,0.1"
```

To export generated training tables directly to JSONL or Parquet files:

```bash
t1d-analytics export-training-data \
    --db t1d_analytics.duckdb \
    --table sft_data \
    --format parquet \
    --output-file ./training_output/sft_data.parquet
```

### 4. Extract, Transform, Load (ETL) & Training for Gemma-4-SQL

You can ingest the generated training pairs into `gemma-4-sql` either via the built-in `t1d-analytics` bridge subcommands or directly through the `gemma-4-sql` CLI:

```bash
# Option A: Using the t1d-analytics Gemma-4-SQL bridge
t1d-analytics bridge-gemma-sql check
t1d-analytics bridge-gemma-sql etl --stage sft --db t1d_analytics.duckdb --table sft_data --output-dir ./shards
t1d-analytics bridge-gemma-sql train --stage sft --db t1d_analytics.duckdb --table sft_data

# Option B: End-to-end automated orchestration via t1d-analytics train
t1d-analytics train \
    --db t1d_analytics.duckdb \
    --output-dir ./training_output \
    --samples-per-table 50 \
    --backend gemma-4-sql \
    --use-lora \
    --execute-training

# Option C: Direct gemma-4-sql CLI invocation
gemma-4-sql etl sft --duckdb-path t1d_analytics.duckdb --duckdb-table sft_data
```

### 5. Evaluate Holdout Execution Accuracy

Evaluate the fine-tuned model against a holdout benchmark set of natural-language clinical questions and gold SQL queries:

```bash
t1d-analytics evaluate \
    --db t1d_analytics.duckdb \
    --test-set ./test_benchmarks.json \
    --model gemma4 \
    --output ./evaluation_report.md
```
