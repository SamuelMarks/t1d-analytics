# End-to-End Training of Gemma-4-SQL on T1D Analytics Datasets

This report outlines the comprehensive steps required to execute an end-to-end training pipeline (Pretraining, Supervised Fine-Tuning/Post-training, and Direct Preference Optimization) of the `gemma-4` architecture specifically optimized for Text-to-SQL tasks on Type 1 Diabetes (T1D) clinical trial datasets. The process integrates tools from both the `t1d-analytics` and `gemma-4-sql` repositories.

## 1. Environment & Prerequisites Setup

### 1.1 Infrastructure Provisioning
To handle large-scale LLM training efficiently, we recommend utilizing Google Cloud TPU instances (e.g., v4-8 or multi-node clusters) via the `libscript` multicloud orchestration system.

```bash
# Clone libscript and install requirements
git clone https://github.com/SamuelMarks/libscript.git ~/.libscript
cd ~/.libscript
./libscript.sh install cloud-providers/gcp/cli latest

# Provision a TPU VM with persistent storage
export TPU_NAME="gemma-t1d-node"
export TPU_DATA_DISK_SIZE="500" 
./stacks/ml-training/tpu-vm-eval-node/setup.sh
```

### 1.2 Repository Installation
On the training instance, install the required frameworks. Both `t1d-analytics` (for data generation) and `gemma-4-sql` (for training orchestration) are needed.

```bash
# Inside the TPU VM
pip install -e /path/to/t1d-analytics
pip install -e /path/to/gemma-4-sql[all]
```

---

## 2. Dataset Acquisition & Preparation (`t1d-analytics`)

The foundation of the training process is extracting domain-specific T1D data and procedurally generating synthetic training pairs using a local LLM.

### 2.1 Download and Load T1D Clinical Data
First, retrieve the public T1D datasets and load them into a local DuckDB instance.
```bash
# 1. Download datasets
t1d-analytics download

# 2. Extract downloaded zip files
t1d-analytics extract --data-dir ./data

# 3. Parse and load into DuckDB
t1d-analytics load --data-dir ./data --db t1d_analytics.duckdb
```

### 2.2 Procedural Generation of Training Pairs
Leverage a local LLM (e.g., `ollama run gemma4`) to inspect the T1D DuckDB schema and generate high-quality `(Prompt, SQL)` pairs. This step generates the `pretrain_data`, `sft_data`, and `dpo_data` tables within the `t1d_analytics.duckdb` database.

```bash
# Generate 1000 pairs per table
t1d-analytics generate-training-data --db t1d_analytics.duckdb --num-pairs 1000 --model gemma4
```

---

## 3. Extract, Transform, Load (ETL) via `gemma-4-sql`

The `gemma-4-sql` ETL pipeline uses Google's `grain` library to normalize the DuckDB data into backend-optimized formats (e.g., for MaxText or JAX).

```bash
# 1. ETL for Pretraining
gemma-4-sql etl pretrain \
    --duckdb-path t1d_analytics.duckdb \
    --duckdb-table pretrain_data \
    --backend maxtext

# 2. ETL for Supervised Fine-Tuning (SFT)
gemma-4-sql etl sft \
    --duckdb-path t1d_analytics.duckdb \
    --duckdb-table sft_data \
    --backend maxtext

# 3. ETL for Direct Preference Optimization (DPO)
gemma-4-sql etl posttrain \
    --duckdb-path t1d_analytics.duckdb \
    --duckdb-table dpo_data \
    --backend maxtext
```

---

## 4. End-to-End Training Pipeline (`gemma-4-sql`)

We utilize the MaxText backend for highly optimized XLA execution on TPU architectures.

### Phase 1: Continuous Pretraining (Domain Adaptation)
Adapt the base Google `gemma-4` weights to exclusively understand T1D schemas and SQL syntax.

```bash
gemma-4-sql pretrain \
    --model google/gemma-4 \
    --dataset t1d_analytics.duckdb \
    --epochs 5 \
    --learning-rate 5e-5 \
    --backend maxtext
```

### Phase 2: Supervised Fine-Tuning (SFT)
Perform instruction tuning so the model maps natural language questions (e.g., "What is the average Time In Range for Patient X?") to precise SQL. We use Parameter-Efficient Fine-Tuning (PEFT/LoRA) to reduce memory overhead.

```bash
# 1. Inject LoRA adapters
gemma-4-sql peft --model my-t1d-pretrained-gemma --target-modules q_proj,v_proj --lora-r 16 --backend maxtext

# 2. Execute SFT
gemma-4-sql sft \
    --model my-t1d-pretrained-gemma \
    --dataset t1d_analytics.duckdb \
    --epochs 3 \
    --backend maxtext
```

### Phase 3: Direct Preference Optimization (DPO)
Align the model's behavior by contrasting the chosen (correct) SQL queries against rejected (incorrect/hallucinated) SQL queries generated in Step 2.2.

```bash
gemma-4-sql dpo \
    --model my-t1d-sft-gemma \
    --dataset t1d_analytics.duckdb \
    --beta 0.1 \
    --backend maxtext
```

---

## 5. Evaluation and Deployment

### 5.1 Evaluate Execution Accuracy
Unlike general NLP models, Text-to-SQL models must be evaluated on Execution Accuracy (EX). We use the `gemma-4-sql` LiveDatabaseEngine to run generated queries against the DuckDB instance.

```bash
gemma-4-sql evaluate \
    --model my-t1d-dpo-gemma \
    --db-type duckdb \
    --db-path t1d_analytics.duckdb \
    --dataset t1d_analytics.duckdb \
    --backend maxtext
```

### 5.2 Export Artifacts to Google Cloud Storage
Once training and evaluation are complete, export the model weights (e.g., to `safetensors` or `orbax`) and upload them to persistent GCS storage before deprovisioning the ephemeral TPU compute.

```bash
# Export the weights
gemma-4-sql export --model my-t1d-dpo-gemma --path ./exported/gemma-4-t1d --backend maxtext

# Upload to GCS
gcloud storage cp -r ./exported/gemma-4-t1d gs://gemma-4-sql-artifacts-12345/
```
