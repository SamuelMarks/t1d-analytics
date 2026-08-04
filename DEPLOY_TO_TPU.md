# Deploying and Training T1D Analytics on Google Cloud TPUs

This document provides a comprehensive guide to deploying, training, and serving the `t1d-analytics` Text-to-SQL Large Language Model on Google Cloud TPUs utilizing the `libscript` infrastructure automation ecosystem. 

This guide details how to leverage `libscript` to orchestrate both single-node and multi-node (distributed) training architectures for the T1D domain pipeline.

---

## 0. Parameters & Environment Variables

Before running the commands in this guide, export the following environment variables. These parameters configure the hardware, model, datasets, and storage buckets used across the workflows.

```bash
# HuggingFace Token (Required for gated models like Gemma)
export HF_TOKEN="your_huggingface_token_here"

# Model & Dataset Configuration
export MODEL_NAME="google/gemma-4"
export DATASET_NAME="t1d-analytics"
export DUCKDB_PATH="t1d_analytics.duckdb"
export DUCKDB_TABLE="pretrain_data"

# Google Cloud Project Configuration
export GCP_PROJECT_ID="your_google_cloud_project_id"
export GCP_ZONE="us-central2-b"

# TPU Hardware Configuration
export TPU_ZONE="us-central2-b"
export TPU_ACCELERATOR_TYPE="v4-8"
export TPU_SCHEDULING_TYPE="spot" # Change to "on-demand" if using on-demand TPUs
export TPU_USE_QUEUED_RESOURCE="true" # Set to true to wait for spot capacity
export TPU_COUNT="1" # Set >1 for multiple nodes

# Single-Node TPU VM Config
export TPU_NAME="t1d-train-node"
export SERVE_TPU_NAME="t1d-serve-node"
export TPU_DATA_DISK_SIZE="500" # Size in GB for persistent disk caching

# Multi-Node Distributed GKE/XPK Config
export XPK_CLUSTER_NAME="t1d-training-cluster"
export WORKLOAD_NAME="t1d-full-train"
export NUM_SLICES="4"

# Object Storage for Asset Persistence
# Choose a unique bucket name, for example:
export BUCKET_NAME="gs://t1d-analytics-artifacts-123456789"
```

---

## 0.5. TFRC (TPU Research Cloud) Authorized Capacity

As part of the TPU Research Cloud (TFRC) grant for this project, you have been authorized to use the following specific compute capacities. To ensure your workloads properly leverage the grant, adhere strictly to these configurations.

You MUST set your `TPU_ZONE`, `TPU_SCHEDULING_TYPE`, and `TPU_ACCELERATOR_TYPE` perfectly to match one of the following allocations:

### Cloud TPU v4 (us-central2-b)
- **32 on-demand chips:** `TPU_SCHEDULING_TYPE="on-demand"`, `TPU_ZONE="us-central2-b"`, `TPU_ACCELERATOR_TYPE="v4-32"` (or smaller slices like `v4-8`, `v4-16`)
- **32 spot chips:** `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="us-central2-b"`, `TPU_ACCELERATOR_TYPE="v4-32"` (or smaller slices)

### Cloud TPU v5e (europe-west4-b & us-central1-a)
- **64 spot chips (Europe):** `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="europe-west4-b"`, `TPU_ACCELERATOR_TYPE="v5litepod-64"` (or smaller slices)
- **64 spot chips (US Central):** `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="us-central1-a"`, `TPU_ACCELERATOR_TYPE="v5litepod-64"` (or smaller slices)

### Cloud TPU v6e (europe-west4-a & us-east1-d)
- **64 spot chips (Europe):** `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="europe-west4-a"`, `TPU_ACCELERATOR_TYPE="v6e-64"` (or smaller slices)
- **64 spot chips (US East):** `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="us-east1-d"`, `TPU_ACCELERATOR_TYPE="v6e-64"` (or smaller slices)

When using spot instances, always remember to export `TPU_USE_QUEUED_RESOURCE="true"`.

---

## 1. Prerequisites

First, clone the `libscript` repository, as all infrastructure provisioning commands must be executed from its root directory.

```bash
git clone https://github.com/SamuelMarks/libscript.git ~/.libscript
cd ~/.libscript

# Install the Google Cloud CLI component and authenticate
./libscript.sh install cloud-providers/gcp/cli latest

# Install the HuggingFace CLI and XPK orchestrator
./libscript.sh install toolchains/huggingface-cli latest
./libscript.sh install toolchains/xpk latest
```

---

## 2. Approach A: Single-Node TPU VMs (Prototyping & PEFT)

**Best for:** Rapid prototyping, parameter-efficient fine-tuning (LoRA / QLoRA), Supervised Fine-Tuning (SFT), and DuckDB ETL for T1D models.
**Hardware:** Single TPU VM (e.g., `v4-8` or `v5litepod-8`).

With `libscript`, you can utilize the `ml-training/tpu-vm-eval-node` stack. This stack automatically creates a TPU VM with an attached persistent disk for dataset caching, mounts GCS via `gcsfuse`, executes your ML loop inside a detached `tmux` session, and forwards TensorBoard (6006) to your local machine.

### Step 2.1: Provision the TPU VM

```bash
# Creates the VM idempotently using the variables defined in Section 0
./stacks/ml-training/tpu-vm-eval-node/setup.sh
```

### Step 2.2: Upload the DuckDB Database to GCS

Once you have generated the training pairs and prepared the `t1d_analytics.duckdb` file locally, upload it to Google Cloud Storage (GCS) so that it is accessible to the training environment via the `gcsfuse` mount.

```bash
gcloud storage cp t1d_analytics.duckdb $BUCKET_NAME/t1d_analytics.duckdb
```

### Step 2.2b: Alternative: Direct Caching (Jaeb Dataset & Rsync)

To prevent massive and repetitive downloads of the raw Jaeb datasets on ephemeral TPU compute, we recommend bypassing GCS entirely. You can prepare the DuckDB database locally and push the datasets and DuckDB file directly to the TPU VM using `rsync` or `scp`.

**1. Prepare Data Locally:**
Use the `t1d-analytics` CLI pipeline to download, extract, load into DuckDB, and generate the synthetic SFT/DPO pairs:
```bash
# 1. Download datasets
t1d-analytics download -o ./data/jaeb_raw

# 2. Extract zip archives
t1d-analytics extract -d ./data/jaeb_raw

# 3. Load CSVs into DuckDB
t1d-analytics load -d ./data/jaeb_raw --db t1d_analytics.duckdb

# 4. Generate synthetic Text-to-SQL training pairs
t1d-analytics generate-training-data --db t1d_analytics.duckdb --num-pairs 100 --model gemma4
```

**2. Push Cache directly to the TPU VM:**
Get your TPU instance's IP address and execute the transfer:

```bash
export TPU_IP=$(gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$TPU_ZONE --format="value(networkEndpoints[0].accessConfig.externalIp)")

# Using rsync
rsync -avzP ./data/jaeb_raw/ t1d_analytics.duckdb $USER@$TPU_IP:~/ml_data/

# Alternative using scp
scp -r ./data/jaeb_raw/ t1d_analytics.duckdb $USER@$TPU_IP:~/ml_data/
```

### Step 2.3: Execute Workloads

You can now dispatch native CLI commands using the stack's deployment script.

**Example 1: Supervised Fine-Tuning (SFT) with LoRA via MaxText**
```bash
./libscript.sh gcp/tpu-vm ssh "$TPU_NAME" "
  export HF_TOKEN=\$HF_TOKEN
  
  # 1. Apply PEFT/LoRA adapters
  gemma-4-sql peft --model \$MODEL_NAME --target-modules q_proj,v_proj --lora-r 16 --backend maxtext
  
  # 2. Run the SFT loop using the mounted DuckDB file
  gemma-4-sql sft --model \$MODEL_NAME --dataset /mnt/ml_data/t1d_analytics.duckdb --backend maxtext
"
```

---

## 3. Approach B: Distributed Training (XPK + GKE)

**Best for:** Pre-training, full-parameter continuous fine-tuning, and massive datasets requiring multi-slice TPU Pods.
**Hardware:** GKE Cluster with Kueue/JobSet orchestration managing TPU node pools.

### Step 3.1: Provision the GKE TPU Cluster

```bash
# Creates the cluster via XPK with Kueue configured
./libscript.sh gcp/gke-tpu-cluster create "$XPK_CLUSTER_NAME"
```

### Step 3.2: Submit a Distributed Training Workload

```bash
# Add xpk to your path locally
export PATH=\"./installed/xpk/bin:\$PATH\"

# Dispatch a multi-slice MaxText pretraining job
xpk workload create \
  --cluster \"$XPK_CLUSTER_NAME\" \
  --workload \"$WORKLOAD_NAME\" \
  --tpu-type \"$TPU_ACCELERATOR_TYPE\" \
  --num-slices \"$NUM_SLICES\" \
  --env \"HF_TOKEN=$HF_TOKEN\" \
  --docker-image \"gcr.io/$GCP_PROJECT_ID/gemma-4-sql-runtime:latest\" \
  --command \"gemma-4-sql pretrain --model $MODEL_NAME --dataset /mnt/ml_data/t1d_analytics.duckdb --backend maxtext\"
```

---

## 4. Serving, Chat, & Agentic Inference

### Option A: Serving API using `libscript` Stacks
Deploy a high-throughput vLLM API server natively:

```bash
# Deploy to a Single TPU VM
export TPU_NAME="$SERVE_TPU_NAME"
./stacks/ai-serving/tpu-vm-vllm/setup.sh
./stacks/ai-serving/tpu-vm-vllm/deploy.sh
```

### Option B: DuckDB UDF Integration
Embed the model directly into a DuckDB instance running on your TPU to query T1D analytics:

```bash
./libscript.sh gcp/tpu-vm ssh "$SERVE_TPU_NAME" "
  gemma-4-sql embed-duckdb --model /mnt/ml_data/t1d-finetuned --db-path /mnt/ml_data/t1d_analytics.duckdb --prompt 'How many patients had hypo events yesterday?'
"
```

---

## 5. Persisting Assets to Object Storage

TPU VMs are ephemeral. Once your training run completes, export and upload the artifacts.

```bash
./libscript.sh gcp/tpu-vm ssh "$TPU_NAME" "
  # Export the trained model to safetensors
  gemma-4-sql export --model /mnt/ml_data/t1d-finetuned --path ./exported/t1d-pt --backend pytorch
  
  # Upload to GCS
  gcloud storage cp -r ./exported/t1d-pt \$BUCKET_NAME/
"
```

---

## 6. Deprovisioning Ephemeral Infrastructure

To avoid runaway costs, destroy all compute resources once the assets are safely stored.

### Teardown: Single TPU VMs
```bash
# If mounted manually, safely unmount first
./libscript.sh gcp/tpu-vm ssh "$TPU_NAME" "~/.libscript/libscript.sh storage-layers/gcsfuse unmount /mnt/ml_data" || true

# Delete the TPU VM
./libscript.sh gcp/tpu-vm delete "$TPU_NAME"
./libscript.sh gcp/tpu-vm delete "$SERVE_TPU_NAME"
```

### Teardown: GKE TPU Clusters (XPK)
```bash
./libscript.sh gcp/gke-tpu-cluster delete "$XPK_CLUSTER_NAME"
```
