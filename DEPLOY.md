# Stanford HealthPlatform - Deployment Guide

This repository contains the configuration and source code for the Stanford HealthPlatform research tools. This document provides step-by-step instructions on how to use libscript to deploy, backup, and restore this specific stack across major cloud providers.

## Pre-requisites

Ensure libscript is installed and you are authenticated to your chosen cloud provider (defaults loaded from `.env`).

- **AWS**: aws configure
- **Default Provider (`$PRIMARY_CLOUD_DISPLAY_NAME`)**: `$PRIMARY_CLOUD_LOGIN_CMD`
- **GCP**: gcloud auth login

## T1D Domain Pipeline

For users specifically looking to deploy the T1D domain pipeline for Text-to-SQL training with Gemma-4-SQL, please refer to the [T1D Integration Guide](T1D_INTEGRATION_GUIDE.md) for a comprehensive step-by-step tutorial.

### TPU Research Cloud (TFRC) Deployments

If you are deploying the training pipeline using a TFRC grant, ensure you provide the correct spot or on-demand scheduling flags prior to provisioning via libscript.

You have access to the following TFRC capacity. Ensure your `TPU_ZONE`, `TPU_ACCELERATOR_TYPE`, and `TPU_SCHEDULING_TYPE` exactly match one of these configurations:

**Cloud TPU v4 (us-central2-b)**

- Up to 32 chips On-Demand: `TPU_SCHEDULING_TYPE="on-demand"`, `TPU_ZONE="us-central2-b"`
- Up to 32 chips Spot: `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="us-central2-b"`

**Cloud TPU v5e (europe-west4-b & us-central1-a)**

- Up to 64 chips Spot (Europe): `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="europe-west4-b"`
- Up to 64 chips Spot (US Central): `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="us-central1-a"`

**Cloud TPU v6e (europe-west4-a & us-east1-d)**

- Up to 64 chips Spot (Europe): `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="europe-west4-a"`
- Up to 64 chips Spot (US East): `TPU_SCHEDULING_TYPE="spot"`, `TPU_ZONE="us-east1-d"`

Example configuration for a v4-8 spot instance:

```bash
export TPU_SCHEDULING_TYPE="spot"
export TPU_USE_QUEUED_RESOURCE="true"
export TPU_ZONE="us-central2-b"
export TPU_ACCELERATOR_TYPE="v4-8"
```

### Remote Data Caching Optimization (Jaeb Dataset)

To prevent massive and repetitive downloads on ephemeral TPU compute, we recommend preparing the DuckDB database locally first, and then using `rsync` or `scp` to push the dataset and your pre-loaded DuckDB instances directly to the nodes before training.

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

**2. Push Cache via Rsync or SCP:**

```bash
# Using rsync (recommended for delta transfers and resume support)
rsync -avzP ./data/jaeb_raw/ t1d_analytics.duckdb $USER@$TPU_IP:~/t1d-analytics/

# Alternative using scp
scp -r ./data/jaeb_raw/ t1d_analytics.duckdb $USER@$TPU_IP:~/t1d-analytics/
```

## 1. Initial Deployment

### Automated Stack Provisioning (`provision`)

To provision the cloud infrastructure, mount the persistent data disk (`$DATA_DISK_MOUNT`), install dependencies (`python 3.12`, `nodejs 20.x`, `duckdb`, `nginx`, `ollama`, `uv`), initialize the DuckDB analytics database via `provision_db`, build the frontend, configure Nginx SPA and API reverse-proxy ingress via `netctl`, and start the `t1d-api` and `t1d-web` services:

**Default Cloud Provider (`$CLOUD_PROVIDER` from `.env`):**

```sh
./libscript.sh provision "$CLOUD_PROVIDER" healthplatform-node "$RG_NAME" "$LOCATION" ./ t1d-analytics
```

**AWS:**

```sh
./libscript.sh provision aws healthplatform-node healthplatform-vpc us-east-1 ./ t1d-analytics
```

**GCP:**

```sh
./libscript.sh provision gcp healthplatform-node healthplatform-project us-central1-a ./ t1d-analytics
```

### Local & Declarative Stack Orchestration

You can also orchestrate dependencies, databases, frontend builds, services, reverse-proxy ingress, and ML workflows directly from `libscript.json`:

```sh
# Resolve and install stack dependencies + run install hooks
./libscript.sh install-deps libscript.json

# Provision DuckDB analytics database and emit .env.libscript
./_lib/databases/provision_db.sh libscript.json

# Compile and stage frontend SPA assets
./_lib/app-servers/provision_frontend.sh --app-dir . --manifest libscript.json

# Emit or apply Nginx reverse-proxy and SPA ingress configuration
./_lib/networking/netctl/cli.sh apply libscript.json --format nginx

# Build frontend, provision DuckDB, start services, and apply Nginx ingress
./libscript.sh start
./libscript.sh status
./libscript.sh health
./libscript.sh logs
./libscript.sh stop

# Execute declarative ML training or inference (supports --target local|gcp-tpu-vm|gcp-gpu-vm|gke-xpk)
./libscript.sh train libscript.json --dry-run
./libscript.sh infer libscript.json --dry-run
./libscript.sh serve-ml libscript.json --target gcp-tpu-vm --backend vllm --dry-run
./libscript.sh eval-ml libscript.json --target gke-xpk --backend maxtext --dry-run
```

### Direct Node Provisioning (Low-Level Provider Subcommands)

If you need to create compute nodes or node groups directly via `libscript` provider routers:

**AWS:**

```sh
./libscript.sh aws node-group create healthplatform-node 1 ami-ubuntu-lts healthplatform-vpc --tags Key=Project,Value=StanfordResearch
```

**Default Cloud Provider (`$CLOUD_PROVIDER`):**

```sh
./libscript.sh "$CLOUD_PROVIDER" node create healthplatform-node "$PRIMARY_CLOUD_VM_IMAGE" "$RG_NAME" --vnet-name healthplatform-vnet --tags Project=StanfordResearch
```

**GCP:**

```sh
./libscript.sh gcp node create healthplatform-node ubuntu-2404-lts healthplatform-project --network healthplatform-vpc --labels project=stanfordresearch
```

## 2. Provisioning the Backup Target

Before performing backups, ensure you have an object storage bucket available:

**AWS S3:**

```sh
./libscript.sh aws storage create stanford-healthplatform-backups
```

**Default Cloud Object Storage (`$CLOUD_PROVIDER`):**

```sh
./libscript.sh "$CLOUD_PROVIDER" storage create "$PRIMARY_CLOUD_STORAGE_ACCOUNT"
```

**GCP Cloud Storage:**

```sh
./libscript.sh gcp storage create stanford-healthplatform-backups
```

## 3. Backing Up Application State

We need to selectively backup the DuckDB analytics database (`$DATA_DISK_MOUNT`), PostgreSQL directories, and Let's Encrypt certificates before tearing down the instance.

Using the explicit path retention feature:

```sh
./libscript.sh backup healthplatform-node --target "$CLOUD_PROVIDER" --paths "$DATA_DISK_MOUNT /var/lib/postgresql/data /etc/letsencrypt" --keep-last 5
```

## 4. Deprovisioning (With Data & IP Retention)

When the active research phase is paused, we can tear down the compute instances to save costs while retaining the Static IPs (so DNS mapping remains intact) and the underlying data disks (for rapid resumption).

**AWS:**

```sh
./libscript.sh deprovision aws healthplatform-node healthplatform-vpc us-east-1 --retain-ip --retain-data
```

**Default Cloud Provider (`$CLOUD_PROVIDER`):**

```sh
./libscript.sh deprovision "$CLOUD_PROVIDER" healthplatform-node "$RG_NAME" "$LOCATION" --retain-ip --retain-data
```

**GCP:**

```sh
./libscript.sh deprovision gcp healthplatform-node healthplatform-project us-central1-a --retain-ip --retain-data
```

## 5. Restoration & Reprovisioning

To bring the environment back online, we use the restore or `--from-backup` provisioning command. This will map the retained IP back to the new instance, re-attach the persisted data disks, and pull the designated backup archive to restore application state.

```sh
./libscript.sh restore healthplatform-node --from-backup latest
```

## 6. Automated Deployment & Health Verification (`test.sh` / `test.cmd`)

To automate provisioning, remote service inspection, and health validation across providers:

```bash
# Test deployment in mock mode locally (zero cloud resources consumed)
./test.sh --unit-mock

# Force live cloud execution with strict pre-flight credential enforcement
./test.sh --live-cloud

# Test deployment using .env defaults or override provider/region
CLOUD_PROVIDER="$CLOUD_PROVIDER" RG_NAME="$RG_NAME" LOCATION="$LOCATION" ./test.sh

# Run remote test suite as part of cloud verification
./test.sh --run-remote-tests

# Fall back to local execution without cloud provisioning
./test.sh --local
```

For Windows environments, use `test.cmd` with identical parameters and flags:

```cmd
test.cmd --unit-mock
test.cmd --live-cloud
test.cmd --local
test.cmd --run-remote-tests
```

### Reference: CLI Flags

| Flag                                       | Description                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--unit-mock`, `--mock-cloud`, `--dry-run` | Executes the full verification workflow in mock mode, validating signal handlers and smoke assertions without spending cloud resources.           |
| `--live-cloud`                             | Enforces live cloud provisioning; performs mandatory pre-flight CLI checks for `$PRIMARY_CLOUD_CLI_NAME`, `gcloud`, or `aws` and exits immediately if unauthenticated. |
| `--local`, `-l`                            | Runs the test suite locally on the current workstation using `pytest` and `npm test`.                                                             |
| `--run-remote-tests`, `-t`                 | Runs `pytest` remotely on the provisioned compute node after health checks pass.                                                                  |

### Reference: Environment Variables

| Variable             | Default (from `.env`)              | Description                                                   |
| -------------------- | ---------------------------------- | ------------------------------------------------------------- |
| `CLOUD_PROVIDER`     | `$DEFAULT_CLOUD_PROVIDER`          | Target cloud provider (`$DEFAULT_CLOUD_PROVIDER`, `gcp`, or `aws`). |
| `LIBSCRIPT_ROOT_DIR` | `$HOME/repos/libscript`            | Path to the Libscript root repository.                        |
| `LIBSCRIPT_CLI`      | `$LIBSCRIPT_ROOT_DIR/libscript.sh` | Path to Libscript executable wrapper.                         |
| `NODE_NAME`          | `test-t1d-analytics-node`          | Name of compute node to allocate.                             |
| `RG_NAME`            | `$RG_NAME`                         | Resource group or GCP project ID.                             |
| `LOCATION`           | `$LOCATION`                        | Cloud region / datacenter zone.                               |
| `REMOTE_DEST`        | `t1d-analytics-test`               | Target remote destination directory.                          |
| `T1D_MOCK_TPU`       | `0`                                | When `1`, simulates TPU workload creation and status polling. |
| `T1D_MOCK_GCS`       | `0`                                | When `1`, simulates GCS upload and checkpoint downloads.      |

### Reference: Exit Codes

| Code | Status  | Meaning                                                                         |
| ---- | ------- | ------------------------------------------------------------------------------- |
| `0`  | Success | Provisioning, remote smoke checks, and deprovisioning succeeded.                |
| `1`  | Failure | Cloud provisioning failed, credentials were missing, or health check timed out. |
