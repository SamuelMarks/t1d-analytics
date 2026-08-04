# Stanford HealthPlatform - Deployment Guide

This repository contains the configuration and source code for the Stanford HealthPlatform research tools. This document provides step-by-step instructions on how to use libscript to deploy, backup, and restore this specific stack across major cloud providers.

## Pre-requisites

Ensure libscript is installed and you are authenticated to your chosen cloud provider.

- **AWS**: aws configure
- **Azure**: az login
- **GCP**: gcloud auth login

## T1D Domain Pipeline

For users specifically looking to deploy the T1D domain pipeline for Text-to-SQL training with Gemma-4-SQL, please refer to the [T1D Integration Guide](T1D_INTEGRATION_GUIDE.md) and the granular [Execution Checklist](TODO_PLAN1.md) for a comprehensive step-by-step tutorial. 

### TPU Research Cloud (TFRC) Deployments
If you are deploying the training pipeline using a TFRC grant, ensure you provide the correct spot scheduling flags prior to provisioning via libscript:
```bash
export TPU_SCHEDULING_TYPE="spot"
export TPU_USE_QUEUED_RESOURCE="true"
export TPU_ZONE="us-central2-b" # MUST explicitly match your grant
export TPU_ACCELERATOR_TYPE="v4-8" # MUST explicitly match your grant
```

### Remote Data Caching Optimization
To prevent massive downloads on ephemeral TPU compute, use `rsync` to push your pre-loaded DuckDB instances directly to the nodes before training:
```bash
rsync -avzP t1d_analytics.duckdb $USER@$TPU_IP:~/t1d-analytics/
```

## 1. Initial Deployment

To provision the infrastructure and deploy the stack:

**AWS:**

```sh
./libscript.sh cloud aws node-group create healthplatform-node 1 ami-ubuntu-lts healthplatform-vpc --tags Key=Project,Value=StanfordResearch
```

**Azure:**

```sh
./libscript.sh cloud azure node create healthplatform-node Ubuntu2204 healthplatform-rg --vnet-name healthplatform-vnet --tags Project=StanfordResearch
```

**GCP:**

```sh
./libscript.sh cloud gcp node create healthplatform-node ubuntu-2204-lts healthplatform-project --network healthplatform-vpc --labels project=stanfordresearch
```

## 2. Provisioning the Backup Target

Before performing backups, ensure you have an object storage bucket available:

**AWS S3:**

```sh
./libscript.sh cloud aws storage create stanford-healthplatform-backups
```

**Azure Blob Storage:**

```sh
./libscript.sh cloud azure storage create stanfordhealthbackups
```

**GCP Cloud Storage:**

```sh
./libscript.sh cloud gcp storage create stanford-healthplatform-backups
```

## 3. Backing Up Application State

We need to selectively backup the shared Postgres database directories and Let's Encrypt certificates before tearing down the instance.

Using the explicit path retention feature:

```sh
./libscript.sh cloud backup healthplatform-node --target azure --paths "/var/lib/postgresql/data /etc/letsencrypt" --keep-last 5
```

## 4. Deprovisioning (With Data & IP Retention)

When the active research phase is paused, we can tear down the compute instances to save costs while retaining the Static IPs (so DNS mapping remains intact) and the underlying data disks (for rapid resumption).

**AWS:**

```sh
./libscript.sh cloud deprovision aws healthplatform-node healthplatform-vpc us-east-1 --retain-ip --retain-data
```

**Azure:**

```sh
./libscript.sh cloud deprovision azure healthplatform-node healthplatform-rg eastus --retain-ip --retain-data
```

**GCP:**

```sh
./libscript.sh cloud deprovision gcp healthplatform-node healthplatform-project us-central1-a --retain-ip --retain-data
```

## 5. Restoration & Reprovisioning

To bring the environment back online, we use the restore command. This will map the retained IP back to the new instance, re-attach the persisted data disks, and pull the designated backup archive to restore the Postgres and LetsEncrypt state.

```sh
./libscript.sh cloud restore healthplatform-node --from-backup latest
```
