# Deploying to TPU

This document outlines how to deploy the prepared datasets for Gemma-4-SQL training on TPU instances.

## Uploading the DuckDB Database to GCS

Once you have generated the training pairs and prepared the `t1d.duckdb` file, upload it to Google Cloud Storage (GCS) so that it is accessible to the training environment.

```bash
gcloud storage cp t1d.duckdb gs://your-bucket-name/t1d.duckdb
```

## Configuring TPU Instances

You need to configure the `ml-training/tpu-vm-eval-node` stack (via `libscript`) to mount the GCS bucket using `gcsfuse`. This allows the remote TPU instances (using `maxtext` and `jax`) to access the DuckDB file seamlessly.

1. Ensure the TPU instance has the correct service account permissions to access the GCS bucket.
2. Use `libscript` to deploy the environment and configure the mount:

```bash
# Example configuration to mount the bucket
./libscript.sh cloud gcp node configure tpu-node --mount-gcs gs://your-bucket-name /mnt/gcs
```

Once mounted, point your training scripts to the file located at `/mnt/gcs/t1d.duckdb`.
