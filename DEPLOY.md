# Deployment Guide

This project is orchestrated using [LibScript](https://github.com/SamuelMarks/libscript), a zero-dependency, cross-platform provisioning framework. It uses a `libscript.json` file to declare infrastructure and software dependencies, which are then resolved and installed natively.

## Prerequisites

Ensure you have a local copy of LibScript available.
For example, if it is located at `~/repos/libscript`:

```bash
export LIBSCRIPT_PATH="$HOME/repos/libscript/libscript.sh"
```

## 1. Configure the Environment

This project defines its system dependencies (e.g., Python, NodeJS, Nginx) as well as its PaaS configuration (hooks, daemon services, ingress routes) in the `libscript.json` file at the root of the repository. Review this file to ensure the requested component versions align with your target environment.

If you need to customize ports, paths, or install methods, export the relevant LibScript variables before proceeding:

```bash
# Example overrides:
export NGINX_LISTEN_PORT=8080
export LIBSCRIPT_LOG_LEVEL=1
```

## 2. Install Dependencies

Run the `install-deps` command to automatically parse the JSON manifest, download, configure, and install the required stack components natively on your machine:

```bash
$LIBSCRIPT_PATH install-deps
```

_Note: LibScript may prompt for `sudo` privileges if it needs to install system-level packages (like `apt` or `brew` packages) or configure daemon services._

## 3. Start the Stack

Once dependencies are installed, you can orchestrate the entire application. The `start` command automatically executes the ETL data pipelines (unless already run), compiles the frontend, daemonizes the FastAPI server natively via `systemd` or `launchd`, and configures the Nginx reverse proxy.

```bash
$LIBSCRIPT_PATH start
```

### Skipping ETL Hooks

If you have already downloaded and compiled the datasets or want to iterate quickly without executing the `build` and `pre_start` hooks, you can pass the `--no-hooks` flag:

```bash
$LIBSCRIPT_PATH start --no-hooks
```

## 4. Managing the Deployment

LibScript acts as a process and service manager for the components it installs. You can use it to monitor the health and logs of your deployment:

**Check component status and health:**

```bash
$LIBSCRIPT_PATH status
$LIBSCRIPT_PATH health
```

**Tail real-time logs for all services:**

```bash
$LIBSCRIPT_PATH logs -f
```

**Stop the stack:**

```bash
$LIBSCRIPT_PATH stop
```

## 5. Remote Cloud Deployment (Azure, AWS, GCP)

You can deploy the entire `t1d-analytics` stack to major cloud providers (Azure, AWS, Google Cloud) using LibScript's native multi-cloud orchestration primitives. The workflow below uses Azure (`./libscript.sh azure`) as an example, but the exact same core primitives (`network`, `firewall`, `node`) map identically to AWS (`./libscript.sh aws`) and GCP (`./libscript.sh gcp`). This provisions the infrastructure, pushes the codebase, installs dependencies, and gracefully tears everything down when finished.

### Prerequisites
Ensure you are authenticated with the Azure CLI (`az login`) and have sufficient permissions to create network, compute, and DNS resources.

### Provision Infrastructure (Azure Example)

From your local LibScript repository root, run:

```bash
# Define our target environment
export RG="t1d-analytics-rg"
export NODE="t1d-web-node"
export LOCATION="eastus"

# Create the Resource Group
az group create --name "$RG" --location "$LOCATION"

# 1. Create the Network
./libscript.sh azure network create t1d-vnet "$RG" --location "$LOCATION"

# 2. Create the Firewall (Opening SSH, HTTP, and HTTPS)
./libscript.sh azure firewall create t1d-nsg "$RG" "22 80 443" --location "$LOCATION"

# 3. Create the Node (Passing Azure-native flags for a data-science workload)
./libscript.sh azure node create "$NODE" "Ubuntu2204" "$RG" \
    --size Standard_D4s_v3 \
    --os-disk-size-gb 128 \
    --vnet-name t1d-vnet \
    --nsg t1d-nsg
```

### Deploy Stack

Now that the infrastructure is up, push the orchestration framework and the application code, then execute the deployment natively.

```bash
# 1. Sync the LibScript core to the node (~/libscript)
./libscript.sh azure node sync "$NODE" "$RG"

# 2. Copy your local t1d-analytics repository to the remote node
./libscript.sh azure node exec "$NODE" "$RG" "mkdir -p ~/t1d-analytics"
./libscript.sh azure node deploy "$NODE" "$RG" "../new_research/stanford/t1d-analytics/" "~/t1d-analytics/"

# 3. Map DNS for Let's Encrypt TLS Validation
./libscript.sh azure dns map-node "$NODE" "$RG" "t1d-analytics.stanford.edu" "stanford.edu" "stanford-dns-rg"

# 4. Resolve and install all dependencies (Python, Node, Nginx)
./libscript.sh azure node exec "$NODE" "$RG" "cd ~/t1d-analytics && sudo ~/libscript/libscript.sh install-deps"

# 4. Start the application stack (Daemonizes FastAPI, Configures Nginx, runs ETL)
./libscript.sh azure node exec "$NODE" "$RG" "cd ~/t1d-analytics && sudo ~/libscript/libscript.sh start"
```

### Deprovision and Teardown

When your research session is complete, gracefully shut down the application stack and delete the infrastructure layers in reverse order:

```bash
# 1. Gracefully stop the t1d-analytics stack and its daemonized services
./libscript.sh azure node exec "$NODE" "$RG" "cd ~/t1d-analytics && sudo ~/libscript/libscript.sh stop"

# 2. Delete the Node (and its associated OS disk)
./libscript.sh azure node delete "$NODE" "$RG"

# 3. Delete the Firewall (NSG)
./libscript.sh azure firewall delete t1d-nsg "$RG"

# 4. Delete the Network (VNET)
./libscript.sh azure network delete t1d-vnet "$RG"

# 5. (Optional) Purge the entire Resource Group
az group delete --name "$RG" --yes --no-wait
```
