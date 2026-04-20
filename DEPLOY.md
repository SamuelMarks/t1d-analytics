# Deployment Guide

This project is orchestrated using [LibScript](https://github.com/samuel/libscript), a zero-dependency, cross-platform provisioning framework. It uses a `libscript.json` file to declare infrastructure and software dependencies, which are then resolved and installed natively.

## Prerequisites

Ensure you have a local copy of LibScript available.
For example, if it is located at `~/repos/libscript`:

```bash
export LIBSCRIPT_PATH="$HOME/repos/libscript/libscript.sh"
```

## 1. Configure the Environment

This project defines its system dependencies (e.g., Python, NodeJS, PostgreSQL, Redis, Nginx) in the `libscript.json` file at the root of the repository. Review this file to ensure the requested component versions align with your target environment.

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

Once dependencies are installed, start the background services defined in the stack (e.g., databases, caches, web servers):

```bash
$LIBSCRIPT_PATH start
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
