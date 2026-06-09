#!/bin/sh

set -e

LIBSCRIPT_ROOT_DIR="${LIBSCRIPT_ROOT_DIR:-$HOME/repos/libscript}"
LIBSCRIPT_CLI="$LIBSCRIPT_ROOT_DIR/libscript.sh"
NODE_NAME="${NODE_NAME:-test-t1d-analytics-node}"
RG_NAME="${RG_NAME:-rg-analytics-test11}"
LOCATION="${LOCATION:-eastus}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DEST="${REMOTE_DEST:-t1d-analytics-test}"

cleanup() {
    echo "Cleaning up resources..."
    "$LIBSCRIPT_CLI" deprovision azure "$NODE_NAME" "$RG_NAME" "$LOCATION" "$REPO_DIR" "$REMOTE_DEST" || echo "Cleanup failed, but continuing."
    echo "Deprovisioning complete."
}

trap cleanup EXIT INT TERM

echo "Starting deployment test for t1d-analytics..."

# Provision the stack
"$LIBSCRIPT_CLI" provision azure "$NODE_NAME" "$RG_NAME" "$LOCATION" "$REPO_DIR" "$REMOTE_DEST"

# Check if node is up by running a simple remote command
echo "Verifying remote execution..."
"$LIBSCRIPT_ROOT_DIR/_lib/cloud-providers/azure/cli.sh" node exec "$NODE_NAME" "$RG_NAME" "echo 'Deployment test successful'"

echo "Test passed."

