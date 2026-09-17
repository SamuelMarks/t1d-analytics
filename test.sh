#!/bin/sh

set -e

# Configuration with configurable defaults
CLOUD_PROVIDER="${CLOUD_PROVIDER:-azure}"
LIBSCRIPT_ROOT_DIR="${LIBSCRIPT_ROOT_DIR:-$HOME/repos/libscript}"
LIBSCRIPT_CLI="${LIBSCRIPT_CLI:-$LIBSCRIPT_ROOT_DIR/libscript.sh}"
NODE_NAME="${NODE_NAME:-test-t1d-analytics-node}"
RG_NAME="${RG_NAME:-rg-analytics-test}"
LOCATION="${LOCATION:-eastus}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DEST="${REMOTE_DEST:-t1d-analytics-test}"

RUN_LOCAL=0
RUN_REMOTE_TESTS=0
MOCK_CLOUD=0
LIVE_CLOUD=0

for arg in "$@"; do
    case "$arg" in
        --unit-mock|--mock-cloud|--dry-run)
            MOCK_CLOUD=1
            ;;
        --live-cloud)
            LIVE_CLOUD=1
            ;;
        --local|-l)
            RUN_LOCAL=1
            ;;
        --run-remote-tests|-t)
            RUN_REMOTE_TESTS=1
            ;;
        --help|-h)
            echo "Usage: $0 [--unit-mock|--mock-cloud] [--live-cloud] [--local|-l] [--run-remote-tests|-t] [args]"
            echo "Options:"
            echo "  --unit-mock, --mock-cloud  Simulate cloud operations locally without cloud resources."
            echo "  --live-cloud               Force live cloud provisioning and execution; fails if auth missing."
            echo "  --local, -l                Run tests locally using installed pytest/npm."
            echo "  --run-remote-tests, -t     Run pytest directly on remote compute instance after provisioning."
            echo "Environment variables:"
            echo "  CLOUD_PROVIDER     Cloud provider (azure, gcp, aws). Default: azure"
            echo "  LIBSCRIPT_ROOT_DIR Path to Libscript repo. Default: \$HOME/repos/libscript"
            echo "  NODE_NAME          Name of compute node. Default: test-t1d-analytics-node"
            echo "  RG_NAME            Resource group or project name. Default: rg-analytics-test"
            echo "  LOCATION           Cloud region/datacenter. Default: eastus"
            echo "  REMOTE_DEST        Destination folder on remote node. Default: t1d-analytics-test"
            exit 0
            ;;
    esac
done

if [ "$MOCK_CLOUD" -eq 1 ]; then
    echo "=========================================================="
    echo "Running in Mock Cloud mode (--mock-cloud specified)..."
    echo "Simulating provisioning on $CLOUD_PROVIDER ($NODE_NAME)..."
    echo "=========================================================="
    MOCK_CLEANED_UP=0
    mock_cleanup() {
        if [ "$MOCK_CLEANED_UP" -eq 0 ]; then
            MOCK_CLEANED_UP=1
            echo "Mock deprovisioning cloud resources on $CLOUD_PROVIDER ($NODE_NAME)..."
            echo "Deprovisioning complete."
        fi
    }
    trap mock_cleanup EXIT INT TERM

    echo "Verifying remote execution..."
    echo "Deployment test successful"
    echo "Verifying backend API health status on remote host..."
    echo '{"status": "healthy", "version": "0.1.0"}'
    echo "Verifying frontend web service on remote host..."
    echo "HTTP/1.1 200 OK"
    if [ "$RUN_REMOTE_TESTS" -eq 1 ]; then
        echo "Running test suite on mock remote instance..."
        pytest -q
    fi
    echo "Deployment and remote health verification passed successfully."
    exit 0
fi

if [ "$RUN_LOCAL" -eq 1 ] || [ ! -x "$LIBSCRIPT_CLI" ]; then
    echo "=========================================================="
    if [ "$RUN_LOCAL" -eq 1 ]; then
        echo "Running test suite locally (--local specified)..."
    else
        echo "Libscript CLI not found at '$LIBSCRIPT_CLI'."
        echo "Falling back to local test suite..."
    fi
    echo "=========================================================="
    pytest
    if [ -d "web" ]; then
        (cd web && npm test)
    fi
    echo "Local test suite completed successfully."
    exit 0
fi

CLEANED_UP=0
cleanup() {
    if [ "$CLEANED_UP" -eq 0 ]; then
        CLEANED_UP=1
        echo "Cleaning up cloud resources on $CLOUD_PROVIDER ($NODE_NAME)..."
        "$LIBSCRIPT_CLI" deprovision "$CLOUD_PROVIDER" "$NODE_NAME" "$RG_NAME" "$LOCATION" "$REPO_DIR" "$REMOTE_DEST" || echo "Cleanup failed, but continuing."
        echo "Deprovisioning complete."
    fi
}

trap cleanup EXIT INT TERM

check_cloud_credentials() {
    echo "Verifying CLI authentication credentials for '$CLOUD_PROVIDER'..."
    case "$CLOUD_PROVIDER" in
        azure)
            if command -v az >/dev/null 2>&1; then
                if ! az account show >/dev/null 2>&1; then
                    echo "WARNING: Azure CLI not authenticated. Run 'az login'." >&2
                    if [ "$LIVE_CLOUD" -eq 1 ]; then
                        echo "ERROR: --live-cloud requires authenticated Azure credentials." >&2
                        exit 1
                    fi
                fi
            elif [ "$LIVE_CLOUD" -eq 1 ]; then
                echo "ERROR: Azure CLI ('az') not found." >&2
                exit 1
            fi
            ;;
        gcp)
            if command -v gcloud >/dev/null 2>&1; then
                if ! gcloud auth print-access-token >/dev/null 2>&1; then
                    echo "WARNING: Google Cloud CLI not authenticated. Run 'gcloud auth login'." >&2
                    if [ "$LIVE_CLOUD" -eq 1 ]; then
                        echo "ERROR: --live-cloud requires authenticated GCP credentials." >&2
                        exit 1
                    fi
                fi
            elif [ "$LIVE_CLOUD" -eq 1 ]; then
                echo "ERROR: Google Cloud CLI ('gcloud') not found." >&2
                exit 1
            fi
            ;;
        aws)
            if command -v aws >/dev/null 2>&1; then
                if ! aws sts get-caller-identity >/dev/null 2>&1; then
                    echo "WARNING: AWS CLI not authenticated. Run 'aws configure'." >&2
                    if [ "$LIVE_CLOUD" -eq 1 ]; then
                        echo "ERROR: --live-cloud requires authenticated AWS credentials." >&2
                        exit 1
                    fi
                fi
            elif [ "$LIVE_CLOUD" -eq 1 ]; then
                echo "ERROR: AWS CLI ('aws') not found." >&2
                exit 1
            fi
            ;;
    esac
}

START_TIME=$(date +%s)
check_cloud_credentials

echo "Starting $CLOUD_PROVIDER deployment test for t1d-analytics..."
echo "Node: $NODE_NAME, Target: $RG_NAME ($LOCATION)"

# Provision the stack
if ! "$LIBSCRIPT_CLI" provision "$CLOUD_PROVIDER" "$NODE_NAME" "$RG_NAME" "$LOCATION" "$REPO_DIR" "$REMOTE_DEST"; then
    echo "ERROR: Provisioning failed on $CLOUD_PROVIDER." >&2
    exit 1
fi

remote_exec() {
    _CMD="$1"
    if [ -f "$LIBSCRIPT_ROOT_DIR/_lib/cloud-providers/$CLOUD_PROVIDER/cli.sh" ]; then
        "$LIBSCRIPT_ROOT_DIR/_lib/cloud-providers/$CLOUD_PROVIDER/cli.sh" node exec "$NODE_NAME" "$RG_NAME" "$_CMD"
    else
        "$LIBSCRIPT_CLI" cloud "$CLOUD_PROVIDER" node exec "$NODE_NAME" "$RG_NAME" "$_CMD"
    fi
}

# 1. Verify remote node execution connectivity
echo "Verifying remote execution..."
if ! remote_exec "echo 'Deployment test successful'"; then
    echo "ERROR: Remote connectivity test failed on $CLOUD_PROVIDER ($NODE_NAME)." >&2
    exit 1
fi

# 2. Verify backend API health endpoint
echo "Verifying backend API health status on remote host..."
if ! remote_exec "curl -sf http://localhost:8000/api/status || curl -sf http://127.0.0.1:8000/api/status"; then
    echo "WARNING: Backend API health endpoint check failed. Gathering diagnostic logs..."
    remote_exec "journalctl -u t1d-api -n 30 --no-pager 2>/dev/null || docker logs --tail 30 t1d-backend 2>/dev/null || true"
    exit 1
fi

# 3. Verify frontend response
echo "Verifying frontend web service on remote host..."
if ! remote_exec "curl -sf -I http://localhost:3000/ || curl -sf -I http://127.0.0.1:3000/"; then
    echo "WARNING: Frontend check failed. Gathering diagnostic logs..."
    remote_exec "journalctl -u t1d-web -n 30 --no-pager 2>/dev/null || docker logs --tail 30 t1d-frontend 2>/dev/null || true"
    exit 1
fi

# 4. Optional: Run full test suite on remote compute instance
if [ "$RUN_REMOTE_TESTS" -eq 1 ]; then
    echo "Running test suite on remote instance..."
    if ! remote_exec "cd $REMOTE_DEST && pytest -q"; then
        echo "ERROR: Remote test suite failed." >&2
        exit 1
    fi
    echo "Remote test suite passed."
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Deployment and remote health verification passed successfully in ${DURATION}s."
