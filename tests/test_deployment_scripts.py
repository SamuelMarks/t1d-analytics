"""Comprehensive unit tests for test.sh and test.cmd cloud deployment scripts."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SH_PATH = REPO_ROOT / "test.sh"
TEST_CMD_PATH = REPO_ROOT / "test.cmd"
WINE_BINARY = shutil.which("wine") or "/Users/samuel/.local/bin/wine"
HAS_WINE = Path(WINE_BINARY).is_file() and os.access(WINE_BINARY, os.X_OK)


def make_executable(script_path: Path, content: str) -> None:
    """
    Write an executable shell script to disk.

    Args:
    ----
        script_path: Path where the script should be written.
        content: Shell script body content.

    """
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(
        script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


def run_test_sh(
    args: list[str],
    env_overrides: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Execute test.sh with isolated environment overrides.

    Args:
    ----
        args: List of command-line arguments.
        env_overrides: Optional environment variables dictionary.
        cwd: Optional working directory for the process.

    Returns:
    -------
        subprocess.CompletedProcess with text output.

    """
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["/bin/sh", str(TEST_SH_PATH)] + args,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


def test_test_sh_help() -> None:
    """Test that test.sh --help and -h display usage and options."""
    for flag in ["--help", "-h"]:
        res = run_test_sh([flag])
        assert res.returncode == 0
        assert "Usage: " in res.stdout
        assert "--unit-mock, --mock-cloud" in res.stdout
        assert "--live-cloud" in res.stdout
        assert "--local, -l" in res.stdout
        assert "CLOUD_PROVIDER" in res.stdout


def test_test_sh_mock_cloud_modes(tmp_path: Path) -> None:
    """
    Test test.sh mock cloud modes with and without remote test execution.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_pytest = bin_dir / "pytest"
    make_executable(fake_pytest, "#!/bin/sh\necho 'fake pytest passed'\nexit 0\n")

    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    for flag in ["--unit-mock", "--mock-cloud", "--dry-run"]:
        res = run_test_sh([flag], env_overrides=env)
        assert res.returncode == 0
        assert "Running in Mock Cloud mode" in res.stdout
        assert "Deployment test successful" in res.stdout
        assert (
            "Deployment and remote health verification passed successfully."
            in res.stdout
        )
        assert "Mock deprovisioning cloud resources" in res.stdout
        assert "Deprovisioning complete." in res.stdout

    # With remote tests
    for flag in ["--run-remote-tests", "-t"]:
        res_t = run_test_sh(["--mock-cloud", flag], env_overrides=env)
        assert res_t.returncode == 0
        assert "Running test suite on mock remote instance..." in res_t.stdout


def test_test_sh_local_tests(tmp_path: Path) -> None:
    """
    Test test.sh --local flag and automatic fallback when Libscript CLI is missing.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_pytest = bin_dir / "pytest"
    make_executable(fake_pytest, "#!/bin/sh\necho 'local pytest executed'\nexit 0\n")
    fake_npm = bin_dir / "npm"
    make_executable(fake_npm, "#!/bin/sh\necho 'local npm test executed'\nexit 0\n")

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LIBSCRIPT_CLI": str(tmp_path / "nonexistent-libscript.sh"),
    }

    # 1. Fallback when CLI missing
    workdir_with_web = tmp_path / "repo1"
    workdir_with_web.mkdir()
    (workdir_with_web / "web").mkdir()
    res_fallback = run_test_sh([], env_overrides=env, cwd=workdir_with_web)
    assert res_fallback.returncode == 0
    assert "Libscript CLI not found" in res_fallback.stdout
    assert "Local test suite completed successfully." in res_fallback.stdout

    # 2. Explicit --local and -l without web folder
    workdir_no_web = tmp_path / "repo2"
    workdir_no_web.mkdir()
    for flag in ["--local", "-l"]:
        res_local = run_test_sh([flag], env_overrides=env, cwd=workdir_no_web)
        assert res_local.returncode == 0
        assert "Running test suite locally (--local specified)..." in res_local.stdout
        assert "Local test suite completed successfully." in res_local.stdout


def test_test_sh_cloud_credentials_verification(tmp_path: Path) -> None:
    """
    Test check_cloud_credentials for azure, gcp, aws, and unknown providers.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_cli = tmp_path / "libscript.sh"
    make_executable(fake_cli, "#!/bin/sh\nexit 1\n")

    # Helper script creator
    def create_tool(name: str, exit_code: int) -> None:
        """
        Create a mock tool executable with a fixed exit code.

        Args:
        ----
            name: Tool binary name to write.
            exit_code: Desired exit code for the tool.

        """
        p = bin_dir / name
        make_executable(p, f"#!/bin/sh\nexit {exit_code}\n")

    # 1. Azure credential checks
    create_tool("az", 0)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CLOUD_PROVIDER": "azure",
        "LIBSCRIPT_CLI": str(fake_cli),
    }
    # az succeeds -> continues to provision (which fails with exit 1)
    res = run_test_sh([], env_overrides=env)
    assert "Verifying CLI authentication credentials for 'azure'..." in res.stdout
    assert "ERROR: Provisioning failed on azure." in res.stderr

    # az fails -> warning emitted
    create_tool("az", 1)
    res_warn = run_test_sh([], env_overrides=env)
    assert "WARNING: Azure CLI not authenticated. Run 'az login'." in res_warn.stderr

    # az fails with --live-cloud -> exit 1 with error
    res_live_fail = run_test_sh(["--live-cloud"], env_overrides=env)
    assert res_live_fail.returncode == 1
    assert (
        "ERROR: --live-cloud requires authenticated Azure credentials."
        in res_live_fail.stderr
    )

    # Prepare an isolated PATH environment without cloud CLIs (az, gcloud, aws)
    # to avoid host tools installed in standard system directories like /usr/bin.
    isolated_bin = tmp_path / "isolated_bin"
    isolated_bin.mkdir()
    for tool_name in [
        "date",
        "dirname",
        "basename",
        "sh",
        "echo",
        "pwd",
        "cat",
        "grep",
        "[",
        "test",
    ]:
        tool_src = shutil.which(tool_name)
        if tool_src:
            dest = isolated_bin / tool_name
            try:
                dest.symlink_to(tool_src)
            except OSError:
                shutil.copy2(tool_src, dest)

    # az missing with --live-cloud
    (bin_dir / "az").unlink()
    env_clean = {
        "PATH": str(isolated_bin),
        "CLOUD_PROVIDER": "azure",
        "LIBSCRIPT_CLI": str(fake_cli),
    }
    res_az_missing = run_test_sh(["--live-cloud"], env_overrides=env_clean)
    assert res_az_missing.returncode == 1
    assert "ERROR: Azure CLI ('az') not found." in res_az_missing.stderr

    # 2. GCP credential checks
    create_tool("gcloud", 0)
    env["CLOUD_PROVIDER"] = "gcp"
    res_gcp = run_test_sh([], env_overrides=env)
    assert "Verifying CLI authentication credentials for 'gcp'..." in res_gcp.stdout

    create_tool("gcloud", 1)
    res_gcp_warn = run_test_sh([], env_overrides=env)
    assert "WARNING: Google Cloud CLI not authenticated." in res_gcp_warn.stderr

    res_gcp_live_fail = run_test_sh(["--live-cloud"], env_overrides=env)
    assert res_gcp_live_fail.returncode == 1
    assert (
        "ERROR: --live-cloud requires authenticated GCP credentials."
        in res_gcp_live_fail.stderr
    )

    (bin_dir / "gcloud").unlink()
    res_gcloud_missing = run_test_sh(
        ["--live-cloud"],
        env_overrides={**env_clean, "CLOUD_PROVIDER": "gcp"},
    )
    assert res_gcloud_missing.returncode == 1
    assert "ERROR: Google Cloud CLI ('gcloud') not found." in res_gcloud_missing.stderr

    # 3. AWS credential checks
    create_tool("aws", 0)
    env["CLOUD_PROVIDER"] = "aws"
    res_aws = run_test_sh([], env_overrides=env)
    assert "Verifying CLI authentication credentials for 'aws'..." in res_aws.stdout

    create_tool("aws", 1)
    res_aws_warn = run_test_sh([], env_overrides=env)
    assert "WARNING: AWS CLI not authenticated." in res_aws_warn.stderr

    res_aws_live_fail = run_test_sh(["--live-cloud"], env_overrides=env)
    assert res_aws_live_fail.returncode == 1
    assert (
        "ERROR: --live-cloud requires authenticated AWS credentials."
        in res_aws_live_fail.stderr
    )

    (bin_dir / "aws").unlink()
    res_aws_missing = run_test_sh(
        ["--live-cloud"],
        env_overrides={**env_clean, "CLOUD_PROVIDER": "aws"},
    )
    assert res_aws_missing.returncode == 1
    assert "ERROR: AWS CLI ('aws') not found." in res_aws_missing.stderr

    # 4. Unknown cloud provider
    env["CLOUD_PROVIDER"] = "custom_cloud"
    res_custom = run_test_sh([], env_overrides=env)
    assert (
        "Verifying CLI authentication credentials for 'custom_cloud'..."
        in res_custom.stdout
    )


def test_test_sh_remote_execution_branches(tmp_path: Path) -> None:
    """
    Test remote execution verification steps, failures, and diagnostic log collection.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    cli_script = tmp_path / "libscript.sh"

    # 1. Connectivity test fails
    cli_code_conn_fail = """#!/bin/sh
cmd="$1"
if [ "$cmd" = "provision" ]; then exit 0; fi
if [ "$cmd" = "deprovision" ]; then exit 0; fi
if [ "$cmd" = "cloud" ]; then
    # Remote execution connectivity check fails
    exit 1
fi
exit 0
"""
    make_executable(cli_script, cli_code_conn_fail)
    env = {
        "LIBSCRIPT_CLI": str(cli_script),
        "CLOUD_PROVIDER": "custom_cloud",
    }
    res_conn_fail = run_test_sh([], env_overrides=env)
    assert res_conn_fail.returncode == 1
    assert "ERROR: Remote connectivity test failed" in res_conn_fail.stderr

    # 2. Backend API health check fails
    cli_code_backend_fail = """#!/bin/sh
cmd="$1"
if [ "$cmd" = "provision" ]; then exit 0; fi
if [ "$cmd" = "deprovision" ]; then exit 0; fi
if [ "$cmd" = "cloud" ]; then
    shift 6
    remote_cmd="$1"
    case "$remote_cmd" in
        *Deployment*) exit 0 ;;
        *api/status*) exit 1 ;;
        *journalctl*|*docker*) echo "mock diagnostic logs collected"; exit 0 ;;
    esac
fi
exit 0
"""
    make_executable(cli_script, cli_code_backend_fail)
    res_back_fail = run_test_sh([], env_overrides=env)
    assert res_back_fail.returncode == 1
    assert (
        "WARNING: Backend API health endpoint check failed. Gathering diagnostic logs..."
        in res_back_fail.stdout
    )
    assert "mock diagnostic logs collected" in res_back_fail.stdout

    # 3. Frontend check fails
    cli_code_frontend_fail = """#!/bin/sh
cmd="$1"
if [ "$cmd" = "provision" ]; then exit 0; fi
if [ "$cmd" = "deprovision" ]; then exit 0; fi
if [ "$cmd" = "cloud" ]; then
    shift 6
    remote_cmd="$1"
    case "$remote_cmd" in
        *Deployment*) exit 0 ;;
        *api/status*) exit 0 ;;
        *:3000*) exit 1 ;;
        *journalctl*|*docker*) echo "frontend diagnostic logs collected"; exit 0 ;;
    esac
fi
exit 0
"""
    make_executable(cli_script, cli_code_frontend_fail)
    res_front_fail = run_test_sh([], env_overrides=env)
    assert res_front_fail.returncode == 1
    assert (
        "WARNING: Frontend check failed. Gathering diagnostic logs..."
        in res_front_fail.stdout
    )
    assert "frontend diagnostic logs collected" in res_front_fail.stdout

    # 4. Remote tests fail
    cli_code_remote_fail = """#!/bin/sh
cmd="$1"
if [ "$cmd" = "provision" ]; then exit 0; fi
if [ "$cmd" = "deprovision" ]; then exit 0; fi
if [ "$cmd" = "cloud" ]; then
    shift 6
    remote_cmd="$1"
    case "$remote_cmd" in
        *pytest*) exit 1 ;;
        *) exit 0 ;;
    esac
fi
exit 0
"""
    make_executable(cli_script, cli_code_remote_fail)
    res_rem_fail = run_test_sh(["--run-remote-tests"], env_overrides=env)
    assert res_rem_fail.returncode == 1
    assert "ERROR: Remote test suite failed." in res_rem_fail.stderr

    # 5. Full success with provider-specific cli.sh and remote tests
    libscript_root = tmp_path / "libscript"
    provider_dir = libscript_root / "_lib" / "cloud-providers" / "custom_cloud"
    provider_dir.mkdir(parents=True)
    provider_cli = provider_dir / "cli.sh"
    provider_code = """#!/bin/sh
echo "custom provider cli executing: $*"
exit 0
"""
    make_executable(provider_cli, provider_code)

    cli_code_success = """#!/bin/sh
cmd="$1"
if [ "$cmd" = "provision" ]; then exit 0; fi
if [ "$cmd" = "deprovision" ]; then
    echo "deprovisioning triggered"
    exit 0
fi
exit 0
"""
    make_executable(cli_script, cli_code_success)
    env_success = {
        "LIBSCRIPT_CLI": str(cli_script),
        "LIBSCRIPT_ROOT_DIR": str(libscript_root),
        "CLOUD_PROVIDER": "custom_cloud",
    }
    res_success = run_test_sh(["-t"], env_overrides=env_success)
    assert res_success.returncode == 0
    assert "custom provider cli executing:" in res_success.stdout
    assert "Remote test suite passed." in res_success.stdout
    assert (
        "Deployment and remote health verification passed successfully in "
        in res_success.stdout
    )
    assert "deprovisioning triggered" in res_success.stdout


def test_test_sh_deprovision_failure_handling(tmp_path: Path) -> None:
    """
    Test that cleanup handles deprovision command failures gracefully.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    cli_script = tmp_path / "libscript.sh"
    cli_code = """#!/bin/sh
cmd="$1"
if [ "$cmd" = "provision" ]; then exit 1; fi
if [ "$cmd" = "deprovision" ]; then exit 1; fi
"""
    make_executable(cli_script, cli_code)
    env = {
        "LIBSCRIPT_CLI": str(cli_script),
        "CLOUD_PROVIDER": "custom_cloud",
    }
    res = run_test_sh([], env_overrides=env)
    assert res.returncode == 1
    assert "Cleanup failed, but continuing." in res.stdout


def test_test_cmd_static_validation() -> None:
    """Statically verify test.cmd batch file labels, branches, and flags."""
    assert TEST_CMD_PATH.is_file()
    content = TEST_CMD_PATH.read_text(encoding="utf-8")

    expected_labels = [
        ":parse_args",
        ":show_help",
        ":after_args",
        ":cleanup",
        ":run_mock_cloud",
        ":run_local_tests",
    ]
    for label in expected_labels:
        assert label in content, f"Missing label {label} in test.cmd"

    # Verify command patterns
    assert 'call "%LIBSCRIPT_CLI%" provision' in content
    assert 'call "%LIBSCRIPT_CLI%" deprovision' in content
    assert 'call "%LIBSCRIPT_CLI%" cloud' in content
    assert "echo Deployment test successful" in content
    assert "api/status" in content
    assert "npm test" in content


def test_test_cmd_execution_with_wine() -> None:
    """Test test.cmd execution under wine cmd if wine is available."""
    if not HAS_WINE:
        return

    # 1. Test --help and -h
    for flag in ["--help", "-h"]:
        res_help = subprocess.run(
            [WINE_BINARY, "cmd", "/c", "test.cmd", flag],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Usage: test.cmd" in res_help.stdout
        assert "--unit-mock" in res_help.stdout
        assert "--live-cloud" in res_help.stdout

    # 2. Test mock cloud modes
    for flag in ["--mock-cloud", "--unit-mock", "--dry-run"]:
        res_mock = subprocess.run(
            [WINE_BINARY, "cmd", "/c", "test.cmd", flag],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Running in Mock Cloud mode" in res_mock.stdout
        assert "Deployment test successful" in res_mock.stdout
        assert "Deprovisioning complete." in res_mock.stdout
