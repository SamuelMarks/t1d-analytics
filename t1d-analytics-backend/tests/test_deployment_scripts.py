"""Comprehensive unit tests for test.sh and test.cmd cloud deployment scripts."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_SH_PATH = REPO_ROOT / "test.sh"
TEST_CMD_PATH = REPO_ROOT / "test.cmd"
DOTENV_PATH = REPO_ROOT / ".env"
DOTENV_SECRETS_TPL_PATH = REPO_ROOT / ".env.secrets.tpl"
DOTENV_SECRETS_PATH = REPO_ROOT / ".env.secrets"


def load_dotenv_dict(
    paths: tuple[Path, ...] = (
        DOTENV_PATH,
        DOTENV_SECRETS_TPL_PATH,
        DOTENV_SECRETS_PATH,
    ),
) -> dict[str, str]:
    """
    Parse the repository .env, .env.secrets.tpl, and .env.secrets files into a dictionary.

    Args:
    ----
        paths: Tuple of dotenv file paths to read in order.

    Returns:
    -------
        dict[str, str]: Parsed environment variables.

    """
    env_map: dict[str, str] = {}
    for env_path in paths:
        if env_path.is_file():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                env_map[key.strip()] = val.strip().strip('"')
    return env_map


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
    (workdir_with_web / "t1d-analytics-web").mkdir()
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
    Test check_cloud_credentials for default provider (.env), gcp, aws, and unknown providers.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    dotenv_vars = load_dotenv_dict()
    default_provider = dotenv_vars["PRIMARY_CLOUD_PROVIDER"]
    default_cli = dotenv_vars["PRIMARY_CLOUD_CLI_NAME"]
    default_display = dotenv_vars["PRIMARY_CLOUD_DISPLAY_NAME"]
    default_login = dotenv_vars["PRIMARY_CLOUD_LOGIN_CMD"]

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

    # 1. Default provider credential checks (from .env)
    create_tool(default_cli, 0)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CLOUD_PROVIDER": default_provider,
        "LIBSCRIPT_CLI": str(fake_cli),
    }
    # CLI succeeds -> continues to provision (which fails with exit 1)
    res = run_test_sh([], env_overrides=env)
    assert (
        f"Verifying CLI authentication credentials for '{default_provider}'..."
        in res.stdout
    )
    assert f"ERROR: Provisioning failed on {default_provider}." in res.stderr

    # CLI fails -> warning emitted
    create_tool(default_cli, 1)
    res_warn = run_test_sh([], env_overrides=env)
    assert (
        f"WARNING: {default_display} CLI not authenticated. Run '{default_login}'."
        in res_warn.stderr
    )

    # CLI fails with --live-cloud -> exit 1 with error
    res_live_fail = run_test_sh(["--live-cloud"], env_overrides=env)
    assert res_live_fail.returncode == 1
    assert (
        f"ERROR: --live-cloud requires authenticated {default_display} credentials."
        in res_live_fail.stderr
    )

    # Prepare an isolated PATH environment without cloud CLIs
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
        assert tool_src is not None
        dest = isolated_bin / tool_name
        dest.symlink_to(tool_src)

    # CLI missing with --live-cloud
    (bin_dir / default_cli).unlink()
    env_clean = {
        "PATH": str(isolated_bin),
        "CLOUD_PROVIDER": default_provider,
        "LIBSCRIPT_CLI": str(fake_cli),
    }
    res_default_missing = run_test_sh(["--live-cloud"], env_overrides=env_clean)
    assert res_default_missing.returncode == 1
    assert (
        f"ERROR: {default_display} CLI ('{default_cli}') not found."
        in res_default_missing.stderr
    )

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
if [ "$cmd" = "custom_cloud" ] || [ "$cmd" = "cloud" ]; then
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
if [ "$cmd" = "custom_cloud" ]; then shift 5; elif [ "$cmd" = "cloud" ]; then shift 6; fi
if [ "$cmd" = "custom_cloud" ] || [ "$cmd" = "cloud" ]; then
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
if [ "$cmd" = "custom_cloud" ]; then shift 5; elif [ "$cmd" = "cloud" ]; then shift 6; fi
if [ "$cmd" = "custom_cloud" ] || [ "$cmd" = "cloud" ]; then
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
if [ "$cmd" = "custom_cloud" ]; then shift 5; elif [ "$cmd" = "cloud" ]; then shift 6; fi
if [ "$cmd" = "custom_cloud" ] || [ "$cmd" = "cloud" ]; then
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
        ":check_cloud_credentials",
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
    assert "exit /b !EXIT_CODE!" in content


def test_test_cmd_control_flow_and_arguments() -> None:
    """Verify structured CLI flag and control-flow parity between test.sh and test.cmd."""
    import re

    sh_content = TEST_SH_PATH.read_text(encoding="utf-8")
    cmd_content = TEST_CMD_PATH.read_text(encoding="utf-8")

    # 1. Extract CLI flags supported by test.sh's case "$arg" block and test.cmd's :parse_args block
    sh_case_match = re.search(r'case "\$arg" in(.*?)esac', sh_content, flags=re.DOTALL)
    assert sh_case_match is not None
    sh_flags = set(re.findall(r"(--[a-z-]+|-[hlt])\b", sh_case_match.group(1)))
    cmd_flags = set(re.findall(r'if "%~1"=="(--[a-z-]+|-[hlt])"', cmd_content))
    assert sh_flags == cmd_flags

    # 2. Verify every goto target in test.cmd resolves to a defined label
    cmd_gotos = {
        target
        for target in re.findall(r"\bgoto\s+:?([a-zA-Z0-9_]+)", cmd_content)
        if target.lower() != "eof"
    }
    cmd_labels = set(
        re.findall(r"^:([a-zA-Z0-9_]+)\b", cmd_content, flags=re.MULTILINE)
    )
    assert cmd_gotos <= cmd_labels

    # 3. Verify default configuration variable names match between test.sh and test.cmd
    expected_config_vars = {
        "CLOUD_PROVIDER",
        "LIBSCRIPT_ROOT_DIR",
        "LIBSCRIPT_CLI",
        "NODE_NAME",
        "RG_NAME",
        "LOCATION",
        "REMOTE_DEST",
        "RUN_LOCAL",
        "RUN_REMOTE_TESTS",
        "MOCK_CLOUD",
        "LIVE_CLOUD",
    }
    for var_name in expected_config_vars:
        assert re.search(rf"\b{var_name}=", sh_content)
        assert re.search(rf'\bset "{var_name}=', cmd_content)

    assert "Usage: %0" in cmd_content
    assert "Running in Mock Cloud mode" in cmd_content
    assert "Deprovisioning complete." in cmd_content
    assert "exit /b 0" in cmd_content
    assert "exit /b 1" in cmd_content
    assert (
        "--live-cloud requires authenticated %PRIMARY_CLOUD_DISPLAY_NAME% credentials."
        in cmd_content
    )
    assert "--live-cloud requires authenticated GCP credentials." in cmd_content
    assert "--live-cloud requires authenticated AWS credentials." in cmd_content


def test_libscript_config_and_docs_consistency() -> None:
    """Verify libscript.json, .env, and deployment markdown docs use current 5-tier component paths."""
    import json

    dotenv_vars = load_dotenv_dict()
    assert "REMOTE_USER" in dotenv_vars
    assert "DATA_DISK_MOUNT" in dotenv_vars
    assert "T1D_DB_PATH" in dotenv_vars

    secrets_tpl_path = REPO_ROOT / ".env.secrets.tpl"
    assert secrets_tpl_path.is_file()
    secrets_tpl_content = secrets_tpl_path.read_text(encoding="utf-8")
    assert "T1D_API_KEY=" in secrets_tpl_content
    assert "OPENAI_API_KEY=" in secrets_tpl_content
    assert "HF_TOKEN=" in secrets_tpl_content

    gitignore_content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env.secrets" in gitignore_content
    assert "!.env.secrets.tpl" in gitignore_content

    libscript_json_path = REPO_ROOT / "libscript.json"
    raw_libscript = libscript_json_path.read_text(encoding="utf-8")
    assert dotenv_vars["REMOTE_USER"] not in raw_libscript
    config = json.loads(raw_libscript)
    assert config["name"] == "t1d-analytics"
    assert ".env" in config["env_files"]
    assert ".env.secrets" in config["env_files"]
    assert "deps" in config
    assert config["deps"]["python"] == "3.12"
    assert config["deps"]["duckdb"] == "latest"
    assert config["deps"]["postgres"] == "18"
    assert "databases" in config
    db_engines = [db["engine"] for db in config["databases"]]
    assert "postgres" in db_engines
    assert "duckdb" in db_engines
    assert "frontends" in config
    assert config["frontends"]["web"]["source_dir"] in ("t1d-analytics-web", "web")
    assert config["frontends"]["web"]["output_dir"] == "dist"
    assert config["ingress"]["provider"] == "nginx"
    assert config["ingress"]["routes"]["/api"]["proxy"] in (
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
    )
    assert config["ingress"]["routes"]["/docs"]["proxy"] in (
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
    )
    assert config["ingress"]["routes"]["/openapi.json"]["proxy"] in (
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
    )
    assert config["ingress"]["routes"]["/"]["spa"] is True
    assert config["services"]["t1d-api"]["port"] in (8000, 8001)
    assert ".env" in config["services"]["t1d-api"]["env_files"]
    assert ".env.secrets" in config["services"]["t1d-api"]["env_files"]
    assert ".env.libscript" in config["services"]["t1d-api"]["env_files"]
    assert ".env" in config["services"]["t1d-web"]["env_files"]
    assert ".env.secrets" in config["services"]["t1d-web"]["env_files"]
    assert ".env.libscript" in config["services"]["t1d-web"]["env_files"]
    assert isinstance(config["hooks"]["install"][0], dict)
    assert "ml" in config
    assert config["ml"]["backend"] == "maxtext"
    assert config["ml"]["tpu_type"] == "v4-8"
    assert config["ml"]["scheduling"] == "spot"
    assert config["ml"]["slices"] == 1
    assert "train_command" in config["ml"]
    assert "infer_command" in config["ml"]

    required = config["dependencies"]["required"]

    db_names = [d["name"] for d in required.get("databases", [])]
    server_names = [s["name"] for s in required.get("servers", [])]
    toolchain_names = [t["name"] for t in required.get("toolchains", [])]

    assert "duckdb" in db_names
    assert "nginx" in server_names
    assert "ollama" in server_names
    assert "nodejs" in toolchain_names
    assert "python" in toolchain_names
    assert "uv" in toolchain_names

    readme_md = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "$LIBSCRIPT_PATH install-deps libscript.json" in readme_md
    assert "$LIBSCRIPT_PATH train libscript.json" in readme_md

    deploy_md = (REPO_ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert './libscript.sh provision "$CLOUD_PROVIDER"' in deploy_md
    assert "./libscript.sh deprovision aws" in deploy_md
    assert './libscript.sh deprovision "$CLOUD_PROVIDER"' in deploy_md
    assert "./libscript.sh deprovision gcp" in deploy_md
    assert "./libscript.sh train libscript.json" in deploy_md
    assert "./_lib/databases/provision_db.sh libscript.json" in deploy_md
    assert "./_lib/app-servers/provision_frontend.sh" in deploy_md
    assert "./_lib/networking/netctl/cli.sh apply libscript.json" in deploy_md
    assert "./libscript.sh cloud " not in deploy_md

    deploy_tpu_md = (REPO_ROOT / "DEPLOY_TO_TPU.md").read_text(encoding="utf-8")
    assert "cloud-providers/gcp-cli" in deploy_tpu_md
    assert "./_lib/ml-training/setup.sh" in deploy_tpu_md
    assert "./payloads/ai/gcp-tpu-vm/setup.sh" in deploy_tpu_md
    assert "./_lib/ai-serving/setup.sh" in deploy_tpu_md
    assert "./_lib/ai-serving/deploy.sh" in deploy_tpu_md
    assert "./topologies/gke-tpu-cluster/cli.sh" in deploy_tpu_md
    assert "./payloads/ai/gke-xpk-training/deploy.sh" in deploy_tpu_md
    assert "./payloads/ai/gke-xpk-inference/deploy.sh" in deploy_tpu_md
    assert "./libscript.sh gcp-tpu-vm" in deploy_tpu_md
    assert "./libscript.sh train" in deploy_tpu_md
    assert "./libscript.sh serve-ml" in deploy_tpu_md
    assert "./stacks/" not in deploy_tpu_md
    assert "gcp/tpu-vm" not in deploy_tpu_md
    assert "gcp/gke-tpu-cluster" not in deploy_tpu_md

    report_md = (REPO_ROOT / "REPORT.md").read_text(encoding="utf-8")
    assert "cloud-providers/gcp-cli" in report_md
    assert "./_lib/ml-training/setup.sh" in report_md
    assert "./libscript.sh train" in report_md
    assert "./stacks/" not in report_md

    sample_env = (
        REPO_ROOT
        / "t1d-analytics-backend"
        / "tests"
        / "test_data"
        / "sample_dotenv.env"
    )
    sample_env.parent.mkdir(parents=True, exist_ok=True)
    sample_env.write_text(
        '# comment\n\nNO_EQUALS\nSAMPLE_KEY="sample_val"\n', encoding="utf-8"
    )
    try:
        parsed = load_dotenv_dict((sample_env, sample_env.parent / "nonexistent.env"))
        assert parsed == {"SAMPLE_KEY": "sample_val"}
    finally:
        sample_env.unlink(missing_ok=True)
