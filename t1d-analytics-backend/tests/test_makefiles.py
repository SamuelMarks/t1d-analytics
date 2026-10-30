"""Comprehensive unit tests for Makefile and make.bat automation files."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAKEFILE_PATH = REPO_ROOT / "Makefile"
MAKE_BAT_PATH = REPO_ROOT / "make.bat"


def run_make(
    target: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """
    Execute a make dry-run command for a specific target.

    Args:
    ----
        target: The Makefile target name to execute with dry-run.
        env_overrides: Optional dictionary of environment variables.

    Returns:
    -------
        subprocess.CompletedProcess containing the execution results.

    """
    env = os.environ.copy()
    env.pop("T1D_DB_PATH", None)
    env.pop("DATA_DIR", None)
    if env_overrides:
        env.update(env_overrides)
    cmd = ["make", "-n", "-f", str(MAKEFILE_PATH), target]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )


def test_makefile_help() -> None:
    """Test that make help prints available commands."""
    res = subprocess.run(
        ["make", "-f", str(MAKEFILE_PATH), "help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Available commands:" in res.stdout
    for expected in [
        "setup",
        "setup_native",
        "run_native",
        "test_native",
        "clean_native",
        "setup_docker",
        "build_docker",
        "run_docker",
        "test_docker",
        "clean_docker",
        "install_db",
        "init_db",
        "seed_db",
        "setup_db",
        "db_docker",
        "seed_docker",
        "setup_db_docker",
        "setup_backend",
        "build_backend",
        "dev_backend",
        "serve_backend",
        "backend_docker",
        "setup_backend_docker",
        "setup_frontend",
        "build_frontend",
        "dev_frontend",
        "serve_frontend",
        "frontend_docker",
        "setup_frontend_docker",
        "install_base",
        "install_deps",
        "build",
        "build_docs",
        "serve",
        "start_stack",
        "stop_stack",
        "status_stack",
        "health_stack",
        "logs_stack",
        "train_ml",
        "infer_ml",
        "test",
        "fmt",
    ]:
        assert expected in res.stdout

    # Cover the env_overrides branch
    run_make("help", env_overrides={"TEST_MAKE_ENV": "1"})


def test_makefile_docker_targets() -> None:
    """Test Makefile docker-related recipes in dry-run mode."""
    res_build = run_make("build_docker")
    assert "docker-compose -f docker-compose.yml build" in res_build.stdout

    res_db = run_make("db_docker")
    assert "docker-compose -f docker-compose.yml up -d ollama" in res_db.stdout

    res_seed = run_make("seed_docker")
    assert (
        "docker-compose -f docker-compose.yml run --rm backend sh -c" in res_seed.stdout
    )
    assert "/app/data/t1d.duckdb" in res_seed.stdout
    assert "t1d-analytics load" in res_seed.stdout
    assert "t1d-analytics doctor" in res_seed.stdout

    res_setup_db = run_make("setup_db_docker")
    assert (
        "docker-compose -f docker-compose.yml run --rm backend sh -c"
        in res_setup_db.stdout
    )
    assert "/app/data/t1d.duckdb" in res_setup_db.stdout
    assert "t1d-analytics load" in res_setup_db.stdout
    assert "t1d-analytics doctor" in res_setup_db.stdout

    res_setup_back = run_make("setup_backend_docker")
    assert "docker-compose -f docker-compose.yml build backend" in res_setup_back.stdout

    res_back_docker = run_make("backend_docker")
    assert (
        "docker-compose -f docker-compose.yml up --build -d ollama backend"
        in res_back_docker.stdout
    )

    res_setup_front = run_make("setup_frontend_docker")
    assert (
        'docker-compose -f docker-compose.yml run --rm frontend sh -c "npm install && npm run build"'
        in res_setup_front.stdout
    )

    res_front_docker = run_make("frontend_docker")
    assert (
        "docker-compose -f docker-compose.yml up --build -d frontend"
        in res_front_docker.stdout
    )

    res_setup_docker = run_make("setup_docker")
    assert "docker-compose -f docker-compose.yml build" in res_setup_docker.stdout
    assert (
        'docker-compose -f docker-compose.yml run --rm frontend sh -c "npm install && npm run build"'
        in res_setup_docker.stdout
    )
    assert "t1d-analytics load" in res_setup_docker.stdout

    res_run = run_make("run_docker")
    assert "docker-compose -f docker-compose.yml up" in res_run.stdout

    res_test = run_make("test_docker")
    assert (
        "docker-compose -f docker-compose.yml run --rm backend pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests"
        in res_test.stdout
    )
    assert (
        "docker-compose -f docker-compose.yml run --rm frontend npm run test"
        in res_test.stdout
    )

    res_clean = run_make("clean_docker")
    assert "docker-compose -f docker-compose.yml down -v" in res_clean.stdout


def test_makefile_install_and_native_setup_targets() -> None:
    """Test Makefile dependency installation and native setup recipes."""
    res_base = run_make("install_base")
    assert "install python 3.12" in res_base.stdout
    assert "install nodejs 20.x" in res_base.stdout
    assert "install uv latest" in res_base.stdout
    assert "install duckdb latest" in res_base.stdout
    assert "install ollama latest" in res_base.stdout
    assert "python3 -m pip install --upgrade pip" in res_base.stdout
    assert "npm install -g npm" in res_base.stdout

    res_deps = run_make("install_deps")
    assert (
        "python3 -m pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt"
        in res_deps.stdout
    )
    assert "npm install" in res_deps.stdout

    res_install_db = run_make("install_db")
    assert "install duckdb latest" in res_install_db.stdout
    assert "install ollama latest" in res_install_db.stdout

    res_init_db = run_make("init_db")
    assert "mkdir -p data" in res_init_db.stdout
    assert "duckdb.connect('data/t1d.duckdb')" in res_init_db.stdout

    res_seed_db = run_make("seed_db")
    assert "mkdir -p data" in res_seed_db.stdout
    assert (
        "t1d-analytics load --data-dir data --db data/t1d.duckdb" in res_seed_db.stdout
    )
    assert "t1d-analytics doctor --db data/t1d.duckdb" in res_seed_db.stdout

    res_setup_db = run_make("setup_db")
    assert "install duckdb latest" in res_setup_db.stdout
    assert "mkdir -p data" in res_setup_db.stdout
    assert "duckdb.connect('data/t1d.duckdb')" in res_setup_db.stdout
    assert (
        "t1d-analytics load --data-dir data --db data/t1d.duckdb" in res_setup_db.stdout
    )
    assert "t1d-analytics doctor --db data/t1d.duckdb" in res_setup_db.stdout

    res_setup_backend = run_make("setup_backend")
    assert "python3 -m pip install --upgrade pip" in res_setup_backend.stdout
    assert (
        "python3 -m pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt"
        in res_setup_backend.stdout
    )
    assert "python3 -m pip install -e t1d-analytics-backend" in res_setup_backend.stdout

    res_build_backend = run_make("build_backend")
    assert "python3 -m pip install -e t1d-analytics-backend" in res_build_backend.stdout

    res_setup_frontend = run_make("setup_frontend")
    assert (
        "cd t1d-analytics-web && npm install && npm run build"
        in res_setup_frontend.stdout
    )

    res_build_frontend = run_make("build_frontend")
    assert "cd t1d-analytics-web && npm run build" in res_build_frontend.stdout

    res_setup_native = run_make("setup_native")
    assert "python3 -m pip install -e t1d-analytics-backend" in res_setup_native.stdout
    assert (
        "cd t1d-analytics-web && npm install && npm run build"
        in res_setup_native.stdout
    )
    assert (
        "t1d-analytics load --data-dir data --db data/t1d.duckdb"
        in res_setup_native.stdout
    )


def test_makefile_build_docs_default_and_custom(tmp_path: Path) -> None:
    """
    Test Makefile build_docs recipe with default and overridden DOCS_DIR.

    Args:
    ----
        tmp_path: Temporary directory fixture provided by pytest.

    """
    res_default = run_make("build_docs")
    assert "mkdir -p docs" in res_default.stdout
    assert (
        "interrogate -vv --fail-under=100 t1d-analytics-backend/src/t1d_analytics"
        in res_default.stdout
    )
    assert (
        "python3 -m sphinx -b html t1d-analytics-backend/docs_src docs"
        in res_default.stdout
    )
    assert "cd t1d-analytics-web && npm run docs" in res_default.stdout

    custom_dir = "custom_docs_output"
    res_custom = subprocess.run(
        [
            "make",
            "-n",
            "-f",
            str(MAKEFILE_PATH),
            f"DOCS_DIR={custom_dir}",
            "build_docs",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"mkdir -p {custom_dir}" in res_custom.stdout
    assert (
        f"python3 -m sphinx -b html t1d-analytics-backend/docs_src {custom_dir}"
        in res_custom.stdout
    )


def test_makefile_build_serve_stack_test_fmt() -> None:
    """Test build, serve, dev/serve backend/frontend, stack, ml, test, clean, and fmt targets."""
    res_build = run_make("build")
    assert "npm run build" in res_build.stdout
    assert "pip install -e t1d-analytics-backend" in res_build.stdout

    res_serve = run_make("serve")
    assert (
        "T1D_DB_PATH=data/t1d.duckdb DEBUG=1 uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000"
        in res_serve.stdout
    )

    res_dev_backend = run_make("dev_backend")
    assert (
        "T1D_DB_PATH=data/t1d.duckdb uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000"
        in res_dev_backend.stdout
    )

    res_serve_backend = run_make("serve_backend")
    assert (
        "T1D_DB_PATH=data/t1d.duckdb uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000"
        in res_serve_backend.stdout
    )

    res_dev_frontend = run_make("dev_frontend")
    assert "cd t1d-analytics-web && npm run dev" in res_dev_frontend.stdout

    res_serve_frontend = run_make("serve_frontend")
    assert "cd t1d-analytics-web && npm run dev" in res_serve_frontend.stdout

    res_run_native = run_make("run_native")
    assert (
        "T1D_DB_PATH=data/t1d.duckdb DEBUG=1 uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000"
        in res_run_native.stdout
    )

    for stack_target, cmd_verb in [
        ("start_stack", "start"),
        ("stop_stack", "stop"),
        ("status_stack", "status"),
        ("health_stack", "health"),
        ("logs_stack", "logs"),
        ("train_ml", "train"),
        ("infer_ml", "infer"),
    ]:
        res_stack = run_make(stack_target)
        assert f'libscript.sh" {cmd_verb} libscript.json' in res_stack.stdout

    res_test = run_make("test")
    assert "pytest" in res_test.stdout
    assert "npm run test" in res_test.stdout

    res_test_native = run_make("test_native")
    assert "pytest" in res_test_native.stdout
    assert "npm run test" in res_test_native.stdout

    res_clean_native = run_make("clean_native")
    assert "rm -rf t1d-analytics-web/dist" in res_clean_native.stdout

    res_fmt = run_make("fmt")
    assert "ruff check" in res_fmt.stdout
    assert "ruff format" in res_fmt.stdout
    assert "npm run format" in res_fmt.stdout


def test_makefile_venv_activate_detection(tmp_path: Path) -> None:
    """
    Test Makefile VENV_ACTIVATE expansion when a virtual environment activate script exists.

    Args:
    ----
        tmp_path: Temporary directory fixture for isolated environment testing.

    """
    fake_venv = tmp_path / ".venv" / "bin"
    fake_venv.mkdir(parents=True)
    activate_file = fake_venv / "activate"
    activate_file.write_text("#!/bin/sh\n")

    res = subprocess.run(
        ["make", "-n", "-f", str(MAKEFILE_PATH), "test"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert ". .venv/bin/activate &&" in res.stdout
    assert "pytest" in res.stdout


def _parse_makefile_structure(
    content: str,
) -> tuple[set[str], dict[str, tuple[list[str], list[str]]]]:
    """
    Parse `.PHONY` declarations and target rules from a Makefile string.

    Args:
    ----
        content: Raw Makefile file contents.

    Returns:
    -------
        A tuple of `(phony_targets, targets_map)` where `targets_map` maps each
        target name to `(prerequisites, recipe_lines)`.

    """
    import re

    phony_targets: set[str] = set()
    targets_map: dict[str, tuple[list[str], list[str]]] = {}
    current_target: str | None = None

    for raw_line in content.splitlines():
        if raw_line.startswith(".PHONY:"):
            phony_targets.update(raw_line.partition(":")[2].split())
            current_target = None
            continue
        target_match = re.match(r"^([a-zA-Z0-9_]+):(.*)$", raw_line)
        if target_match:
            current_target = target_match.group(1)
            prereqs = target_match.group(2).strip().split()
            targets_map[current_target] = (prereqs, [])
        elif current_target is not None and raw_line.startswith("\t"):
            targets_map[current_target][1].append(raw_line.strip())
        elif raw_line.strip() and not raw_line.startswith("#"):
            current_target = None

    return phony_targets, targets_map


def _parse_make_bat_structure(
    content: str,
) -> tuple[set[str], dict[str, str], dict[str, list[str]]]:
    """
    Parse usage targets, CLI dispatch table, and `:label` blocks from `make.bat`.

    Args:
    ----
        content: Raw `make.bat` file contents.

    Returns:
    -------
        A tuple of `(usage_targets, dispatch_map, label_blocks)` where `dispatch_map`
        maps CLI target argument to label name and `label_blocks` maps label name to
        its body lines.

    """
    import re

    usage_match = re.search(r"::\s*make\.bat\s*\[([^\]]+)\]", content)
    usage_targets = (
        set(usage_match.group(1).split("|")) if usage_match is not None else set()
    )

    dispatch_map: dict[str, str] = dict(
        re.findall(r'if "%~1"=="([^"]+)"\s+goto\s+([a-zA-Z0-9_]+)', content)
    )

    label_blocks: dict[str, list[str]] = {}
    current_label: str | None = None
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        label_match = re.match(r"^:([a-zA-Z0-9_]+)\b", stripped)
        if label_match:
            current_label = label_match.group(1)
            label_blocks[current_label] = []
        elif current_label is not None and stripped and not stripped.startswith("::"):
            label_blocks[current_label].append(stripped)

    return usage_targets, dispatch_map, label_blocks


def test_make_bat_static_validation() -> None:
    """Verify structured target, prerequisite, and command parity between Makefile and make.bat."""
    assert MAKEFILE_PATH.is_file()
    assert MAKE_BAT_PATH.is_file()

    makefile_content = MAKEFILE_PATH.read_text(encoding="utf-8")
    bat_content = MAKE_BAT_PATH.read_text(encoding="utf-8")

    phony_targets, makefile_targets = _parse_makefile_structure(makefile_content)
    usage_targets, dispatch_map, label_blocks = _parse_make_bat_structure(bat_content)

    # 1. Exact target set parity across Makefile .PHONY, Makefile rules, and make.bat dispatch
    assert phony_targets == set(makefile_targets.keys())
    assert set(dispatch_map.keys()) == set(makefile_targets.keys())
    assert usage_targets == set(makefile_targets.keys())
    for arg_name, label_name in dispatch_map.items():
        assert arg_name == label_name
        assert label_name in label_blocks, f"Missing :{label_name} block in make.bat"

    # Helper labels in make.bat must also exist
    for helper_label in ("find_root", "root_found", "ensure_venv", "activate_venv"):
        assert helper_label in label_blocks

    # 2. Composite/alias targets in Makefile must delegate to the same prerequisites in make.bat
    for target_name, (prereqs, recipe_lines) in makefile_targets.items():
        if prereqs and not recipe_lines:
            bat_body = "\n".join(label_blocks[target_name])
            for prereq in prereqs:
                assert f"call :{prereq}" in bat_body, (
                    f"Target '{target_name}' in make.bat missing 'call :{prereq}' "
                    f"to match Makefile prerequisite"
                )

    # 3. Scoped per-label command verification in make.bat matching Makefile semantics
    expected_label_commands: dict[str, list[str]] = {
        "build_docker": ["%DOCKER_COMPOSE% -f %COMPOSE_FILE% build"],
        "db_docker": ["%DOCKER_COMPOSE% -f %COMPOSE_FILE% up -d ollama"],
        "seed_docker": [
            "t1d-analytics seed --data-dir /app/data --db /app/data/t1d.duckdb",
            "t1d-analytics load --data-dir /app/data --db /app/data/t1d.duckdb",
            "t1d-analytics doctor --db /app/data/t1d.duckdb",
        ],
        "setup_backend_docker": ["%DOCKER_COMPOSE% -f %COMPOSE_FILE% build backend"],
        "backend_docker": [
            "%DOCKER_COMPOSE% -f %COMPOSE_FILE% up --build -d ollama backend"
        ],
        "setup_frontend_docker": [
            '%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm frontend sh -c "npm install && npm run build"'
        ],
        "frontend_docker": [
            "%DOCKER_COMPOSE% -f %COMPOSE_FILE% up --build -d frontend"
        ],
        "run_docker": ["%DOCKER_COMPOSE% -f %COMPOSE_FILE% up"],
        "test_docker": [
            "%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm backend pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests",
            "%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm frontend npm run test",
        ],
        "clean_docker": ["%DOCKER_COMPOSE% -f %COMPOSE_FILE% down -v"],
        "install_base": [
            r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install python 3.12',
            r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install nodejs 20.x',
            r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install uv latest',
            r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install duckdb latest',
            r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install ollama latest',
            "pip install --upgrade pip",
        ],
        "install_deps": [
            "pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt",
            "npm install",
        ],
        "init_db": ["import duckdb; duckdb.connect"],
        "seed_db": [
            "t1d-analytics seed --data-dir %DATA_DIR% --db %T1D_DB_PATH%",
            "t1d-analytics load --data-dir %DATA_DIR% --db %T1D_DB_PATH%",
            "t1d-analytics doctor --db %T1D_DB_PATH%",
        ],
        "build_backend": ["pip install -e t1d-analytics-backend"],
        "dev_backend": [
            "uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000"
        ],
        "build_frontend": ["npm run build"],
        "dev_frontend": ["npm run dev"],
        "build_docs": [
            "interrogate -vv --fail-under=100 t1d-analytics-backend/src/t1d_analytics",
            "python3 -m sphinx -b html t1d-analytics-backend/docs_src %DOCS_DIR%",
            "npm run docs",
        ],
        "serve": [
            "set DEBUG=1",
            "uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000",
        ],
        "start_stack": ['call "%LIBSCRIPT_CLI%" start libscript.json'],
        "stop_stack": ['call "%LIBSCRIPT_CLI%" stop libscript.json'],
        "status_stack": ['call "%LIBSCRIPT_CLI%" status libscript.json'],
        "health_stack": ['call "%LIBSCRIPT_CLI%" health libscript.json'],
        "logs_stack": ['call "%LIBSCRIPT_CLI%" logs libscript.json'],
        "train_ml": ['call "%LIBSCRIPT_CLI%" train libscript.json'],
        "infer_ml": ['call "%LIBSCRIPT_CLI%" infer libscript.json'],
        "test": ["pytest", "npm run test"],
        "clean_native": ['rmdir /S /Q "t1d-analytics-web\\dist"'],
        "fmt": ["ruff check", "ruff format", "npm run format"],
    }

    for label_name, expected_cmds in expected_label_commands.items():
        block_text = "\n".join(label_blocks[label_name])
        for cmd_snippet in expected_cmds:
            assert cmd_snippet in block_text, (
                f"Label :{label_name} in make.bat missing expected command: {cmd_snippet}"
            )


def test_make_bat_execution_with_wine() -> None:
    """Verify that every dispatched target in make.bat maps to a defined label that terminates cleanly."""
    bat_content = MAKE_BAT_PATH.read_text(encoding="utf-8")
    _, dispatch_map, label_blocks = _parse_make_bat_structure(bat_content)

    assert len(dispatch_map) >= 40
    for target_name, label_name in dispatch_map.items():
        assert label_name in label_blocks, f"Goto target {target_name} missing label"
        block_lines = label_blocks[label_name]
        assert block_lines, f"Label :{label_name} has empty body"
        assert block_lines[-1] in ("goto :eof", "exit /b 0", "exit /b 1"), (
            f"Label :{label_name} does not terminate with 'goto :eof' (found: {block_lines[-1]})"
        )

    assert 'if "%~1"=="" goto help' in bat_content
    assert "echo Unknown target: %~1" in bat_content


def test_readme_makefile_usage_table() -> None:
    """Verify that README.md includes the Makefile usage section and documents every Makefile target."""
    import re

    makefile_content = MAKEFILE_PATH.read_text(encoding="utf-8")
    _, makefile_targets = _parse_makefile_structure(makefile_content)

    readme_content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## 🧰 Makefile Usage" in readme_content
    assert re.search(
        r"\|\s*Target\s*\|\s*Category\s*\|\s*Description\s*\|", readme_content
    )

    documented_targets = set(
        re.findall(r"^\|\s*`([a-zA-Z0-9_]+)`\s*\|", readme_content, flags=re.MULTILINE)
    )
    missing_in_readme = set(makefile_targets.keys()) - documented_targets
    assert not missing_in_readme, (
        f"Targets defined in Makefile but undocumented in README.md: {missing_in_readme}"
    )


def test_shell_scripts_syntax_validation() -> None:
    """Verify that make.sh and test.sh pass bash -n syntax validation."""
    for script_name in ("make.sh", "test.sh"):
        script_path = REPO_ROOT / script_name
        assert script_path.exists()
        res = subprocess.run(
            ["bash", "-n", str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"{script_name} syntax error: {res.stderr}"
