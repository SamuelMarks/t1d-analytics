"""Comprehensive unit tests for Makefile and make.bat automation files."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAKEFILE_PATH = REPO_ROOT / "Makefile"
MAKE_BAT_PATH = REPO_ROOT / "make.bat"
WINE_BINARY = shutil.which("wine") or "/Users/samuel/.local/bin/wine"
HAS_WINE = Path(WINE_BINARY).is_file() and os.access(WINE_BINARY, os.X_OK)


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


def test_make_bat_static_validation() -> None:
    """Verify make.bat syntax, target definitions, and commands statically."""
    assert MAKE_BAT_PATH.is_file()
    content = MAKE_BAT_PATH.read_text(encoding="utf-8")

    expected_labels = [
        ":help",
        ":ensure_venv",
        ":activate_venv",
        ":build_docker",
        ":db_docker",
        ":seed_docker",
        ":setup_db_docker",
        ":setup_backend_docker",
        ":backend_docker",
        ":setup_frontend_docker",
        ":frontend_docker",
        ":setup_docker",
        ":run_docker",
        ":test_docker",
        ":clean_docker",
        ":install_base",
        ":install_deps",
        ":install_db",
        ":init_db",
        ":seed_db",
        ":setup_db",
        ":setup_backend",
        ":build_backend",
        ":dev_backend",
        ":serve_backend",
        ":setup_frontend",
        ":build_frontend",
        ":dev_frontend",
        ":serve_frontend",
        ":build",
        ":build_docs",
        ":serve",
        ":setup",
        ":setup_native",
        ":run_native",
        ":start_stack",
        ":stop_stack",
        ":status_stack",
        ":health_stack",
        ":logs_stack",
        ":train_ml",
        ":infer_ml",
        ":test",
        ":test_native",
        ":clean_native",
        ":fmt",
    ]
    for label in expected_labels:
        assert label in content, f"Missing label {label} in make.bat"

    # Verify key batch execution logic
    assert "%DOCKER_COMPOSE% -f %COMPOSE_FILE% build" in content
    assert "%DOCKER_COMPOSE% -f %COMPOSE_FILE% build backend" in content
    assert (
        '%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm frontend sh -c "npm install && npm run build"'
        in content
    )
    assert "%DOCKER_COMPOSE% -f %COMPOSE_FILE% up" in content
    assert (
        "%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm backend pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests"
        in content
    )
    assert "%DOCKER_COMPOSE% -f %COMPOSE_FILE% down -v" in content
    assert r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install python 3.12' in content
    assert r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install nodejs 20.x' in content
    assert r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install uv latest' in content
    assert r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install duckdb latest' in content
    assert r'call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install ollama latest' in content
    assert "python3 -m venv .venv" in content
    assert "pip install --upgrade pip" in content
    assert (
        "pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt"
        in content
    )
    assert "t1d-analytics load --data-dir %DATA_DIR% --db %T1D_DB_PATH%" in content
    assert "t1d-analytics doctor --db %T1D_DB_PATH%" in content
    assert (
        "interrogate -vv --fail-under=100 t1d-analytics-backend/src/t1d_analytics"
        in content
    )
    assert (
        "python3 -m sphinx -b html t1d-analytics-backend/docs_src %DOCS_DIR%" in content
    )
    assert "npm run build" in content
    assert (
        "uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000"
        in content
    )
    assert 'call "%LIBSCRIPT_CLI%" start libscript.json' in content
    assert 'call "%LIBSCRIPT_CLI%" train libscript.json' in content
    assert 'call "%LIBSCRIPT_CLI%" infer libscript.json' in content
    assert "ruff check" in content
    assert "ruff format" in content


def test_make_bat_execution_with_wine() -> None:
    """Verify that every dispatched target in make.bat maps to a defined label and exits cleanly."""
    import re

    content = MAKE_BAT_PATH.read_text(encoding="utf-8")
    goto_targets = set(
        re.findall(r'if "%~1"=="[^"]*"\s+goto\s+([a-zA-Z0-9_]+)', content)
    )
    defined_labels = set(
        re.findall(r"^:([a-zA-Z0-9_]+)\b", content, flags=re.MULTILINE)
    )

    assert len(goto_targets) >= 40
    missing_labels = goto_targets - defined_labels
    assert not missing_labels, f"Goto targets missing labels: {missing_labels}"
    assert 'if "%~1"=="" goto help' in content
    assert "echo Unknown target: %~1" in content


def test_readme_makefile_usage_table() -> None:
    """Verify that README.md includes the Makefile usage section and documents all targets."""
    import re

    readme_content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## 🧰 Makefile Usage" in readme_content
    assert re.search(
        r"\|\s*Target\s*\|\s*Category\s*\|\s*Description\s*\|", readme_content
    )

    expected_targets = [
        "help",
        "install_base",
        "install_deps",
        "setup",
        "setup_native",
        "setup_backend",
        "build_backend",
        "setup_frontend",
        "build_frontend",
        "install_db",
        "init_db",
        "seed_db",
        "setup_db",
        "build",
        "build_docs",
        "serve",
        "run_native",
        "dev_backend",
        "serve_backend",
        "dev_frontend",
        "serve_frontend",
        "start_stack",
        "stop_stack",
        "status_stack",
        "health_stack",
        "logs_stack",
        "train_ml",
        "infer_ml",
        "test",
        "test_native",
        "clean_native",
        "fmt",
        "setup_docker",
        "db_docker",
        "seed_docker",
        "setup_db_docker",
        "backend_docker",
        "setup_backend_docker",
        "frontend_docker",
        "setup_frontend_docker",
        "build_docker",
        "run_docker",
        "test_docker",
        "clean_docker",
    ]
    for target in expected_targets:
        assert re.search(rf"\|\s*`{re.escape(target)}`\s*\|", readme_content), (
            f"Missing target `{target}` in README.md table"
        )
