"""
Orchestration script to cross-platform test the T1D Analytics project.

This module sets up backend and frontend environments, executes the
associated tests, gathers and merges coverage, and updates the shields
within the project's README.md.
"""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


def compute_combined_coverage(
    cov_data: DefaultDict[str, DefaultDict[int, int]],
) -> Tuple[int, int, str]:
    """
    Format merged LCOV coverage records and compute total found and hit line counts.

    Args:
    ----
        cov_data: Mapping from source filepath to `{line_number: hit_count}`.

    Returns:
    -------
        Tuple[int, int, str]: `(total_found, total_hit, formatted_lcov_text)`.

    """
    total_found = 0
    total_hit = 0
    lines_out: List[str] = []

    for filename, lines in cov_data.items():
        lines_out.append(f"SF:{filename}\n")
        lf = len(lines)
        lh = 0
        for line_num, hits in sorted(lines.items()):
            lines_out.append(f"DA:{line_num},{hits}\n")
            if hits > 0:
                lh += 1
        lines_out.append(f"LF:{lf}\n")
        lines_out.append(f"LH:{lh}\n")
        lines_out.append("end_of_record\n")

        total_found += lf
        total_hit += lh

    return total_found, total_hit, "".join(lines_out)


def update_readme_badges(
    content: str,
    test_cov: int,
    doc_cov: int,
) -> Tuple[str, bool]:
    """
    Update Coverage and Docs shields in `README.md` content.

    Args:
    ----
        content: Existing `README.md` markdown content.
        test_cov: Integer test coverage percentage (0–100).
        doc_cov: Integer docstring coverage percentage (0–100).

    Returns:
    -------
        Tuple[str, bool]: `(updated_content, license_shield_found)`.

    """
    test_color = (
        "brightgreen" if test_cov >= 90 else "yellow" if test_cov >= 70 else "red"
    )
    test_shield = f"[![Coverage](https://img.shields.io/badge/Coverage-{test_cov}%25-{test_color}.svg)](#)"

    doc_color = "brightgreen" if doc_cov >= 90 else "yellow" if doc_cov >= 70 else "red"
    doc_shield = (
        f"[![Docs](https://img.shields.io/badge/Docs-{doc_cov}%25-{doc_color}.svg)](#)"
    )

    cleaned = re.sub(r"\[!\[Coverage\].*\n?", "", content)
    cleaned = re.sub(r"\[!\[Docs\].*\n?", "", cleaned)

    license_pattern = r"(\[!\[License\][^\n]+\n)"
    replacement = rf"\1{test_shield}\n{doc_shield}\n"

    if re.search(license_pattern, cleaned):
        return re.sub(license_pattern, replacement, cleaned), True
    return cleaned, False


def build_test_commands(
    npm_cmd: str,
    pytest_cmd: str,
) -> Dict[str, Sequence[str]]:
    """
    Construct the standard CLI command vectors used by the test runner.

    Args:
    ----
        npm_cmd: Resolved `npm` executable path.
        pytest_cmd: Resolved `pytest` executable path.

    Returns:
    -------
        Dict[str, Sequence[str]]: Named command vectors for web and backend test stages.

    """
    return {
        "web_install": [npm_cmd, "install"],
        "web_build": [npm_cmd, "run", "build"],
        "backend_pytest": [
            pytest_cmd,
            "-c",
            "t1d-analytics-backend/pyproject.toml",
            "t1d-analytics-backend/tests/",
            "--cov-report=json",
            "--cov-report=lcov",
        ],
        "web_coverage": [npm_cmd, "run", "coverage"],
    }


def run_cmd(
    cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None
) -> None:
    """
    Run a shell command as a subprocess, failing the script if the exit code is non-zero.

    Args:
    ----
        cmd: A list of strings comprising the command to be executed.
        env: An optional dictionary of environment variables.
        cwd: An optional string representing the working directory to execute the command in.

    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main() -> None:
    """
    Execute the main entrypoint for the test orchestration script.

    Performs building, tests, server orchestration, E2E checks, and coverage processing.
    """
    print("Running web install and build...")
    npm_cmd = shutil.which("npm") or "npm"
    env = os.environ.copy()
    python_bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = f"{python_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = os.path.abspath(os.path.join("t1d-analytics-backend", "src"))

    pytest_cmd = shutil.which("pytest", path=env["PATH"]) or "pytest"
    cmds = build_test_commands(npm_cmd=npm_cmd, pytest_cmd=pytest_cmd)

    run_cmd(list(cmds["web_install"]), cwd="t1d-analytics-web")
    run_cmd(list(cmds["web_build"]), cwd="t1d-analytics-web")

    print("Running python tests and coverage...")
    run_cmd(
        list(cmds["backend_pytest"]),
        env=env,
    )

    print("Running frontend tests...")
    run_cmd(list(cmds["web_coverage"]), cwd="t1d-analytics-web")

    print("Running e2e tests...")
    os.makedirs("t1d-analytics-backend/tests/test_data", exist_ok=True)

    with open("t1d-analytics-backend/tests/test_data/patients.csv", "w") as f:
        f.write("id,name,age\n1,Alice,30\n2,Bob,40\n")

    with open("t1d-analytics-backend/tests/test_data/tiny.csv", "w") as f:
        f.write("id,name,value\n1,Test A,100\n2,Test B,200\n")

    for db_artifact in ("ci_test.duckdb", "ci_test.duckdb.wal"):
        if os.path.exists(db_artifact):
            os.remove(db_artifact)

    system_python = sys.executable
    run_cmd(
        [
            system_python,
            "-c",
            (
                "from t1d_analytics.seed import seed_database; "
                "seed_database(data_dir='t1d-analytics-backend/tests/test_data/', db_path='ci_test.duckdb', sessions_db_path='t1d-analytics-backend/tests/test_data/sessions.duckdb', overwrite_csvs=True)"
            ),
        ],
        env=env,
    )

    import time
    import urllib.request

    backend_env = env.copy()
    backend_env["T1D_DB_PATH"] = "ci_test.duckdb"
    backend_env["T1D_SESSIONS_DB_PATH"] = (
        "t1d-analytics-backend/tests/test_data/sessions.duckdb"
    )
    env["T1D_API_PORT"] = "8000"

    print("Starting backend server...")
    backend_proc = subprocess.Popen(
        [system_python, "-m", "uvicorn", "t1d_analytics.api:app", "--port", "8000"],
        env=backend_env,
    )

    print("Starting frontend server...")
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1"],
        cwd="t1d-analytics-web",
        env=env,
    )

    print("Waiting for backend DB to connect and API to be ready...")
    backend_ready = False
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/api/ready")
            backend_ready = True
            break
        except Exception:
            time.sleep(1)

    if not backend_ready:
        print("Backend failed to start.")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(1)

    print("Waiting for frontend dev server to be ready...")
    frontend_ready = False
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:5173")
            frontend_ready = True
            break
        except Exception:
            time.sleep(1)

    if not frontend_ready:
        print("Frontend failed to start.")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(1)

    print("Servers ready. Running playwright tests...")
    npx_cmd = shutil.which("npx") or "npx"
    test_result = subprocess.run(
        [npx_cmd, "playwright", "test"], cwd="t1d-analytics-web"
    )
    coverage_result = subprocess.run(
        [npx_cmd, "nyc", "report", "--reporter=lcov", "--report-dir=coverage/e2e"],
        cwd="t1d-analytics-web",
    )

    print("Tearing down servers...")
    backend_proc.terminate()
    frontend_proc.terminate()
    backend_proc.wait()
    frontend_proc.wait()

    if test_result.returncode != 0:
        print(f"E2E tests failed with exit code {test_result.returncode}")
        sys.exit(test_result.returncode)
    if coverage_result.returncode != 0:
        print(f"E2E coverage failed with exit code {coverage_result.returncode}")
        sys.exit(coverage_result.returncode)

    print("Merging coverage...")
    cov_data: DefaultDict[str, DefaultDict[int, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    lcov_files = [
        "coverage.lcov",
        os.path.join("t1d-analytics-web", "coverage", "lcov.info"),
        os.path.join("t1d-analytics-web", "coverage", "e2e", "lcov.info"),
    ]

    for lcov_file in lcov_files:
        try:
            with open(lcov_file, "r") as f:
                current_file: str = ""
                for line in f:
                    line = line.strip()
                    if line.startswith("SF:"):
                        current_file = line[3:]
                    elif line.startswith("DA:") and current_file:
                        parts = line[3:].split(",")
                        line_num = int(parts[0])
                        hits = int(parts[1])
                        cov_data[current_file][line_num] = max(
                            cov_data[current_file][line_num], hits
                        )
        except FileNotFoundError:
            print(f"Warning: {lcov_file} not found.")

    total_found, total_hit, combined_lcov_text = compute_combined_coverage(cov_data)
    with open("combined-coverage.lcov", "w") as out:
        out.write(combined_lcov_text)

    if total_found == 0:
        print("No lines found in coverage report.")
        sys.exit(1)

    cov = (total_hit / total_found) * 100.0
    print(f"Combined Coverage: {cov:.2f}%")
    if cov < 100.0:
        print(f"Coverage is below 100% threshold! (Actual: {cov:.2f}%)")
        sys.exit(1)

    print("Generating shields in README.md...")
    try:
        with open("coverage.json") as f:
            data = json.load(f)
            test_cov = int(data["totals"]["percent_covered"])
    except Exception as e:
        print(f"Failed to read coverage.json: {e}")
        sys.exit(1)

    try:
        interrogate_cmd = shutil.which("interrogate", path=env["PATH"]) or "interrogate"
        result = subprocess.run(
            [interrogate_cmd, "t1d-analytics-backend/src/t1d_analytics"],
            capture_output=True,
            text=True,
            env=env,
        )
        match = re.search(r"actual: ([0-9.]+)%", result.stdout)
        doc_cov = int(float(match.group(1))) if match else 0
    except Exception as e:
        print(f"Failed to run interrogate: {e}")
        sys.exit(1)

    readme_path = "README.md"
    with open(readme_path, "r") as f:
        content = f.read()

    updated_content, license_found = update_readme_badges(
        content=content,
        test_cov=test_cov,
        doc_cov=doc_cov,
    )
    if not license_found:
        print("License shield not found in README.md")

    with open(readme_path, "w") as f:
        f.write(updated_content)

    print("Pre-commit checks passed successfully!")


if __name__ == "__main__":
    main()
