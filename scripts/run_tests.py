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
from typing import DefaultDict, Dict, List, Optional


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
    run_cmd([npm_cmd, "install"], cwd="web")
    run_cmd([npm_cmd, "run", "build"], cwd="web")

    print("Running python tests and coverage...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")

    pytest_cmd = shutil.which("pytest") or "pytest"
    run_cmd([pytest_cmd, "tests/", "--cov-report=json", "--cov-report=lcov"], env=env)

    print("Running frontend tests...")
    run_cmd([npm_cmd, "run", "coverage"], cwd="web")

    print("Running e2e tests...")
    os.makedirs("tests/test_data", exist_ok=True)

    with open("tests/test_data/patients.csv", "w") as f:
        f.write("id,name,age\n1,Alice,30\n2,Bob,40\n")

    with open("tests/test_data/tiny.csv", "w") as f:
        f.write("id,value\n1,100\n2,200\n")

    system_python = shutil.which("python3") or shutil.which("python") or "python"
    run_cmd(
        [
            system_python,
            "-c",
            "from t1d_analytics.analytics import load_data_to_duckdb; load_data_to_duckdb('tests/test_data/', 'ci_test.duckdb')",
        ],
        env=env,
    )

    import time
    import urllib.request

    backend_env = env.copy()
    backend_env["T1D_DB_PATH"] = "ci_test.duckdb"

    print("Starting backend server...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "t1d_analytics.api:app", "--port", "8000"],
        env=backend_env,
    )

    print("Starting frontend server...")
    frontend_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd="web", env=env)

    print("Waiting for backend DB to connect and API to be ready...")
    backend_ready = False
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:8000/api/schema")
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
            urllib.request.urlopen("http://localhost:5173")
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
    test_result = subprocess.run([npx_cmd, "playwright", "test"], cwd="web")
    coverage_result = subprocess.run(
        [npx_cmd, "nyc", "report", "--reporter=lcov", "--report-dir=coverage/e2e"],
        cwd="web",
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
        os.path.join("web", "coverage", "lcov.info"),
        os.path.join("web", "coverage", "e2e", "lcov.info"),
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

    total_found = 0
    total_hit = 0

    with open("combined-coverage.lcov", "w") as out:
        for filename, lines in cov_data.items():
            out.write(f"SF:{filename}\n")
            lf = len(lines)
            lh = 0
            for line_num, hits in sorted(lines.items()):
                out.write(f"DA:{line_num},{hits}\n")
                if hits > 0:
                    lh += 1
            out.write(f"LF:{lf}\n")
            out.write(f"LH:{lh}\n")
            out.write("end_of_record\n")

            total_found += lf
            total_hit += lh

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

    test_color = (
        "brightgreen" if test_cov >= 90 else "yellow" if test_cov >= 70 else "red"
    )
    test_shield = f"[![Coverage](https://img.shields.io/badge/Coverage-{test_cov}%25-{test_color}.svg)](#)"

    try:
        interrogate_cmd = shutil.which("interrogate") or "interrogate"
        result = subprocess.run(
            [interrogate_cmd, "src/t1d_analytics"],
            capture_output=True,
            text=True,
            env=env,
        )
        match = re.search(r"actual: ([0-9.]+)%", result.stdout)
        doc_cov = int(float(match.group(1))) if match else 0
    except Exception as e:
        print(f"Failed to run interrogate: {e}")
        sys.exit(1)

    doc_color = "brightgreen" if doc_cov >= 90 else "yellow" if doc_cov >= 70 else "red"
    doc_shield = (
        f"[![Docs](https://img.shields.io/badge/Docs-{doc_cov}%25-{doc_color}.svg)](#)"
    )

    readme_path = "README.md"
    with open(readme_path, "r") as f:
        content = f.read()

    content = re.sub(r"\[!\[Coverage\].*\n?", "", content)
    content = re.sub(r"\[!\[Docs\].*\n?", "", content)

    license_pattern = r"(\[!\[License\][^\n]+\n)"
    replacement = rf"\1{test_shield}\n{doc_shield}\n"

    if re.search(license_pattern, content):
        content = re.sub(license_pattern, replacement, content)
    else:
        print("License shield not found in README.md")

    with open(readme_path, "w") as f:
        f.write(content)

    print("Pre-commit checks passed successfully!")


if __name__ == "__main__":
    main()
