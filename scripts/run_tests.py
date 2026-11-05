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
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class CoverageMetrics:
    """
    Aggregated line coverage metrics across merged LCOV reports.

    Attributes
    ----------
        total_found: Total instrumented lines discovered (`LF`).
        total_hit: Total lines executed at least once (`LH`).
        formatted_lcov: Merged LCOV report text.

    """

    total_found: int
    total_hit: int
    formatted_lcov: str

    @property
    def percent(self) -> float:
        """
        Calculate the combined line coverage percentage (`0.0` to `100.0`).

        Returns
        -------
            float: Combined coverage percentage, or `0.0` if `total_found == 0`.

        """
        if self.total_found == 0:
            return 0.0
        return (self.total_hit / self.total_found) * 100.0


@dataclass(frozen=True)
class CommandResult:
    """
    Structured result from executing a subprocess command.

    Attributes
    ----------
        command: Executed command argument vector.
        returncode: Process exit status code.
        stdout: Captured standard output text when requested.

    """

    command: Tuple[str, ...]
    returncode: int
    stdout: str = ""


@dataclass(frozen=True)
class TestSuiteConfig:
    """
    Configuration options controlling full-stack test orchestration and badge updates.

    Attributes
    ----------
        skip_readme_update: When True, skips mutating `README.md` badges on disk.
        coverage_threshold: Minimum required combined LCOV line coverage percentage.
        backend_ready_url: Readiness URL for the FastAPI backend server.
        frontend_ready_url: Readiness URL for the Vite frontend dev server.

    """

    __test__ = False

    skip_readme_update: bool = False
    coverage_threshold: float = 100.0
    backend_ready_url: str = "http://127.0.0.1:8000/api/ready"
    frontend_ready_url: str = "http://127.0.0.1:5173"


def parse_lcov_files(
    lcov_paths: Sequence[Union[str, Path]],
) -> DefaultDict[str, DefaultDict[int, int]]:
    """
    Parse one or more LCOV files from disk and aggregate maximum hit counts per file and line.

    Args:
    ----
        lcov_paths: Sequence of paths to LCOV files on disk.

    Returns:
    -------
        DefaultDict[str, DefaultDict[int, int]]: Mapping of `filepath -> {line_number: max_hits}`.

    """
    cov_data: DefaultDict[str, DefaultDict[int, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    for lcov_file in lcov_paths:
        try:
            with open(lcov_file, encoding="utf-8") as f:
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
    return cov_data


class _UrlOpenResultProtocol:
    """Marker protocol for `urlopen_fn` return values in `wait_for_service_ready`."""


def wait_for_service_ready(
    url: str,
    max_attempts: int = 30,
    sleep_seconds: float = 1.0,
    urlopen_fn: Callable[
        ..., Union[_UrlOpenResultProtocol, str, bytes, None]
    ] = urllib.request.urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """
    Poll an HTTP health/readiness endpoint until it responds or `max_attempts` is exhausted.

    Args:
    ----
        url: Target HTTP URL to poll.
        max_attempts: Maximum polling iterations.
        sleep_seconds: Delay in seconds between failed attempts.
        urlopen_fn: Injectable URL opener callable.
        sleep_fn: Injectable sleep callable.

    Returns:
    -------
        bool: True if the endpoint responded without raising an exception, False otherwise.

    """
    for _ in range(max_attempts):
        try:
            urlopen_fn(url)
            return True
        except (urllib.error.URLError, OSError, ValueError, RuntimeError):
            sleep_fn(sleep_seconds)
    return False


def extract_pytest_coverage_percent(coverage_json_path: Union[str, Path]) -> int:
    """
    Read pytest `coverage.json` and return the integer total covered percentage.

    Args:
    ----
        coverage_json_path: Path to the `coverage.json` report file.

    Returns:
    -------
        int: Integer coverage percentage (0–100).

    """
    with open(coverage_json_path, encoding="utf-8") as f:
        data = json.load(f)
        return int(float(data["totals"]["percent_covered"]))


def extract_interrogate_coverage_percent(stdout_text: str) -> int:
    """
    Parse `interrogate` CLI output to extract the integer docstring coverage percentage.

    Args:
    ----
        stdout_text: Raw standard output string from `interrogate`.

    Returns:
    -------
        int: Parsed integer docstring coverage percentage, or 0 if not found.

    """
    match = re.search(r"actual: ([0-9.]+)%", stdout_text)
    return int(float(match.group(1))) if match else 0


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


def parse_lcov_totals(lcov_text: str) -> Tuple[int, int]:
    """
    Parse LCOV text directly and return `(total_found, total_hit)` line counts.

    Args:
    ----
        lcov_text: Raw LCOV report text.

    Returns:
    -------
        Tuple[int, int]: `(total_lines_found, total_lines_hit)`.

    """
    cov_data: DefaultDict[str, DefaultDict[int, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    current_file = ""
    for raw_line in lcov_text.splitlines():
        line = raw_line.strip()
        if line.startswith("SF:"):
            current_file = line[3:]
        elif line.startswith("DA:") and current_file:
            parts = line[3:].split(",")
            line_num = int(parts[0])
            hits = int(parts[1])
            cov_data[current_file][line_num] = max(
                cov_data[current_file][line_num], hits
            )
    total_found, total_hit, _ = compute_combined_coverage(cov_data)
    return total_found, total_hit


def resolve_badge_color(percent: float) -> str:
    """
    Determine the shields.io badge color for a given coverage percentage.

    Args:
    ----
        percent: Coverage percentage (`0.0` to `100.0`).

    Returns:
    -------
        str: `'brightgreen'`, `'yellow'`, or `'red'`.

    """
    if percent >= 90:
        return "brightgreen"
    if percent >= 70:
        return "yellow"
    return "red"


def update_readme_shields(
    readme_text: str,
    test_cov: float,
    doc_cov: float,
) -> str:
    """
    Rewrite the Coverage and Docs shields in `readme_text` and return the updated Markdown string.

    Args:
    ----
        readme_text: Existing `README.md` Markdown content.
        test_cov: Test coverage percentage.
        doc_cov: Documentation coverage percentage.

    Returns:
    -------
        str: Updated `README.md` Markdown string.

    """
    updated, _ = update_readme_badges(
        content=readme_text,
        test_cov=int(test_cov),
        doc_cov=int(doc_cov),
    )
    return updated


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
    test_color = resolve_badge_color(test_cov)
    test_shield = f"[![Coverage](https://img.shields.io/badge/Coverage-{test_cov}%25-{test_color}.svg)](#)"

    doc_color = resolve_badge_color(doc_cov)
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


def evaluate_coverage_metrics(
    lcov_paths: Sequence[Union[str, Path]],
) -> CoverageMetrics:
    """
    Parse and merge LCOV files into a `CoverageMetrics` summary.

    Args:
    ----
        lcov_paths: Sequence of LCOV report file paths.

    Returns:
    -------
        CoverageMetrics: Merged coverage metrics and formatted LCOV string.

    """
    cov_data = parse_lcov_files(lcov_paths)
    total_found, total_hit, combined_lcov_text = compute_combined_coverage(cov_data)
    return CoverageMetrics(
        total_found=total_found,
        total_hit=total_hit,
        formatted_lcov=combined_lcov_text,
    )


def sync_readme_shields_on_disk(
    readme_path: Union[str, Path],
    test_cov: int,
    doc_cov: int,
) -> bool:
    """
    Read `readme_path`, update its Coverage and Docs shields, and write the updated content back to disk.

    Args:
    ----
        readme_path: Path to `README.md`.
        test_cov: Integer test coverage percentage (0–100).
        doc_cov: Integer docstring coverage percentage (0–100).

    Returns:
    -------
        bool: True if the license shield anchor was found in `README.md`, False otherwise.

    """
    with open(readme_path, encoding="utf-8") as f:
        content = f.read()

    updated_content, license_found = update_readme_badges(
        content=content,
        test_cov=test_cov,
        doc_cov=doc_cov,
    )
    if not license_found:
        print("License shield not found in README.md")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    return license_found


def run_cmd(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """
    Run a shell command as a subprocess, failing the script if the exit code is non-zero.

    Args:
    ----
        cmd: A list of strings comprising the command to be executed.
        env: An optional dictionary of environment variables.
        cwd: An optional string representing the working directory to execute the command in.
        timeout: Optional timeout in seconds for the subprocess.

    Returns:
    -------
        CommandResult: Structured result when the command exits with code 0.

    """
    print(f"Running: {' '.join(cmd)}")
    if timeout is not None:
        result = subprocess.run(cmd, env=env, cwd=cwd, timeout=timeout)
    else:
        result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)
    return CommandResult(command=tuple(cmd), returncode=result.returncode)


def main(
    skip_readme_update: bool = False,
    workspace_root: Optional[Union[str, Path]] = None,
) -> None:
    """
    Execute the main entrypoint for the test orchestration script.

    Performs building, tests, server orchestration, E2E checks, and coverage processing.

    Args:
    ----
        skip_readme_update: When True (or when `T1D_SKIP_README_SHIELD_UPDATE=1` is set),
            skips modifying `README.md` on disk.
        workspace_root: Optional workspace directory root (defaults to current working directory).

    """
    root = Path(workspace_root) if workspace_root is not None else Path()
    web_dir = root / "t1d-analytics-web"
    test_data_dir = root / "t1d-analytics-backend" / "tests" / "test_data"

    suite_cfg = TestSuiteConfig(
        skip_readme_update=skip_readme_update
        or os.environ.get("T1D_SKIP_README_SHIELD_UPDATE") == "1"
    )
    print("Running web install and build...")
    npm_cmd = shutil.which("npm") or "npm"
    env = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{python_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str((root / "t1d-analytics-backend" / "src").resolve())

    pytest_cmd = shutil.which("pytest", path=env["PATH"]) or "pytest"
    cmds = build_test_commands(npm_cmd=npm_cmd, pytest_cmd=pytest_cmd)

    run_cmd(list(cmds["web_install"]), cwd=str(web_dir))
    run_cmd(list(cmds["web_build"]), cwd=str(web_dir))

    print("Running python tests and coverage...")
    run_cmd(
        list(cmds["backend_pytest"]),
        env=env,
    )

    print("Running frontend tests...")
    run_cmd(list(cmds["web_coverage"]), cwd=str(web_dir))

    print("Running e2e tests...")
    shutil.rmtree(web_dir / ".nyc_output", ignore_errors=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    (test_data_dir / "patients.csv").write_text(
        "id,name,age\n1,Alice,30\n2,Bob,40\n", encoding="utf-8"
    )
    (test_data_dir / "tiny.csv").write_text(
        "id,name,value\n1,Test A,100\n2,Test B,200\n", encoding="utf-8"
    )

    for db_artifact in ("ci_test.duckdb", "ci_test.duckdb.wal"):
        db_p = root / db_artifact
        if db_p.exists():
            db_p.unlink()

    ci_db_str = str(root / "ci_test.duckdb")
    sessions_db_str = str(test_data_dir / "sessions.duckdb")
    system_python = sys.executable
    run_cmd(
        [
            system_python,
            "-c",
            (
                "from t1d_analytics.seed import seed_database; "
                f"seed_database(data_dir={str(test_data_dir)!r}, db_path={ci_db_str!r}, sessions_db_path={sessions_db_str!r}, overwrite_csvs=True)"
            ),
        ],
        env=env,
    )

    backend_env = env.copy()
    backend_env["T1D_DB_PATH"] = ci_db_str
    backend_env["T1D_SESSIONS_DB_PATH"] = sessions_db_str
    env["T1D_API_PORT"] = "8000"

    print("Starting backend server...")
    backend_proc = subprocess.Popen(
        [system_python, "-m", "uvicorn", "t1d_analytics.api:app", "--port", "8000"],
        env=backend_env,
    )

    print("Starting frontend server...")
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1"],
        cwd=str(web_dir),
        env=env,
    )

    print("Waiting for backend DB to connect and API to be ready...")
    backend_ready = wait_for_service_ready(
        "http://127.0.0.1:8000/api/ready",
        urlopen_fn=urllib.request.urlopen,
        sleep_fn=time.sleep,
    )

    if not backend_ready:
        print("Backend failed to start.")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(1)

    print("Waiting for frontend dev server to be ready...")
    frontend_ready = wait_for_service_ready(
        "http://127.0.0.1:5173",
        urlopen_fn=urllib.request.urlopen,
        sleep_fn=time.sleep,
    )

    if not frontend_ready:
        print("Frontend failed to start.")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(1)

    print("Servers ready. Running playwright tests...")
    npx_cmd = shutil.which("npx") or "npx"
    test_result = subprocess.run([npx_cmd, "playwright", "test"], cwd=str(web_dir))
    coverage_result = subprocess.run(
        [npx_cmd, "nyc", "report", "--reporter=lcov", "--report-dir=coverage/e2e"],
        cwd=str(web_dir),
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
    lcov_files: Sequence[Union[str, Path]] = [
        root / "coverage.lcov",
        web_dir / "coverage" / "lcov.info",
        web_dir / "coverage" / "e2e" / "lcov.info",
    ]
    metrics = evaluate_coverage_metrics(lcov_files)
    (root / "combined-coverage.lcov").write_text(
        metrics.formatted_lcov, encoding="utf-8"
    )

    if metrics.total_found == 0:
        print("No lines found in coverage report.")
        sys.exit(1)

    cov = metrics.percent
    print(f"Combined Coverage: {cov:.2f}%")
    if cov < suite_cfg.coverage_threshold:
        print(f"Coverage is below 100% threshold! (Actual: {cov:.2f}%)")
        sys.exit(1)

    if not suite_cfg.skip_readme_update:
        print("Generating shields in README.md...")
        try:
            test_cov = extract_pytest_coverage_percent(root / "coverage.json")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
            print(f"Failed to read coverage.json: {e}")
            sys.exit(1)

        try:
            interrogate_cmd = (
                shutil.which("interrogate", path=env["PATH"]) or "interrogate"
            )
            result = subprocess.run(
                [
                    interrogate_cmd,
                    str(root / "t1d-analytics-backend" / "src" / "t1d_analytics"),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            doc_cov = extract_interrogate_coverage_percent(result.stdout)
        except (
            OSError,
            ValueError,
            TypeError,
            subprocess.SubprocessError,
            RuntimeError,
        ) as e:
            print(f"Failed to run interrogate: {e}")
            sys.exit(1)

        sync_readme_shields_on_disk(
            readme_path=root / "README.md",
            test_cov=test_cov,
            doc_cov=doc_cov,
        )

    print("Pre-commit checks passed successfully!")


if __name__ == "__main__":
    main()
