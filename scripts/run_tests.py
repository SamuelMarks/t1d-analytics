"""
Orchestration script to cross-platform test the T1D Analytics project.

This module sets up backend and frontend environments, executes unit and E2E
test suites, gathers and merges line, branch, and function LCOV coverage,
verifies Python AST function execution coverage and zero forbidden coverage
suppressions, and updates shields within the project's `README.md`.
"""

import ast
import collections
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

FORBIDDEN_COVERAGE_IGNORE_RE = re.compile(
    r"pragma:\s*no\s*cover|v8\s+ignore|istanbul\s+ignore|c8\s+ignore",
    re.IGNORECASE,
)


@dataclass
class FileLcovData:
    """
    Aggregated line, function, and branch coverage counters for a single source file.

    Attributes:
        lines: Mapping of `line_number -> max_hit_count`.
        functions: Mapping of `function_name -> (start_line, max_hit_count)`.
        branches: Mapping of `(line_num, block_id, branch_id) -> max_hit_count` (`-1` if never evaluated).

    """

    lines: collections.defaultdict[int, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    functions: dict[str, tuple[int, int]] = field(default_factory=dict)
    branches: dict[tuple[int, str, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageMetrics:
    """
    Aggregated line, function, and branch coverage metrics across merged LCOV reports.

    Attributes:
        total_found: Total instrumented lines discovered (`LF`).
        total_hit: Total lines executed at least once (`LH`).
        formatted_lcov: Merged LCOV report text preserving `FN`, `FNDA`, `BRDA`, and `DA` records.
        functions_found: Total instrumented functions discovered (`FNF`).
        functions_hit: Total functions executed at least once (`FNH`).
        branches_found: Total instrumented branches discovered (`BRF`).
        branches_hit: Total branches taken at least once (`BRH`).

    """

    total_found: int
    total_hit: int
    formatted_lcov: str
    functions_found: int = 0
    functions_hit: int = 0
    branches_found: int = 0
    branches_hit: int = 0

    @property
    def percent(self) -> float:
        """
        Calculate the combined line coverage percentage (`0.0` to `100.0`).

        Returns:
            float: Combined line coverage percentage, or `0.0` if `total_found == 0`.

        """
        if self.total_found == 0:
            return 0.0
        return (self.total_hit / self.total_found) * 100.0

    @property
    def function_percent(self) -> float:
        """
        Calculate the combined function coverage percentage (`0.0` to `100.0`).

        Returns:
            float: Combined function coverage percentage (`100.0` when `functions_found == 0`).

        """
        if self.functions_found == 0:
            return 100.0
        return (self.functions_hit / self.functions_found) * 100.0

    @property
    def branch_percent(self) -> float:
        """
        Calculate the combined branch coverage percentage (`0.0` to `100.0`).

        Returns:
            float: Combined branch coverage percentage (`100.0` when `branches_found == 0`).

        """
        if self.branches_found == 0:
            return 100.0
        return (self.branches_hit / self.branches_found) * 100.0

    def meets_threshold(self, threshold: float) -> bool:
        """
        Return True only when line, function, and branch coverage all meet or exceed `threshold`.

        Args:
            threshold: Required minimum coverage percentage (`0.0` to `100.0`).

        Returns:
            bool: True when line, function, and branch coverage each meet `threshold`.

        """
        return (
            self.percent >= threshold
            and self.function_percent >= threshold
            and self.branch_percent >= threshold
        )


@dataclass(frozen=True)
class CommandResult:
    """
    Structured result from executing a subprocess command.

    Attributes:
        command: Executed command argument vector.
        returncode: Process exit status code.
        stdout: Captured standard output text when requested.

    """

    command: tuple[str, ...]
    returncode: int
    stdout: str = ""


@runtime_checkable
class ServerProcessProtocol(Protocol):
    """Structural protocol for background server processes managed during E2E execution."""

    def terminate(self) -> None:
        """Send graceful termination signal to the process."""
        ...

    def kill(self) -> None:
        """Send forceful kill signal to the process."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process termination and return exit code."""
        ...

    def poll(self) -> int | None:
        """Return exit code if terminated, or None if still running."""
        ...


@runtime_checkable
class CommandRunnerProtocol(Protocol):
    """Injectable protocol for running CLI commands and spawning background server processes."""

    def run(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        capture_output: bool = False,
        text: bool = False,
    ) -> CommandResult:
        """Execute a synchronous command and return its `CommandResult`."""
        ...

    def spawn_server(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ServerProcessProtocol:
        """Spawn a background server process returning a `ServerProcessProtocol` handle."""
        ...


class SubprocessCommandRunner:
    """Production `CommandRunnerProtocol` backed by `subprocess.run` and `subprocess.Popen`."""

    def run(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        capture_output: bool = False,
        text: bool = False,
    ) -> CommandResult:
        """
        Execute a command via `subprocess.run` and wrap the result in `CommandResult`.

        Args:
            cmd: Command argument sequence.
            env: Optional environment mapping.
            cwd: Optional working directory path.
            timeout: Optional timeout in seconds.
            capture_output: Whether to capture stdout and stderr.
            text: Whether to decode stdout/stderr as UTF-8 text.

        Returns:
            CommandResult: Exit code and captured stdout.

        """
        cmd_list = list(cmd)
        proc = subprocess.run(
            cmd_list,
            env=env,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            check=False,
        )
        stdout_str = proc.stdout if isinstance(proc.stdout, str) else ""
        return CommandResult(
            command=tuple(cmd_list),
            returncode=proc.returncode,
            stdout=stdout_str,
        )

    def spawn_server(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ServerProcessProtocol:
        """
        Spawn a background server process via `subprocess.Popen`.

        Args:
            cmd: Command argument sequence.
            env: Optional environment mapping.
            cwd: Optional working directory path.

        Returns:
            ServerProcessProtocol: Spawned `Popen` process handle.

        """
        return subprocess.Popen(list(cmd), env=env, cwd=cwd)


@dataclass(frozen=True)
class TestSuiteConfig:
    """
    Configuration options controlling full-stack test orchestration and badge updates.

    Attributes:
        skip_readme_update: When True, skips mutating `README.md` badges on disk.
        coverage_threshold: Minimum required combined LCOV line, branch, and function coverage percentage.
        backend_ready_url: Readiness URL for the FastAPI backend server.
        frontend_ready_url: Readiness URL for the Vite frontend dev server.

    """

    __test__ = False

    skip_readme_update: bool = False
    coverage_threshold: float = 100.0
    backend_ready_url: str = "http://127.0.0.1:8000/api/ready"
    frontend_ready_url: str = "http://127.0.0.1:5173"


def _ingest_lcov_record_line(file_data: FileLcovData, line: str) -> None:
    """
    Parse a single `DA:`, `FN:`, `FNDA:`, or `BRDA:` LCOV record line into `file_data`.

    Args:
        file_data: Mutable per-file coverage accumulator.
        line: Trimmed LCOV line belonging to the current `SF:` section.

    """
    if line.startswith("DA:"):
        parts = line[3:].split(",")
        line_num = int(parts[0])
        hits = int(parts[1])
        file_data.lines[line_num] = max(file_data.lines[line_num], hits)
    elif line.startswith("FN:"):
        fn_parts = line[3:].split(",", 2)
        if len(fn_parts) >= 2:
            fn_line = int(fn_parts[0])
            fn_name = (
                fn_parts[2]
                if len(fn_parts) == 3 and fn_parts[1].isdigit()
                else ",".join(fn_parts[1:])
            )
            prev = file_data.functions.get(fn_name)
            file_data.functions[fn_name] = (fn_line, prev[1] if prev else 0)
    elif line.startswith("FNDA:"):
        fnda_parts = line[5:].split(",", 1)
        if len(fnda_parts) == 2:
            fn_hits = int(fnda_parts[0])
            fn_name = fnda_parts[1]
            prev = file_data.functions.get(fn_name, (1, 0))
            file_data.functions[fn_name] = (prev[0], max(prev[1], fn_hits))
    elif line.startswith("BRDA:"):
        br_parts = line[5:].split(",")
        if len(br_parts) == 4:
            br_key = (int(br_parts[0]), br_parts[1], br_parts[2])
            taken = 0 if br_parts[3] == "-" else int(br_parts[3])
            file_data.branches[br_key] = max(file_data.branches.get(br_key, 0), taken)


def parse_lcov_text_into(
    lcov_text: str,
    cov_store: collections.defaultdict[str, FileLcovData],
) -> None:
    """
    Parse LCOV text (`SF:`, `FN:`, `FNDA:`, `BRDA:`, `DA:`) into `cov_store`.

    Args:
        lcov_text: Raw LCOV report text.
        cov_store: Destination mapping of `filepath -> FileLcovData`.

    """
    current_file = ""
    for raw_line in lcov_text.splitlines():
        line = raw_line.strip()
        if line.startswith("SF:"):
            current_file = line[3:]
        elif line == "end_of_record":
            current_file = ""
        elif current_file:
            _ingest_lcov_record_line(cov_store[current_file], line)


def parse_lcov_files(
    lcov_paths: Sequence[str | Path],
) -> collections.defaultdict[str, FileLcovData]:
    """
    Parse one or more LCOV files from disk and merge line, function, and branch hit counts.

    Args:
        lcov_paths: Sequence of paths to LCOV files on disk.

    Returns:
        DefaultDict[str, FileLcovData]: Mapping of `filepath -> FileLcovData`.

    """
    cov_store: collections.defaultdict[str, FileLcovData] = collections.defaultdict(
        FileLcovData
    )
    for lcov_file in lcov_paths:
        try:
            with open(lcov_file, encoding="utf-8") as f:
                parse_lcov_text_into(f.read(), cov_store)
        except FileNotFoundError:
            print(f"Warning: {lcov_file} not found.")
    return cov_store


def _format_single_file_lcov(
    filename: str,
    file_data: FileLcovData,
) -> tuple[str, int, int, int, int, int, int]:
    """
    Render merged LCOV records for a single source file and return `(text, lf, lh, fnf, fnh, brf, brh)`.

    Args:
        filename: Source file path string.
        file_data: Aggregated line, function, and branch counters.

    Returns:
        Tuple[str, int, int, int, int, int, int]: Rendered LCOV section and found/hit totals.

    """
    records: list[str] = [f"SF:{filename}"]

    fnf = len(file_data.functions)
    fnh = 0
    for fn_name, (fn_line, _fn_hits) in sorted(
        file_data.functions.items(), key=lambda item: (item[1][0], item[0])
    ):
        records.append(f"FN:{fn_line},{fn_name}")
    for fn_name, (_fn_line, fn_hits) in sorted(
        file_data.functions.items(), key=lambda item: (item[1][0], item[0])
    ):
        records.append(f"FNDA:{fn_hits},{fn_name}")
        if fn_hits > 0:
            fnh += 1
    if fnf > 0:
        records.append(f"FNF:{fnf}")
        records.append(f"FNH:{fnh}")

    brf = len(file_data.branches)
    brh = 0
    for (b_line, b_block, b_id), b_hits in sorted(file_data.branches.items()):
        records.append(f"BRDA:{b_line},{b_block},{b_id},{b_hits}")
        if b_hits > 0:
            brh += 1
    if brf > 0:
        records.append(f"BRF:{brf}")
        records.append(f"BRH:{brh}")

    lf = len(file_data.lines)
    lh = 0
    for line_num, hits in sorted(file_data.lines.items()):
        records.append(f"DA:{line_num},{hits}")
        if hits > 0:
            lh += 1
    records.append(f"LF:{lf}")
    records.append(f"LH:{lh}")
    records.append("end_of_record")
    records.append("")
    line_sep = """
"""
    return line_sep.join(records), lf, lh, fnf, fnh, brf, brh


def evaluate_lcov_store(
    cov_store: collections.defaultdict[str, FileLcovData],
) -> CoverageMetrics:
    """
    Compute `CoverageMetrics` (line, function, and branch totals plus formatted LCOV) from `cov_store`.

    Args:
        cov_store: Mapping of `filepath -> FileLcovData`.

    Returns:
        CoverageMetrics: Complete line, function, and branch coverage summary.

    """
    total_lf = 0
    total_lh = 0
    total_fnf = 0
    total_fnh = 0
    total_brf = 0
    total_brh = 0
    chunks: list[str] = []

    for filename, file_data in cov_store.items():
        chunk, lf, lh, fnf, fnh, brf, brh = _format_single_file_lcov(
            filename, file_data
        )
        chunks.append(chunk)
        total_lf += lf
        total_lh += lh
        total_fnf += fnf
        total_fnh += fnh
        total_brf += brf
        total_brh += brh

    return CoverageMetrics(
        total_found=total_lf,
        total_hit=total_lh,
        formatted_lcov="".join(chunks),
        functions_found=total_fnf,
        functions_hit=total_fnh,
        branches_found=total_brf,
        branches_hit=total_brh,
    )


def evaluate_coverage_metrics(
    lcov_paths: Sequence[str | Path],
) -> CoverageMetrics:
    """
    Parse and merge LCOV files on disk into a `CoverageMetrics` summary.

    Args:
        lcov_paths: Sequence of LCOV report file paths.

    Returns:
        CoverageMetrics: Merged line, function, and branch coverage summary.

    """
    return evaluate_lcov_store(parse_lcov_files(lcov_paths))


def evaluate_lcov_text(lcov_text: str) -> CoverageMetrics:
    """
    Parse raw LCOV text directly into a `CoverageMetrics` summary.

    Args:
        lcov_text: Raw LCOV report text.

    Returns:
        CoverageMetrics: Parsed line, function, and branch coverage summary.

    """
    cov_store: collections.defaultdict[str, FileLcovData] = collections.defaultdict(
        FileLcovData
    )
    parse_lcov_text_into(lcov_text, cov_store)
    return evaluate_lcov_store(cov_store)


def _is_stub_or_abstract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    in_protocol_class: bool,
) -> bool:
    """
    Return True if `node` is a `Protocol` method, `@abstractmethod` / `@overload`, or pure docstring/`...` stub.

    Args:
        node: Python AST function or async function node.
        in_protocol_class: Whether `node` is declared directly inside a `Protocol` class.

    Returns:
        bool: True when `node` has no executable runtime body to cover.

    """
    if in_protocol_class:
        return True
    for dec in node.decorator_list:
        dec_name = (
            dec.id
            if isinstance(dec, ast.Name)
            else (dec.attr if isinstance(dec, ast.Attribute) else "")
        )
        if dec_name in ("abstractmethod", "overload"):
            return True
    stmts = list(node.body)
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]
    if not stmts:
        return True
    return (
        len(stmts) == 1
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and stmts[0].value.value is Ellipsis
    )


def _executable_body_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[int]:
    """
    Collect line numbers of executable statements in `node.body` (excluding leading docstring).

    Args:
        node: Python AST function or async function definition.

    Returns:
        Set[int]: Set of 1-based statement line numbers inside `node`.

    """
    stmts = list(node.body)
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]
    return {
        sub.lineno
        for stmt in stmts
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.stmt)
    }


def _audit_functions_in_ast(
    tree: ast.Module,
    rel_label: str,
    executed_lines: set[int],
) -> tuple[int, int, list[str]]:
    """
    Walk `tree` and check every concrete `def` / `async def` against `executed_lines`.

    Args:
        tree: Parsed module AST.
        rel_label: Display path label for uncovered function reporting.
        executed_lines: Set of executed line numbers recorded in `coverage.json`.

    Returns:
        Tuple[int, int, List[str]]: `(total_functions, covered_functions, uncovered_names)`.

    """
    total = 0
    covered = 0
    uncovered: list[str] = []

    def _visit_scope(body: Sequence[ast.stmt], prefix: str, in_protocol: bool) -> None:
        """Recursively inspect class and function definitions within `body`."""
        nonlocal total, covered
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                is_proto = any(
                    (isinstance(b, ast.Name) and b.id == "Protocol")
                    or (isinstance(b, ast.Attribute) and b.attr == "Protocol")
                    for b in stmt.bases
                )
                next_prefix = f"{prefix}.{stmt.name}" if prefix else stmt.name
                _visit_scope(stmt.body, next_prefix, is_proto)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}.{stmt.name}" if prefix else stmt.name
                if not _is_stub_or_abstract_function(stmt, in_protocol):
                    body_lines = _executable_body_lines(stmt)
                    total += 1
                    if body_lines.intersection(executed_lines):
                        covered += 1
                    else:
                        uncovered.append(f"{rel_label}:{stmt.lineno}:{qual}")
                _visit_scope(stmt.body, qual, False)

    _visit_scope(tree.body, "", False)
    return total, covered, uncovered


def audit_python_function_coverage(
    coverage_json_path: str | Path,
    source_paths: Sequence[str | Path],
) -> tuple[int, int, list[str]]:
    """
    Verify every concrete Python `def`/`async def` in `source_paths` has executed lines in `coverage.json`.

    Args:
        coverage_json_path: Path to pytest `coverage.json` containing per-file `executed_lines`.
        source_paths: Sequence of Python source directories or `.py` files to audit.

    Returns:
        Tuple[int, int, List[str]]: `(total_functions, covered_functions, uncovered_qualnames)`.

    """
    with open(coverage_json_path, encoding="utf-8") as f:
        raw_cov = json.load(f)

    files_section = raw_cov.get("files", {})
    if not files_section:
        return 0, 0, []

    executed_by_resolved: dict[str, set[int]] = {}
    for file_key, file_payload in files_section.items():
        exec_list = file_payload.get("executed_lines", [])
        executed_by_resolved[str(Path(file_key).resolve())] = {
            int(ln) for ln in exec_list
        }

    py_files: list[Path] = []
    for src_item in source_paths:
        p = Path(src_item)
        if p.is_file() and p.suffix == ".py":
            py_files.append(p)
        elif p.is_dir():
            py_files.extend(sorted(p.rglob("*.py")))

    total_fns = 0
    covered_fns = 0
    uncovered_all: list[str] = []
    for py_file in py_files:
        resolved_str = str(py_file.resolve())
        if resolved_str not in executed_by_resolved:
            continue
        source_text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(py_file))
        f_tot, f_cov, f_unc = _audit_functions_in_ast(
            tree, str(py_file), executed_by_resolved[resolved_str]
        )
        total_fns += f_tot
        covered_fns += f_cov
        uncovered_all.extend(f_unc)

    return total_fns, covered_fns, uncovered_all


def audit_forbidden_coverage_ignores(
    source_roots: Sequence[str | Path],
) -> list[str]:
    """
    Scan Python and TypeScript source trees for forbidden coverage-exclusion comments.

    Rejects `# pragma: no cover`, `v8 ignore`, `istanbul ignore`, and `c8 ignore`.

    Args:
        source_roots: Sequence of source directories or files to scan.

    Returns:
        List[str]: List of `filepath:line_number` violations (empty when clean).

    """
    violations: list[str] = []
    for src_root in source_roots:
        root_p = Path(src_root)
        if not root_p.exists():
            continue
        candidates = (
            [root_p]
            if root_p.is_file()
            else [
                p
                for p in sorted(root_p.rglob("*"))
                if p.is_file() and p.suffix in (".py", ".ts")
            ]
        )
        for file_path in candidates:
            for idx, line in enumerate(
                file_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if FORBIDDEN_COVERAGE_IGNORE_RE.search(line):
                    violations.append(f"{file_path}:{idx}")
    return violations


class _UrlOpenResultProtocol:
    """Marker protocol for `urlopen_fn` return values in `wait_for_service_ready`."""


def wait_for_service_ready(
    url: str,
    max_attempts: int = 30,
    sleep_seconds: float = 1.0,
    urlopen_fn: Callable[
        ..., _UrlOpenResultProtocol | str | bytes | None
    ] = urllib.request.urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """
    Poll an HTTP health/readiness endpoint until it responds or `max_attempts` is exhausted.

    Args:
        url: Target HTTP URL to poll.
        max_attempts: Maximum polling iterations.
        sleep_seconds: Delay in seconds between failed attempts.
        urlopen_fn: Injectable URL opener callable.
        sleep_fn: Injectable sleep callable.

    Returns:
        bool: True if the endpoint responded without raising an exception, False otherwise.

    """
    for _ in range(max_attempts):
        try:
            urlopen_fn(url)
            return True
        except (urllib.error.URLError, OSError, ValueError, RuntimeError):
            sleep_fn(sleep_seconds)
    return False


@contextlib.contextmanager
def managed_server_processes(
    procs: Sequence[ServerProcessProtocol],
    wait_timeout: float = 5.0,
) -> Iterator[None]:
    """
    Ensure spawned background server processes are terminated and waited on exit or failure.

    Args:
        procs: Sequence of active `ServerProcessProtocol` server handles.
        wait_timeout: Graceful termination wait timeout in seconds before `kill()` fallback.

    Yields:
        None: Executes the enclosed E2E block before guaranteed process cleanup.

    """
    try:
        yield
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def extract_pytest_coverage_percent(coverage_json_path: str | Path) -> int:
    """
    Read pytest `coverage.json` and return the integer total covered percentage.

    Args:
        coverage_json_path: Path to the `coverage.json` report file.

    Returns:
        int: Integer coverage percentage (0–100).

    """
    with open(coverage_json_path, encoding="utf-8") as f:
        data = json.load(f)
        return int(float(data["totals"]["percent_covered"]))


def extract_interrogate_coverage_percent(stdout_text: str) -> int:
    """
    Parse `interrogate` CLI output to extract the integer docstring coverage percentage.

    Args:
        stdout_text: Raw standard output string from `interrogate`.

    Returns:
        int: Parsed integer docstring coverage percentage, or 0 if not found.

    """
    match = re.search(r"(?:actual|TOTAL):\s*([0-9.]+)%", stdout_text)
    return int(float(match.group(1))) if match else 0


def resolve_badge_color(percent: float) -> str:
    """
    Determine the shields.io badge color for a given coverage percentage.

    Args:
        percent: Coverage percentage (`0.0` to `100.0`).

    Returns:
        str: `'brightgreen'`, `'yellow'`, or `'red'`.

    """
    if percent >= 90:
        return "brightgreen"
    if percent >= 70:
        return "yellow"
    return "red"


def update_readme_badges(
    content: str,
    test_cov: int,
    doc_cov: int,
) -> tuple[str, bool]:
    """
    Update Coverage and Docs shields in `README.md` content.

    Args:
        content: Existing `README.md` markdown content.
        test_cov: Integer test coverage percentage (0–100).
        doc_cov: Integer docstring coverage percentage (0–100).

    Returns:
        Tuple[str, bool]: `(updated_content, license_shield_found)`.

    """
    test_color = resolve_badge_color(test_cov)
    test_shield = f"[![Coverage](https://img.shields.io/badge/Coverage-{test_cov}%25-{test_color}.svg)](#)"

    doc_color = resolve_badge_color(doc_cov)
    doc_shield = (
        f"[![Docs](https://img.shields.io/badge/Docs-{doc_cov}%25-{doc_color}.svg)](#)"
    )

    line_sep = """
"""
    kept_lines = [
        ln
        for ln in content.splitlines(keepends=True)
        if not ln.lstrip().startswith("[![Coverage]")
        and not ln.lstrip().startswith("[![Docs]")
    ]
    out_lines: list[str] = []
    found_license = False
    for ln in kept_lines:
        out_lines.append(ln)
        if not found_license and ln.lstrip().startswith("[![License]"):
            if not ln.endswith(line_sep):
                out_lines[-1] = ln + line_sep
            out_lines.append(f"{test_shield}{line_sep}")
            out_lines.append(f"{doc_shield}{line_sep}")
            found_license = True
    return "".join(out_lines), found_license


def build_test_commands(
    npm_cmd: str,
    pytest_cmd: str,
) -> dict[str, Sequence[str]]:
    """
    Construct the standard CLI command vectors used by the test runner.

    Args:
        npm_cmd: Resolved `npm` executable path.
        pytest_cmd: Resolved `pytest` executable path.

    Returns:
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


def sync_readme_shields_on_disk(
    readme_path: str | Path,
    test_cov: int,
    doc_cov: int,
) -> bool:
    """
    Read `readme_path`, update its Coverage and Docs shields, and write the updated content back to disk.

    Args:
        readme_path: Path to `README.md`.
        test_cov: Integer test coverage percentage (0–100).
        doc_cov: Integer docstring coverage percentage (0–100).

    Returns:
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
    cmd: list[str],
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    runner: CommandRunnerProtocol | None = None,
) -> CommandResult:
    """
    Run a shell command via `CommandRunnerProtocol`, failing the script if the exit code is non-zero.

    Args:
        cmd: A list of strings comprising the command to be executed.
        env: An optional dictionary of environment variables.
        cwd: An optional string representing the working directory to execute the command in.
        timeout: Optional timeout in seconds for the subprocess.
        runner: Optional injectable `CommandRunnerProtocol` instance.

    Returns:
        CommandResult: Structured result when the command exits with code 0.

    """
    active_runner = runner if runner is not None else SubprocessCommandRunner()
    print(f"Running: {' '.join(cmd)}")
    result = active_runner.run(cmd, env=env, cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result


def _execute_e2e_stage(
    root: Path,
    web_dir: Path,
    npm_cmd: str,
    env: dict[str, str],
    suite_cfg: TestSuiteConfig,
    runner: CommandRunnerProtocol,
    urlopen_fn: Callable[
        ..., _UrlOpenResultProtocol | str | bytes | None
    ] = urllib.request.urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Seed an isolated temporary E2E workspace, launch backend/frontend servers safely, and run Playwright + NYC.

    Args:
        root: Repository workspace root path.
        web_dir: Frontend `t1d-analytics-web` directory path.
        npm_cmd: Resolved `npm` executable path.
        env: Base environment mapping.
        suite_cfg: Orchestration configuration with readiness URLs.
        runner: Active `CommandRunnerProtocol` implementation.
        urlopen_fn: Injectable HTTP readiness opener callable.
        sleep_fn: Injectable sleep callable for readiness polling.

    """
    shutil.rmtree(web_dir / ".nyc_output", ignore_errors=True)
    for stale_artifact in ("ci_test.duckdb", "ci_test.duckdb.wal"):
        stale_p = root / stale_artifact
        if stale_p.exists():
            stale_p.unlink()

    with tempfile.TemporaryDirectory(prefix="t1d_e2e_") as tmp_e2e_str:
        e2e_dir = Path(tmp_e2e_str)
        (e2e_dir / "patients.csv").write_text(
            """id,name,age
1,Alice,30
2,Bob,40
""",
            encoding="utf-8",
        )
        (e2e_dir / "tiny.csv").write_text(
            """id,name,value
1,Test A,100
2,Test B,200
""",
            encoding="utf-8",
        )
        ci_db_str = str(e2e_dir / "ci_test.duckdb")
        sessions_db_str = str(e2e_dir / "sessions.duckdb")
        seed_env = env.copy()
        seed_env["T1D_EXTRA_DB_ROOTS"] = str(e2e_dir)
        system_python = sys.executable
        run_cmd(
            [
                system_python,
                "-c",
                (
                    "from t1d_analytics.seed import seed_database; "
                    f"seed_database(data_dir={str(e2e_dir)!r}, db_path={ci_db_str!r}, sessions_db_path={sessions_db_str!r}, overwrite_csvs=True)"
                ),
            ],
            env=seed_env,
            runner=runner,
        )

        backend_env = seed_env.copy()
        backend_env["T1D_DB_PATH"] = ci_db_str
        backend_env["T1D_SESSIONS_DB_PATH"] = sessions_db_str

        front_env = env.copy()
        front_env["T1D_API_PORT"] = "8000"
        front_env["VITE_COVERAGE"] = "true"

        print("Starting backend server...")
        backend_proc = runner.spawn_server(
            [system_python, "-m", "uvicorn", "t1d_analytics.api:app", "--port", "8000"],
            env=backend_env,
        )
        print("Starting frontend server...")
        frontend_proc = runner.spawn_server(
            [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1"],
            cwd=str(web_dir),
            env=front_env,
        )

        with managed_server_processes([backend_proc, frontend_proc]):
            print("Waiting for backend DB to connect and API to be ready...")
            if not wait_for_service_ready(
                suite_cfg.backend_ready_url,
                urlopen_fn=urlopen_fn,
                sleep_fn=sleep_fn,
            ):
                print("Backend failed to start.")
                sys.exit(1)

            print("Waiting for frontend dev server to be ready...")
            if not wait_for_service_ready(
                suite_cfg.frontend_ready_url,
                urlopen_fn=urlopen_fn,
                sleep_fn=sleep_fn,
            ):
                print("Frontend failed to start.")
                sys.exit(1)

            print("Servers ready. Running playwright tests...")
            npx_cmd = shutil.which("npx") or "npx"
            test_result = runner.run([npx_cmd, "playwright", "test"], cwd=str(web_dir))
            coverage_result = runner.run(
                [
                    npx_cmd,
                    "nyc",
                    "report",
                    "--reporter=lcov",
                    "--report-dir=coverage/e2e",
                ],
                cwd=str(web_dir),
            )

        if test_result.returncode != 0:
            print(f"E2E tests failed with exit code {test_result.returncode}")
            sys.exit(test_result.returncode)
        if coverage_result.returncode != 0:
            print(f"E2E coverage failed with exit code {coverage_result.returncode}")
            sys.exit(coverage_result.returncode)


def _verify_combined_coverage_and_ast(
    root: Path,
    web_dir: Path,
    suite_cfg: TestSuiteConfig,
) -> CoverageMetrics:
    """
    Validate forbidden coverage ignores, merge LCOV line/branch/function coverage, and verify Python AST function coverage.

    Args:
        root: Repository workspace root path.
        web_dir: Frontend `t1d-analytics-web` directory path.
        suite_cfg: Orchestration configuration with required coverage threshold.

    Returns:
        CoverageMetrics: Validated combined coverage metrics.

    """
    ignore_violations = audit_forbidden_coverage_ignores(
        [
            root / "t1d-analytics-backend" / "src",
            web_dir / "src",
        ]
    )
    if ignore_violations:
        print(f"Forbidden coverage ignore directives found: {ignore_violations}")
        sys.exit(1)

    print("Merging coverage...")
    lcov_files: Sequence[str | Path] = [
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

    print(
        f"Combined Coverage: lines={metrics.percent:.2f}%, "
        f"functions={metrics.function_percent:.2f}%, "
        f"branches={metrics.branch_percent:.2f}%"
    )
    if not metrics.meets_threshold(suite_cfg.coverage_threshold):
        print(
            f"Coverage is below {suite_cfg.coverage_threshold:.2f}% threshold! "
            f"(lines={metrics.percent:.2f}%, functions={metrics.function_percent:.2f}%, "
            f"branches={metrics.branch_percent:.2f}%)"
        )
        sys.exit(1)

    cov_json_path = root / "coverage.json"
    if cov_json_path.exists():
        try:
            _tot_fn, _cov_fn, uncovered_py_fns = audit_python_function_coverage(
                cov_json_path,
                [
                    root / "t1d-analytics-backend" / "src" / "t1d_analytics",
                    root / "scripts",
                ],
            )
        except (OSError, ValueError, SyntaxError) as exc:
            print(f"Failed to audit Python function coverage: {exc}")
            sys.exit(1)
        if uncovered_py_fns:
            print(f"Uncovered Python functions detected: {uncovered_py_fns}")
            sys.exit(1)

    return metrics


def _update_shields_stage(
    root: Path,
    env: dict[str, str],
    runner: CommandRunnerProtocol,
) -> None:
    """
    Extract pytest and interrogate percentages and update `README.md` shields on disk.

    Args:
        root: Repository workspace root path.
        env: Subprocess environment dictionary.
        runner: Active `CommandRunnerProtocol` implementation.

    """
    print("Generating shields in README.md...")
    try:
        test_cov = extract_pytest_coverage_percent(root / "coverage.json")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"Failed to read coverage.json: {e}")
        sys.exit(1)

    try:
        interrogate_cmd = shutil.which("interrogate", path=env["PATH"]) or "interrogate"
        pyproject_cfg = root / "t1d-analytics-backend" / "pyproject.toml"
        interrogate_args = [interrogate_cmd]
        if pyproject_cfg.exists():
            interrogate_args.extend(["-c", str(pyproject_cfg)])
        interrogate_args.extend(["-vv", "--fail-under=100"])
        target_appended = False
        for target_rel in (
            root / "t1d-analytics-backend" / "src" / "t1d_analytics",
            root / "t1d-analytics-backend" / "tests",
            root / "t1d-analytics-backend" / "docs_src",
            root / "scripts",
        ):
            if target_rel.exists():
                interrogate_args.append(str(target_rel))
                target_appended = True
        if not target_appended:
            interrogate_args.append(
                str(root / "t1d-analytics-backend" / "src" / "t1d_analytics")
            )
        result = runner.run(
            interrogate_args,
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


def main(
    skip_readme_update: bool = False,
    workspace_root: str | Path | None = None,
    suite_config: TestSuiteConfig | None = None,
    runner: CommandRunnerProtocol | None = None,
    urlopen_fn: Callable[
        ..., _UrlOpenResultProtocol | str | bytes | None
    ] = urllib.request.urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Execute the main entrypoint for the test orchestration script.

    Performs building, unit tests, isolated server orchestration, E2E checks,
    line/branch/function coverage verification, and optional shield updates.

    Args:
        skip_readme_update: When True (or when `T1D_SKIP_README_SHIELD_UPDATE=1` is set),
            skips modifying `README.md` on disk.
        workspace_root: Optional workspace directory root (defaults to current working directory).
        suite_config: Optional `TestSuiteConfig` overriding URLs or thresholds.
        runner: Optional injectable `CommandRunnerProtocol` implementation.
        urlopen_fn: Injectable HTTP readiness opener callable.
        sleep_fn: Injectable sleep callable for readiness polling.

    """
    active_runner = runner if runner is not None else SubprocessCommandRunner()
    root = Path(workspace_root) if workspace_root is not None else Path()
    web_dir = root / "t1d-analytics-web"

    suite_cfg = suite_config or TestSuiteConfig(
        skip_readme_update=skip_readme_update
        or os.environ.get("T1D_SKIP_README_SHIELD_UPDATE") == "1"
    )
    print("Running web install and build...")
    npm_cmd = shutil.which("npm") or "npm"
    env = os.environ.copy()
    env.pop("T1D_SKIP_README_SHIELD_UPDATE", None)
    python_bin_dir = str(Path(sys.executable).absolute().parent)
    env["PATH"] = f"{python_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str((root / "t1d-analytics-backend" / "src").resolve())

    pytest_cmd = shutil.which("pytest", path=env["PATH"]) or "pytest"
    cmds = build_test_commands(npm_cmd=npm_cmd, pytest_cmd=pytest_cmd)

    run_cmd(list(cmds["web_install"]), cwd=str(web_dir), runner=active_runner)
    run_cmd(list(cmds["web_build"]), cwd=str(web_dir), runner=active_runner)

    print("Running python tests and coverage...")
    run_cmd(list(cmds["backend_pytest"]), env=env, runner=active_runner)

    print("Running frontend tests...")
    run_cmd(list(cmds["web_coverage"]), cwd=str(web_dir), runner=active_runner)

    print("Running e2e tests...")
    _execute_e2e_stage(
        root,
        web_dir,
        npm_cmd,
        env,
        suite_cfg,
        active_runner,
        urlopen_fn=urlopen_fn,
        sleep_fn=sleep_fn,
    )
    _verify_combined_coverage_and_ast(root, web_dir, suite_cfg)

    if not suite_cfg.skip_readme_update:
        _update_shields_stage(root, env, active_runner)

    print("Pre-commit checks passed successfully!")


if __name__ == "__main__":
    main()
