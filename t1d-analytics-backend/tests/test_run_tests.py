"""
Tests for test orchestration script (`scripts/run_tests.py`).

Verifies LCOV line/branch/function coverage merging, Python AST function
execution auditing, forbidden coverage-ignore detection, safe child server
lifecycle cleanup, isolated E2E scratch directories, and README shield updates.
"""

import json
import runpy
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.run_tests import (
    CommandResult,
    CommandRunnerProtocol,
    CoverageMetrics,
    ServerProcessProtocol,
    SubprocessCommandRunner,
    TestSuiteConfig,
    audit_forbidden_coverage_ignores,
    audit_python_function_coverage,
    build_test_commands,
    evaluate_coverage_metrics,
    evaluate_lcov_text,
    extract_interrogate_coverage_percent,
    extract_pytest_coverage_percent,
    main,
    managed_server_processes,
    parse_lcov_files,
    resolve_badge_color,
    run_cmd,
    update_readme_badges,
    wait_for_service_ready,
)

NL = "\n"
PRAGMA_NO_COVER = "# pragma: " + "no cover"
V8_IGNORE = "/* v8 " + "ignore next */"


def test_lcov_line_function_branch_merge_and_thresholds(tmp_path: Path) -> None:
    """Test merging FN/FNDA, BRDA, and DA records and enforcing 100% across all three dimensions."""
    lcov_a = tmp_path / "backend.lcov"
    lcov_b = tmp_path / "frontend.lcov"
    lcov_a.write_text(
        NL.join(
            [
                "SF:src/mod.py",
                "FN:1,alpha",
                "FN:5,8,beta",
                "FN:9",
                "FNDA:1,alpha",
                "FNDA:0,beta",
                "FNDA:bad",
                "BRDA:2,0,0,1",
                "BRDA:2,0,1,-",
                "BRDA:short",
                "DA:1,1",
                "DA:2,0",
                "end_of_record",
                "DA:999,1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lcov_b.write_text(
        NL.join(
            [
                "SF:src/mod.py",
                "FNDA:3,beta",
                "FN:12,gamma",
                "FNDA:2,gamma",
                "BRDA:2,0,1,4",
                "DA:1,2",
                "DA:2,1",
                "end_of_record",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_coverage_metrics([lcov_a, lcov_b, tmp_path / "missing.lcov"])
    assert metrics.total_found == 2
    assert metrics.total_hit == 2
    assert metrics.functions_found == 3
    assert metrics.functions_hit == 3
    assert metrics.branches_found == 2
    assert metrics.branches_hit == 2
    assert metrics.percent == 100.0
    assert metrics.function_percent == 100.0
    assert metrics.branch_percent == 100.0
    assert metrics.meets_threshold(100.0) is True
    assert "FN:1,alpha" in metrics.formatted_lcov
    assert "FNDA:3,beta" in metrics.formatted_lcov
    assert "FNF:3" in metrics.formatted_lcov
    assert "FNH:3" in metrics.formatted_lcov
    assert "BRDA:2,0,1,4" in metrics.formatted_lcov
    assert "BRF:2" in metrics.formatted_lcov
    assert "BRH:2" in metrics.formatted_lcov

    partial_fn = evaluate_lcov_text(
        NL.join(
            [
                "SF:a.py",
                "FN:1,fn1",
                "FNDA:0,fn1",
                "DA:1,1",
                "DA:3,0",
                "end_of_record",
                "",
            ]
        )
    )
    assert partial_fn.percent == 50.0
    assert partial_fn.function_percent == 0.0
    assert partial_fn.meets_threshold(100.0) is False

    partial_br = evaluate_lcov_text(
        NL.join(["SF:a.py", "BRDA:1,0,0,-", "DA:1,1", "end_of_record", ""])
    )
    assert partial_br.percent == 100.0
    assert partial_br.branch_percent == 0.0
    assert partial_br.meets_threshold(100.0) is False

    empty_m = CoverageMetrics(total_found=0, total_hit=0, formatted_lcov="")
    assert empty_m.percent == 0.0
    assert empty_m.function_percent == 100.0
    assert empty_m.branch_percent == 100.0
    assert len(parse_lcov_files([lcov_a])) == 1


def test_python_ast_function_coverage_and_forbidden_ignores(tmp_path: Path) -> None:
    """Test audit_python_function_coverage and audit_forbidden_coverage_ignores on real source files."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    mod_file = pkg_dir / "sample_mod.py"
    single_file = tmp_path / "standalone.py"
    untracked_file = pkg_dir / "untracked.py"
    untracked_file.write_text(
        NL.join(["def never_in_json() -> int:", "    return 1", ""]),
        encoding="utf-8",
    )

    mod_source = NL.join(
        [
            '"""Sample module."""',
            "import abc",
            "import typing",
            "from typing import Protocol, overload",
            "",
            "class MyProto(Protocol):",
            "    def proto_method(self) -> int:",
            '        """Docstring only stub."""',
            "",
            "class DottedProto(typing.Protocol):",
            "    def proto_ellipsis(self) -> int:",
            "        ...",
            "",
            "class BaseAbs:",
            "    @abc.abstractmethod",
            "    def abs_attr(self) -> None:",
            "        raise NotImplementedError",
            "",
            "    @staticmethod",
            "    def concrete_static() -> int:",
            '        """Static helper."""',
            "        return 1",
            "",
            "@overload",
            "def ov_fn(x: int) -> int: ...",
            "",
            "def ov_fn(x: int) -> int:",
            '    """Concrete implementation."""',
            "    return x + 1",
            "",
            "def doc_only_fn() -> None:",
            '    """Pure docstring stub."""',
            "",
            "def ellipsis_only_fn() -> None:",
            '    """Docstring plus ellipsis."""',
            "    ...",
            "",
            "async def async_missed() -> int:",
            '    """Uncovered async function."""',
            "    return 42",
            "",
        ]
    )
    mod_file.write_text(mod_source, encoding="utf-8")
    single_file.write_text(
        NL.join(["def standalone_covered() -> int:", "    return 7", ""]),
        encoding="utf-8",
    )

    cov_json = tmp_path / "coverage.json"
    cov_json.write_text('{"totals": {"percent_covered": 100}}', encoding="utf-8")
    assert audit_python_function_coverage(cov_json, [pkg_dir]) == (0, 0, [])

    cov_payload = {
        "files": {
            str(mod_file): {"executed_lines": [22, 29]},
            str(single_file): {"executed_lines": [2]},
        },
        "totals": {"percent_covered": 90},
    }
    cov_json.write_text(json.dumps(cov_payload), encoding="utf-8")
    tot, cov, unc = audit_python_function_coverage(
        cov_json, [pkg_dir, single_file, tmp_path / "nonexistent_dir"]
    )
    assert tot == 4
    assert cov == 3
    assert len(unc) == 1
    assert "async_missed" in unc[0]

    # Forbidden coverage ignore scanner
    clean_ts = pkg_dir / "clean.ts"
    clean_ts.write_text(f"export const x: number = 1;{NL}", encoding="utf-8")
    assert (
        audit_forbidden_coverage_ignores([pkg_dir, single_file, tmp_path / "missing"])
        == []
    )

    bad_py = pkg_dir / "bad.py"
    bad_py.write_text(f"x = 1  {PRAGMA_NO_COVER}{NL}", encoding="utf-8")
    bad_ts = tmp_path / "bad.ts"
    bad_ts.write_text(
        f"{V8_IGNORE}{NL}export const y = 2;{NL}",
        encoding="utf-8",
    )
    violations = audit_forbidden_coverage_ignores([pkg_dir, bad_ts])
    assert len(violations) == 2


def test_badges_commands_ready_and_managed_processes() -> None:
    """Test badge formatting, command generation, readiness polling, and stubborn process kill fallback."""
    raw_readme_no_trailing_nl = "[![License](https://example.com/lic.svg)](#)"
    updated_green, found = update_readme_badges(raw_readme_no_trailing_nl, 100, 100)
    assert found is True
    assert "Coverage-100%25-brightgreen.svg" in updated_green
    assert "Docs-100%25-brightgreen.svg" in updated_green

    replaced_twice, found_twice = update_readme_badges(updated_green, 80, 60)
    assert found_twice is True
    assert replaced_twice.count("[![Coverage]") == 1
    assert "Coverage-80%25-yellow.svg" in replaced_twice
    assert "Docs-60%25-red.svg" in replaced_twice

    no_lic = f"# No license{NL}"
    updated_none, found_none = update_readme_badges(no_lic, 75, 80)
    assert found_none is False
    assert updated_none == no_lic

    assert resolve_badge_color(95.0) == "brightgreen"
    assert resolve_badge_color(75.0) == "yellow"
    assert resolve_badge_color(50.0) == "red"

    cmds = build_test_commands(npm_cmd="/usr/bin/npm", pytest_cmd="/usr/bin/pytest")
    assert cmds["web_install"] == ["/usr/bin/npm", "install"]
    assert "--cov-report=lcov" in cmds["backend_pytest"]

    calls = 0

    def flaky_open(url: str) -> str:
        """Fail once before succeeding."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("not ready")
        return url

    assert (
        wait_for_service_ready(
            "http://localhost:8000",
            max_attempts=3,
            urlopen_fn=flaky_open,
            sleep_fn=lambda _: None,
        )
        is True
    )
    assert (
        wait_for_service_ready(
            "http://localhost:8000",
            max_attempts=2,
            urlopen_fn=lambda _: (_ for _ in ()).throw(OSError("down")),
            sleep_fn=lambda _: None,
        )
        is False
    )


class FakeServerProcess:
    """Explicit contract-verified fake implementing `ServerProcessProtocol`."""

    def __init__(self, *, stubborn: bool = False) -> None:
        """
        Initialize fake server process.

        Args:
            stubborn: When True, first `wait(timeout=...)` raises `TimeoutExpired` until `kill()`.

        """
        self.terminated = False
        self.killed = False
        self.waited = 0
        self._stubborn = stubborn

    def terminate(self) -> None:
        """Record termination request."""
        self.terminated = True

    def kill(self) -> None:
        """Record forceful kill request."""
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        """
        Wait for fake process exit, optionally simulating timeout before `kill()`.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            int: Simulated exit code `0`.

        """
        self.waited += 1
        if self._stubborn and not self.killed and timeout is not None:
            raise subprocess.TimeoutExpired(cmd="fake-server", timeout=timeout)
        return 0

    def poll(self) -> int | None:
        """
        Return `0` once terminated/killed or `None` while active.

        Returns:
            int | None: Simulated poll status.

        """
        return 0 if (self.terminated or self.killed) else None


class FakeCommandRunner:
    """Explicit contract-verified fake implementing `CommandRunnerProtocol`."""

    def __init__(
        self,
        *,
        interrogate_stdout: str = "actual: 100.0%",
        on_run: Callable[[Sequence[str]], CommandResult] | None = None,
    ) -> None:
        """
        Initialize fake command runner.

        Args:
            interrogate_stdout: Default stdout returned for `interrogate` invocations.
            on_run: Optional callback overriding `run()` behavior per command vector.

        """
        self.commands: list[list[str]] = []
        self.spawned_servers: list[tuple[list[str], FakeServerProcess]] = []
        self._interrogate_stdout = interrogate_stdout
        self._on_run = on_run

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
        Record synchronous command invocation and return configured `CommandResult`.

        Args:
            cmd: Command argument sequence.
            env: Optional environment mapping.
            cwd: Optional working directory path.
            timeout: Optional timeout in seconds.
            capture_output: Whether stdout capture was requested.
            text: Whether text decoding was requested.

        Returns:
            CommandResult: Simulated command result.

        """
        _ = (env, cwd, timeout, capture_output, text)
        self.commands.append(list(cmd))
        if self._on_run is not None:
            return self._on_run(cmd)
        joined = " ".join(cmd)
        stdout = self._interrogate_stdout if "interrogate" in joined else ""
        return CommandResult(command=tuple(cmd), returncode=0, stdout=stdout)

    def spawn_server(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ServerProcessProtocol:
        """
        Record spawned background server command and return a `FakeServerProcess`.

        Args:
            cmd: Command argument sequence.
            env: Optional environment mapping.
            cwd: Optional working directory path.

        Returns:
            ServerProcessProtocol: Tracked fake server process.

        """
        _ = (env, cwd)
        proc = FakeServerProcess()
        self.spawned_servers.append((list(cmd), proc))
        return proc


def test_real_http_server_and_stubborn_process_cleanup() -> None:
    """Verify real mini-HTTP-server readiness and real child-process cleanup on normal exit and failure."""
    real_runner = SubprocessCommandRunner()
    assert isinstance(real_runner, CommandRunnerProtocol)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    http_script = (
        "import http.server\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "    def log_message(self, fmt, *args): pass\n"
        f"http.server.HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"
    )
    stubborn_script = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )

    http_proc = real_runner.spawn_server([sys.executable, "-c", http_script])
    stubborn_proc = real_runner.spawn_server([sys.executable, "-c", stubborn_script])
    assert isinstance(http_proc, ServerProcessProtocol)

    with managed_server_processes([http_proc, stubborn_proc], wait_timeout=0.15):
        assert (
            wait_for_service_ready(
                f"http://127.0.0.1:{port}/",
                max_attempts=50,
                sleep_seconds=0.05,
            )
            is True
        )

    assert http_proc.poll() is not None
    assert stubborn_proc.poll() is not None

    # Verify child process cleanup when an exception is raised inside the managed block
    fail_proc = real_runner.spawn_server(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    with pytest.raises(RuntimeError, match="boom during e2e"):
        with managed_server_processes([fail_proc], wait_timeout=1.0):
            assert fail_proc.poll() is None
            raise RuntimeError("boom during e2e")
    assert fail_proc.poll() is not None


def test_run_cmd_success_and_failure() -> None:
    """Test run_cmd exit code 0 return and non-zero SystemExit propagation."""
    res_ok = run_cmd([sys.executable, "-c", "print('ok')"], timeout=10.0)
    assert isinstance(res_ok, CommandResult)
    assert res_ok.returncode == 0

    res_no_timeout = run_cmd([sys.executable, "-c", "pass"])
    assert res_no_timeout.returncode == 0

    with pytest.raises(SystemExit) as exc_info:
        run_cmd([sys.executable, "-c", "import sys; sys.exit(42)"])
    assert exc_info.value.code == 42


def test_main_full_success_and_custom_suite_config(tmp_path: Path) -> None:
    """Test main happy path with CommandRunnerProtocol fake, isolated E2E temp dir, and custom URLs."""
    (tmp_path / "ci_test.duckdb").write_text("stale", encoding="utf-8")
    backend_dir = tmp_path / "t1d-analytics-backend"
    (backend_dir / "src" / "t1d_analytics").mkdir(parents=True)
    (backend_dir / "pyproject.toml").write_text(
        f"[tool.interrogate]{NL}", encoding="utf-8"
    )
    mod_p = backend_dir / "src" / "t1d_analytics" / "ok.py"
    mod_p.write_text(
        NL.join(['"""Doc."""', "def f() -> int:", "    return 1", ""]),
        encoding="utf-8",
    )

    (tmp_path / "coverage.lcov").write_text(
        NL.join(
            [
                "SF:file1.py",
                "FN:2,f",
                "FNDA:1,f",
                "BRDA:2,0,0,1",
                "DA:1,2",
                "DA:2,3",
                "end_of_record",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        json.dumps(
            {
                "files": {str(mod_p): {"executed_lines": [3]}},
                "totals": {"percent_covered": 100},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        NL.join(
            [
                "[![License](https://img.shields.io/badge/license.svg)](#)",
                "Some readme content",
                "",
            ]
        ),
        encoding="utf-8",
    )

    polled_urls: list[str] = []

    def tracking_urlopen(url: str) -> str:
        """Record readiness URLs polled by main()."""
        polled_urls.append(url)
        return "ok"

    custom_cfg = TestSuiteConfig(
        skip_readme_update=False,
        coverage_threshold=100.0,
        backend_ready_url="http://127.0.0.1:18000/api/ready",
        frontend_ready_url="http://127.0.0.1:15173",
    )
    fake_runner = FakeCommandRunner(interrogate_stdout="actual: 100.0%")
    assert isinstance(fake_runner, CommandRunnerProtocol)

    with patch(
        "shutil.which",
        side_effect=lambda name, *args, **kwargs: f"/usr/bin/{name}",
    ):
        main(
            workspace_root=tmp_path,
            suite_config=custom_cfg,
            runner=fake_runner,
            urlopen_fn=tracking_urlopen,
            sleep_fn=lambda _: None,
        )
        main(
            skip_readme_update=True,
            workspace_root=tmp_path,
            runner=fake_runner,
            urlopen_fn=tracking_urlopen,
            sleep_fn=lambda _: None,
        )

    assert "http://127.0.0.1:18000/api/ready" in polled_urls
    assert "http://127.0.0.1:15173" in polled_urls
    assert len(fake_runner.commands) >= 8
    assert len(fake_runner.spawned_servers) == 4
    assert all(
        proc.terminated and proc.waited >= 1 for _, proc in fake_runner.spawned_servers
    )
    assert not (tmp_path / "ci_test.duckdb").exists()
    assert not (tmp_path / "t1d-analytics-backend" / "tests" / "test_data").exists()
    assert (tmp_path / "combined-coverage.lcov").exists()
    assert "Coverage-100%25-brightgreen.svg" in (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )


def test_main_failure_branches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main error exits across server readiness, E2E failures, coverage gates, AST audit, and shields."""
    # 1. Backend fails readiness -> both spawned procs terminated and waited
    r1 = FakeCommandRunner()
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=r1,
                urlopen_fn=lambda _: (_ for _ in ()).throw(
                    OSError("Connection refused")
                ),
                sleep_fn=lambda _: None,
            )
    assert exc.value.code == 1
    assert len(r1.spawned_servers) == 2
    assert all(proc.terminated and proc.waited == 1 for _, proc in r1.spawned_servers)

    # 2. Frontend fails readiness -> both spawned procs terminated and waited
    r2 = FakeCommandRunner()
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=r2,
                urlopen_fn=lambda u: (
                    "ok" if "8000" in u else (_ for _ in ()).throw(OSError("down"))
                ),
                sleep_fn=lambda _: None,
            )
    assert exc.value.code == 1
    assert all(proc.terminated and proc.waited == 1 for _, proc in r2.spawned_servers)

    # 3. Playwright fails & NYC fails -> server cleanup still runs
    for fail_kw, expected_code in (("playwright", 2), ("nyc", 5)):

        def _fail_e2e_cmd(
            cmd: Sequence[str],
            target_kw: str = fail_kw,
            code_val: int = expected_code,
        ) -> CommandResult:
            """Return failing exit code for target E2E tool command."""
            return CommandResult(
                command=tuple(cmd),
                returncode=code_val if target_kw in cmd else 0,
            )

        r_e2e = FakeCommandRunner(on_run=_fail_e2e_cmd)
        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit) as exc:
                main(
                    workspace_root=tmp_path,
                    runner=r_e2e,
                    urlopen_fn=lambda _: "ok",
                    sleep_fn=lambda _: None,
                )
        assert exc.value.code == expected_code
        assert all(proc.terminated for _, proc in r_e2e.spawned_servers)

    # 4. Forbidden coverage ignore found in src/
    bad_src = tmp_path / "t1d-analytics-backend" / "src"
    bad_src.mkdir(parents=True)
    bad_file = bad_src / "smell.py"
    bad_file.write_text(f"x = 1  {PRAGMA_NO_COVER}{NL}", encoding="utf-8")
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1
    bad_file.unlink()

    # 5. Missing lcov -> total_found == 0
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1

    # 6. Below threshold (branch uncovered)
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:f.py", "BRDA:1,0,0,-", "DA:1,1", "end_of_record", ""]),
        encoding="utf-8",
    )
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1

    # 7. AST function audit detects uncovered Python function & invalid JSON
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:f.py", "DA:1,1", "end_of_record", ""]),
        encoding="utf-8",
    )
    py_pkg = bad_src / "t1d_analytics"
    py_pkg.mkdir(parents=True)
    uncov_py = py_pkg / "uncov.py"
    uncov_py.write_text(
        NL.join(["def missed_fn() -> int:", "    return 9", ""]),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        json.dumps(
            {
                "files": {str(uncov_py): {"executed_lines": []}},
                "totals": {"percent_covered": 100},
            }
        ),
        encoding="utf-8",
    )
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1

    (tmp_path / "coverage.json").write_text("{corrupt", encoding="utf-8")
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1
    (tmp_path / "coverage.json").unlink()
    uncov_py.unlink()
    py_pkg.rmdir()

    # 8. Missing coverage.json during shields stage
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1

    # 9. Interrogate failure during shields stage (with pyproject.toml present and no target dirs)
    (tmp_path / "t1d-analytics-backend" / "pyproject.toml").write_text(
        "", encoding="utf-8"
    )
    (tmp_path / "coverage.json").write_text(
        '{"totals": {"percent_covered": 85}}', encoding="utf-8"
    )

    def fail_interrogate(cmd: Sequence[str]) -> CommandResult:
        """Raise RuntimeError when interrogate runs."""
        if "interrogate" in cmd:
            raise RuntimeError("Interrogate failed")
        return CommandResult(command=tuple(cmd), returncode=0)

    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc:
            main(
                workspace_root=tmp_path,
                runner=FakeCommandRunner(on_run=fail_interrogate),
                urlopen_fn=lambda _: "ok",
                sleep_fn=lambda _: None,
            )
        assert exc.value.code == 1

    # 10. README without license shield + non-matching interrogate output + FakeServerProcess stubborn path
    stubborn_fake = FakeServerProcess(stubborn=True)
    assert stubborn_fake.poll() is None
    with managed_server_processes([stubborn_fake], wait_timeout=0.01):
        pass
    assert stubborn_fake.killed is True
    assert stubborn_fake.poll() == 0

    readme_path = tmp_path / "README.md"
    readme_path.write_text(f"# Readme{NL}No shield.{NL}", encoding="utf-8")
    with patch("shutil.which", return_value=None):
        main(
            workspace_root=tmp_path,
            runner=FakeCommandRunner(interrogate_stdout="unknown format"),
            urlopen_fn=lambda _: "ok",
            sleep_fn=lambda _: None,
        )
    out = capsys.readouterr().out
    assert "License shield not found in README.md" in out
    assert extract_pytest_coverage_percent(tmp_path / "coverage.json") == 85
    assert extract_interrogate_coverage_percent("unknown") == 0


def test_module_dunder_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test execution when scripts.run_tests is invoked as __main__ via runpy in a temporary cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:file1.py", "DA:1,1", "LF:1", "LH:1", "end_of_record", ""]),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        '{"totals": {"percent_covered": 100}}', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        NL.join(["[![License](https://img.shields.io/badge/license.svg)](#)", ""]),
        encoding="utf-8",
    )

    def _dunder_run(
        cmd: Sequence[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Return 100% interrogate stdout for __main__ execution test."""
        _ = (env, cwd, timeout, capture_output, text, check)
        return subprocess.CompletedProcess(
            args=list(cmd), returncode=0, stdout="TOTAL: 100.0%"
        )

    fake_runner = FakeCommandRunner(interrogate_stdout="TOTAL: 100.0%")
    sys.modules.pop("scripts.run_tests", None)
    with (
        patch("sys.argv", ["run_tests.py"]),
        patch("subprocess.Popen", side_effect=fake_runner.spawn_server),
        patch("urllib.request.urlopen", return_value="ok"),
        patch("subprocess.run", side_effect=_dunder_run),
    ):
        runpy.run_module("scripts.run_tests", run_name="__main__")
    assert "Pre-commit checks passed successfully!" in capsys.readouterr().out
    assert "Coverage-100%25-brightgreen.svg" in (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )
