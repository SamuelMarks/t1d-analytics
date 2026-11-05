"""
Tests for test orchestration script (scripts/run_tests.py).

Verifies command execution, error handling, coverage gathering,
merging, threshold evaluations, and README shield generation against real
temporary workspace directories (`tmp_path`).
"""

import collections
import runpy
import sys
from pathlib import Path
from typing import DefaultDict, Sequence
from unittest.mock import MagicMock, patch

import pytest
from scripts.run_tests import (
    CommandResult,
    CoverageMetrics,
    TestSuiteConfig,
    build_test_commands,
    compute_combined_coverage,
    extract_interrogate_coverage_percent,
    extract_pytest_coverage_percent,
    main,
    parse_lcov_files,
    parse_lcov_totals,
    resolve_badge_color,
    run_cmd,
    update_readme_badges,
    update_readme_shields,
    wait_for_service_ready,
)

from t1d_analytics.sql_utils import SqlValue

NL = chr(10)


def test_pure_orchestration_helpers() -> None:
    """Test compute_combined_coverage, update_readme_badges, and build_test_commands directly."""
    cov_map: DefaultDict[str, DefaultDict[int, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    cov_map["src/a.py"][1] = 2
    cov_map["src/a.py"][2] = 0
    total_found, total_hit, text = compute_combined_coverage(cov_map)
    assert total_found == 2
    assert total_hit == 1
    assert text == NL.join(
        ["SF:src/a.py", "DA:1,2", "DA:2,0", "LF:2", "LH:1", "end_of_record", ""]
    )

    raw_readme = NL.join(
        ["[![License](https://example.com/lic.svg)](#)", "# Title", ""]
    )
    updated_green, found = update_readme_badges(raw_readme, 100, 100)
    assert found is True
    assert "Coverage-100%25-brightgreen.svg" in updated_green
    assert "Docs-100%25-brightgreen.svg" in updated_green

    no_lic = f"# No license{NL}"
    updated_yellow, found_y = update_readme_badges(no_lic, 75, 80)
    assert found_y is False
    assert updated_yellow == no_lic

    cmds = build_test_commands(npm_cmd="/usr/bin/npm", pytest_cmd="/usr/bin/pytest")
    assert cmds["web_install"] == ["/usr/bin/npm", "install"]
    assert cmds["web_build"] == ["/usr/bin/npm", "run", "build"]
    assert "--cov-report=json" in cmds["backend_pytest"]

    lcov_raw = NL.join(
        ["SF:src/x.py", "DA:1,2", "DA:2,0", "DA:1,5", "end_of_record", "unrelated", ""]
    )
    assert parse_lcov_totals(lcov_raw) == (2, 1)
    assert resolve_badge_color(95.0) == "brightgreen"
    assert resolve_badge_color(75.0) == "yellow"
    assert resolve_badge_color(50.0) == "red"
    assert "Coverage-95%25-brightgreen" in update_readme_shields(
        NL.join(["# Title", "[![License](http://lic)]", ""]), 95.0, 100.0
    )

    empty_metrics = CoverageMetrics(total_found=0, total_hit=0, formatted_lcov="")
    assert empty_metrics.percent == 0.0
    full_metrics = CoverageMetrics(
        total_found=10, total_hit=10, formatted_lcov=f"SF:x{NL}"
    )
    assert full_metrics.percent == 100.0
    cfg = TestSuiteConfig(skip_readme_update=True)
    assert cfg.skip_readme_update is True
    assert cfg.coverage_threshold == 100.0


def test_real_filesystem_lcov_merge_and_readme_update(tmp_path: Path) -> None:
    """Test LCOV file aggregation and README shield updates against real files in tmp_path."""
    lcov_a = tmp_path / "backend.lcov"
    lcov_b = tmp_path / "frontend.lcov"
    lcov_a.write_text(
        NL.join(["SF:src/mod.py", "DA:1,1", "DA:2,0", "end_of_record", ""]),
        encoding="utf-8",
    )
    lcov_b.write_text(
        NL.join(["SF:src/mod.py", "DA:1,2", "DA:2,1", "end_of_record", ""]),
        encoding="utf-8",
    )

    cov_map = parse_lcov_files([lcov_a, lcov_b, tmp_path / "missing.lcov"])
    total_found, total_hit, merged_lcov = compute_combined_coverage(cov_map)
    assert total_found == 2
    assert total_hit == 2
    merged_path = tmp_path / "combined-coverage.lcov"
    merged_path.write_text(merged_lcov, encoding="utf-8")
    assert NL.join(["DA:1,2", "DA:2,1", "LF:2", "LH:2"]) in merged_path.read_text(
        encoding="utf-8"
    )

    cov_json = tmp_path / "coverage.json"
    cov_json.write_text('{"totals": {"percent_covered": 98.7}}', encoding="utf-8")
    assert extract_pytest_coverage_percent(cov_json) == 98
    assert extract_interrogate_coverage_percent("actual: 100.0% PASSED") == 100
    assert extract_interrogate_coverage_percent("no match") == 0

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

    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        NL.join(
            [
                "# T1D",
                "[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pct = int((total_hit / total_found) * 100)
    updated_readme, found = update_readme_badges(
        readme_file.read_text(encoding="utf-8"), pct, 100
    )
    assert found is True
    readme_file.write_text(updated_readme, encoding="utf-8")
    disk_readme = readme_file.read_text(encoding="utf-8")
    assert "Coverage-100%25-brightgreen.svg" in disk_readme
    assert "Docs-100%25-brightgreen.svg" in disk_readme


def test_run_cmd_success() -> None:
    """Test run_cmd when the command returns exit code 0."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        res = run_cmd(["echo", "hello"], env={"VAR": "1"}, cwd=".")
        assert isinstance(res, CommandResult)
        assert res.returncode == 0
        mock_run.assert_called_once_with(["echo", "hello"], env={"VAR": "1"}, cwd=".")

        res_timeout = run_cmd(
            ["echo", "timed"], env={"VAR": "1"}, cwd=".", timeout=30.0
        )
        assert res_timeout.returncode == 0
        mock_run.assert_called_with(
            ["echo", "timed"], env={"VAR": "1"}, cwd=".", timeout=30.0
        )


def test_run_cmd_failure() -> None:
    """Test run_cmd when the command exits with a non-zero code."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=42)
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["false"])
        assert exc_info.value.code == 42


def test_main_full_success(tmp_path: Path) -> None:
    """Test main execution through the happy path against real temporary files."""
    (tmp_path / "ci_test.duckdb").write_text("stale", encoding="utf-8")
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["DA:99,1", "SF:file1.py", "DA:1,2", "DA:2,3", "end_of_record", ""]),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        '{"totals": {"percent_covered": 100}}', encoding="utf-8"
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

    with (
        patch(
            "shutil.which",
            side_effect=lambda name, *args, **kwargs: f"/usr/bin/{name}",
        ),
        patch("scripts.run_tests.run_cmd") as mock_rc,
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run") as mock_run,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_proc_backend = MagicMock()
        mock_proc_frontend = MagicMock()
        mock_popen.side_effect = [
            mock_proc_backend,
            mock_proc_frontend,
            mock_proc_backend,
            mock_proc_frontend,
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout="actual: 100.0%")

        main(workspace_root=tmp_path)
        main(skip_readme_update=True, workspace_root=tmp_path)

        assert mock_rc.call_count >= 5
        assert not (tmp_path / "ci_test.duckdb").exists()
        assert (
            tmp_path / "t1d-analytics-backend" / "tests" / "test_data" / "patients.csv"
        ).exists()
        assert (tmp_path / "combined-coverage.lcov").exists()
        assert "Coverage-100%25-brightgreen.svg" in (tmp_path / "README.md").read_text(
            encoding="utf-8"
        )
        assert mock_proc_backend.terminate.called
        assert mock_proc_frontend.terminate.called


def test_main_backend_fail_to_start(tmp_path: Path) -> None:
    """Test main exits if backend server fails healthcheck."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", side_effect=OSError("Connection refused")),
        patch("time.sleep"),
    ):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_popen.side_effect = [mock_backend, mock_frontend]

        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 1
        assert mock_backend.terminate.called
        assert mock_frontend.terminate.called


def test_main_frontend_fail_to_start(tmp_path: Path) -> None:
    """Test main exits if frontend server fails healthcheck."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("time.sleep"),
    ):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_popen.side_effect = [mock_backend, mock_frontend]

        def urlopen_side_effect(url: str) -> MagicMock:
            """Handle URL open requests returning success for backend and failure for frontend."""
            if "8000" in url:
                return MagicMock()
            raise OSError("Frontend down")

        mock_urlopen.side_effect = urlopen_side_effect

        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 1
        assert mock_backend.terminate.called
        assert mock_frontend.terminate.called


def test_main_playwright_fail(tmp_path: Path) -> None:
    """Test main exits if playwright test returns non-zero."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run") as mock_run,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_popen.side_effect = [mock_backend, mock_frontend]
        mock_run.side_effect = [
            MagicMock(returncode=2),
            MagicMock(returncode=0),
        ]

        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 2
        assert mock_backend.terminate.called
        assert mock_frontend.terminate.called


def test_main_nyc_report_fail(tmp_path: Path) -> None:
    """Test main exits if nyc report returns non-zero."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run") as mock_run,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_popen.side_effect = [mock_backend, mock_frontend]
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=5),
        ]

        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 5


def test_main_lcov_not_found_and_total_found_zero(tmp_path: Path) -> None:
    """Test main handles missing lcov files on disk and exits when total_found == 0."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]
        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 1


def test_main_lcov_lines_uncovered_and_below_threshold(tmp_path: Path) -> None:
    """Test main exits when combined coverage on disk has uncovered lines (< 100%)."""
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:some_file.py", "DA:1,1", "DA:2,0", "end_of_record", ""]),
        encoding="utf-8",
    )
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]
        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 1


def test_main_coverage_json_failure(tmp_path: Path) -> None:
    """Test main exits when coverage.json is missing or corrupted on disk."""
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:some_file.py", "DA:1,1", "end_of_record", ""]),
        encoding="utf-8",
    )
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]
        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 1


def test_main_interrogate_failure(tmp_path: Path) -> None:
    """Test main exits when running interrogate fails."""
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:some_file.py", "DA:1,1", "end_of_record", ""]),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        '{"totals": {"percent_covered": 85}}', encoding="utf-8"
    )

    def subprocess_run(
        cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
    ) -> MagicMock:
        """Execute simulated command raising RuntimeError for interrogate."""
        _ = (args, kwargs)
        if "interrogate" in cmd:
            raise RuntimeError("Interrogate not found")
        return MagicMock(returncode=0)

    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
        patch("subprocess.run", side_effect=subprocess_run),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]
        with pytest.raises(SystemExit) as exc:
            main(workspace_root=tmp_path)
        assert exc.value.code == 1


def test_main_badge_colors_and_no_license_shield(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test shield colors (yellow, red) and case where README.md on disk has no license shield."""
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:some_file.py", "DA:1,1", "end_of_record", ""]),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        '{"totals": {"percent_covered": 75}}', encoding="utf-8"
    )
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        NL.join(["# Readme", "No license shield here.", ""]), encoding="utf-8"
    )

    def subprocess_run(
        cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
    ) -> MagicMock:
        """Execute simulated subprocess returning fifty percent documentation coverage."""
        _ = (args, kwargs)
        if "interrogate" in cmd:
            return MagicMock(returncode=0, stdout="actual: 50.0%")
        return MagicMock(returncode=0)

    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
        patch("subprocess.run", side_effect=subprocess_run),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]
        main(workspace_root=tmp_path)

    out = capsys.readouterr().out
    assert "License shield not found in README.md" in out
    assert "Pre-commit checks passed successfully!" in out
    assert "# Readme" in readme_path.read_text(encoding="utf-8")


def test_main_badge_colors_red_and_interrogate_no_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test test_cov < 70 (red) and interrogate output without matching regex on real disk README.md."""
    (tmp_path / "coverage.lcov").write_text(
        NL.join(["SF:some_file.py", "DA:1,1", "end_of_record", ""]),
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        '{"totals": {"percent_covered": 60}}', encoding="utf-8"
    )
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        NL.join(["[![License](https://img.shields.io/badge/license.svg)](#)", ""]),
        encoding="utf-8",
    )

    def subprocess_run(
        cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
    ) -> MagicMock:
        """Execute simulated subprocess returning non-matching interrogate output."""
        _ = (args, kwargs)
        if "interrogate" in cmd:
            return MagicMock(returncode=0, stdout="unknown format")
        return MagicMock(returncode=0)

    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
        patch("subprocess.run", side_effect=subprocess_run),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]
        main(workspace_root=tmp_path)

    out = capsys.readouterr().out
    assert "Pre-commit checks passed successfully!" in out
    written = readme_path.read_text(encoding="utf-8")
    assert "Coverage-60%25-red.svg" in written
    assert "Docs-0%25-red.svg" in written


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

    def subprocess_run(
        cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
    ) -> MagicMock:
        """Mock subprocess.run returning 100% coverage stdout."""
        _ = (cmd, args, kwargs)
        res = MagicMock()
        res.returncode = 0
        res.stdout = "TOTAL: 100.0%"
        return res

    sys.modules.pop("scripts.run_tests", None)
    with (
        patch("sys.argv", ["run_tests.py"]),
        patch("subprocess.Popen", return_value=MagicMock()),
        patch("urllib.request.urlopen", return_value=MagicMock()),
        patch("subprocess.run", side_effect=subprocess_run),
    ):
        runpy.run_module("scripts.run_tests", run_name="__main__")
    assert "Pre-commit checks passed successfully!" in capsys.readouterr().out
    assert "Coverage-100%25-brightgreen.svg" in (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )
