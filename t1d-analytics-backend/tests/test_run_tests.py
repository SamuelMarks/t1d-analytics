"""
Tests for test orchestration script (scripts/run_tests.py).

Verifies command execution, error handling, coverage gathering,
merging, threshold evaluations, and README shield generation.
"""

import collections
import io
from pathlib import Path
from typing import DefaultDict, Sequence, Union
from unittest.mock import MagicMock, mock_open, patch

import pytest
from scripts.run_tests import (
    build_test_commands,
    compute_combined_coverage,
    extract_interrogate_coverage_percent,
    extract_pytest_coverage_percent,
    main,
    parse_lcov_files,
    run_cmd,
    update_readme_badges,
    wait_for_service_ready,
)

from t1d_analytics.sql_utils import SqlValue

FileTarget = Union[str, Path]
OpenReturn = Union[MagicMock, io.StringIO]


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
    assert text == "SF:src/a.py\nDA:1,2\nDA:2,0\nLF:2\nLH:1\nend_of_record\n"

    raw_readme = "[![License](https://example.com/lic.svg)](#)\n# Title\n"
    updated_green, found = update_readme_badges(raw_readme, 100, 100)
    assert found is True
    assert "Coverage-100%25-brightgreen.svg" in updated_green
    assert "Docs-100%25-brightgreen.svg" in updated_green

    updated_yellow, found_y = update_readme_badges("# No license\n", 75, 80)
    assert found_y is False
    assert updated_yellow == "# No license\n"

    cmds = build_test_commands(npm_cmd="/usr/bin/npm", pytest_cmd="/usr/bin/pytest")
    assert cmds["web_install"] == ["/usr/bin/npm", "install"]
    assert cmds["web_build"] == ["/usr/bin/npm", "run", "build"]
    assert "--cov-report=json" in cmds["backend_pytest"]


def test_real_filesystem_lcov_merge_and_readme_update(tmp_path: Path) -> None:
    """Test LCOV file aggregation and README shield updates against real files in tmp_path."""
    lcov_a = tmp_path / "backend.lcov"
    lcov_b = tmp_path / "frontend.lcov"
    lcov_a.write_text(
        "SF:src/mod.py\nDA:1,1\nDA:2,0\nend_of_record\n", encoding="utf-8"
    )
    lcov_b.write_text(
        "SF:src/mod.py\nDA:1,2\nDA:2,1\nend_of_record\n", encoding="utf-8"
    )

    cov_map = parse_lcov_files([lcov_a, lcov_b, tmp_path / "missing.lcov"])
    total_found, total_hit, merged_lcov = compute_combined_coverage(cov_map)
    assert total_found == 2
    assert total_hit == 2
    merged_path = tmp_path / "combined-coverage.lcov"
    merged_path.write_text(merged_lcov, encoding="utf-8")
    assert "DA:1,2\nDA:2,1\nLF:2\nLH:2" in merged_path.read_text(encoding="utf-8")

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
        "# T1D\n[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)\n",
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
        run_cmd(["echo", "hello"], env={"VAR": "1"}, cwd=".")
        mock_run.assert_called_once_with(["echo", "hello"], env={"VAR": "1"}, cwd=".")


def test_run_cmd_failure() -> None:
    """Test run_cmd when the command exits with a non-zero code."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=42)
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["false"])
        assert exc_info.value.code == 42


def test_main_full_success() -> None:
    """Test main execution through the happy path."""
    with (
        patch(
            "shutil.which",
            side_effect=lambda name, *args, **kwargs: f"/usr/bin/{name}",
        ),
        patch("scripts.run_tests.run_cmd") as mock_rc,
        patch("os.makedirs"),
        patch("os.path.exists", side_effect=lambda p: p == "ci_test.duckdb"),
        patch("os.remove") as mock_remove,
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run") as mock_run,
        patch("urllib.request.urlopen") as mock_urlopen,
    ):
        mock_proc_backend = MagicMock()
        mock_proc_frontend = MagicMock()
        mock_popen.side_effect = [mock_proc_backend, mock_proc_frontend]

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0),
            MagicMock(stdout="actual: 100.0%"),
        ]
        mock_urlopen.return_value = MagicMock()

        lcov_content = "\n".join(["DA:99,1", "SF:file1.py", "DA:1,2", "DA:2,3", ""])
        readme_data = "\n".join(
            [
                "[![License](https://img.shields.io/badge/license.svg)](#)",
                "Some readme content",
                "",
            ]
        )

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Dispatch mocked file opening operations for success testing."""
            filename = str(file)
            if "README.md" in filename:
                return io.StringIO(readme_data)
            elif "coverage.json" in filename:
                return io.StringIO('{"totals": {"percent_covered": "100"}}')
            elif "combined-coverage.lcov" in filename:
                return io.StringIO()
            elif "lcov" in filename:
                return io.StringIO(lcov_content)
            return io.StringIO()

        with patch("builtins.open", side_effect=custom_open):
            main()

        assert mock_rc.call_count >= 5
        mock_remove.assert_called_once_with("ci_test.duckdb")
        assert mock_proc_backend.terminate.called
        assert mock_proc_frontend.terminate.called


def test_main_backend_fail_to_start() -> None:
    """Test main exits if backend server fails healthcheck."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("builtins.open", mock_open()),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", side_effect=Exception("Connection refused")),
        patch("time.sleep"),
    ):
        mock_backend = MagicMock()
        mock_frontend = MagicMock()
        mock_popen.side_effect = [mock_backend, mock_frontend]

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert mock_backend.terminate.called
        assert mock_frontend.terminate.called


def test_main_frontend_fail_to_start() -> None:
    """Test main exits if frontend server fails healthcheck."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("builtins.open", mock_open()),
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
            raise Exception("Frontend down")

        mock_urlopen.side_effect = urlopen_side_effect

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert mock_backend.terminate.called
        assert mock_frontend.terminate.called


def test_main_playwright_fail() -> None:
    """Test main exits if playwright test returns non-zero."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("builtins.open", mock_open()),
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
            main()
        assert exc.value.code == 2
        assert mock_backend.terminate.called
        assert mock_frontend.terminate.called


def test_main_nyc_report_fail() -> None:
    """Test main exits if nyc report returns non-zero."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("builtins.open", mock_open()),
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
            main()
        assert exc.value.code == 5


def test_main_lcov_not_found_and_total_found_zero() -> None:
    """Test main handles FileNotFoundError when reading lcov files and exits when total_found == 0."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Simulate missing lcov files by raising FileNotFoundError."""
            filename = str(file)
            if "combined-coverage.lcov" in filename:
                return io.StringIO()
            if "lcov" in filename:
                raise FileNotFoundError()
            return io.StringIO()

        with patch("builtins.open", side_effect=custom_open):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_main_lcov_lines_uncovered() -> None:
    """Test parsing lcov file where line has zero hits to cover line with hits == 0."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,0", "DA:2,0", ""])

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Return lcov contents having zero line hits."""
            filename = str(file)
            if "combined-coverage.lcov" in filename:
                return io.StringIO()
            if "lcov" in filename:
                return io.StringIO(lcov_content)
            return io.StringIO()

        with patch("builtins.open", side_effect=custom_open):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_main_coverage_below_threshold() -> None:
    """Test main exits when combined coverage is below 100%."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", "DA:2,0", ""])

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Return lcov contents resulting in fifty percent coverage."""
            filename = str(file)
            if "combined-coverage.lcov" in filename:
                return io.StringIO()
            if "lcov" in filename:
                return io.StringIO(lcov_content)
            return io.StringIO()

        with patch("builtins.open", side_effect=custom_open):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_main_coverage_json_failure() -> None:
    """Test main exits when reading coverage.json fails."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Simulate corrupted coverage json by raising OSError."""
            filename = str(file)
            if "coverage.json" in filename:
                raise OSError("Corrupt json")
            if "combined-coverage.lcov" in filename:
                return io.StringIO()
            if "lcov" in filename:
                return io.StringIO(lcov_content)
            return io.StringIO()

        with patch("builtins.open", side_effect=custom_open):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_main_interrogate_failure() -> None:
    """Test main exits when running interrogate fails."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        def subprocess_run(
            cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
        ) -> MagicMock:
            """Execute simulated command raising RuntimeError for interrogate."""
            if "interrogate" in cmd:
                raise RuntimeError("Interrogate not found")
            return MagicMock(returncode=0)

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Provide mocked file handlers for interrogate failure tests."""
            filename = str(file)
            if "coverage.json" in filename:
                return io.StringIO('{"totals": {"percent_covered": 85}}')
            if "combined-coverage.lcov" in filename:
                return io.StringIO()
            if "lcov" in filename:
                return io.StringIO(lcov_content)
            return io.StringIO()

        with (
            patch("subprocess.run", side_effect=subprocess_run),
            patch("builtins.open", side_effect=custom_open),
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_main_badge_colors_and_no_license_shield(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test shield colors (yellow, red) and case where README has no license shield."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        def subprocess_run(
            cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
        ) -> MagicMock:
            """Execute simulated subprocess returning fifty percent documentation coverage."""
            if "interrogate" in cmd:
                return MagicMock(returncode=0, stdout="actual: 50.0%")
            return MagicMock(returncode=0)

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])
        readme_without_license = "\n".join(["# Readme", "No license shield here.", ""])
        written_readme: list[str] = []

        class CapturingStringIO(io.StringIO):
            """StringIO subclass that records written content on close."""

            def close(self) -> None:
                """Store buffer value before closing."""
                written_readme.append(self.getvalue())
                super().close()

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Provide mocked open handles for badge color verification tests."""
            filename = str(file)
            if "coverage.json" in filename:
                return io.StringIO('{"totals": {"percent_covered": 75}}')
            if "README.md" in filename and mode == "r":
                return io.StringIO(readme_without_license)
            if "README.md" in filename and mode == "w":
                return CapturingStringIO()
            if filename.endswith(("lcov", "lcov.info")):
                return io.StringIO(lcov_content)
            return io.StringIO()

        with (
            patch("subprocess.run", side_effect=subprocess_run),
            patch("builtins.open", side_effect=custom_open),
        ):
            main()
        out = capsys.readouterr().out
        assert "License shield not found in README.md" in out
        assert "Pre-commit checks passed successfully!" in out
        assert len(written_readme) == 1
        assert "# Readme" in written_readme[0]


def test_main_badge_colors_red_and_interrogate_no_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test test_cov < 70 (red) and interrogate output without matching regex."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        def subprocess_run(
            cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
        ) -> MagicMock:
            """Execute simulated subprocess returning non-matching interrogate output."""
            if "interrogate" in cmd:
                return MagicMock(returncode=0, stdout="unknown format")
            return MagicMock(returncode=0)

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])
        readme_data = "\n".join(
            ["[![License](https://img.shields.io/badge/license.svg)](#)", ""]
        )
        written_red_readme: list[str] = []

        class CapturingRedStringIO(io.StringIO):
            """StringIO subclass that records written red badge content on close."""

            def close(self) -> None:
                """Store buffer value before closing."""
                written_red_readme.append(self.getvalue())
                super().close()

        def custom_open(
            file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
        ) -> OpenReturn:
            """Provide mock file handler for red color badges and interrogate mismatch tests."""
            filename = str(file)
            if "coverage.json" in filename:
                return io.StringIO('{"totals": {"percent_covered": 60}}')
            if "README.md" in filename and mode == "r":
                return io.StringIO(readme_data)
            if "README.md" in filename and mode == "w":
                return CapturingRedStringIO()
            if filename.endswith(("lcov", "lcov.info")):
                return io.StringIO(lcov_content)
            return io.StringIO()

        with (
            patch("subprocess.run", side_effect=subprocess_run),
            patch("builtins.open", side_effect=custom_open),
        ):
            main()
        out = capsys.readouterr().out
        assert "Pre-commit checks passed successfully!" in out
        assert len(written_red_readme) == 1
        assert "Coverage-60%25-red.svg" in written_red_readme[0]
        assert "Docs-0%25-red.svg" in written_red_readme[0]


def test_module_dunder_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Test execution when scripts.run_tests is invoked as __main__ via runpy."""
    import io
    import runpy
    import sys

    readme_data = (
        "![Test Coverage](https://img.shields.io/badge/coverage-50.0%25-red)\n"
        "![Docstring Coverage](https://img.shields.io/badge/docs-100%25-brightgreen)\n"
    )
    lcov_content = "SF:file1.py\nDA:1,1\nLF:1\nLH:1\nend_of_record\n"

    def subprocess_run(
        cmd: Sequence[str], *args: SqlValue, **kwargs: SqlValue
    ) -> MagicMock:
        """Mock subprocess.run returning 100% coverage stdout."""
        res = MagicMock()
        res.returncode = 0
        res.stdout = "TOTAL: 100.0%"
        return res

    def custom_open(
        file: FileTarget, mode: str = "r", *args: SqlValue, **kwargs: SqlValue
    ) -> OpenReturn:
        """Mock file open returning fixture streams based on filename."""
        filename = str(file)
        if mode == "w":
            return io.StringIO()
        if "coverage.json" in filename:
            return io.StringIO('{"totals": {"percent_covered": 100}}')
        if "README.md" in filename:
            return io.StringIO(readme_data)
        return io.StringIO(lcov_content)

    sys.modules.pop("scripts.run_tests", None)
    with (
        patch("sys.argv", ["run_tests.py"]),
        patch("os.makedirs"),
        patch("subprocess.Popen", return_value=MagicMock()),
        patch("urllib.request.urlopen", return_value=MagicMock()),
        patch("subprocess.run", side_effect=subprocess_run),
        patch("builtins.open", side_effect=custom_open),
    ):
        runpy.run_module("scripts.run_tests", run_name="__main__")
    assert "Pre-commit checks passed successfully!" in capsys.readouterr().out
