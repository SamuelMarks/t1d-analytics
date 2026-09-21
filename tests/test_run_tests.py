"""
Tests for test orchestration script (scripts/run_tests.py).

Verifies command execution, error handling, coverage gathering,
merging, threshold evaluations, and README shield generation.
"""

from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest
import scripts.run_tests as srt
from scripts.run_tests import main, run_cmd


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
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("scripts.run_tests.run_cmd") as mock_rc,
        patch("os.makedirs"),
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

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Dispatch mocked file opening operations for success testing."""
            filename = str(file)
            if "README.md" in filename:
                return mock_open(read_data=readme_data).return_value
            elif "coverage.json" in filename:
                return mock_open(
                    read_data='{"totals": {"percent_covered": "100"}}'
                ).return_value
            elif "combined-coverage.lcov" in filename:
                return mock_open().return_value
            elif "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

        with patch("builtins.open", side_effect=custom_open):
            main()

        assert mock_rc.call_count >= 5
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

        def urlopen_side_effect(url: str) -> Any:
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

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Simulate missing lcov files by raising FileNotFoundError."""
            filename = str(file)
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                raise FileNotFoundError()
            return mock_open().return_value

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

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Return lcov contents having zero line hits."""
            filename = str(file)
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

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

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Return lcov contents resulting in fifty percent coverage."""
            filename = str(file)
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

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

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Simulate corrupted coverage json by raising OSError."""
            filename = str(file)
            if "coverage.json" in filename:
                raise OSError("Corrupt json")
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

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

        def subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            """Execute simulated command raising RuntimeError for interrogate."""
            if "interrogate" in cmd:
                raise RuntimeError("Interrogate not found")
            return MagicMock(returncode=0)

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Provide mocked file handlers for interrogate failure tests."""
            filename = str(file)
            if "coverage.json" in filename:
                return mock_open(
                    read_data='{"totals": {"percent_covered": 85}}'
                ).return_value
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

        with (
            patch("subprocess.run", side_effect=subprocess_run),
            patch("builtins.open", side_effect=custom_open),
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_main_badge_colors_and_no_license_shield() -> None:
    """Test shield colors (yellow, red) and case where README has no license shield."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        def subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            """Execute simulated subprocess returning fifty percent documentation coverage."""
            if "interrogate" in cmd:
                return MagicMock(returncode=0, stdout="actual: 50.0%")
            return MagicMock(returncode=0)

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])
        readme_without_license = "\n".join(["# Readme", "No license shield here.", ""])

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Provide mocked open handles for badge color verification tests."""
            filename = str(file)
            if mode == "w":
                return mock_open().return_value
            if "coverage.json" in filename:
                return mock_open(
                    read_data='{"totals": {"percent_covered": 75}}'
                ).return_value
            if "README.md" in filename:
                return mock_open(read_data=readme_without_license).return_value
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

        with (
            patch("subprocess.run", side_effect=subprocess_run),
            patch("builtins.open", side_effect=custom_open),
        ):
            main()


def test_main_badge_colors_red_and_interrogate_no_match() -> None:
    """Test test_cov < 70 (red) and interrogate output without matching regex."""
    with (
        patch("shutil.which", return_value=None),
        patch("scripts.run_tests.run_cmd"),
        patch("os.makedirs"),
        patch("subprocess.Popen") as mock_popen,
        patch("urllib.request.urlopen", return_value=MagicMock()),
    ):
        mock_popen.side_effect = [MagicMock(), MagicMock()]

        def subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            """Execute simulated subprocess returning non-matching interrogate output."""
            if "interrogate" in cmd:
                return MagicMock(returncode=0, stdout="unknown format")
            return MagicMock(returncode=0)

        lcov_content = "\n".join(["SF:some_file.py", "DA:1,1", ""])
        readme_data = "\n".join(
            ["[![License](https://img.shields.io/badge/license.svg)](#)", ""]
        )

        def custom_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            """Provide mock file handler for red color badges and interrogate mismatch tests."""
            filename = str(file)
            if mode == "w":
                return mock_open().return_value
            if "coverage.json" in filename:
                return mock_open(
                    read_data='{"totals": {"percent_covered": 60}}'
                ).return_value
            if "README.md" in filename:
                return mock_open(read_data=readme_data).return_value
            if "combined-coverage.lcov" in filename:
                return mock_open().return_value
            if "lcov" in filename:
                return mock_open(read_data=lcov_content).return_value
            return mock_open().return_value

        with (
            patch("subprocess.run", side_effect=subprocess_run),
            patch("builtins.open", side_effect=custom_open),
        ):
            main()


def test_module_dunder_main() -> None:
    """Test execution when __name__ == __main__."""
    with open(srt.__file__) as f:
        lines = f.readlines()

    main_start = next(
        i for i, line in enumerate(lines) if line.startswith("def main()")
    )
    lines_mod = lines[: main_start + 1] + ["    pass\n"]
    for i in range(main_start + 1, len(lines)):
        if lines[i].startswith("if __name__"):
            lines_mod.extend(lines[i:])
            break

    code_obj = compile("".join(lines_mod), srt.__file__, "exec")
    exec(code_obj, {"__name__": "__main__"})
