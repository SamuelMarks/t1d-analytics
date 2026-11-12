"""Structural parity and behavioral tests for Makefile and make.bat automation targets."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAKEFILE_PATH = REPO_ROOT / "Makefile"
MAKE_BAT_PATH = REPO_ROOT / "make.bat"


def _extract_makefile_targets_and_phony(
    makefile_text: str,
) -> tuple[set[str], set[str]]:
    """
    Parse `.PHONY` declarations and defined target names from `Makefile` text.

    Args:
        makefile_text: Raw contents of `Makefile`.

    Returns:
        tuple[set[str], set[str]]: `(phony_targets, defined_targets)`.

    """
    phony: set[str] = set()
    defined: set[str] = set()
    target_re = re.compile(r"^([A-Za-z0-9_.-]+)\s*:(?!=)")
    for raw_line in makefile_text.splitlines():
        if raw_line.startswith(".PHONY:"):
            phony.update(raw_line.split(":", 1)[1].split())
            continue
        if raw_line.startswith(("	", "#", " ")) or not raw_line.strip():
            continue
        match = target_re.match(raw_line)
        if match:
            name = match.group(1)
            if not name.startswith("."):
                defined.add(name)
    return phony, defined


def _extract_make_bat_dispatched_and_labels(
    make_bat_text: str,
) -> tuple[set[str], set[str]]:
    """
    Parse `if "%~1"=="<target>" goto <label>` dispatch entries and `:label` definitions from `make.bat`.

    Args:
        make_bat_text: Raw contents of `make.bat`.

    Returns:
        tuple[set[str], set[str]]: `(dispatched_targets, defined_labels)`.

    """
    dispatch_re = re.compile(
        r'^\s*if\s+"%~1"=="([A-Za-z0-9_.-]+)"\s+goto\s+([A-Za-z0-9_.-]+)', re.IGNORECASE
    )
    label_re = re.compile(r"^:([A-Za-z0-9_.-]+)\s*$")
    dispatched: set[str] = set()
    labels: set[str] = set()
    for raw_line in make_bat_text.splitlines():
        d_match = dispatch_re.match(raw_line)
        if d_match:
            dispatched.add(d_match.group(1))
        l_match = label_re.match(raw_line.strip())
        if l_match:
            labels.add(l_match.group(1))
    return dispatched, labels


def test_makefile_phony_completeness_and_make_bat_target_parity() -> None:
    """Verify 1-to-1 target parity across Makefile .PHONY, Makefile rules, make.bat dispatchers, and labels."""
    makefile_text = MAKEFILE_PATH.read_text(encoding="utf-8")
    make_bat_text = MAKE_BAT_PATH.read_text(encoding="utf-8")

    phony_targets, makefile_targets = _extract_makefile_targets_and_phony(makefile_text)
    bat_dispatched, bat_labels = _extract_make_bat_dispatched_and_labels(make_bat_text)

    assert len(makefile_targets) >= 40
    assert phony_targets == makefile_targets
    assert bat_dispatched == makefile_targets
    assert bat_dispatched.issubset(bat_labels)


def test_all_makefile_targets_expand_cleanly_in_dry_run() -> None:
    """Verify every Makefile target expands and parses with exit code 0 under `make -n`."""
    _, makefile_targets = _extract_makefile_targets_and_phony(
        MAKEFILE_PATH.read_text(encoding="utf-8")
    )
    env = os.environ.copy()
    env.pop("T1D_DB_PATH", None)
    env.pop("DATA_DIR", None)

    for target in sorted(makefile_targets):
        res = subprocess.run(
            ["make", "-n", "-f", str(MAKEFILE_PATH), target],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert res.returncode == 0, f"Target '{target}' failed dry-run: {res.stderr}"


def test_makefile_behavioral_help_and_init_db_in_tmp_path(tmp_path: Path) -> None:
    """Execute real `make help` and `make init_db` targets against an isolated `tmp_path` directory."""
    _, makefile_targets = _extract_makefile_targets_and_phony(
        MAKEFILE_PATH.read_text(encoding="utf-8")
    )
    help_res = subprocess.run(
        ["make", "-f", str(MAKEFILE_PATH), "help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Available commands:" in help_res.stdout
    for target in sorted(makefile_targets):
        assert target in help_res.stdout

    isolated_data_dir = tmp_path / "make_data"
    isolated_db_path = isolated_data_dir / "isolated.duckdb"
    env = os.environ.copy()
    env["DATA_DIR"] = str(isolated_data_dir)
    env["T1D_DB_PATH"] = str(isolated_db_path)

    init_res = subprocess.run(
        [
            "make",
            "-f",
            str(MAKEFILE_PATH),
            f"DATA_DIR={isolated_data_dir}",
            f"T1D_DB_PATH={isolated_db_path}",
            "init_db",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert init_res.returncode == 0
    assert isolated_db_path.is_file()
    conn = duckdb.connect(str(isolated_db_path), read_only=True)
    try:
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        conn.close()
