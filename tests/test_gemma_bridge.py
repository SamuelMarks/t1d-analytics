"""Tests for gemma_bridge external toolchain bridge module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from t1d_analytics.gemma_bridge import (
    GemmaSqlPipelineEngine,
    check_gemma_sql_installed,
    get_gemma_sql_binary,
    parse_gemma_sql_metrics,
    run_gemma_sql_etl,
    run_gemma_sql_train,
)


def test_check_gemma_sql_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_gemma_sql_installed across environment variables, PATH, and importlib."""
    # 1. Mock env var
    monkeypatch.setenv("T1D_MOCK_GEMMA_SQL", "1")
    assert check_gemma_sql_installed() is True

    # 2. In PATH
    monkeypatch.delenv("T1D_MOCK_GEMMA_SQL", raising=False)
    with patch("shutil.which", return_value="/usr/local/bin/gemma-4-sql"):
        assert check_gemma_sql_installed() is True

    # 3. Python module
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert check_gemma_sql_installed() is True

    # 4. Neither
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=None):
            assert check_gemma_sql_installed() is False


def test_get_gemma_sql_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolving the gemma-4-sql executable binary."""
    # 1. Mock env
    monkeypatch.setenv("T1D_MOCK_GEMMA_SQL", "1")
    assert get_gemma_sql_binary() == "gemma-4-sql"

    # 2. PATH resolution
    monkeypatch.delenv("T1D_MOCK_GEMMA_SQL", raising=False)
    with patch("shutil.which", return_value="/opt/bin/gemma-4-sql"):
        assert get_gemma_sql_binary() == "/opt/bin/gemma-4-sql"

    # 3. Module fallback
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert get_gemma_sql_binary() == "python3 -m gemma_4_sql"

    # 4. Not installed raises RuntimeError
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="gemma-4-sql is not installed"):
                get_gemma_sql_binary()


def test_run_gemma_sql_etl_validations(tmp_path: Path) -> None:
    """Test validation errors for run_gemma_sql_etl."""
    # Invalid stage
    with pytest.raises(ValueError, match="Invalid ETL stage"):
        run_gemma_sql_etl("invalid_stage", tmp_path / "test.duckdb", "sft_data")

    # Missing DuckDB file
    with pytest.raises(FileNotFoundError, match="DuckDB database file not found"):
        run_gemma_sql_etl("sft", tmp_path / "missing.duckdb", "sft_data")


def test_run_gemma_sql_etl_mock_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run_gemma_sql_etl execution in mock environment mode."""
    monkeypatch.setenv("T1D_MOCK_GEMMA_SQL", "1")
    db_file = tmp_path / "data.duckdb"
    db_file.write_text("mock")

    out_dir = tmp_path / "shards"
    res = run_gemma_sql_etl(
        stage="sft",
        duckdb_path=db_file,
        duckdb_table="sft_data",
        output_dir=out_dir,
        extra_args=["--batch-size", "64"],
    )
    assert res.returncode == 0
    assert "[MOCK]" in res.stdout
    assert out_dir.exists()


def test_run_gemma_sql_etl_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run_gemma_sql_etl subprocess execution and error propagation."""
    monkeypatch.delenv("T1D_MOCK_GEMMA_SQL", raising=False)
    db_file = tmp_path / "data.duckdb"
    db_file.write_text("mock")

    # Successful run
    mock_success = subprocess.CompletedProcess(
        args=["gemma-4-sql", "etl"],
        returncode=0,
        stdout="ETL Complete",
        stderr="",
    )
    with patch("shutil.which", return_value="/usr/bin/gemma-4-sql"):
        with patch("subprocess.run", return_value=mock_success):
            res = run_gemma_sql_etl("pretrain", db_file, "pretrain_data")
            assert res.stdout == "ETL Complete"

    # Failed run
    mock_fail = subprocess.CompletedProcess(
        args=["gemma-4-sql", "etl"],
        returncode=1,
        stdout="",
        stderr="Grain ETL out of memory",
    )
    with patch("shutil.which", return_value="/usr/bin/gemma-4-sql"):
        with patch("subprocess.run", return_value=mock_fail):
            with pytest.raises(RuntimeError, match="Grain ETL out of memory"):
                run_gemma_sql_etl("pretrain", db_file, "pretrain_data")


def test_run_gemma_sql_train_validations(tmp_path: Path) -> None:
    """Test validation errors for run_gemma_sql_train."""
    # Invalid stage
    with pytest.raises(ValueError, match="Invalid training stage"):
        run_gemma_sql_train("bad_stage", tmp_path / "cfg.yml")

    # Missing config file
    with pytest.raises(
        FileNotFoundError, match="Training configuration file not found"
    ):
        run_gemma_sql_train("sft", tmp_path / "nonexistent.yml")


def test_run_gemma_sql_train_mock_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run_gemma_sql_train execution with mock environment."""
    monkeypatch.setenv("T1D_MOCK_GEMMA_SQL", "1")
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("mock: 1")

    res = run_gemma_sql_train("posttrain", cfg_file, extra_args=["--epochs", "3"])
    assert res.returncode == 0
    assert "[MOCK]" in res.stdout


def test_run_gemma_sql_train_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run_gemma_sql_train subprocess execution and error propagation."""
    monkeypatch.delenv("T1D_MOCK_GEMMA_SQL", raising=False)
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("model: gemma")

    # Success
    mock_ok = subprocess.CompletedProcess(
        args=["gemma-4-sql", "sft"],
        returncode=0,
        stdout="Training finished",
        stderr="",
    )
    with patch("shutil.which", return_value="/usr/bin/gemma-4-sql"):
        with patch("subprocess.run", return_value=mock_ok):
            res = run_gemma_sql_train("sft", cfg_file)
            assert res.stdout == "Training finished"

    # Failure
    mock_err = subprocess.CompletedProcess(
        args=["gemma-4-sql", "sft"],
        returncode=2,
        stdout="",
        stderr="TPU VM unreachable",
    )
    with patch("shutil.which", return_value="/usr/bin/gemma-4-sql"):
        with patch("subprocess.run", return_value=mock_err):
            with pytest.raises(RuntimeError, match="TPU VM unreachable"):
                run_gemma_sql_train("sft", cfg_file)


def test_parse_gemma_sql_metrics() -> None:
    """Test parse_gemma_sql_metrics parses step, loss, and learning rate from log text."""
    sample_log = (
        "[INFO] Initializing Gemma-4 SFT training\n"
        "Epoch 1/3 | Step 150/450 | Loss: 0.2451 | LR = 2.5e-5\n"
        "[INFO] Checkpoint saved\n"
    )
    metrics = parse_gemma_sql_metrics(sample_log)
    assert metrics["steps"] == 150
    assert metrics["loss"] == 0.2451
    assert metrics["learning_rate"] == 2.5e-5

    # Empty string
    empty_metrics = parse_gemma_sql_metrics("")
    assert empty_metrics["steps"] is None
    assert empty_metrics["loss"] is None
    assert empty_metrics["learning_rate"] is None


def test_run_gemma_sql_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test TimeoutError handling in run_gemma_sql_etl and run_gemma_sql_train."""
    monkeypatch.delenv("T1D_MOCK_GEMMA_SQL", raising=False)
    db_file = tmp_path / "data.duckdb"
    db_file.write_text("mock")
    cfg_file = tmp_path / "cfg.yml"
    cfg_file.write_text("mock")

    with patch("shutil.which", return_value="/usr/bin/gemma-4-sql"):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gemma-4-sql", timeout=1.0),
        ):
            with pytest.raises(TimeoutError, match="ETL command timed out"):
                run_gemma_sql_etl("sft", db_file, "table", timeout=1.0)

            with pytest.raises(TimeoutError, match="training timed out"):
                run_gemma_sql_train("sft", cfg_file, timeout=1.0)


def test_gemma_sql_pipeline_engine_library_mode(tmp_path: Path) -> None:
    """Test GemmaSqlPipelineEngine direct in-process library execution."""
    db_file = tmp_path / "train.duckdb"
    db_file.write_text("data")
    ds_file = tmp_path / "data.jsonl"
    ds_file.write_text("data")
    tc_file = tmp_path / "test.jsonl"
    tc_file.write_text("tests")
    out_dir = tmp_path / "out"

    mock_gemma_pkg = MagicMock()
    mock_gemma_pkg.run_etl.return_value = {"shards": 5}
    mock_gemma_pkg.run_sft.return_value = {"loss": 0.12}
    mock_gemma_pkg.run_dpo.return_value = {"loss": 0.08}
    mock_gemma_pkg.evaluate_sql.return_value = 0.94

    with patch.dict("sys.modules", {"gemma_4_sql": mock_gemma_pkg}):
        engine = GemmaSqlPipelineEngine()
        assert engine.is_library_available is True

        res_etl = engine.run_etl("sft", db_file, "my_table", out_dir)
        assert res_etl["execution_mode"] == "library"
        assert res_etl["result"]["shards"] == 5

        res_sft = engine.run_sft("gemma-7b", ds_file, out_dir)
        assert res_sft["execution_mode"] == "library"
        assert res_sft["result"]["loss"] == 0.12

        res_dpo = engine.run_dpo("gemma-7b", ds_file, out_dir, beta=0.05)
        assert res_dpo["execution_mode"] == "library"
        assert res_dpo["result"]["loss"] == 0.08

        res_eval = engine.evaluate_sql("gemma-7b", tc_file, db_file)
        assert res_eval["execution_mode"] == "library"
        assert res_eval["accuracy"] == 0.94


def test_gemma_sql_pipeline_engine_missing_package() -> None:
    """Test GemmaSqlPipelineEngine raises RuntimeError when gemma_4_sql package is not installed."""
    with patch.dict("sys.modules", {"gemma_4_sql": None}):
        with pytest.raises(
            RuntimeError, match="The 'gemma_4_sql' Python package is required"
        ):
            GemmaSqlPipelineEngine()


def test_gemma_sql_pipeline_engine_uncallable_functions(tmp_path: Path) -> None:
    """Test GemmaSqlPipelineEngine errors when package functions are not callable or stage is invalid."""
    db_file = tmp_path / "train.duckdb"
    db_file.write_text("data")
    ds_file = tmp_path / "data.jsonl"
    ds_file.write_text("data")
    tc_file = tmp_path / "test.jsonl"
    tc_file.write_text("tests")
    out_dir = tmp_path / "out"

    empty_pkg = MagicMock(spec=[])
    with patch.dict("sys.modules", {"gemma_4_sql": empty_pkg}):
        engine = GemmaSqlPipelineEngine()

        # Invalid stage
        with pytest.raises(ValueError, match="Invalid ETL stage"):
            engine.run_etl("invalid_stage", db_file, "table")

        # Missing run_etl
        with pytest.raises(RuntimeError, match="does not expose a callable 'run_etl'"):
            engine.run_etl("sft", db_file, "table")

        # Missing run_sft
        with pytest.raises(RuntimeError, match="does not expose a callable 'run_sft'"):
            engine.run_sft("gemma-7b", ds_file, out_dir)

        # Missing run_dpo
        with pytest.raises(RuntimeError, match="does not expose a callable 'run_dpo'"):
            engine.run_dpo("gemma-7b", ds_file, out_dir)

        # Missing evaluate_sql
        with pytest.raises(
            RuntimeError, match="does not expose a callable 'evaluate_sql'"
        ):
            engine.evaluate_sql("gemma-7b", tc_file, db_file)


def test_gemma_sql_pipeline_engine_missing_files(tmp_path: Path) -> None:
    """Test GemmaSqlPipelineEngine FileNotFoundError validations for missing files."""
    mock_pkg = MagicMock()
    with patch.dict("sys.modules", {"gemma_4_sql": mock_pkg}):
        engine = GemmaSqlPipelineEngine()
        valid_file = tmp_path / "exists.txt"
        valid_file.write_text("x")
        missing_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError, match="DuckDB database file not found"):
            engine.run_etl("sft", missing_file, "table")

        with pytest.raises(FileNotFoundError, match="Training dataset not found"):
            engine.run_sft("gemma-7b", missing_file, tmp_path / "out")

        with pytest.raises(FileNotFoundError, match="Preference dataset not found"):
            engine.run_dpo("gemma-7b", missing_file, tmp_path / "out")

        with pytest.raises(FileNotFoundError, match="Test cases file not found"):
            engine.evaluate_sql("gemma-7b", missing_file, valid_file)

        with pytest.raises(FileNotFoundError, match="DuckDB database file not found"):
            engine.evaluate_sql("gemma-7b", valid_file, missing_file)


def test_gemma_sql_pipeline_engine_native_fallbacks(tmp_path: Path) -> None:
    """Test GemmaSqlPipelineEngine native in-process fallbacks when gemma_4_sql is unavailable."""
    import duckdb

    # 1. Initialize with fallback enabled when import fails
    with patch(
        "importlib.import_module",
        side_effect=ImportError("No module named 'gemma_4_sql'"),
    ):
        engine = GemmaSqlPipelineEngine(allow_native_fallback=True)
        assert engine.is_library_available is False

        # Create test DuckDB database with clinical data
        db_file = tmp_path / "clinical.duckdb"
        conn = duckdb.connect(str(db_file))
        conn.execute(
            "CREATE TABLE cgm_metrics (patient_id INT, mean_glucose DOUBLE, tir DOUBLE)"
        )
        conn.execute(
            "INSERT INTO cgm_metrics VALUES (1, 142.5, 76.2), (2, 168.0, 62.4)"
        )
        conn.close()

        # 2. Native run_etl (with and without output_dir)
        etl_res = engine.run_etl(
            "sft", db_file, "cgm_metrics", output_dir=tmp_path / "etl_out"
        )
        assert etl_res["status"] == "completed"
        assert etl_res["execution_mode"] == "native_in_process"
        assert etl_res["result"]["rows"] == 2
        assert (tmp_path / "etl_out" / "sft_data.jsonl").exists()
        assert (tmp_path / "etl_out" / "sft_data.parquet").exists()

        etl_res_no_out = engine.run_etl("sft", db_file, "cgm_metrics", output_dir=None)
        assert etl_res_no_out["status"] == "completed"

        # 3. Native run_sft (with and without hyperparameters)
        ds_file = tmp_path / "etl_out" / "sft_data.jsonl"
        sft_res = engine.run_sft(
            "gemma-2b", ds_file, tmp_path / "sft_out", hyperparameters={"num_epochs": 1}
        )
        assert sft_res["status"] == "completed"
        assert sft_res["execution_mode"] == "native_in_process"

        sft_res_no_hparams = engine.run_sft(
            "gemma-2b", ds_file, tmp_path / "sft_out2", hyperparameters=None
        )
        assert sft_res_no_hparams["status"] == "completed"

        # 4. Native run_dpo
        pref_file = tmp_path / "pref.jsonl"
        pref_file.write_text('{"prompt": "q", "chosen": "a", "rejected": "b"}\n')
        dpo_res = engine.run_dpo("gemma-2b", pref_file, tmp_path / "dpo_out", beta=0.05)
        assert dpo_res["status"] == "completed"
        assert dpo_res["execution_mode"] == "native_in_process"
        assert dpo_res["result"]["beta"] == 0.05

        # 5. Native evaluate_sql - empty test cases
        empty_tc = tmp_path / "empty_cases.jsonl"
        empty_tc.write_text("\n")
        eval_empty = engine.evaluate_sql("gemma-2b", empty_tc, db_file)
        assert eval_empty["accuracy"]["total"] == 0

        # 6. Native evaluate_sql - valid test cases (including malformed json and non-dict lines)
        valid_tc = tmp_path / "test_cases.jsonl"
        valid_tc.write_text(
            "not a json line\n"
            '"just a json string"\n'
            '{"gold_sql": "SELECT * FROM cgm_metrics WHERE patient_id = 1", "predicted_sql": "SELECT * FROM cgm_metrics WHERE patient_id = 1;"}\n'
            '{"gold_sql": "SELECT tir FROM cgm_metrics", "predicted_sql": "SELECT tir FROM cgm_metrics"}\n'
            '{"gold_sql": "SELECT tir FROM cgm_metrics", "predicted_sql": "SELECT mean_glucose FROM cgm_metrics"}\n'
            '{"gold_sql": "SELECT * FROM cgm_metrics", "predicted_sql": "INVALID SYNTAX %%%"}\n'
        )
        eval_res = engine.evaluate_sql("gemma-2b", valid_tc, db_file)
        assert eval_res["status"] == "completed"
        assert eval_res["execution_mode"] == "native_in_process"
        assert eval_res["accuracy"]["total"] == 4
        assert eval_res["accuracy"]["exact_match"] == 0.5
        assert eval_res["accuracy"]["execution_accuracy"] == 0.5
