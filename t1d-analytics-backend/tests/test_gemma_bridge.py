"""Tests for gemma_bridge external toolchain bridge module."""

import subprocess
import types
from pathlib import Path
from typing import Dict, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

from t1d_analytics.gemma_bridge import (
    EtlPayloadDict,
    EvalAccuracyDict,
    GemmaSqlBridge,
    GemmaSqlPipelineEngine,
    InProcessDuckDbGemmaFallback,
    SubprocessGemmaToolchain,
    _classify_sql_error_hint,
    _extract_schema_map_from_text,
    check_gemma_sql_installed,
    execute_agentic_nl_to_sql,
    get_gemma_sql_binary,
    parse_gemma_sql_metrics,
    prune_schema_with_gemma_rag,
    resolve_local_gemma_sql_repo,
    run_gemma_sql_etl,
    run_gemma_sql_train,
    validate_sql_tokens,
)
from t1d_analytics.job_manager import JobMetricValue


class _FakeGemmaLibraryModule:
    """Realistic contract-verifying fake module implementing _GemmaLibraryProtocol."""

    def run_etl(
        self,
        stage: str,
        *,
        duckdb_path: str,
        duckdb_table: str,
        output_dir: Optional[str] = None,
    ) -> EtlPayloadDict:
        """Validate ETL arguments and return structured shard metadata."""
        assert stage in ("pretrain", "sft", "posttrain")
        assert Path(duckdb_path).exists()
        assert duckdb_table.isidentifier()
        return {
            "table": duckdb_table,
            "rows": 10,
            "output_dir": output_dir or "",
            "shards": 5,
        }

    def run_sft(
        self,
        *,
        model: str,
        dataset: str,
        output_dir: str,
        hyperparameters: Dict[str, Union[int, float, str, bool]],
    ) -> Dict[str, JobMetricValue]:
        """Validate SFT arguments and return structured training metrics."""
        assert model
        assert Path(dataset).exists()
        assert Path(output_dir).exists()
        assert isinstance(hyperparameters, dict)
        return {"loss": 0.12, "steps": 10}

    def run_dpo(
        self,
        *,
        model: str,
        dataset: str,
        output_dir: str,
        beta: float,
    ) -> Dict[str, JobMetricValue]:
        """Validate DPO arguments and return structured alignment metrics."""
        assert model
        assert Path(dataset).exists()
        assert Path(output_dir).exists()
        assert beta > 0.0
        return {"loss": 0.08, "beta": beta}

    def evaluate_sql(
        self,
        *,
        model: str,
        test_cases: str,
        duckdb_path: str,
    ) -> Union[EvalAccuracyDict, float]:
        """Validate SQL evaluation arguments and return accuracy score."""
        assert model
        assert Path(test_cases).exists()
        assert Path(duckdb_path).exists()
        return 0.94


def test_check_gemma_sql_installed() -> None:
    """Test check_gemma_sql_installed across PATH binary discovery and importlib spec lookup."""
    # 1. Found in PATH
    with patch("shutil.which", return_value="/usr/local/bin/gemma-4-sql"):
        assert check_gemma_sql_installed() is True

    # 2. Not in PATH, found as Python module
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert check_gemma_sql_installed() is True

    # 3. Neither in PATH nor installed as Python module
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=None):
            assert check_gemma_sql_installed() is False


def test_get_gemma_sql_binary() -> None:
    """Test resolving the gemma-4-sql executable binary across PATH, module fallback, and error."""
    # 1. PATH resolution
    with patch("shutil.which", return_value="/opt/bin/gemma-4-sql"):
        assert get_gemma_sql_binary() == "/opt/bin/gemma-4-sql"

    # 2. Python module fallback
    with patch("shutil.which", return_value=None):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert get_gemma_sql_binary() == "python3 -m gemma_4_sql"

    # 3. Not installed raises RuntimeError
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


def test_run_gemma_sql_etl_subprocess(tmp_path: Path) -> None:
    """Test run_gemma_sql_etl subprocess execution and error propagation."""
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

    with (
        patch("shutil.which", return_value=None),
        patch("importlib.util.find_spec", return_value=MagicMock()),
        patch("subprocess.run", return_value=mock_success) as mock_mod_run,
    ):
        res_mod = run_gemma_sql_etl("pretrain", db_file, "pretrain_data")
        assert res_mod.stdout == "ETL Complete"
        assert mock_mod_run.call_args[0][0][:3] == ["python3", "-m", "gemma_4_sql"]

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


def test_run_gemma_sql_train_subprocess(tmp_path: Path) -> None:
    """Test run_gemma_sql_train subprocess execution and error propagation."""
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


def test_run_gemma_sql_timeouts(tmp_path: Path) -> None:
    """Test TimeoutError handling in run_gemma_sql_etl and run_gemma_sql_train."""
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
    """Test GemmaSqlPipelineEngine direct in-process library execution against contract-verifying fake."""
    db_file = tmp_path / "train.duckdb"
    db_file.write_text("data")
    ds_file = tmp_path / "data.jsonl"
    ds_file.write_text("data")
    tc_file = tmp_path / "test.jsonl"
    tc_file.write_text("tests")
    out_dir = tmp_path / "out"

    fake_gemma_pkg = _FakeGemmaLibraryModule()
    assert GemmaSqlBridge is GemmaSqlPipelineEngine
    assert isinstance(SubprocessGemmaToolchain(), SubprocessGemmaToolchain)
    assert isinstance(InProcessDuckDbGemmaFallback(), InProcessDuckDbGemmaFallback)

    with patch("importlib.import_module", return_value=fake_gemma_pkg):
        engine = GemmaSqlPipelineEngine()
        assert engine.is_library_available is True

        res_etl = engine.run_etl("sft", db_file, "my_table", out_dir)
        assert res_etl["execution_mode"] == "library"
        assert res_etl["result"]["shards"] == 5
        assert res_etl["result"]["table"] == "my_table"

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
    # 1. Initialize with fallback enabled when import fails
    import duckdb

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2]
    p_tok = patch(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: mock_tok
    )
    p_hf = patch(
        "t1d_analytics.training_runner.HuggingFaceCausalLMFactory.create_model",
        lambda *a, **kw: MagicMock(),
    )
    p_tok.start()
    p_hf.start()
    try:
        _run_native_fallbacks_assertions(tmp_path, duckdb)
    finally:
        p_hf.stop()
        p_tok.stop()


def _run_native_fallbacks_assertions(tmp_path: Path, duckdb: types.ModuleType) -> None:
    """Execute assertions for GemmaSqlPipelineEngine native in-process fallbacks."""
    # Create test DuckDB database with clinical data

    with patch(
        "t1d_analytics.gemma_bridge.importlib.import_module",
        side_effect=ImportError("No module named 'gemma_4_sql'"),
    ):
        engine = GemmaSqlPipelineEngine(allow_native_fallback=True)
    assert engine.is_library_available is False
    db_file = tmp_path / "clinical.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute(
        "CREATE TABLE cgm_metrics (patient_id INT, mean_glucose DOUBLE, tir DOUBLE)"
    )
    conn.execute("INSERT INTO cgm_metrics VALUES (1, 142.5, 76.2), (2, 168.0, 62.4)")
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

    with pytest.raises(ValueError, match="Invalid DuckDB table identifier"):
        engine.run_etl("sft", db_file, "cgm_metrics; DROP TABLE cgm_metrics;")

    # Test run_etl when table has no columns (rel.description is None/empty)
    with patch(
        "duckdb.DuckDBPyConnection.execute",
        return_value=MagicMock(description=None, fetchall=lambda: []),
    ):
        etl_empty_cols = engine.run_etl(
            "sft", db_file, "cgm_metrics", output_dir=tmp_path / "etl_empty_cols"
        )
        assert etl_empty_cols["status"] == "completed"

    # 3. Native run_sft (with and without hyperparameters, and with dry_run=False)
    ds_file = tmp_path / "etl_out" / "sft_data.jsonl"
    sft_res = engine.run_sft(
        "gemma-2b", ds_file, tmp_path / "sft_out", hyperparameters={"num_epochs": 1}
    )
    assert sft_res["status"] == "completed"
    assert sft_res["execution_mode"] == "native_in_process"

    sft_res_full = engine.run_sft(
        "gemma-2b",
        ds_file,
        tmp_path / "sft_out_full",
        hyperparameters={
            "num_epochs": 1,
            "dry_run": False,
            "use_tiny_model": True,
        },
        dry_run=False,
    )
    assert sft_res_full["status"] == "completed"
    assert sft_res_full["result"]["status"] == "completed"
    assert (
        tmp_path / "sft_out_full" / "checkpoint-final" / "training_metrics.json"
    ).exists()

    sft_res_no_hparams = engine.run_sft(
        "gemma-2b", ds_file, tmp_path / "sft_out2", hyperparameters=None
    )
    assert sft_res_no_hparams["status"] == "completed"

    # 4. Native run_dpo (including empty lines, malformed JSON, non-dict JSON, and empty rejected_sql)
    pref_file = tmp_path / "pref.jsonl"
    pref_file.write_text(
        "\n"
        "not-json\n"
        "[1, 2, 3]\n"
        '{"prompt": "q1", "chosen_sql": "SELECT COUNT(*) FROM p", "rejected_sql": "SELECT * FROM p WHERE 1=0"}\n'
        '{"prompt": "q2", "chosen": "SELECT id FROM p", "rejected": ""}\n'
        '{"prompt": "q3", "chosen": ""}\n'
    )
    dpo_res = engine.run_dpo("gemma-2b", pref_file, tmp_path / "dpo_out", beta=0.05)
    assert dpo_res["status"] == "completed"
    assert dpo_res["execution_mode"] == "native_in_process"
    assert "result" in dpo_res
    assert dpo_res["result"]["status"] == "dry_run_completed"
    assert dpo_res["result"]["dpo_pairs"] == 2
    dpo_loss_val = dpo_res["result"]["dpo_loss"]
    reward_margin_val = dpo_res["result"]["reward_margin"]
    assert isinstance(dpo_loss_val, (int, float))
    assert isinstance(reward_margin_val, (int, float))
    assert float(dpo_loss_val) > 0.0
    assert float(reward_margin_val) > 0.0

    dpo_res_full = engine.run_dpo(
        "gemma-2b",
        pref_file,
        tmp_path / "dpo_out_full",
        beta=0.05,
        dry_run=False,
    )
    assert dpo_res_full["result"]["status"] == "completed"
    assert (
        tmp_path / "dpo_out_full" / "checkpoint-final" / "training_metrics.json"
    ).exists()

    # Also test DPO when no valid chosen_sql pairs exist in file
    pref_empty = tmp_path / "pref_empty.jsonl"
    pref_empty.write_text('{"prompt": "p", "chosen_sql": ""}\n')
    dpo_zero_res = engine.run_dpo(
        "gemma-2b", pref_empty, tmp_path / "dpo_zero_out", beta=0.1
    )
    assert dpo_zero_res["result"]["dpo_pairs"] == 0
    assert dpo_zero_res["result"]["dpo_loss"] == 0.0
    assert dpo_zero_res["result"]["reward_margin"] == 0.0

    # 5. Native evaluate_sql - empty test cases
    empty_tc = tmp_path / "empty_cases.jsonl"
    empty_tc.write_text("\n")
    eval_empty = engine.evaluate_sql("gemma-2b", empty_tc, db_file)
    assert isinstance(eval_empty["accuracy"], dict)
    assert eval_empty["accuracy"]["total"] == 0

    # 6. Native evaluate_sql - valid test cases (including malformed json, non-dict lines, and missing prediction)
    valid_tc = tmp_path / "test_cases.jsonl"
    valid_tc.write_text(
        "not a json line\n"
        '"just a json string"\n'
        '{"gold_sql": "SELECT * FROM cgm_metrics WHERE patient_id = 1", "predicted_sql": "SELECT * FROM cgm_metrics WHERE patient_id = 1;"}\n'
        '{"gold_sql": "SELECT tir FROM cgm_metrics", "predicted_sql": "SELECT tir FROM cgm_metrics"}\n'
        '{"gold_sql": "SELECT tir FROM cgm_metrics", "predicted_sql": "SELECT mean_glucose FROM cgm_metrics"}\n'
        '{"gold_sql": "SELECT * FROM cgm_metrics", "predicted_sql": "INVALID SYNTAX %%%"}\n'
        '{"gold_sql": "SELECT * FROM cgm_metrics", "predicted_sql": ""}\n'
    )
    eval_res = engine.evaluate_sql("gemma-2b", valid_tc, db_file)
    assert eval_res["status"] == "completed"
    assert eval_res["execution_mode"] == "native_in_process"
    assert isinstance(eval_res["accuracy"], dict)
    assert eval_res["accuracy"]["total"] == 5
    assert eval_res["accuracy"]["exact_match"] == 0.4
    assert eval_res["accuracy"]["execution_accuracy"] == 0.4

    # 7. Native evaluate_sql - unexpected exception during execution
    with patch(
        "duckdb.DuckDBPyConnection.execute", side_effect=Exception("Test mock error")
    ):
        eval_err = engine.evaluate_sql("gemma-2b", valid_tc, db_file)
        assert eval_err["status"] == "completed"
        assert isinstance(eval_err["accuracy"], dict)
        assert eval_err["accuracy"]["total"] == 5
        assert eval_err["accuracy"]["execution_accuracy"] == 0.0


def test_run_gemma_sql_etl_extra_args(tmp_path: Path) -> None:
    """Test run_gemma_sql_etl with extra_args."""
    import duckdb

    from t1d_analytics.gemma_bridge import run_gemma_sql_etl

    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE t (id INT)")
    conn.close()

    with patch("t1d_analytics.gemma_bridge.get_gemma_sql_binary", return_value="echo"):
        res = run_gemma_sql_etl(
            "sft",
            db_path,
            "t",
            output_dir=tmp_path / "out",
            extra_args=["--verbose", "true"],
        )
        assert res.returncode == 0


def test_run_gemma_sql_train_extra_args(tmp_path: Path) -> None:
    """Test run_gemma_sql_train with extra_args."""
    from t1d_analytics.gemma_bridge import run_gemma_sql_train

    cfg = tmp_path / "config.yaml"
    cfg.write_text("epochs: 1")

    with patch("t1d_analytics.gemma_bridge.get_gemma_sql_binary", return_value="echo"):
        res = run_gemma_sql_train("sft", str(cfg), extra_args=["--debug"])
        assert res.returncode == 0


def test_resolve_local_gemma_sql_repo(tmp_path: Path) -> None:
    """Test resolve_local_gemma_sql_repo across found, already in sys.path, and missing paths."""
    import sys

    fake_root = tmp_path / "workspace" / "sub" / "project"
    fake_root.mkdir(parents=True)
    gemma_pkg = tmp_path / "workspace" / "gemma-4-sql" / "src" / "gemma_4_sql"
    gemma_pkg.mkdir(parents=True)
    (gemma_pkg / "__init__.py").write_text("")

    fake_file = fake_root / "module.py"
    fake_file.write_text("")

    clean_path = [p for p in sys.path if str(tmp_path) not in p]
    with patch("sys.path", clean_path):
        found = resolve_local_gemma_sql_repo(search_root=fake_file)
        assert found == gemma_pkg.parent
        assert str(gemma_pkg.parent) in sys.path

        # Second call when already in sys.path
        found_again = resolve_local_gemma_sql_repo(search_root=fake_root)
        assert found_again == gemma_pkg.parent

    # Missing repo reaching filesystem root
    with patch("pathlib.Path.exists", return_value=False):
        assert resolve_local_gemma_sql_repo(search_root=Path("/a/b")) is None


def test_extract_schema_map_and_prune_schema_with_gemma_rag() -> None:
    """Test _extract_schema_map_from_text and prune_schema_with_gemma_rag across text, DDL, SDK, and fallback."""
    assert prune_schema_with_gemma_rag("", "Table: patients") == "Table: patients"
    assert prune_schema_with_gemma_rag("query", "   ") == "   "
    assert prune_schema_with_gemma_rag("query", "no tables here") == "no tables here"

    nl = chr(10)
    schema_text = nl.join(
        [
            "Table: patients",
            "Columns:",
            "  - subject_id (INTEGER)",
            "  - age (INTEGER)",
            "  -",
            "Table:",
            "Table: cgm_readings",
            "Columns:",
            "  - subject_id (INTEGER)",
            "  - glucose (DOUBLE)",
        ]
    )
    parsed_map = _extract_schema_map_from_text(schema_text)
    assert "patients" in parsed_map
    assert "cgm_readings" in parsed_map

    # Test single-line Columns: format emitted by get_database_schema and inline Table: | Columns: format
    canonical_schema = nl.join(
        [
            "-- Table: lab_results | Columns: patient_id, hba1c, ",
            "-- Table: empty_inline | NoCols: x",
            "-- Table:   | Columns: ignored",
            "Table: visits",
            "Columns: visit_id (INTEGER), patient_id (INTEGER), ",
            "Sample row: (1, 10)",
        ]
    )
    canonical_map = _extract_schema_map_from_text(canonical_schema)
    assert canonical_map["lab_results"] == ["patient_id", "hba1c"]
    assert canonical_map["empty_inline"] == []
    assert canonical_map["visits"] == ["visit_id", "patient_id"]

    ddl_text = (
        'CREATE TABLE "patients" (subject_id INT, age INT, , PRIMARY KEY (subject_id));'
        + nl
        + "CREATE TABLE IF NOT EXISTS visits (visit_id INT, patient_id INT);"
    )
    ddl_map = _extract_schema_map_from_text(ddl_text)
    assert ddl_map["patients"] == ["subject_id", "age"]
    assert ddl_map["visits"] == ["visit_id", "patient_id"]

    # 1. Using gemma_4_sql.sdk.rag module
    mock_rag = MagicMock()
    mock_rag.retrieve_relevant_schema.return_value = (
        "-- Relevant Schema Context:"
        + nl
        + "-- Table: patients | Columns: subject_id, age"
    )
    with patch("importlib.import_module", return_value=mock_rag):
        enriched = prune_schema_with_gemma_rag("average age of patients", schema_text)
        assert "-- Relevant Schema Context:" in enriched
        assert schema_text in enriched

    # 2. When retrieve_relevant_schema returns empty string -> falls back to built-in ranking
    mock_rag_empty = MagicMock()
    mock_rag_empty.retrieve_relevant_schema.return_value = "   "
    with patch("importlib.import_module", return_value=mock_rag_empty):
        enriched_fb = prune_schema_with_gemma_rag("glucose readings", schema_text)
        assert "-- Table: cgm_readings" in enriched_fb

    from t1d_analytics.api_schemas import ColumnInfo, TableSchemaModel
    from t1d_analytics.gemma_bridge import score_sql_completion_logprob

    # 3. When import_module raises -> falls back to built-in ranking (with and without matching words)
    with patch("importlib.import_module", side_effect=ImportError("no rag")):
        enriched_err = prune_schema_with_gemma_rag("unrelated words xyz", ddl_text)
        assert "-- Relevant Schema Context:" in enriched_err

        structured_models = [
            TableSchemaModel(
                table_name="patients",
                columns=[
                    ColumnInfo(name="subject_id", type="INTEGER"),
                    ColumnInfo(name="age", type="DOUBLE"),
                ],
                sample_row="(1, 24.5)",
            )
        ]
        enriched_models = prune_schema_with_gemma_rag("patients age", structured_models)
        assert "-- Table: patients | Columns: subject_id, age" in enriched_models
        assert prune_schema_with_gemma_rag("", structured_models) != ""
        assert prune_schema_with_gemma_rag("patients", []) == ""

        good_score = score_sql_completion_logprob(
            "Count patients by age",
            "SELECT COUNT(subject_id), age FROM patients GROUP BY age",
        )
        bad_syntax = score_sql_completion_logprob(
            "Count patients by age",
            "SELECT COUNT(subject_id",
        )
        non_readonly = score_sql_completion_logprob(
            "Count patients by age",
            "DROP TABLE patients",
        )
        empty_score = score_sql_completion_logprob("Count patients", "   ")
        assert good_score > bad_syntax
        assert good_score > non_readonly
        assert bad_syntax > empty_score


def test_validate_sql_tokens() -> None:
    """Test validate_sql_tokens with gemma_4_sql.tokenization.SQLTokenizer and byte fallback."""
    mock_tok_mod = MagicMock()
    mock_tok_instance = MagicMock()
    mock_tok_instance.encode.return_value = [10, 20, 30]
    mock_tok_mod.SQLTokenizer.return_value = mock_tok_instance

    with patch("importlib.import_module", return_value=mock_tok_mod):
        assert validate_sql_tokens("SELECT 1") == [10, 20, 30]

    # Non-list return from encode falls back to UTF-8 bytes
    mock_tok_instance.encode.return_value = "not-a-list"
    with patch("importlib.import_module", return_value=mock_tok_mod):
        assert validate_sql_tokens("AB") == [65, 66]

    # ImportError falls back to UTF-8 bytes
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        assert validate_sql_tokens("AB") == [65, 66]

    from t1d_analytics.sql_agent import validate_sql_tokens as raw_validate_sql_tokens

    with patch("importlib.import_module", side_effect=ImportError("missing")):
        assert raw_validate_sql_tokens("AB", repo_resolver=None) == [65, 66]


def test_classify_sql_error_hint_and_execute_agentic_nl_to_sql() -> None:
    """Test _classify_sql_error_hint and execute_agentic_nl_to_sql self-correction loop."""
    assert "column names" in _classify_sql_error_hint(
        "Binder Error: Referenced column 'bad_col' not found"
    )
    assert "table names" in _classify_sql_error_hint(
        "Catalog Error: Table with name bad_tbl does not exist"
    )
    assert _classify_sql_error_hint("Syntax error near SELECT") == ""

    # 1. Succeeds on second attempt after self-correction
    prompts_seen = []

    def fake_gen(prompt: str) -> str:
        """Generate candidate SQL across retry turns."""
        nl = chr(10)
        prompts_seen.append(prompt)
        if len(prompts_seen) == 1:
            return f"```sql{nl}SELECT bad_col FROM patients{nl}```"
        return f"```sql{nl}SELECT COUNT(*) AS cnt FROM patients{nl}```"

    from t1d_analytics.sql_utils import SqlValue

    def fake_exec(db: str, sql: str) -> list[dict[str, SqlValue]]:
        """Execute candidate SQL and fail if bad_col is present."""
        if "bad_col" in sql:
            raise RuntimeError("Binder Error: Referenced column 'bad_col' not found")
        return [{"cnt": 100}]

    res = execute_agentic_nl_to_sql(
        "test.duckdb", "count patients", fake_gen, fake_exec, max_retries=2
    )
    assert res["success"] is True
    assert res["attempts"] == 2
    assert res["rows"] == [{"cnt": 100}]
    assert res["error"] is None

    # 2. Exhausts retries and returns failure
    res_fail = execute_agentic_nl_to_sql(
        "test.duckdb",
        "count patients",
        lambda _: "SELECT bad_col FROM patients",
        fake_exec,
        max_retries=2,
    )
    assert res_fail["success"] is False
    assert res_fail["attempts"] == 2
    assert res_fail["error"] is not None
