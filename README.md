<div align="center">
  <h1>🩸 T1D Analytics Suite</h1>

  <p><strong>An end-to-end toolchain for downloading, parsing, querying, and visualizing Type 1 Diabetes (T1D) clinical trial datasets.</strong></p>

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](#)
[![Docs](https://img.shields.io/badge/Docs-100%25-brightgreen.svg)](#)
[![CI](https://github.com/SamuelMarks/t1d-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/t1d-analytics/actions/workflows/ci.yml)

</div>

---

This project provides a robust CLI for data retrieval, a highly optimized DuckDB-powered analytics engine, local LLM integration for translating natural language into SQL, a lightweight FastAPI backend, and a responsive Vanilla TypeScript web interface.

## 📑 Table of Contents

- [Features](#-features)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Makefile Usage](#-makefile-usage)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Documentation](#-documentation)
- [License](#-license)

## 🚀 Features

- **Automated Data Retrieval & Seeding**: Robust CLI tools to safely download, extract, seed baseline cohorts, and catalog complex clinical trial datasets (including CSV, TXT, SAS `.sas7bdat`/`.xpt`, and SPSS `.sav`).
- **Local DuckDB Analytics**: Automatically parses raw clinical trial files (with encoding and delimiter detection) into a pooled, read-only validated local DuckDB database with data quality profiling and export to Excel (`.xlsx`), Parquet, CSV, and JSON.
- **Multi-Provider Natural Language to SQL & Rule-Based Fallback**: Integrated with [Ollama](https://ollama.com/) (`gemma:2b`, `llama3.2:1b`, `phi4-mini`, `gemma4`) and Mozilla's [any-llm](https://github.com/mozilla-ai/any-llm) (`huggingface`, `openai`, `anthropic`, `google`), with automatic schema-guided rule-based SQL fallback and diagnostic reporting when LLM inference is unavailable.
- **Gemma-4-SQL Training Pipeline**: Provides CLI and API tools to procedurally generate stratified training datasets (`pretrain`, `sft`, and `dpo`), bridge directly to `gemma-4-sql`, and orchestrate local CPU/GPU or Google Cloud TPU (`remote-tpu-maxtext`) training jobs.
- **FastAPI Backend**: A modular, localized REST, asynchronous job polling (`/api/chat/jobs`, `/api/eval/nl-to-sql/jobs`), and Server-Sent Events (SSE) streaming API serving chat sessions, SQL execution, clinical query templates (`/api/templates`), table profiling, streaming exports, background training jobs, optional API key / Bearer authentication, route-tiered rate limiting, and deep health diagnostics.
- **Vanilla TypeScript Web UI**: A responsive dark-mode web application built without heavy frameworks (No React/Angular/Vue). Powered by a strongly typed `AppEventBus`, composable controllers for chat sessions (`/chats/<id>` shareable deep-linking), a Visual Cohort Builder, an Academic Table 1 & Clinical SQL Template Picker, a Schema Explorer & Database Switcher, responsive `viewBox` SVG CGM Ambulatory Glucose Profile (AGP) and bar/line charts, paginated formula-injection-safe result exports, multi-provider LLM configuration, and 8-locale internationalization (`en`, `es`, `fr`, `de`, `zh`, `ja`, `ar`, `he` with RTL support).
- **Uncompromising Quality**: Enforces strict 100% test coverage (lines, branches, functions, statements) and 100% documentation coverage across both the Python backend and TypeScript frontend.

## 🏗️ Architecture & Tech Stack

The repository is structured into distinct, decoupled functional layers:

```text
t1d-analytics/
├── libscript.json                # Declarative LibScript multi-cloud, service, and TPU manifest
├── Makefile / make.sh / make.bat # Cross-platform native and Docker lifecycle automation
├── test.sh / test.cmd            # Cloud deployment, TLS, and remote health verification harness
├── scripts/                      # Full-stack test, AST, & LCOV coverage runner (run_tests.py)
├── t1d-analytics-backend/        # Python Backend Package (Python 3.12+)
│   ├── pyproject.toml            # Python package metadata and strict tool configuration
│   ├── docs_src/                 # Sphinx documentation sources
│   ├── src/t1d_analytics/        # Python Backend Core
│   │   ├── api.py                        # FastAPI application factory, middleware, and lifespan
│   │   ├── api_dependencies.py           # AppServices DI container, auth, and CORS dependencies
│   │   ├── api_exports.py                # Bounded-memory DuckDB COPY & cursor streaming exports
│   │   ├── api_schemas.py                # Canonical Pydantic API request/response wire schemas
│   │   ├── routers/                      # Modular routers (chat, session, status, table, training)
│   │   ├── cli.py / cli_parser.py / cli_handlers.py # 15-subcommand typed CLI & handlers
│   │   ├── downloader.py / parser.py / archive_extractor.py # Authenticated download & extraction
│   │   ├── ingestion.py                  # DuckDB tabular loader, manifest, and directory watcher
│   │   ├── clinical_file_readers.py      # CSV/TXT/SAS (.sas7bdat/.xpt)/SPSS (.sav) readers
│   │   ├── ingestion_study_resolver.py   # Clinical trial study & ontology mapping
│   │   ├── seed.py / jaeb_cohort_sql.py / seed_verifier.py # Cohort seeding & canonical views
│   │   ├── analytics.py                  # DuckDB schema inspection, table profiling, and REPL
│   │   ├── db_security.py / duckdb_pool.py / rate_limiter.py # Path guards, pooling, rate limits
│   │   ├── sql_executor.py / sql_utils.py / sql_agent.py # Read-only SQL execution & sanitization
│   │   ├── chat_service.py               # Chat service facade
│   │   ├── chat_turn_executor.py         # Sync, async-job, and holdout evaluation turn executor
│   │   ├── nl_to_sql_translator.py       # NL-to-SQL translation, EXPLAIN validation, and repair
│   │   ├── sse_chat_streamer.py          # Disconnect-aware SSE token and SQL result streaming
│   │   ├── sql_fallback_synthesizer.py   # Deterministic schema-guided fallback SQL synthesis
│   │   ├── clinical_prompts.py / clinical_query_catalog.py # Prompts & 12 gold SQL templates
│   │   ├── table1_specs.py / table1_sql_compiler.py / table1_formatter.py / table1_builder.py
│   │   ├── llm_client.py / ollama_discovery.py / llm_stream_protocols.py # Multi-provider LLM I/O
│   │   ├── gemma_bridge.py / gemma_cli_toolchain.py / gemma_library_adapter.py / gemma_inprocess_fallback.py
│   │   ├── hf_models.py                  # Typed PyTorch, HuggingFace, BitsAndBytes & PEFT LoRA factories
│   │   ├── training_data.py / training_dataset_exporter.py / training_dpo_miner.py # SFT/DPO data
│   │   ├── training_runner.py / tpu_orchestrator.py / gcs_storage.py # CPU/GPU/TPU training & GCS
│   │   ├── job_manager.py                # Async chat/eval and training job lifecycle managers
│   │   ├── session_store.py              # Normalized DuckDB chat session & message persistence
│   │   ├── diagnostics.py / errors.py    # Readiness diagnostics and structured T1DError hierarchy
│   │   └── models.py / i18n.py           # Pydantic domain models and gettext localization
│   └── tests/                    # Pytest unit and integration test suite (100% Coverage)
└── t1d-analytics-web/            # Vanilla TypeScript Web Interface
    ├── package.json              # Frontend package manifest and scripts
    ├── src/                      # Typed AppEventBus, state, API client/types, SVG charts, Table 1, CSS
    │   └── controllers/          # Composable modal, sidebar, schema, template, model, & status controllers
    ├── tests/                    # Vitest DOM and state unit tests (100% Coverage)
    ├── tests-e2e/                # Playwright mocked E2E and accessibility tests
    └── tests-e2e-full/           # Playwright full-stack E2E tests against live FastAPI + DuckDB
```

- **Database**: DuckDB (with optional PostgreSQL and Valkey in declarative LibScript deployments)
- **Backend API**: Python 3.12+, FastAPI, Uvicorn, Pydantic
- **LLM Engine**: Ollama (`gemma:2b`, `llama3.2:1b`, `phi4-mini`, `gemma4`) & `any-llm` (`huggingface`, `openai`, `anthropic`, `google`)
- **Frontend**: HTML5, Vanilla CSS, TypeScript, Vite, Marked, Highlight.js, DOMPurify, i18next
- **Testing**: Pytest, Vitest, Playwright (`@axe-core/playwright`)

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.12+** (Tested and optimized for Python 3.12)
2. **Node.js** (v20+ recommended) and `npm`
3. **Ollama**: Installed and running locally with a supported model pulled (e.g., `gemma:2b` for lightweight local execution or `gemma4` for full-scale generation):
   ```bash
   ollama pull gemma:2b
   # Or for larger hardware:
   ollama pull gemma4
   ```

## ⚡ Quick Start

### 1. Setup the Python Backend

Clone the repository and prepare your virtual environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all backend dependencies (add [training] for local LoRA / QLoRA fine-tuning)
pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt
pip install -e "t1d-analytics-backend[training]"
```

### 2. Setup the Web Frontend

Navigate to the `t1d-analytics-web` directory and install the required Node modules:

```bash
cd t1d-analytics-web
npm install
```

### 3. Run the Stack

Please refer to the [USAGE.md](USAGE.md) for detailed instructions on launching the API, running the CLI tools, and starting the Vite development server.

## 🧰 Makefile Usage

Cross-platform `Makefile` / `make.sh` (POSIX/macOS/Linux) and `make.bat` (Windows) entry points are provided to automate environment provisioning, dependency installation, database ingestion, building, serving, testing, and Docker workflows.

```bash
# POSIX / macOS / Linux
make <target>
# Or via the POSIX shell wrapper:
./make.sh <target>

# Windows (Command Prompt / PowerShell)
.\make.bat <target>
```

### Common Workflows

```bash
# Full native setup (backend virtualenv, frontend build, DuckDB init & data load)
make setup

# Start the unified local server (FastAPI serving built frontend assets on port 8000)
make serve

# Full Docker setup and execution
make setup_docker
make run_docker
```

### Configurable Environment Variables

You can override the default paths and binaries either via environment variables or inline `make` arguments:

| Variable             | Default                                           | Description                                                     |
| :------------------- | :------------------------------------------------ | :-------------------------------------------------------------- |
| `DOCS_DIR`           | `docs`                                            | Output directory for generated Sphinx and TypeDoc documentation |
| `DATA_DIR`           | `data`                                            | Directory containing clinical trial datasets to ingest          |
| `T1D_DB_PATH`        | `$(DATA_DIR)/t1d.duckdb`                          | Path to the target DuckDB database file                         |
| `T1D_DB_SEARCH_DIR`  | `/Volumes/TOSHIBA_EXT/stanford/t1d:/mnt/data/t1d` | Search directories for discovering T1D DuckDB databases         |
| `COMPOSE_FILE`       | `docker-compose.yml`                              | Docker Compose manifest file                                    |
| `DOCKER_COMPOSE`     | `docker-compose`                                  | Docker Compose binary or command                                |
| `LIBSCRIPT_ROOT_DIR` | `~/repos/libscript`                               | Path to local LibScript installation (used by `install_base`)   |
| `LIBSCRIPT_CLI`      | `$(LIBSCRIPT_ROOT_DIR)/libscript.sh`              | Path to local LibScript CLI binary                              |

### Target Reference

| Target                  | Category     | Description                                                                                                                                       |
| :---------------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `help`                  | General      | Show available commands and usage descriptions                                                                                                    |
| `install_base`          | Provisioning | Install language runtimes (Python 3.12, Node.js 20.x), database (DuckDB), and tools (`uv`, `ollama`) via LibScript, and initialize `.venv`        |
| `install_deps`          | Provisioning | Install local Python backend requirements and frontend `npm` dependencies                                                                         |
| `setup`                 | Native Setup | Setup native backend, frontend, and DuckDB database (`setup_backend` + `setup_frontend` + `setup_db`)                                             |
| `setup_native`          | Native Setup | Alias for `setup` (fully native stack setup)                                                                                                      |
| `setup_backend`         | Native Setup | Create virtual environment and install the native Python backend package in editable mode                                                         |
| `build_backend`         | Build & Docs | Install the Python backend package in editable mode                                                                                               |
| `setup_frontend`        | Native Setup | Install `npm` dependencies and build native frontend production assets                                                                            |
| `build_frontend`        | Build & Docs | Build the Vite frontend production bundle                                                                                                         |
| `install_db`            | Database     | Install native DuckDB and Ollama via LibScript                                                                                                    |
| `init_db`               | Database     | Initialize the native DuckDB database file (`T1D_DB_PATH`)                                                                                        |
| `seed_db`               | Database     | Load clinical trial datasets from `DATA_DIR` into DuckDB and run `t1d-analytics doctor`                                                           |
| `setup_db`              | Native Setup | Initialize the native DuckDB database file, load datasets from `DATA_DIR`, and run `t1d-analytics doctor` (`install_db` + `init_db` + `seed_db`)  |
| `build`                 | Build & Docs | Build the frontend bundle and install the Python backend package (`build_frontend` + `build_backend`)                                             |
| `build_docs`            | Build & Docs | Verify 100% docstring coverage (`interrogate`) and build backend (Sphinx) and frontend (TypeDoc) docs into `DOCS_DIR`                             |
| `serve`                 | Run (Native) | Serve the frontend behind the backend static file server (`DEBUG=1`) on port `8000`                                                               |
| `run_native`            | Run (Native) | Alias for `serve` (run the native stack)                                                                                                          |
| `dev_backend`           | Run (Native) | Start the native FastAPI backend server with hot-reload on port `8000`                                                                            |
| `serve_backend`         | Run (Native) | Alias for `dev_backend`                                                                                                                           |
| `dev_frontend`          | Run (Native) | Start the native Vite frontend development server                                                                                                 |
| `serve_frontend`        | Run (Native) | Alias for `dev_frontend`                                                                                                                          |
| `start_stack`           | LibScript    | Start the declarative LibScript stack (`libscript.json`)                                                                                          |
| `stop_stack`            | LibScript    | Stop the declarative LibScript stack (`libscript.json`)                                                                                           |
| `status_stack`          | LibScript    | Inspect status of the declarative LibScript stack (`libscript.json`)                                                                              |
| `health_stack`          | LibScript    | Run health checks on the declarative LibScript stack (`libscript.json`)                                                                           |
| `logs_stack`            | LibScript    | Tail logs for the declarative LibScript stack (`libscript.json`)                                                                                  |
| `train_ml`              | LibScript    | Run declarative ML/MPAX benchmark training via LibScript (`libscript.json`)                                                                       |
| `infer_ml`              | LibScript    | Run declarative ML/MPAX arena inference/scoring via LibScript (`libscript.json`)                                                                  |
| `test`                  | Quality      | Run backend (`pytest`) and frontend (`npm run test`) test suites locally                                                                          |
| `test_native`           | Quality      | Alias for `test`                                                                                                                                  |
| `clean_native`          | Quality      | Clean local frontend/backend build, test, coverage, and documentation artifacts                                                                   |
| `fmt`                   | Quality      | Lint and format backend (`ruff check --fix`, `ruff format`) and frontend (`npm run format`) code                                                  |
| `setup_docker`          | Docker       | Build containers, build frontend assets, and initialize the Docker DuckDB database (`build_docker` + `setup_frontend_docker` + `setup_db_docker`) |
| `db_docker`             | Docker       | Start the Ollama backing container via Docker Compose                                                                                             |
| `seed_docker`           | Docker       | Initialize and load the DuckDB database inside Docker and run `t1d-analytics doctor`                                                              |
| `setup_db_docker`       | Docker       | Alias for `seed_docker`                                                                                                                           |
| `backend_docker`        | Docker       | Build and start the Ollama and backend containers via Docker Compose                                                                              |
| `setup_backend_docker`  | Docker       | Build the backend Docker image                                                                                                                    |
| `frontend_docker`       | Docker       | Build and start the frontend container via Docker Compose                                                                                         |
| `setup_frontend_docker` | Docker       | Install and build frontend assets inside Docker                                                                                                   |
| `build_docker`          | Docker       | Build all Docker containers via `$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build`                                                                      |
| `run_docker`            | Docker       | Start the Docker containers via `$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up`                                                                         |
| `test_docker`           | Docker       | Run backend and frontend test suites inside Docker containers                                                                                     |
| `clean_docker`          | Docker       | Stop and remove Docker containers and volumes via `$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v`                                                  |

## 🧪 Testing

The project enforces a strict 100% test coverage and 100% documentation coverage requirement across all modules.

**Unified Full-Stack Test & 100% Combined LCOV Runner:**

```bash
python scripts/run_tests.py
# In CI (without mutating README badges on disk):
T1D_SKIP_README_SHIELD_UPDATE=1 python scripts/run_tests.py
```

**Backend (Python):**

```bash
pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests/ --cov-report=json --cov-report=lcov --cov-fail-under=100
```

**Frontend Unit Tests (TypeScript / Vitest):**

```bash
cd t1d-analytics-web
npm run coverage
```

**End-to-End Tests (Playwright Mocked & Full-Stack):**

```bash
cd t1d-analytics-web
npx playwright test
npm run test:e2e:full
```

## ⚙️ Deployment

This project leverages [LibScript](https://github.com/SamuelMarks/libscript) for complete, native PaaS and AI/TPU orchestration without Docker.

```bash
export LIBSCRIPT_DIR="$HOME/repos/libscript"
[ -d "$LIBSCRIPT_DIR" ] || git clone --depth=1 https://github.com/SamuelMarks/libscript "$LIBSCRIPT_DIR"
export LIBSCRIPT_PATH="$LIBSCRIPT_DIR/libscript.sh"

# 1. Install toolchains, database, and server dependencies natively (Python 3.12, Node.js 20.x, DuckDB, Nginx, Ollama, uv)
$LIBSCRIPT_PATH install-deps libscript.json

# 2. Build frontend, provision DuckDB, apply Nginx SPA/API ingress, and start daemonized services
$LIBSCRIPT_PATH start libscript.json
$LIBSCRIPT_PATH status libscript.json

# 3. Execute declarative ML training or inference (supports --target local|gcp-tpu-vm|gcp-gpu-vm|gke-xpk)
$LIBSCRIPT_PATH train libscript.json --dry-run
$LIBSCRIPT_PATH infer libscript.json --dry-run
```

For full cloud and TPU orchestration details, see [DEPLOY.md](DEPLOY.md) and [DEPLOY_TO_TPU.md](DEPLOY_TO_TPU.md).

## 📖 Documentation

Dive deeper into the suite's capabilities with our comprehensive guides:

- [**USAGE.md**](USAGE.md): Detailed instructions on using the CLI, API, and web frontend.
- [**REPORT.md**](REPORT.md): End-to-End Training Report.
- [**T1D_INTEGRATION_GUIDE.md**](T1D_INTEGRATION_GUIDE.md): Integration guide for specialized model workflows.
- [**DEPLOY.md**](DEPLOY.md): Guide for deploying the main application stack.
- [**DEPLOY_TO_TPU.md**](DEPLOY_TO_TPU.md): Instructions for executing training pipelines on Google Cloud TPUs.

---

## ⚖️ License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
