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
- **FastAPI Backend**: A modular, localized REST and Server-Sent Events (SSE) streaming API serving chat sessions, SQL execution, table profiling, background training jobs, and deep health diagnostics.
- **Vanilla TypeScript Web UI**: A responsive dark-mode web application built without heavy frameworks (No React/Angular/Vue). Includes composable controllers for chat sessions (`#chat-<id>` deep-linking), a Visual Cohort Builder, a Clinical Prompt & SQL Template Library, a Schema Explorer & Database Switcher, CGM Ambulatory Glucose Profile (AGP) metrics/charts, multi-provider LLM configuration, and 8-locale internationalization (`en`, `es`, `fr`, `de`, `zh`, `ja`, `ar`, `he` with RTL support).
- **Uncompromising Quality**: Enforces strict 100% test coverage (lines, branches, functions, statements) and 100% documentation coverage across both the Python backend and TypeScript frontend.

## 🏗️ Architecture & Tech Stack

The repository is structured into distinct, decoupled functional layers:

```text
t1d-analytics/
├── scripts/                      # Full-stack test & LCOV coverage runner (run_tests.py)
├── t1d-analytics-backend/        # Python Backend Package
│   ├── pyproject.toml            # Python package metadata and strict tool configuration
│   ├── docs_src/                 # Sphinx documentation sources
│   ├── src/t1d_analytics/        # Python Backend Core
│   │   ├── api.py                # FastAPI application factory, middleware, and lifespan
│   │   ├── routers/              # Modular API routers (chat, sessions, status, tables, training)
│   │   ├── cli.py                # Command-line interface entry point
│   │   ├── cli_parser.py         # Strongly-typed CLI argument parser (15 subcommands)
│   │   ├── cli_handlers.py       # CLI subcommand handlers
│   │   ├── downloader.py         # Authenticated dataset scraping and concurrent downloading
│   │   ├── archive_extractor.py  # Safe zip extraction with decompression limits
│   │   ├── parser.py             # HTML dataset metadata parser
│   │   ├── ingestion.py          # DuckDB CSV/TXT/SAS/SPSS loader and directory watcher
│   │   ├── seed.py               # Baseline T1D clinical cohort seeding and canonical views
│   │   ├── analytics.py          # DuckDB schema inspection, table profiling, and REPL
│   │   ├── chat_service.py       # NL-to-SQL translation, SSE streaming, and evaluation
│   │   ├── sql_fallback_synthesizer.py # Deterministic schema-guided fallback SQL synthesis
│   │   ├── llm_client.py         # Ollama and any-llm multi-provider client
│   │   ├── gemma_bridge.py       # Bridge to external gemma-4-sql ETL and training toolchains
│   │   ├── training_data.py      # Synthetic Pretrain/SFT/DPO pair generation and export
│   │   ├── training_runner.py    # Local CPU/GPU, HuggingFace, and TPU training execution
│   │   ├── tpu_orchestrator.py   # Google Cloud TPU VM and XPK workload orchestration
│   │   ├── job_manager.py        # Background training job lifecycle manager
│   │   ├── session_store.py      # Persistent DuckDB chat session storage
│   │   ├── diagnostics.py        # System, database, and LLM readiness diagnostics
│   │   ├── models.py             # Pydantic domain and training models
│   │   └── i18n.py               # Backend gettext internationalization (en, ja, ar, he)
│   └── tests/                    # Pytest unit and integration test suite (100% Coverage)
└── t1d-analytics-web/            # Vanilla TypeScript Web Interface
    ├── package.json              # Frontend package manifest and scripts
    ├── src/                      # Core UI coordinator, state, CGM metrics, charts, and CSS
    │   └── controllers/          # Composable modal, sidebar, schema, model, and status controllers
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

# Install all backend dependencies
pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt
pip install -e t1d-analytics-backend
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

A cross-platform `Makefile` (POSIX/macOS/Linux) and `make.bat` (Windows) are provided to automate environment provisioning, dependency installation, database ingestion, building, serving, testing, and Docker workflows.

```bash
# POSIX / macOS / Linux
make <target>

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
```

**Backend (Python):**

```bash
pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests/ --cov=t1d_analytics --cov-report=term-missing --cov-fail-under=100
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
$LIBSCRIPT_PATH start
$LIBSCRIPT_PATH status

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
