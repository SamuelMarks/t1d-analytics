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
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Documentation](#-documentation)
- [License](#-license)

## 🚀 Features

- **Automated Data Retrieval**: Robust CLI tools to safely download, extract, and catalog complex clinical trial datasets.
- **Local DuckDB Analytics**: Automatically parses raw `.csv` and `.txt` trial files (with robust encoding and delimiter detection) into a highly optimized, local DuckDB database.
- **Natural Language to SQL**: Integrated with [Ollama](https://ollama.com/) and Mozilla's [any-llm](https://github.com/mozilla-ai/any-llm) to translate plain English clinical questions into valid DuckDB SQL queries using the `gemma4` model.
- **Gemma-4-SQL Training Pipeline**: Provides tools to procedurally generate training datasets (`pretrain`, `sft`, and `dpo`) and a comprehensive integration guide for executing end-to-end training and deployment of specialized T1D models on Google Cloud TPUs via `gemma-4-sql`.
- **FastAPI Backend**: A lightweight, highly documented REST API serving as the bridge between the DuckDB engine, the local LLM, and the frontend clients.
- **Vanilla TypeScript Web UI**: A clean, responsive, dark-mode web application built without heavy frameworks (No React/Angular/Vue). Features full chat session management, dropdown context menus, model selection, and dynamic HTML table rendering for SQL results.
- **Uncompromising Quality**: Enforces strict 100% code coverage across all layers (Python backend and TypeScript frontend) and maintains comprehensive documentation.

## 🏗️ Architecture & Tech Stack

The repository is structured into distinct, decoupled functional layers:

```text
t1d-analytics/
├── src/t1d_analytics/   # Python Backend Core
│   ├── cli.py           # Command-line interface entry points
│   ├── downloader.py    # Handles dataset scraping and downloading
│   ├── parser.py        # Parses complex dataset metadata
│   ├── analytics.py     # DuckDB loading and analytics engine
│   ├── api.py           # FastAPI application endpoints
│   ├── models.py        # Pydantic data models
│   ├── training_data.py # Utilities for Gemma-4-SQL training generation
│   └── i18n.py          # Internationalization support
├── tests/               # Python backend test suite (100% Coverage)
└── web/                 # Vanilla TS Web Interface
    ├── src/             # Core UI, state management, and CSS
    ├── tests/           # Vitest DOM and state tests (100% Coverage)
    └── tests-e2e/       # Playwright end-to-end tests
```

- **Database**: DuckDB
- **Backend API**: Python 3.9+, FastAPI, Uvicorn, Pydantic
- **LLM Engine**: Ollama (gemma4)
- **Frontend**: HTML5, Vanilla CSS, TypeScript, Vite
- **Testing**: Pytest, Vitest, Playwright

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.9+** (Tested and optimized for Python 3.12)
2. **Node.js** (v18+ recommended) and `npm`
3. **Ollama**: Installed and running locally with the `gemma4` model pulled.
   ```bash
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
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Setup the Web Frontend

Navigate to the `web` directory and install the required Node modules:

```bash
cd web
npm install
```

### 3. Run the Stack

Please refer to the [USAGE.md](USAGE.md) for detailed instructions on launching the API, running the CLI tools, and starting the Vite development server.

## 🧪 Testing

The project enforces a strict 100% test coverage requirement to ensure absolute reliability in clinical data handling.

**Backend (Python):**

```bash
pytest tests/ --cov=t1d_analytics --cov-report=term-missing --cov-fail-under=100
```

**Frontend (TypeScript):**

```bash
cd web
npm run coverage
```

**End-to-End Tests (Playwright):**

```bash
cd web
npm run test:e2e
```

## ⚙️ Deployment

This project leverages [LibScript](https://github.com/SamuelMarks/libscript) for complete, native PaaS deployment without Docker.

```bash
export LIBSCRIPT_PATH="$HOME/repos/libscript/libscript.sh"
[ -d "$LIBSCRIPT_PATH" ] || git clone --depth=1 https://github.com/SamuelMarks/libscript "$LIBSCRIPT_PATH"

# 1. Install toolchains (Python 3.12, NodeJS 20, Nginx)
$LIBSCRIPT_PATH install-deps

# 2. Run ETL hooks, build frontend, setup daemons, and configure Nginx
$LIBSCRIPT_PATH start

# Note: You can skip hooks if data is already loaded
$LIBSCRIPT_PATH start --no-hooks
```

For full cloud orchestration details, see [DEPLOY.md](DEPLOY.md).

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
