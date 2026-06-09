T1D Analytics Suite
===================

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](#)
[![Docs](https://img.shields.io/badge/Docs-100%25-brightgreen.svg)](#)
[![CI](https://github.com/SamuelMarks/t1d-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/t1d-analytics/actions/workflows/ci.yml)

An end-to-end toolchain for downloading, parsing, querying, and visualizing Type 1 Diabetes (T1D) clinical trial datasets.

This project provides a robust CLI for data retrieval, a DuckDB-powered analytics engine, a local LLM integration for translating natural language to SQL, a FastAPI backend, and a responsive Vanilla TypeScript web interface.

## 🚀 Features

- **Automated Data Retrieval**: CLI tools to safely download and extract complex clinical trial datasets.
- **Local DuckDB Analytics**: Automatically parses raw `.csv` and `.txt` trial files (with robust encoding and delimiter detection) into a highly optimized, local DuckDB database.
- **Natural Language to SQL**: Integrated with [Ollama](https://ollama.com/) and Mozilla's [any-llm](https://github.com/mozilla-ai/any-llm) to translate plain English questions into valid DuckDB SQL queries using the `gemma4` model.
- **FastAPI Backend**: A lightweight, highly documented REST API serving as the bridge between the DuckDB engine, the LLM, and the frontend.
- **Vanilla TypeScript Web UI**: A clean, responsive, dark-mode web application built without heavy frameworks (No React/Angular/Vue). Features full chat session management, dropdown context menus, model selection, and dynamic HTML table rendering for SQL results.
- **100% Test Coverage**: Both the Python backend (`pytest`) and the TypeScript frontend (`vitest`) maintain strict 100% code coverage across lines, branches, and functions.

## 🏗️ Architecture

The repository is structured into distinct functional layers:

```text
t1d-analytics/
├── src/t1d_analytics/
│   ├── cli.py         # Command-line interface entry points
│   ├── downloader.py  # Handles dataset scraping and downloading
│   ├── parser.py      # Parses complex dataset metadata
│   ├── analytics.py   # DuckDB loading and CLI REPL
│   └── api.py         # FastAPI application endpoints
├── tests/             # Python backend tests (100% Coverage)
└── web/               # Web Interface
    ├── src/           # Vanilla TS, HTML, and CSS
    └── tests/         # Vitest DOM and state tests (100% Coverage)
```

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.9+** (Tested on 3.12)
2. **Node.js** (v18+ recommended) and `npm`
3. **Ollama**: Running locally with the `gemma4` model pulled.
   ```bash
   ollama pull gemma4
   ```

## 📦 Installation

### 1. Python Backend Dependencies

Clone the repository and install the required Python packages into your virtual environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Web Frontend Dependencies

Navigate to the `web` directory and install the Node modules:

```bash
cd web
npm install
```

## 🧪 Testing

The project enforces a strict 100% test coverage requirement.

**Backend (Python):**

```bash
pytest tests/ --cov=t1d_analytics --cov-report=term-missing --cov-fail-under=100
```

**Frontend (TypeScript):**

```bash
cd web
npm run coverage
```

## ⚙️ Deployment

This project uses [LibScript](https://github.com/SamuelMarks/libscript) for complete, native PaaS deployment without Docker.

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

See [DEPLOY.md](DEPLOY.md) for full cloud orchestration details.

## 📖 Next Steps

For detailed instructions on how to use the CLI, spin up the backend API, and interact with the web application, please see the [**USAGE.md**](./USAGE.md) file.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
