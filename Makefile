# ==============================================================================
# T1D Analytics Suite - Makefile
#
# Provides both Docker and Native (non-Docker) targets to set up, build,
# initialize, seed, run, test, and clean the Database (DuckDB), Backend
# (FastAPI), and Frontend (Vite + TypeScript).
# ==============================================================================

.PHONY: help setup setup_native run_native test_native clean_native setup_docker build_docker run_docker test_docker clean_docker db_docker seed_docker setup_db_docker backend_docker frontend_docker setup_backend_docker setup_frontend_docker install_base install_deps install_db init_db seed_db setup_db setup_backend build_backend dev_backend serve_backend setup_frontend build_frontend dev_frontend serve_frontend build build_docs serve start_stack stop_stack status_stack health_stack logs_stack train_ml infer_ml test fmt

DOCS_DIR ?= docs
DATA_DIR ?= data
T1D_DB_PATH ?= $(DATA_DIR)/t1d.duckdb
COMPOSE_FILE ?= docker-compose.yml
DOCKER_COMPOSE ?= docker-compose
LIBSCRIPT_ROOT_DIR ?= $(HOME)/repos/libscript
LIBSCRIPT_CLI ?= $(LIBSCRIPT_ROOT_DIR)/libscript.sh

VENV_ACTIVATE := $(shell for d in .venv venv .venv-* venv-* t1d-analytics-backend/.venv t1d-analytics-backend/venv; do if [ -f "$$d/bin/activate" ]; then echo ". $$d/bin/activate && "; break; fi; done)

help:
	@echo "Available commands:"
	@echo ""
	@echo "  --- Unified Full-Stack Lifecycle (Native vs Docker) ---"
	@echo "  setup                  Alias for setup_native (full native setup: base, db, backend, frontend)"
	@echo "  setup_native           Install native runtimes, deps, initialize/seed DB, and build"
	@echo "  run_native             Run the full stack natively (LibScript stack or DEBUG static server)"
	@echo "  test_native            Run backend and frontend unit tests natively (alias: test)"
	@echo "  clean_native           Clean local build and test artifacts"
	@echo "  setup_docker           Build containers and initialize Docker DuckDB database"
	@echo "  build_docker           Build the docker containers"
	@echo "  run_docker             Run the docker containers"
	@echo "  test_docker            Run tests in docker containers"
	@echo "  clean_docker           Clean up docker containers and volumes"
	@echo ""
	@echo "  --- Database Targets (DuckDB) ---"
	@echo "  install_db             Install native DuckDB and Ollama via LibScript"
	@echo "  init_db                Initialize native DuckDB database file ($(T1D_DB_PATH))"
	@echo "  seed_db                Ingest clinical trial CSVs into DuckDB and run diagnostics natively"
	@echo "  setup_db               Run full native database setup (install_db, init_db, seed_db)"
	@echo "  db_docker              Start only the Ollama backing service container via Docker Compose"
	@echo "  seed_docker            Ingest clinical CSVs and run diagnostics inside the backend Docker container"
	@echo "  setup_db_docker        Initialize and load DuckDB database inside Docker"
	@echo ""
	@echo "  --- Backend Targets (FastAPI) ---"
	@echo "  setup_backend          Create Python venv and install native backend package"
	@echo "  build_backend          Install the backend package in editable mode"
	@echo "  dev_backend            Start the native FastAPI backend server with hot reload"
	@echo "  serve_backend          Alias for dev_backend"
	@echo "  backend_docker         Build and run the backend container via Docker Compose"
	@echo "  setup_backend_docker   Build backend Docker image"
	@echo ""
	@echo "  --- Frontend Targets (Vite + TypeScript) ---"
	@echo "  setup_frontend         Install and build native frontend assets"
	@echo "  build_frontend         Build the frontend production bundle"
	@echo "  dev_frontend           Start the native Vite frontend server"
	@echo "  serve_frontend         Alias for dev_frontend"
	@echo "  frontend_docker        Build and run the frontend container via Docker Compose"
	@echo "  setup_frontend_docker  Install and build frontend assets inside Docker"
	@echo ""
	@echo "  --- Shared / Orchestration / Quality Targets ---"
	@echo "  install_base           Install language runtime, database, and tools"
	@echo "  install_deps           Install local dependencies"
	@echo "  build                  Build the frontend and backend"
	@echo "  build_docs             Build the API docs (override with DOCS_DIR=$(DOCS_DIR))"
	@echo "  serve                  Serve the frontend behind the backend local dir static file server (which is enabled in DEBUG mode only)"
	@echo "  start_stack            Start the declarative LibScript stack (libscript.json)"
	@echo "  stop_stack             Stop the declarative LibScript stack (libscript.json)"
	@echo "  status_stack           Inspect status of the declarative LibScript stack"
	@echo "  health_stack           Run health checks on the declarative LibScript stack"
	@echo "  logs_stack             Tail logs for the declarative LibScript stack"
	@echo "  train_ml               Run declarative ML/MPAX benchmark training via LibScript"
	@echo "  infer_ml               Run declarative ML/MPAX arena inference/scoring via LibScript"
	@echo "  test                   Run tests locally"
	@echo "  fmt                    Format backend and frontend code"
	@echo "  help                   Show help text"

# ------------------------------------------------------------------------------
# Docker Targets
# ------------------------------------------------------------------------------

build_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build

db_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d ollama

seed_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm backend sh -c "mkdir -p /app/data && python3 -c 'import duckdb; duckdb.connect(\"/app/data/t1d.duckdb\").close()' && t1d-analytics seed --data-dir /app/data --db /app/data/t1d.duckdb && t1d-analytics load --data-dir /app/data --db /app/data/t1d.duckdb && t1d-analytics doctor --db /app/data/t1d.duckdb"

setup_db_docker: seed_docker

setup_backend_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build backend

backend_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up --build -d ollama backend

setup_frontend_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm frontend sh -c "npm install && npm run build"

frontend_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up --build -d frontend

setup_docker: build_docker setup_frontend_docker setup_db_docker

run_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up

test_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm backend pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) run --rm frontend npm run test

clean_docker:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v

# ------------------------------------------------------------------------------
# Native Base & Dependency Targets
# ------------------------------------------------------------------------------

install_base:
	@if [ -n "$(LIBSCRIPT_ROOT_DIR)" ] && [ -x "$(LIBSCRIPT_ROOT_DIR)/libscript.sh" ]; then \
		"$(LIBSCRIPT_ROOT_DIR)/libscript.sh" install python 3.12 || true; \
		"$(LIBSCRIPT_ROOT_DIR)/libscript.sh" install nodejs 20.x || true; \
		"$(LIBSCRIPT_ROOT_DIR)/libscript.sh" install uv latest || true; \
		"$(LIBSCRIPT_ROOT_DIR)/libscript.sh" install duckdb latest || true; \
		"$(LIBSCRIPT_ROOT_DIR)/libscript.sh" install ollama latest || true; \
	elif [ -x "$$HOME/repos/libscript/libscript.sh" ]; then \
		"$$HOME/repos/libscript/libscript.sh" install python 3.12 || true; \
		"$$HOME/repos/libscript/libscript.sh" install nodejs 20.x || true; \
		"$$HOME/repos/libscript/libscript.sh" install uv latest || true; \
		"$$HOME/repos/libscript/libscript.sh" install duckdb latest || true; \
		"$$HOME/repos/libscript/libscript.sh" install ollama latest || true; \
	elif command -v libscript.sh >/dev/null 2>&1; then \
		libscript.sh install python 3.12 || true; \
		libscript.sh install nodejs 20.x || true; \
		libscript.sh install uv latest || true; \
		libscript.sh install duckdb latest || true; \
		libscript.sh install ollama latest || true; \
	fi
	@if [ -z "$(VENV_ACTIVATE)" ] && [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	$(VENV_ACTIVATE) python3 -m pip install --upgrade pip
	npm install -g npm

install_deps:
	@if [ -z "$(VENV_ACTIVATE)" ] && [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	$(VENV_ACTIVATE) python3 -m pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt
	cd t1d-analytics-web && npm install

# ------------------------------------------------------------------------------
# Native Database Targets (DuckDB)
# ------------------------------------------------------------------------------

install_db:
	@if [ -f "$(LIBSCRIPT_CLI)" ]; then \
		"$(LIBSCRIPT_CLI)" install duckdb latest || true; \
		"$(LIBSCRIPT_CLI)" install ollama latest || true; \
	fi

init_db:
	mkdir -p $(DATA_DIR)
	$(VENV_ACTIVATE) python3 -c "import duckdb; duckdb.connect('$(T1D_DB_PATH)').close()"

seed_db:
	mkdir -p $(DATA_DIR)
	$(VENV_ACTIVATE) t1d-analytics seed --data-dir $(DATA_DIR) --db $(T1D_DB_PATH)
	$(VENV_ACTIVATE) t1d-analytics load --data-dir $(DATA_DIR) --db $(T1D_DB_PATH)
	$(VENV_ACTIVATE) t1d-analytics doctor --db $(T1D_DB_PATH)

setup_db: install_db init_db seed_db

# ------------------------------------------------------------------------------
# Native Backend Targets (FastAPI)
# ------------------------------------------------------------------------------

setup_backend:
	@if [ -z "$(VENV_ACTIVATE)" ] && [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	$(VENV_ACTIVATE) python3 -m pip install --upgrade pip
	$(VENV_ACTIVATE) python3 -m pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt
	$(VENV_ACTIVATE) python3 -m pip install -e t1d-analytics-backend

build_backend:
	$(VENV_ACTIVATE) python3 -m pip install -e t1d-analytics-backend

dev_backend:
	$(VENV_ACTIVATE) T1D_DB_PATH=$(T1D_DB_PATH) uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000

serve_backend: dev_backend

# ------------------------------------------------------------------------------
# Native Frontend Targets (Vite + TypeScript)
# ------------------------------------------------------------------------------

setup_frontend:
	cd t1d-analytics-web && npm install && npm run build

build_frontend:
	cd t1d-analytics-web && npm run build

dev_frontend:
	cd t1d-analytics-web && npm run dev

serve_frontend: dev_frontend

# ------------------------------------------------------------------------------
# Composite Native Targets
# ------------------------------------------------------------------------------

build: build_frontend build_backend

build_docs:
	mkdir -p $(DOCS_DIR)
	$(VENV_ACTIVATE) interrogate -vv --fail-under=100 t1d-analytics-backend/src/t1d_analytics
	$(VENV_ACTIVATE) python3 -m sphinx -b html t1d-analytics-backend/docs_src $(DOCS_DIR)
	cd t1d-analytics-web && npm run docs
	@echo "Docs built in $(DOCS_DIR)"

serve:
	$(VENV_ACTIVATE) T1D_DB_PATH=$(T1D_DB_PATH) DEBUG=1 uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000

setup: setup_backend setup_frontend setup_db

setup_native: setup

run_native: serve

start_stack:
	"$(LIBSCRIPT_CLI)" start libscript.json

stop_stack:
	"$(LIBSCRIPT_CLI)" stop libscript.json

status_stack:
	"$(LIBSCRIPT_CLI)" status libscript.json

health_stack:
	"$(LIBSCRIPT_CLI)" health libscript.json

logs_stack:
	"$(LIBSCRIPT_CLI)" logs libscript.json

train_ml:
	"$(LIBSCRIPT_CLI)" train libscript.json

infer_ml:
	"$(LIBSCRIPT_CLI)" infer libscript.json

test:
	$(VENV_ACTIVATE) pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests
	cd t1d-analytics-web && npm run test

test_native: test

clean_native:
	rm -rf t1d-analytics-web/dist t1d-analytics-web/coverage .pytest_cache .coverage t1d-analytics-backend/.pytest_cache $(DOCS_DIR)

fmt:
	$(VENV_ACTIVATE) ruff check --config t1d-analytics-backend/pyproject.toml --fix t1d-analytics-backend scripts
	$(VENV_ACTIVATE) ruff format --config t1d-analytics-backend/pyproject.toml t1d-analytics-backend scripts
	cd t1d-analytics-web && npm run format
