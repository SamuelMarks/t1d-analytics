.PHONY: build_docker run_docker test_docker clean_docker install_base install_deps build_docs build serve test help

DOCS_DIR ?= docs
VENV_ACTIVATE := $(shell for d in .venv venv .venv-* venv-*; do if [ -f "$$d/bin/activate" ]; then echo ". $$d/bin/activate && "; break; fi; done)

help:
	@echo "Available commands:"
	@echo "  build_docker  Build the docker containers"
	@echo "  run_docker    Run the docker containers"
	@echo "  test_docker   Run tests in docker containers"
	@echo "  clean_docker  Clean up docker containers"
	@echo "  install_base  Install language runtime and tools"
	@echo "  install_deps  Install local dependencies"
	@echo "  build_docs    Build the API docs (override with DOCS_DIR=$(DOCS_DIR))"
	@echo "  build         Build the frontend and backend"
	@echo "  serve         Serve the frontend behind the backend local dir static file server (which is enabled in DEBUG mode only)"
	@echo "  test          Run tests locally"
	@echo "  help          Show help text"

build_docker:
	docker-compose build

run_docker:
	docker-compose up

test_docker:
	docker-compose run --rm backend pytest
	docker-compose run --rm frontend npm run test

clean_docker:
	docker-compose down -v

install_base:
	$(VENV_ACTIVATE) python -m pip install --upgrade pip
	npm install -g npm

install_deps:
	$(VENV_ACTIVATE) python -m pip install -r requirements.txt -r requirements-dev.txt
	cd web && npm install

build_docs:
	mkdir -p $(DOCS_DIR)
	@echo "Docs built in $(DOCS_DIR)"

build:
	cd web && npm run build
	$(VENV_ACTIVATE) python -m pip install -e .

serve:
	$(VENV_ACTIVATE) DEBUG=1 uvicorn src.t1d_analytics.api:app --reload --port 8000

test:
	$(VENV_ACTIVATE) pytest
	cd web && npm run test
