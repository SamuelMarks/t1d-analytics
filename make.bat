@echo off
:: ## Overview
:: Cross-platform build, database provisioning, dependency installation, stack orchestration,
:: and test runner for Windows (supporting both Docker and Native non-Docker workflows).
::
:: ## Usage
:: make.bat [help|setup|setup_native|run_native|test_native|clean_native|setup_docker|build_docker|run_docker|test_docker|clean_docker|db_docker|seed_docker|setup_db_docker|backend_docker|frontend_docker|setup_backend_docker|setup_frontend_docker|install_base|install_deps|install_db|init_db|seed_db|setup_db|setup_backend|build_backend|dev_backend|serve_backend|setup_frontend|build_frontend|dev_frontend|serve_frontend|build_docs|build|serve|start_stack|stop_stack|status_stack|health_stack|logs_stack|train_ml|infer_ml|test|fmt]

setlocal EnableDelayedExpansion
set "THIS_FILE=%~f0"

if "%DOCS_DIR%"=="" set DOCS_DIR=docs
if "%DATA_DIR%"=="" set DATA_DIR=data
if "%T1D_DB_PATH%"=="" set T1D_DB_PATH=%DATA_DIR%/t1d.duckdb
if "%COMPOSE_FILE%"=="" set COMPOSE_FILE=docker-compose.yml
if "%DOCKER_COMPOSE%"=="" set DOCKER_COMPOSE=docker-compose
if "%LIBSCRIPT_ROOT_DIR%"=="" set "LIBSCRIPT_ROOT_DIR=%USERPROFILE%\repos\libscript"
if "%LIBSCRIPT_CLI%"=="" set "LIBSCRIPT_CLI=%LIBSCRIPT_ROOT_DIR%\libscript.cmd"

if "%~1"=="" goto help
if "%~1"=="help" goto help
if "%~1"=="setup" goto setup
if "%~1"=="setup_native" goto setup_native
if "%~1"=="run_native" goto run_native
if "%~1"=="test_native" goto test_native
if "%~1"=="clean_native" goto clean_native
if "%~1"=="setup_docker" goto setup_docker
if "%~1"=="build_docker" goto build_docker
if "%~1"=="run_docker" goto run_docker
if "%~1"=="test_docker" goto test_docker
if "%~1"=="clean_docker" goto clean_docker
if "%~1"=="db_docker" goto db_docker
if "%~1"=="seed_docker" goto seed_docker
if "%~1"=="setup_db_docker" goto setup_db_docker
if "%~1"=="backend_docker" goto backend_docker
if "%~1"=="frontend_docker" goto frontend_docker
if "%~1"=="setup_backend_docker" goto setup_backend_docker
if "%~1"=="setup_frontend_docker" goto setup_frontend_docker
if "%~1"=="install_base" goto install_base
if "%~1"=="install_deps" goto install_deps
if "%~1"=="install_db" goto install_db
if "%~1"=="init_db" goto init_db
if "%~1"=="seed_db" goto seed_db
if "%~1"=="setup_db" goto setup_db
if "%~1"=="setup_backend" goto setup_backend
if "%~1"=="build_backend" goto build_backend
if "%~1"=="dev_backend" goto dev_backend
if "%~1"=="serve_backend" goto serve_backend
if "%~1"=="setup_frontend" goto setup_frontend
if "%~1"=="build_frontend" goto build_frontend
if "%~1"=="dev_frontend" goto dev_frontend
if "%~1"=="serve_frontend" goto serve_frontend
if "%~1"=="build_docs" goto build_docs
if "%~1"=="build" goto build
if "%~1"=="serve" goto serve
if "%~1"=="start_stack" goto start_stack
if "%~1"=="stop_stack" goto stop_stack
if "%~1"=="status_stack" goto status_stack
if "%~1"=="health_stack" goto health_stack
if "%~1"=="logs_stack" goto logs_stack
if "%~1"=="train_ml" goto train_ml
if "%~1"=="infer_ml" goto infer_ml
if "%~1"=="test" goto test
if "%~1"=="fmt" goto fmt

echo Unknown target: %~1
goto help

:: ## help
:: Displays available targets and usage instructions.
:help
echo Available commands:
echo.
echo   --- Unified Full-Stack Lifecycle (Native vs Docker) ---
echo   setup                  Alias for setup_native (full native setup: base, db, backend, frontend)
echo   setup_native           Install native runtimes, deps, initialize/seed DB, and build
echo   run_native             Run the full stack natively (LibScript stack or DEBUG static server)
echo   test_native            Run backend and frontend unit tests natively (alias: test)
echo   clean_native           Clean local build and test artifacts
echo   setup_docker           Build containers and initialize Docker DuckDB database
echo   build_docker           Build the docker containers
echo   run_docker             Run the docker containers
echo   test_docker            Run tests in docker containers
echo   clean_docker           Clean up docker containers and volumes
echo.
echo   --- Database Targets (DuckDB) ---
echo   install_db             Install native DuckDB and Ollama via LibScript
echo   init_db                Initialize native DuckDB database file (%T1D_DB_PATH%)
echo   seed_db                Ingest clinical trial CSVs into DuckDB and run diagnostics natively
echo   setup_db               Run full native database setup (install_db, init_db, seed_db)
echo   db_docker              Start only the Ollama backing service container via Docker Compose
echo   seed_docker            Ingest clinical CSVs and run diagnostics inside the backend Docker container
echo   setup_db_docker        Initialize and load DuckDB database inside Docker
echo.
echo   --- Backend Targets (FastAPI) ---
echo   setup_backend          Create Python venv and install native backend package
echo   build_backend          Install the backend package in editable mode
echo   dev_backend            Start the native FastAPI backend server with hot reload
echo   serve_backend          Alias for dev_backend
echo   backend_docker         Build and run the backend container via Docker Compose
echo   setup_backend_docker   Build backend Docker image
echo.
echo   --- Frontend Targets (Vite + TypeScript) ---
echo   setup_frontend         Install and build native frontend assets
echo   build_frontend         Build the frontend production bundle
echo   dev_frontend           Start the native Vite frontend server
echo   serve_frontend         Alias for dev_frontend
echo   frontend_docker        Build and run the frontend container via Docker Compose
echo   setup_frontend_docker  Install and build frontend assets inside Docker
echo.
echo   --- Shared / Orchestration / Quality Targets ---
echo   install_base           Install language runtime, database, and tools
echo   install_deps           Install local dependencies
echo   build                  Build the frontend and backend
echo   build_docs             Build the API docs (override with DOCS_DIR=%%DOCS_DIR%%)
echo   serve                  Serve the frontend behind the backend local dir static file server (which is enabled in DEBUG mode only)
echo   start_stack            Start the declarative LibScript stack (libscript.json)
echo   stop_stack             Stop the declarative LibScript stack (libscript.json)
echo   status_stack           Inspect status of the declarative LibScript stack
echo   health_stack           Run health checks on the declarative LibScript stack
echo   logs_stack             Tail logs for the declarative LibScript stack
echo   train_ml               Run declarative ML/MPAX benchmark training via LibScript
echo   infer_ml               Run declarative ML/MPAX arena inference/scoring via LibScript
echo   test                   Run tests locally
echo   fmt                    Format backend and frontend code
echo   help                   Show help text
goto :eof

:: ## ensure_venv
:: Creates a Python virtual environment if not present.
:ensure_venv
for /d %%d in (.venv venv .venv-* venv-* t1d-analytics-backend\.venv t1d-analytics-backend\venv) do (
    if exist "%%d\Scripts\activate.bat" (
        goto :eof
    )
)
python3 -m venv .venv
goto :eof

:: ## activate_venv
:: Activates the local Python virtual environment if present.
:activate_venv
for /d %%d in (.venv venv .venv-* venv-* t1d-analytics-backend\.venv t1d-analytics-backend\venv) do (
    if exist "%%d\Scripts\activate.bat" (
        call "%%d\Scripts\activate.bat"
        goto :eof
    )
)
goto :eof

:: -----------------------------------------------------------------------------
:: Docker Targets
:: -----------------------------------------------------------------------------

:: ## build_docker
:: Builds the Docker Compose containers.
:build_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% build
goto :eof

:: ## db_docker
:: Starts the Ollama backing container via Docker Compose.
:db_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% up -d ollama
goto :eof

:: ## seed_docker
:: Runs DuckDB CSV ingestion and diagnostics inside the backend Docker container.
:seed_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm backend sh -c "mkdir -p /app/data && python3 -c 'import duckdb; duckdb.connect(\"/app/data/t1d.duckdb\").close()' && t1d-analytics seed --data-dir /app/data --db /app/data/t1d.duckdb && t1d-analytics load --data-dir /app/data --db /app/data/t1d.duckdb && t1d-analytics doctor --db /app/data/t1d.duckdb"
goto :eof

:: ## setup_db_docker
:: Initializes and seeds the Docker DuckDB database.
:setup_db_docker
call :seed_docker
goto :eof

:: ## setup_backend_docker
:: Builds the backend Docker image.
:setup_backend_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% build backend
goto :eof

:: ## backend_docker
:: Builds and starts the ollama and backend containers via Docker Compose.
:backend_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% up --build -d ollama backend
goto :eof

:: ## setup_frontend_docker
:: Installs and builds frontend assets inside the frontend Docker container.
:setup_frontend_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm frontend sh -c "npm install && npm run build"
goto :eof

:: ## frontend_docker
:: Builds and starts the frontend container via Docker Compose.
:frontend_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% up --build -d frontend
goto :eof

:: ## setup_docker
:: Builds all Docker images, frontend assets, and initializes the containerized database.
:setup_docker
call :build_docker
if errorlevel 1 exit /b 1
call :setup_frontend_docker
if errorlevel 1 exit /b 1
call :setup_db_docker
if errorlevel 1 exit /b 1
goto :eof

:: ## run_docker
:: Starts the Docker Compose containers.
:run_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% up
goto :eof

:: ## test_docker
:: Runs backend and frontend tests inside Docker Compose containers.
:test_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm backend pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests
if errorlevel 1 exit /b 1
%DOCKER_COMPOSE% -f %COMPOSE_FILE% run --rm frontend npm run test
if errorlevel 1 exit /b 1
goto :eof

:: ## clean_docker
:: Tears down Docker Compose containers and volumes.
:clean_docker
%DOCKER_COMPOSE% -f %COMPOSE_FILE% down -v
goto :eof

:: -----------------------------------------------------------------------------
:: Native Base & Dependency Targets
:: -----------------------------------------------------------------------------

:: ## install_base
:: Installs base language runtimes, tools, and databases via LibScript.
:install_base
if exist "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" (
    call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install python 3.12
    call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install nodejs 20.x
    call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install uv latest
    call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install duckdb latest
    call "%LIBSCRIPT_ROOT_DIR%\libscript.cmd" install ollama latest
)
call :ensure_venv
call :activate_venv
python3 -m pip install --upgrade pip
npm install -g npm
goto :eof

:: ## install_deps
:: Installs Python backend and Node.js frontend dependencies.
:install_deps
call :ensure_venv
call :activate_venv
python3 -m pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt
if errorlevel 1 exit /b 1
cd t1d-analytics-web
call npm install
if errorlevel 1 (
    cd ..
    exit /b 1
)
cd ..
goto :eof

:: -----------------------------------------------------------------------------
:: Native Database Targets (DuckDB)
:: -----------------------------------------------------------------------------

:: ## install_db
:: Installs native DuckDB and Ollama engines via LibScript.
:install_db
if exist "%LIBSCRIPT_CLI%" (
    call "%LIBSCRIPT_CLI%" install duckdb latest
    call "%LIBSCRIPT_CLI%" install ollama latest
)
goto :eof

:: ## init_db
:: Initializes the native DuckDB database file.
:init_db
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
call :activate_venv
python3 -c "import duckdb; duckdb.connect('%T1D_DB_PATH%').close()"
if errorlevel 1 exit /b 1
goto :eof

:: ## seed_db
:: Ingests clinical trial CSV datasets into DuckDB and runs diagnostics.
:seed_db
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
call :activate_venv
t1d-analytics seed --data-dir %DATA_DIR% --db %T1D_DB_PATH%
if errorlevel 1 exit /b 1
t1d-analytics load --data-dir %DATA_DIR% --db %T1D_DB_PATH%
if errorlevel 1 exit /b 1
t1d-analytics doctor --db %T1D_DB_PATH%
if errorlevel 1 exit /b 1
goto :eof

:: ## setup_db
:: Runs the complete native database setup lifecycle (install_db, init_db, seed_db).
:setup_db
call :install_db
if errorlevel 1 exit /b 1
call :init_db
if errorlevel 1 exit /b 1
call :seed_db
if errorlevel 1 exit /b 1
goto :eof

:: -----------------------------------------------------------------------------
:: Native Backend Targets (FastAPI)
:: -----------------------------------------------------------------------------

:: ## setup_backend
:: Creates a Python virtual environment if missing and installs backend packages.
:setup_backend
call :ensure_venv
call :activate_venv
python3 -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python3 -m pip install -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt
if errorlevel 1 exit /b 1
python3 -m pip install -e t1d-analytics-backend
if errorlevel 1 exit /b 1
goto :eof

:: ## build_backend
:: Installs the Python backend package in editable mode.
:build_backend
call :activate_venv
python3 -m pip install -e t1d-analytics-backend
if errorlevel 1 exit /b 1
goto :eof

:: ## dev_backend
:: Starts the local FastAPI backend server with hot reload.
:dev_backend
call :activate_venv
set T1D_DB_PATH=%T1D_DB_PATH%
uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000
goto :eof

:: ## serve_backend
:: Alias for dev_backend.
:serve_backend
call :dev_backend
goto :eof

:: -----------------------------------------------------------------------------
:: Native Frontend Targets (Vite + TypeScript)
:: -----------------------------------------------------------------------------

:: ## setup_frontend
:: Installs Node.js frontend dependencies and builds production bundle.
:setup_frontend
cd t1d-analytics-web
call npm install
if errorlevel 1 (
    cd ..
    exit /b 1
)
call npm run build
if errorlevel 1 (
    cd ..
    exit /b 1
)
cd ..
goto :eof

:: ## build_frontend
:: Builds the Vite frontend production distribution.
:build_frontend
cd t1d-analytics-web
call npm run build
if errorlevel 1 (
    cd ..
    exit /b 1
)
cd ..
goto :eof

:: ## dev_frontend
:: Starts the local Vite development server.
:dev_frontend
cd t1d-analytics-web
call npm run dev
cd ..
goto :eof

:: ## serve_frontend
:: Alias for dev_frontend.
:serve_frontend
call :dev_frontend
goto :eof

:: -----------------------------------------------------------------------------
:: Composite Native Targets
:: -----------------------------------------------------------------------------

:: ## build
:: Builds the frontend and installs the Python backend package.
:build
call :build_frontend
if errorlevel 1 exit /b 1
call :build_backend
if errorlevel 1 exit /b 1
goto :eof

:: ## build_docs
:: Audits backend and frontend docstring coverage and generates documentation.
:build_docs
if not exist "%DOCS_DIR%" mkdir "%DOCS_DIR%"
call :activate_venv
interrogate -vv --fail-under=100 t1d-analytics-backend/src/t1d_analytics
if errorlevel 1 goto :eof
python3 -m sphinx -b html t1d-analytics-backend/docs_src %DOCS_DIR%
if errorlevel 1 goto :eof
cd t1d-analytics-web
call npm run docs
cd ..
echo Docs built in %DOCS_DIR%
goto :eof

:: ## serve
:: Starts the local FastAPI backend server in DEBUG mode serving static frontend assets.
:serve
call :activate_venv
set T1D_DB_PATH=%T1D_DB_PATH%
set DEBUG=1
uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --port 8000
goto :eof

:: ## setup
:: Executes full native setup across backend, frontend, and DuckDB database.
:setup
call :setup_backend
if errorlevel 1 exit /b 1
call :setup_frontend
if errorlevel 1 exit /b 1
call :setup_db
if errorlevel 1 exit /b 1
goto :eof

:: ## setup_native
:: Alias for setup.
:setup_native
call :setup
goto :eof

:: ## run_native
:: Runs the full application stack natively.
:run_native
call :serve
goto :eof

:: ## start_stack
:: Starts the declarative LibScript stack.
:start_stack
call "%LIBSCRIPT_CLI%" start libscript.json
goto :eof

:: ## stop_stack
:: Stops the declarative LibScript stack.
:stop_stack
call "%LIBSCRIPT_CLI%" stop libscript.json
goto :eof

:: ## status_stack
:: Inspects status of the declarative LibScript stack.
:status_stack
call "%LIBSCRIPT_CLI%" status libscript.json
goto :eof

:: ## health_stack
:: Runs health checks on the declarative LibScript stack.
:health_stack
call "%LIBSCRIPT_CLI%" health libscript.json
goto :eof

:: ## logs_stack
:: Outputs logs for the declarative LibScript stack.
:logs_stack
call "%LIBSCRIPT_CLI%" logs libscript.json
goto :eof

:: ## train_ml
:: Runs declarative ML/MPAX benchmark training via LibScript.
:train_ml
call "%LIBSCRIPT_CLI%" train libscript.json
goto :eof

:: ## infer_ml
:: Runs declarative ML/MPAX arena inference/scoring via LibScript.
:infer_ml
call "%LIBSCRIPT_CLI%" infer libscript.json
goto :eof

:: ## test
:: Runs the local backend pytest suite and Vite frontend unit tests.
:test
call :activate_venv
pytest -c t1d-analytics-backend/pyproject.toml t1d-analytics-backend/tests
if errorlevel 1 exit /b 1
cd t1d-analytics-web
call npm run test
if errorlevel 1 (
    cd ..
    exit /b 1
)
cd ..
goto :eof

:: ## test_native
:: Alias for test.
:test_native
call :test
goto :eof

:: ## clean_native
:: Removes local frontend/backend build and test artifacts.
:clean_native
if exist "t1d-analytics-web\dist" rmdir /S /Q "t1d-analytics-web\dist"
if exist "t1d-analytics-web\coverage" rmdir /S /Q "t1d-analytics-web\coverage"
if exist ".pytest_cache" rmdir /S /Q ".pytest_cache"
if exist "t1d-analytics-backend\.pytest_cache" rmdir /S /Q "t1d-analytics-backend\.pytest_cache"
if exist ".coverage" del /F /Q ".coverage"
if exist "%DOCS_DIR%" rmdir /S /Q "%DOCS_DIR%"
goto :eof

:: ## fmt
:: Formats Python backend and Vite frontend code.
:fmt
call :activate_venv
ruff check --config t1d-analytics-backend/pyproject.toml --fix t1d-analytics-backend scripts
ruff format --config t1d-analytics-backend/pyproject.toml t1d-analytics-backend scripts
cd t1d-analytics-web
call npm run format
cd ..
goto :eof
