@echo off
setlocal

if "%DOCS_DIR%"=="" set DOCS_DIR=docs

if "%~1"=="" goto help
if "%~1"=="help" goto help
if "%~1"=="build_docker" goto build_docker
if "%~1"=="run_docker" goto run_docker
if "%~1"=="test_docker" goto test_docker
if "%~1"=="clean_docker" goto clean_docker
if "%~1"=="install_base" goto install_base
if "%~1"=="install_deps" goto install_deps
if "%~1"=="build_docs" goto build_docs
if "%~1"=="build" goto build
if "%~1"=="serve" goto serve
if "%~1"=="test" goto test

echo Unknown target: %~1
goto help

:help
echo Available commands:
echo   build_docker  Build the docker containers
echo   run_docker    Run the docker containers
echo   test_docker   Run tests in docker containers
echo   clean_docker  Clean up docker containers
echo   install_base  Install language runtime and tools
echo   install_deps  Install local dependencies
echo   build_docs    Build the API docs (override with DOCS_DIR=%%DOCS_DIR%%)
echo   build         Build the frontend and backend
echo   serve         Serve the frontend behind the backend local dir static file server (which is enabled in DEBUG mode only)
echo   test          Run tests locally
echo   help          Show help text
goto :eof

:build_docker
docker-compose build
goto :eof

:run_docker
docker-compose up
goto :eof

:test_docker
docker-compose run --rm backend pytest
docker-compose run --rm frontend npm run test
goto :eof

:clean_docker
docker-compose down -v
goto :eof

:activate_venv
for /d %%d in (.venv venv .venv-* venv-*) do (
    if exist "%%d\Scripts\activate.bat" (
        call "%%d\Scripts\activate.bat"
        goto :eof
    )
)
goto :eof

:install_base
call :activate_venv
python -m pip install --upgrade pip
npm install -g npm
goto :eof

:install_deps
call :activate_venv
python -m pip install -r requirements.txt -r requirements-dev.txt
cd web
call npm install
cd ..
goto :eof

:build_docs
if not exist "%DOCS_DIR%" mkdir "%DOCS_DIR%"
echo Docs built in %DOCS_DIR%
goto :eof

:build
cd web
call npm run build
cd ..
call :activate_venv
python -m pip install -e .
goto :eof

:serve
call :activate_venv
set DEBUG=1
uvicorn src.t1d_analytics.api:app --reload --port 8000
goto :eof

:test
call :activate_venv
pytest
cd web
call npm run test
cd ..
goto :eof
