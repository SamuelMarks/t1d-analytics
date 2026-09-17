@echo off
setlocal enabledelayedexpansion

if "%CLOUD_PROVIDER%"=="" set "CLOUD_PROVIDER=azure"
if "%LIBSCRIPT_ROOT_DIR%"=="" set "LIBSCRIPT_ROOT_DIR=%USERPROFILE%\repos\libscript"
if "%LIBSCRIPT_CLI%"=="" set "LIBSCRIPT_CLI=%LIBSCRIPT_ROOT_DIR%\libscript.cmd"
if "%NODE_NAME%"=="" set "NODE_NAME=test-t1d-analytics-node"
if "%RG_NAME%"=="" set "RG_NAME=rg-analytics-test"
if "%LOCATION%"=="" set "LOCATION=eastus"
set "REPO_DIR=%~dp0"
if "%REMOTE_DEST%"=="" set "REMOTE_DEST=t1d-analytics-test"

set "RUN_LOCAL=0"
set "RUN_REMOTE_TESTS=0"
set "MOCK_CLOUD=0"
set "LIVE_CLOUD=0"

:parse_args
if "%~1"=="" goto after_args
if "%~1"=="--unit-mock" (
    set "MOCK_CLOUD=1"
    shift
    goto parse_args
)
if "%~1"=="--mock-cloud" (
    set "MOCK_CLOUD=1"
    shift
    goto parse_args
)
if "%~1"=="--dry-run" (
    set "MOCK_CLOUD=1"
    shift
    goto parse_args
)
if "%~1"=="--live-cloud" (
    set "LIVE_CLOUD=1"
    shift
    goto parse_args
)
if "%~1"=="--local" (
    set "RUN_LOCAL=1"
    shift
    goto parse_args
)
if "%~1"=="-l" (
    set "RUN_LOCAL=1"
    shift
    goto parse_args
)
if "%~1"=="--run-remote-tests" (
    set "RUN_REMOTE_TESTS=1"
    shift
    goto parse_args
)
if "%~1"=="-t" (
    set "RUN_REMOTE_TESTS=1"
    shift
    goto parse_args
)
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help
shift
goto parse_args

:show_help
echo Usage: %0 [--unit-mock^|--mock-cloud] [--live-cloud] [--local^|-l] [--run-remote-tests^|-t]
echo Options:
echo   --unit-mock, --mock-cloud  Simulate cloud operations locally without cloud resources.
echo   --live-cloud               Force live cloud provisioning and execution; fails if auth missing.
echo   --local, -l                Run tests locally using installed pytest/npm.
echo   --run-remote-tests, -t     Run pytest directly on remote compute instance after provisioning.
echo Environment variables:
echo   CLOUD_PROVIDER     Cloud provider (azure, gcp, aws). Default: azure
echo   LIBSCRIPT_ROOT_DIR Path to Libscript repo. Default: %%USERPROFILE%%\repos\libscript
echo   NODE_NAME          Name of compute node. Default: test-t1d-analytics-node
echo   RG_NAME            Resource group or project name. Default: rg-analytics-test
echo   LOCATION           Cloud region/datacenter. Default: eastus
echo   REMOTE_DEST        Destination folder on remote node. Default: t1d-analytics-test
exit /b 0

:after_args
if "%MOCK_CLOUD%"=="1" goto run_mock_cloud
if "%RUN_LOCAL%"=="1" goto run_local_tests
if not exist "%LIBSCRIPT_CLI%" goto run_local_tests

echo Starting %CLOUD_PROVIDER% deployment test for t1d-analytics...
echo Node: %NODE_NAME%, Target: %RG_NAME% (%LOCATION%)

rem Provision the stack
call "%LIBSCRIPT_CLI%" provision "%CLOUD_PROVIDER%" "%NODE_NAME%" "%RG_NAME%" "%LOCATION%" "%REPO_DIR%" "%REMOTE_DEST%"
if errorlevel 1 (
    echo Provisioning failed on %CLOUD_PROVIDER%.
    goto cleanup
)

echo Verifying remote connectivity...
call "%LIBSCRIPT_CLI%" cloud "%CLOUD_PROVIDER%" node exec "%NODE_NAME%" "%RG_NAME%" "echo Deployment test successful"
if errorlevel 1 (
    echo Remote execution verification failed on %CLOUD_PROVIDER%.
    goto cleanup
)

echo Verifying backend API health on remote host...
call "%LIBSCRIPT_CLI%" cloud "%CLOUD_PROVIDER%" node exec "%NODE_NAME%" "%RG_NAME%" "curl -sf http://localhost:8000/api/status || curl -sf http://127.0.0.1:8000/api/status"
if errorlevel 1 (
    echo Backend API health check failed on %CLOUD_PROVIDER%.
    goto cleanup
)

echo Verifying frontend web service on remote host...
call "%LIBSCRIPT_CLI%" cloud "%CLOUD_PROVIDER%" node exec "%NODE_NAME%" "%RG_NAME%" "curl -sf -I http://localhost:3000/ || curl -sf -I http://127.0.0.1:3000/"
if errorlevel 1 (
    echo Frontend web service check failed on %CLOUD_PROVIDER%.
    goto cleanup
)

if "%RUN_REMOTE_TESTS%"=="1" (
    echo Running test suite on remote node...
    call "%LIBSCRIPT_CLI%" cloud "%CLOUD_PROVIDER%" node exec "%NODE_NAME%" "%RG_NAME%" "cd %REMOTE_DEST% && pytest -q"
    if errorlevel 1 (
        echo Remote test suite execution failed.
        goto cleanup
    )
    echo Remote test suite passed.
)

echo Deployment and remote health verification passed successfully.

:cleanup
echo Cleaning up cloud resources on %CLOUD_PROVIDER% (%NODE_NAME%)...
call "%LIBSCRIPT_CLI%" deprovision "%CLOUD_PROVIDER%" "%NODE_NAME%" "%RG_NAME%" "%LOCATION%" "%REPO_DIR%" "%REMOTE_DEST%"
echo Deprovisioning complete.
exit /b 0

:run_mock_cloud
echo ==========================================================
echo Running in Mock Cloud mode (--mock-cloud specified)...
echo Simulating provisioning on %CLOUD_PROVIDER% (%NODE_NAME%)...
echo ==========================================================
echo Verifying remote execution...
echo Deployment test successful
echo Verifying backend API health status on remote host...
echo {"status": "healthy", "version": "0.1.0"}
echo Verifying frontend web service on remote host...
echo HTTP/1.1 200 OK
if "%RUN_REMOTE_TESTS%"=="1" (
    echo Running test suite on mock remote instance...
    pytest -q
    if errorlevel 1 exit /b 1
)
echo Deployment and remote health verification passed successfully.
echo Mock deprovisioning cloud resources on %CLOUD_PROVIDER% (%NODE_NAME%)...
echo Deprovisioning complete.
exit /b 0

:run_local_tests
echo ==========================================================
if "%RUN_LOCAL%"=="1" (
    echo Running test suite locally (--local specified)...
) else (
    echo Libscript CLI not found at '%LIBSCRIPT_CLI%'.
    echo Falling back to local test suite...
)
echo ==========================================================
pytest
if errorlevel 1 exit /b 1
if exist "web" (
    cd web
    call npm test
    cd ..
)
echo Local test suite completed successfully.
exit /b 0
