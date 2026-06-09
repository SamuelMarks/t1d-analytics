@echo off
setlocal

if "%LIBSCRIPT_ROOT_DIR%"=="" set "LIBSCRIPT_ROOT_DIR=%USERPROFILE%\repos\libscript"
set "LIBSCRIPT_CLI=%LIBSCRIPT_ROOT_DIR%\libscript.cmd"
if "%NODE_NAME%"=="" set "NODE_NAME=test-t1d-analytics-node"
if "%RG_NAME%"=="" set "RG_NAME=rg-analytics-test11"
if "%LOCATION%"=="" set "LOCATION=eastus"
set "REPO_DIR=%~dp0"
if "%REMOTE_DEST%"=="" set "REMOTE_DEST=t1d-analytics-test"

echo Starting deployment test for t1d-analytics...

rem Provision the stack
call "%LIBSCRIPT_CLI%" provision azure "%NODE_NAME%" "%RG_NAME%" "%LOCATION%" "%REPO_DIR%" "%REMOTE_DEST%"
if errorlevel 1 (
    echo Provisioning failed.
    goto cleanup
)

echo Verifying remote execution...
call "%LIBSCRIPT_CLI%" cloud azure node exec "%NODE_NAME%" "%RG_NAME%" "echo Deployment test successful"
if errorlevel 1 (
    echo Remote execution verification failed.
    goto cleanup
)

echo Test passed.

:cleanup
echo Cleaning up resources...
call "%LIBSCRIPT_CLI%" deprovision azure "%NODE_NAME%" "%RG_NAME%" "%LOCATION%" "%REPO_DIR%" "%REMOTE_DEST%"
echo Deprovisioning complete.
exit /b 0

