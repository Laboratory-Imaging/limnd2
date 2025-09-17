@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%") do set "SCRIPT_DIR=%%~fI"
if not "%SCRIPT_DIR:~-1%"=="\\" set "SCRIPT_DIR=%SCRIPT_DIR%\\"
set "REPO_ROOT=%SCRIPT_DIR%..\\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "LOG_FILE=%SCRIPT_DIR%mypy.log"
python -c "import mypy" >nul 2>&1
if errorlevel 1 (
    echo mypy not found. Installing via pip...
    python -m pip install --upgrade mypy || goto :install_error
)
echo Running mypy against "%REPO_ROOT%".
powershell -NoLogo -NoProfile -Command "$repo = '%REPO_ROOT%'; $log = '%LOG_FILE%'; Set-Location -Path $repo; python -m mypy . 2>&1 | Tee-Object -FilePath $log; exit $LASTEXITCODE"
set "EXITCODE=%ERRORLEVEL%"
if "%EXITCODE%"=="0" (
    echo mypy completed successfully. Log saved to "%LOG_FILE%".
) else (
    echo mypy completed with exit code %EXITCODE%. See "%LOG_FILE%" for details.
)
goto :pause_and_exit

:install_error
echo Unable to install mypy. See output above.
set "EXITCODE=1"

:pause_and_exit
echo.
pause
exit /b %EXITCODE%
