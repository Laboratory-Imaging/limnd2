@echo off
setlocal enableextensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "TEST_RESULTS_DIR=%REPO_ROOT%\test_results"
set "HTML_REPORT=%TEST_RESULTS_DIR%\nd2_compatability_test_report.html"
set "COVERAGE_DIR=%TEST_RESULTS_DIR%\nd2_compatability_coverage"

REM Change to the directory of this script (tests_compatability/)
cd /d "%SCRIPT_DIR%"

if not exist "%TEST_RESULTS_DIR%" mkdir "%TEST_RESULTS_DIR%"
if not exist "%COVERAGE_DIR%" mkdir "%COVERAGE_DIR%"

where uv >nul 2>&1
if %errorlevel%==0 (
  echo Syncing development dependencies with uv...
  cd /d "%CD%\.."
  uv sync --extra dev || goto :end
  set "RUNNER=uv run"
) else (
  echo uv not found; falling back to the active Python environment.
  set "RUNNER=python -m"
)

REM Ensure src/ is importable (src layout)
set PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%

echo Running tests and generating HTML report...
%RUNNER% pytest tests_compatability --html="%HTML_REPORT%" --self-contained-html || echo Tests completed with failures.

echo Generating coverage HTML report...
%RUNNER% coverage html -d "%COVERAGE_DIR%" || echo Coverage HTML generation failed.

echo Opening reports in your default browser...
start "" "%HTML_REPORT%"
start "" "%COVERAGE_DIR%\index.html"

echo Done. Reports should now be open.

:end
endlocal
exit /b 0
