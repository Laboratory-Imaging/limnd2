@echo off
setlocal enableextensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "TEST_RESULTS_DIR=%REPO_ROOT%\test_results"
set "HTML_REPORT=%TEST_RESULTS_DIR%\limnd2_test_report.html"
set "COVERAGE_DIR=%TEST_RESULTS_DIR%\limnd2_coverage"

REM Change to the directory of this script (tests/)
cd /d "%SCRIPT_DIR%"

if not exist "%TEST_RESULTS_DIR%" mkdir "%TEST_RESULTS_DIR%"
if not exist "%COVERAGE_DIR%" mkdir "%COVERAGE_DIR%"

echo Checking Python test dependencies...
python -m pip install -U pip >nul 2>&1
python -c "import importlib.util, sys; mods=['pytest','pytest_html','pytest_cov','coverage']; sys.exit(1 if any(importlib.util.find_spec(m) is None for m in mods) else 0)"
if errorlevel 1 (
  echo Installing missing Python test dependencies...
  python -m pip install -U pytest pytest-html pytest-cov coverage || goto :end
) else (
  echo All required Python test dependencies are already installed.
)

REM Ensure src/ is importable (src layout)
set PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%

echo Running tests and generating HTML report...
pytest --html="%HTML_REPORT%" --self-contained-html || echo Tests completed with failures.

echo Generating coverage HTML report...
coverage html -d "%COVERAGE_DIR%" || echo Coverage HTML generation failed.

echo Opening reports in your default browser...
start "" "%HTML_REPORT%"
start "" "%COVERAGE_DIR%\index.html"

echo Done. Reports should now be open.

:end
endlocal
exit /b 0
