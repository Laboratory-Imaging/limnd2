@echo off
setlocal enableextensions

REM Change to the directory of this script (tests_compatability/)
cd /d "%~dp0"

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
set PYTHONPATH=%CD%\..\src;%PYTHONPATH%

if not exist test_report mkdir test_report
set PYTEST_LOG=%CD%\test_report\pytest_output.txt
set COVERAGE_LOG=%CD%\test_report\coverage_output.txt

echo Running tests and generating HTML report...
pytest --html=test_report\report.html --self-contained-html > "%PYTEST_LOG%" 2>&1 || echo Tests completed with failures. See %PYTEST_LOG%.

echo Generating coverage HTML report...
coverage html > "%COVERAGE_LOG%" 2>&1 || echo Coverage HTML generation failed. See %COVERAGE_LOG%.

echo Opening reports in your default browser...
start "" "%CD%\test_report\report.html"
start "" "%CD%\htmlcov\index.html"

echo Done. Reports should now be open.

:end
endlocal
exit /b 0
