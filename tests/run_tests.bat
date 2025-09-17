@echo off
setlocal enableextensions

REM Change to the directory of this script (tests/)
cd /d "%~dp0"

echo Checking Python test dependencies...
python -m pip install -U pip >nul 2>&1
python -c "import importlib, sys; mods=['pytest','pytest_html','pytest_cov','coverage']; sys.exit(1 if any(importlib.util.find_spec(m) is None for m in mods) else 0)"
if errorlevel 1 (
  echo Installing missing Python test dependencies...
  python -m pip install -U pytest pytest-html pytest-cov coverage || goto :end
) else (
  echo All required Python test dependencies are already installed.
)

REM Ensure src/ is importable (src layout)
set PYTHONPATH=%CD%\..\src;%PYTHONPATH%

echo Running tests and generating HTML report...
pytest --html=report.html --self-contained-html || echo Tests completed with failures.

echo Generating coverage HTML report...
coverage html || echo Coverage HTML generation failed.

echo Opening reports in your default browser...
start "" "%CD%\report.html"
start "" "%CD%\htmlcov\index.html"

echo Done. Reports should now be open.

:end
endlocal
exit /b 0
