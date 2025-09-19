# Maintainers Guide

## Installation

### Prerequisites

limnd2 package requires following packages (also listed in `requirements.txt`) to work correctly:

- python>=3.12.0
- numpy
- h5py
- ome_types
- pandas
- tifffile
- imagecodecs

### Installation scripts

This package and its prerequisites can be installed running following commands in Powershell / shell.

#### Windows

``` sh
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/Laboratory-Imaging/Laboratory-Imaging.github.io/refs/heads/main/limnd2/setup_limnd2.bat' -OutFile 'setup_limnd2.bat'; & '.\setup_limnd2.bat'; Remove-Item 'setup_limnd2.bat'"
```

#### Linux / MacOS   // TODO test script when repo is public

``` sh
curl -O https://raw.githubusercontent.com/Laboratory-Imaging/Laboratory-Imaging.github.io/refs/heads/main/limnd2/setup_limnd2.sh && chmod +x setup_limnd2.sh && ./setup_limnd2.sh && rm ./setup.sh
```

### Manual Installation

#### Windows

``` sh
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
python -m venv env
env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
code .
```

### Build & Upload to Private PyPI

``` sh
pip install build setuptools twine
python -m build
twine upload -r local dist\*
```

Using `uv`:
```sh
uv build
uv publish --publish-url http://gaexec:9500 --trusted-publishing never --username "-" --password "-" dist/*
```

### Documentation Preview

``` sh
mkdocs serve
```

## Static Type Checks

Run the batch helpers located in `tests\static_type_check` so they pick up the repository's `pyproject.toml` configuration.

- `tests\static_type_check\run_mypy_check.bat`
  - Ensures `mypy` is installed (installs if missing).
  - Executes `mypy` from the repository root and mirrors output to the console and `tests\static_type_check\mypy.log`.
  - Leaves the window open (`pause`) so you can review results immediately.

- `tests\static_type_check\run_pyright_check.bat`
  - Installs `pyright` on demand.
  - Runs `pyright` against the repo root, streaming output to the console and `tests\static_type_check\pyright.log`.
  - Keeps the window open after completion for quick inspection.

Both scripts exit with the underlying tool's status code (zero for success, non-zero when diagnostics are reported), but they always write the full log next to the scripts.

## Test Suite

Use `tests\run_tests.bat` to execute the full test workflow from the command line.

- Upgrades `pip`, checks for required packages (`pytest`, `pytest-html`, `pytest-cov`, `coverage`), and installs any that are missing.
- Ensures the `src` layout is on `PYTHONPATH` before running tests.
- Invokes `pytest --html=report.html --self-contained-html`, producing `tests\report.html` and also printing failures, if any, to the console.
- Runs `coverage html`, generating the HTML coverage report under `tests\htmlcov` (entry point `tests\htmlcov\index.html`).
- Launches both HTML reports in the default browser automatically and then returns control to the shell.

Running the batch file multiple times overwrites the existing reports; clean them manually if you need a fresh workspace.
