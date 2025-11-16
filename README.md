# limnd2

A Python library for reading and writing `.nd2` files produced by Nikon NIS-Elements Software.

Built upon [tlambert03/nd2](https://github.com/tlambert03/nd2) with a compatible drop-in interface, adding write capabilities and extended metadata support.

## Documentation
Documentation is available [here](https://laboratory-imaging.github.io/limnd2/docs/).

## Installation

### Prerequisites

limnd2 package requires the following core dependencies:

- python>=3.9
- numpy
- ome_types
- tifffile
- imagecodecs
- Pillow

#### Optional Dependencies

For working with results and analytics features, install the optional `results` extras:

- h5py
- pandas

Install with: `pip install limnd2[results]` or `uv pip install limnd2[results]`

### Installation scripts

This package and its prerequisites can be installed running following commands in Powershell / shell.

#### Windows

```powershell
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/Laboratory-Imaging/Laboratory-Imaging.github.io/refs/heads/main/limnd2/setup_limnd2.bat' -OutFile 'setup_limnd2.bat'; & '.\setup_limnd2.bat'; Remove-Item 'setup_limnd2.bat'"
```

#### Linux / MacOS   // TODO test script when repo is public

```sh
curl -O https://raw.githubusercontent.com/Laboratory-Imaging/Laboratory-Imaging.github.io/refs/heads/main/limnd2/setup_limnd2.sh && chmod +x setup_limnd2.sh && ./setup_limnd2.sh && rm ./setup.sh
```


### Manual Installation

This project uses `pyproject.toml` for dependency management and can be installed with either `pip` or `uv`.

#### Using uv (recommended)

```powershell
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
uv venv
# Windows
.venv\Scripts\activate
# Linux/MacOS
# source .venv/bin/activate
uv pip install -e ".[dev]"
code .
```

#### Using pip

```powershell
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
python -m venv env
# Windows
env\Scripts\activate
# Linux/MacOS
# source env/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
code .
```

### Building and Publishing

#### Build the Package

First, build the distribution packages:

```powershell
# Using uv (recommended)
uv build

# Or using pip/build
python -m build
```

This creates `.whl` and `.tar.gz` files in the `dist/` directory.

#### Publishing to PyPI Servers

The project is configured with multiple PyPI server indices in `pyproject.toml`:
- `pypi`: Public PyPI (https://pypi.org)
- `local`: Internal server at http://gaexec:9500
- `aws-pypi`: AWS server at http://18.184.201.85:8080

**Option 1: Using `uv publish` (recommended)**

```powershell
# Publish to local server (no authentication)
uv publish --publish-url http://gaexec:9500 --trusted-publishing never --username "-" --password "-" dist/*

# Publish to AWS server (with authentication via environment variables)
$env:UV_PUBLISH_USERNAME = "your-username"
$env:UV_PUBLISH_PASSWORD = "your-password"
uv publish --publish-url http://18.184.201.85:8080 dist/*

# Or pass credentials directly
uv publish --publish-url http://18.184.201.85:8080 --username "your-username" --password "your-password" dist/*
```

**Option 2: Using `twine` (traditional method)**

First, configure credentials in `~/.pypirc` (Linux/Mac) or `%USERPROFILE%\.pypirc` (Windows):

```ini
[distutils]
index-servers =
    local
    aws-pypi

[local]
repository = http://gaexec:9500
username = -
password = -

[aws-pypi]
repository = http://18.184.201.85:8080
username = your-username
password = your-password
```

Then upload:

```powershell
# Upload to local server
twine upload -r local dist/*

# Upload to AWS server
twine upload -r aws-pypi dist/*
```

> [!WARNING]
> Never commit the `.pypirc` file to version control as it contains credentials. It's already ignored in `.gitignore` by default.

> [!TIP]
> For security, use environment variables for credentials:
> ```powershell
> # Windows PowerShell
> $env:TWINE_USERNAME = "your-username"
> $env:TWINE_PASSWORD = "your-password"
> twine upload -r aws-pypi dist/*
> ```

### Documentation Preview

```sh
mkdocs serve
```

## Development

### Running Tests

This project uses pytest for testing. After installing the dev dependencies (see Manual Installation above), you can run tests in several ways:

#### Test Data Acquisition

Many tests require `.nd2` sample files. The test suite automatically acquires test data using a three-tier fallback strategy:

1. **Network share** (intranet): Copies from `\\server\home\lukas.jirusek\limnd2_test_files` if accessible
2. **S3 download** (public): Downloads and extracts from AWS S3 if network share unavailable (requires `py7zr`)
3. **Skip tests**: Tests requiring sample files are automatically skipped if neither source is available

The test data is automatically downloaded on first test run and cached in `tests/test_files/nd2_files/`. You can also manually set the `LIMND2_TEST_DATA_ROOT` environment variable to point to a custom directory containing `.nd2` files.

#### Using VS Code Testing UI

Once you've installed the dev dependencies with `uv pip install -e ".[dev]"` or `pip install -e ".[dev]"`, you can use VS Code's built-in Testing UI:

1. Click the Testing icon in the Activity Bar (left sidebar)
2. VS Code will automatically discover tests in the `tests/` directory
3. On first run, test data will be automatically downloaded from S3 (if needed)
4. Run individual tests or the entire test suite from the UI

> [!NOTE]
> If you get an error "No module named pytest" when using VS Code's Testing button, make sure you've installed the dev dependencies first.

> [!TIP]
> If VS Code shows "not found" errors for parametrized tests (tests with `[parameter]` suffix), try refreshing the test discovery:
> - Command Palette → `Testing: Refresh Tests`
> - Or click individual parametrized test cases instead of the parent test node

#### Using Command Line

```powershell
# Run all tests with coverage (downloads test data automatically on first run)
pytest

# Run tests with HTML report (Windows)
tests\run_tests.bat

# Run specific test file
pytest tests/test_reader_base.py

# Skip slow tests (marked with @pytest.mark.slow)
pytest -m "not slow"
```

### Running MyPy static type checker

> [!NOTE]
> MyPy is not required for running the package, it is only used for static type checking.
> You can install MyPy with `pip install mypy`.

To run MyPy static type checker, run the following command in the root directory of the repository:

```sh
mypy .
```

MyPy is also run automatically with each commit, you can download latest MyPy report by navigating to [Workflow action for MyPy](https://github.com/Laboratory-Imaging/limnd2/actions/workflows/mypy_check.yml), selecting latest workflow run and by downloading the latest report from the "Artifacts" section at the bottom of the page.

> [!NOTE]
> Those reports are only available for 90 days since the workflow run.
