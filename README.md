# limnd2

`.nd2` (Nikon NIS Elements) file reader and writer in Python.

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

#### Using uv (recommended)

```powershell
uv build
uv publish --publish-url http://gaexec:9500 --trusted-publishing never --username "-" --password "-" dist/*
```

#### Using pip/twine

```powershell
pip install build setuptools twine
python -m build
twine upload -r local dist\*
```

### Documentation Preview

```sh
mkdocs serve
```

## Running MyPy static type checker

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
