# limnd2 package

!!! warning
    This Python package is not yet available for the public, both the package and the documentation is still being worked on.

`.nd2` (Nikon NIS Elements) file reader and writer in Python.

## GitHub repository

Source code for this library can be found [here](https://github.com/Laboratory-Imaging/limnd2)

## Installation

### Prerequisites

limnd2 package requires following packages (also listed in `requirements.txt`) to work correctly:

- python>=3.12.0
- numpy

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

#### Windows

```powershell
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
python -m venv env
env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
code .
```

for building and uploading to private Pypi

```powershell
pip install build setuptools twine
python -m build
twine upload -r local dist\*
```

## API reference

Here are the most important files in this library and an overview of what they contain:

- [nd2.py](nd2.md) - contains classes for opening ND2 files for reading and writing
- [attributes.py](attributes.md) - contains data structures about image attributes (width, height, component count, sequence count, ...)
- [experiment.py](experiment.md) - contains data structures about experiment loops (timeloop, z-stack, multipoint, ...)
    - [experiment_factory.py](experiment_factory.md) - contains helpers for creating experiment data structure
- [metadata.py](metadata.md) - contains data structures about image attributes (width, height, component count, sequence count, ...)
