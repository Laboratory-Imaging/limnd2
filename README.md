# limnd2

`.nd2` (Nikon NIS Elements) file reader and writer in Python.

## Documentation
Documentation is available [here](https://laboratory-imaging.github.io/limnd2/docs/).

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
