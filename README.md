# limnd2

## prerequisites

- min Python version 3.12.0

```bat
git clone https://gitlab.com/ondrejprazsky/limnd2.git
cd limnd2
python -m venv env
env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install build setuptools twine
pip install --editable .
code .
```

## to generate package

Create `~/.pypirc`

```txt
[distutils]
  index-servers =
    pypi
    local

[pypi]
  username: <your_pypi_username>
  password: <your_pypi_passwd>

[local]
  repository: http://gaexec:9500
  username:
  password:
```

Build and upload

```bat
python -m build
twine upload -r local dist\*
```

## to install package

```bat
pip install --trusted-host 192.168.11.200 --index-url http://192.168.11.200:9500/simple/ limnd2
```

or create 
- `$HOME/.config/pip/pip.conf` (on Linux)
- `%APPDATA%\pip\pip.ini` (on Windows)

with the following content:

```toml
[global]
extra-index-url = http://192.168.11.200:9500/simple/

[install]
trusted-host = 192.168.11.200
```
