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
pip install --upgrade build setuptools twine
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
pip install --trusted-host gaexec --index-url http://gaexec:9500/simple/ limnd2
```

or create `~/.pip/pip.conf` with the following content:
```toml
[global]
extra-index-url = http://gaexec:9500/simple/

[install]
trusted-host =
    gaexec
```
