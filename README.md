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
pip install --upgrade setuptools
pip install --upgrade build
pip install --upgrade twine
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

pip install --trusted-host gaexec --index-url http://gaexec:9500/simple/ limnd2
