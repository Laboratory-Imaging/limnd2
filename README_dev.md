# limnd2 development guide

This file is for maintainers and contributors working from a local clone.

For package users, see [README.md](README.md).

## Dependencies and extras

Dependency management is defined in `pyproject.toml`.

- Base runtime: `python>=3.9`, `numpy`, `ome_types`
- Optional runtime extras:
  - `results` (`h5py`, `pandas`)
  - `commonff` (`Pillow`, `tifffile`, `zarr`)
  - `legacy` (`imagecodecs`)
- Full dev environment: `.[dev]` (runtime extras + docs/test/typecheck/build tooling)

## Local development setup

### Using uv (recommended)

```sh
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
uv venv
# Windows: .venv\Scripts\activate
# Linux/MacOS: source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using pip

```sh
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
python -m venv env
# Windows: env\Scripts\activate
# Linux/MacOS: source env/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Build and publish

### Release checklist

1. Bump `version` in `pyproject.toml` under `[project]`.
2. Add or update release entry in `CHANGELOG.toml` for that version.
3. Update version text in user-facing docs if present (for example `README.md` and `docs/index.md` warning blocks).
4. Build distributions (`uv build` or `python -m build`).
5. Publish `dist/*` to required indexes (`local` / `aws-pypi`) using commands below.
6. Apply tagging policy (see below).

### Release changelog (`CHANGELOG.toml`)

GitHub release notes are generated from `CHANGELOG.toml` by the release workflow.

Format:

```toml
[releases."0.3.1"]
title = "Short release title"
message = "One short summary paragraph for this release."
prerelease = false
changes = [
  "Describe the first user-visible change.",
  "Describe the second user-visible change.",
]
```

### Tagging policy

- Pre-`1.0.0` (temporary internal phase):
  - We may publish multiple internal builds while staying on `0.3.0`.
  - GitHub release tags are optional during this phase.
- From `1.0.0` onward (strict policy):
  - One package version = one immutable git tag = one GitHub Release.
  - Never move or reuse a published version tag.
  - If a mistake is found after release, bump to a new version and tag that new commit.

Recommended commands:

```sh
git tag v1.0.0
git push origin v1.0.0
```

Then create a GitHub Release for the same tag (`v1.0.0`).

> [!WARNING]
> Pre-release retagging is allowed only during the temporary pre-`1.0.0` phase.
> Do not retag any published `1.0.0+` version.
>
> If you must retag before `1.0.0`:
> ```sh
> # delete local tag
> git tag -d v0.3.0
>
> # delete remote tag
> git push origin :refs/tags/v0.3.0
>
> # recreate and push tag
> git tag v0.3.0
> git push origin v0.3.0
> ```

### Build distributions

```sh
uv build
# or
python -m build
```

Artifacts are created in `dist/` (`.whl` and `.tar.gz`).

### Publish to package index

The project is configured with multiple indexes in `pyproject.toml`:

- `pypi`: https://pypi.org
- `local`: http://gaexec:9500
- `aws-pypi`: https://pypi.lim-dev.xyz

Using `uv publish`:

```powershell
# local (no auth)
uv publish --publish-url http://gaexec:9500 --trusted-publishing never --username "-" --password "-" dist/*

# aws index
uv publish --publish-url https://pypi.lim-dev.xyz --username "your-username" --password "your-password" dist/*
```

Using `twine`:

```powershell
twine upload -r local dist/*
twine upload -r aws-pypi dist/*
```

> [!WARNING]
> Never commit real credentials in `.pypirc`. `gitignore` only blocks new untracked files; if `.pypirc` is already tracked, untrack it with `git rm --cached .pypirc`.

## Documentation

Preview docs locally:

```sh
mkdocs serve
```

## Testing

### Test data acquisition

Many tests require `.nd2` samples. The suite attempts:

1. network share sync (default: `\\teak\devel\limnd2stk\limnd2_test_files`, or `LIMND2_TEST_DATA_ROOT` if set)
2. S3 ZIP download fallback
3. skip tests requiring sample data

You can also set `LIMND2_TEST_DATA_ROOT` to a local dataset path.

### Run tests

```sh
# default test run (with coverage per pyproject addopts)
pytest

# specific test file
pytest tests/test_suites/limnd2/test_reader_base.py

# skip slow tests
pytest -m "not slow"
```

Windows helper script:

```bat
tests\run_tests.bat
```

## Static type checking

```sh
mypy .
# optional
pyright
```

Workflow reports are also published in GitHub Actions artifacts:

- https://github.com/Laboratory-Imaging/limnd2/actions/workflows/mypy_check.yml

Windows batch helpers:

- `tests\static_type_check\run_mypy_check.bat`
- `tests\static_type_check\run_pyright_check.bat`
