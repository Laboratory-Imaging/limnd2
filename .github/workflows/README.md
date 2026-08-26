# GitHub Workflows

This directory contains the repository's GitHub Actions workflows.

## `ci.yml`

Purpose:
- Run the default fast validation suite for regular development.
- Validate packaging and wheel installation.
- Validate documentation build integrity.

Triggers:
- Push to `main`
- Pull request targeting `main`

What it does:
- Runs `pytest -m "not slow"` on:
  - `ubuntu-latest` with Python `3.11`, `3.12`, `3.13`
  - `windows-latest` with Python `3.11`, `3.12`, `3.13`
- Prefetches ND2 test fixtures used by the default test suite.
- Builds source and wheel distributions with `uv build`.
- Smoke-tests the built wheel in a clean environment.
- Builds docs with `mkdocs build --strict`.

Outputs:
- Uploaded artifact: `ci-dist`
  - contains files from `dist/*`

## `release.yml`

Purpose:
- Validate and publish tagged releases.

Triggers:
- Push of a tag matching `v*`

What it does:
- Verifies the tagged commit is contained in `main`.
- Derives the package version from the pushed Git tag.
- Runs the fast/default pytest suite.
- Builds distributions once.
- Verifies built artifact filenames match the tag-derived version.
- Smoke-tests the built wheel in a clean environment.
- Builds curated release notes from `CHANGELOG.toml` when available.
- Creates a GitHub Release.
- Optionally publishes the same artifacts to the AWS-backed package index.

Outputs:
- Uploaded artifact: `release-dist`
  - contains files from `dist/*`
- Uploaded artifact: `release-body`
  - uploaded only when curated release notes were generated
- GitHub Release
  - created for the pushed tag
- Package publish side effect
  - uploads to `https://pypi.laboratory-imaging.com`
  - only when repository variable `ENABLE_AWS_PUBLISH == 'true'`
  - requires secrets `AWS_PYPI_USERNAME` and `AWS_PYPI_PASSWORD`

## `compatibility.yml`

Purpose:
- Run the broader compatibility/parity test suite that exercises the `nd2`-style compatibility layer.

Triggers:
- Manual run via `workflow_dispatch`
- Weekly schedule
  - cron: `0 3 * * 1` (Mondays at 03:00 UTC)

What it does:
- Installs development dependencies.
- Runs `pytest tests_compatability`.

Outputs:
- No uploaded artifacts currently.
- Workflow status is expected to reflect real compatibility failures.

## `mypy_check.yml`

Purpose:
- Run advisory static type checking without blocking development.

Triggers:
- Push to `main` when Python source or `pyproject.toml` changes
- Pull request targeting `main` when Python source or `pyproject.toml` changes

What it does:
- Installs development dependencies.
- Runs `mypy src/limnd2`.
- Captures output to a log file even if mypy reports errors.

Outputs:
- Uploaded artifact: `mypy-report`
  - contains `mypy.log`

## `deploy_docs.yml`

Purpose:
- Build and publish documentation to the external Pages repository.

Triggers:
- Push to `main`

What it does:
- Builds docs with `mkdocs build --strict`.
- Clones `Laboratory-Imaging/Laboratory-Imaging.github.io`.
- Replaces `limnd2/docs` in that repository with the newly built docs.
- Commits and pushes the update.

Outputs:
- No uploaded artifacts currently.
- Deployment side effect:
  - pushes documentation changes to `Laboratory-Imaging/Laboratory-Imaging.github.io`
- Requires secret:
  - `DOCS_PAGES_DEPLOY_KEY`
