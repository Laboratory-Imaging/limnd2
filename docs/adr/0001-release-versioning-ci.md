# ADR 0001: Release, Versioning, and CI Strategy

## Status

Accepted.

## Context

`limnd2` is a Python library for reading, writing, converting, and exporting
Nikon ND2 microscopy files. The project is moving from an internal pre-release
workflow toward a public, repeatable release process.

The current project state has several release and CI gaps:

- Package version is statically defined in `pyproject.toml`.
- Tag `v0.3.0` exists, but current `main` contains later commits.
- The release workflow force-updates the tag matching the current package
  version on pushes to `main`.
- Package publishing is manually triggered.
- There is no required PR workflow that runs the default `pytest` suite.
- Type checking is advisory and uploads a log artifact.
- Python support metadata is inconsistent: some files refer to Python 3.10,
  while `pyproject.toml` currently requires Python 3.11 or newer.

These issues make it possible for Git tags, package versions, GitHub Releases,
and published artifacts to drift apart.

## Decision

Use trunk-based development with immutable release tags and tag-derived package
versions.

### Branching

`main` is the primary integration branch and must remain releasable. Development
work should happen on short-lived branches and merge through pull requests.

Recommended branch names:

- `feature/<issue>-short-description`
- `fix/<issue>-short-description`
- `docs/<short-description>`
- `chore/<short-description>`
- `release/vX.Y.Z`
- `hotfix/vX.Y.Z`
- `maint/X.Y` only if maintaining older release lines becomes necessary

Prefer squash merges for feature and fix pull requests. Delete branches after
merge. Do not force-push to protected branches.

### Versioning

Git tags are the source of truth for package versions. The project should use
`setuptools_scm` to derive the package version from immutable tags.

Release tags use the `v*` form:

- `v0.4.0`
- `v1.0.0`
- `v1.0.0rc1`

The produced package version must match the tag without the leading `v`.

Examples:

- `v0.4.0` produces package version `0.4.0`.
- `v1.0.0rc1` produces package version `1.0.0rc1`.

Release tags should be annotated:

```sh
git tag -a v0.4.0 -m "Release v0.4.0"
git push origin v0.4.0
```

Published tags must never be moved or reused. If a published release is wrong,
fix it in a new version.

### Python Support

The minimum supported Python version is 3.10.

Package metadata should declare:

```toml
requires-python = ">=3.10"
```

Classifiers should cover Python 3.10 through 3.14. CI should test a pragmatic
matrix of Python 3.10, 3.12, and 3.14.

### Package Publishing

Packages are published only from version tags matching `v*`.

Packages are published to the Laboratory Imaging public package index:

```text
https://pypi.laboratory-imaging.com
```

Packages are not published to the official community PyPI service at
`https://pypi.org`.

Release candidates are published when tagged, for example `v1.0.0rc1`.

Development builds from untagged commits are not published.

### Release Workflow

A single tag-triggered release workflow should:

1. Run on tags matching `v*`.
2. Validate that the tag format is supported.
3. Validate that the package version produced by `setuptools_scm` matches the
   tag.
4. Run the fast/default test suite.
5. Build the source distribution and wheel once.
6. Smoke-test the built wheel in a clean environment.
7. Create a GitHub Release for the tag.
8. Build release notes from `CHANGELOG.toml`.
9. Attach the exact built artifacts to the GitHub Release.
10. Publish the same artifacts to `https://pypi.laboratory-imaging.com`.

The current workflow that force-updates tags on pushes to `main` should be
removed.

### CI

Pull requests and pushes to `main` should run required fast checks:

- Install development dependencies.
- Run `pytest -m "not slow"`.
- Build source distribution and wheel.
- Install the built wheel in a clean environment.
- Smoke-test package import and CLI entry points.
- Build documentation with `mkdocs build --strict`.

The default required CI matrix should include Python 3.10, 3.12, and 3.14.
Linux and Windows should both be represented, because the project handles file
formats and paths that may behave differently across platforms.

### Type Checking

Type checking remains advisory initially.

The typecheck workflow should run `mypy src/limnd2`, upload its log as an
artifact, and avoid blocking merges until the project is clean enough for a
required check.

### Compatibility Tests

The `tests_compatability` suite should run manually and on a schedule. It should
not block every pull request or release initially because it depends on larger
datasets, optional dependencies, and parity behavior that may be slow or noisy.

Maintainers should run the compatibility workflow before important releases.

### Documentation

Documentation should build on pull requests and deploy on merges to `main`.

User-facing documentation should no longer state that changes may be released
without a version bump. Under this strategy, each published artifact is tied to
one immutable version tag.

### Changelog

Curated release notes live in `CHANGELOG.toml`.

Final releases should require a matching changelog entry. Release candidates may
fall back to generated notes if needed, but curated notes are preferred.

The first release after this migration should be `v0.4.0`.

## Consequences

Benefits:

- Git tags, GitHub Releases, package versions, and published artifacts align.
- Published versions become auditable and reproducible.
- Accidental retagging after publication is removed from the workflow.
- Release candidates use the same path as final releases.
- PR checks become a reliable merge gate without blocking on slower
  compatibility coverage.

Costs and follow-up work:

- Packaging metadata must migrate to dynamic versioning with `setuptools_scm`.
- Existing release and publish workflows must be replaced.
- GitHub repository settings must protect `main` and `v*` tags.
- Type checking remains a cleanup item before it can become required.
- Compatibility testing remains advisory until it is fast and stable enough to
  gate releases.

## Required Implementation Steps

1. Update `pyproject.toml` to use `setuptools_scm` and `dynamic = ["version"]`.
2. Set `requires-python = ">=3.10"` and update Python classifiers.
3. Replace release and publish workflows with a single tag-triggered
   `release.yml`.
4. Add required PR CI for tests, build, wheel smoke test, and docs build.
5. Convert the typecheck workflow into an explicitly advisory workflow.
6. Add manual/scheduled compatibility testing.
7. Update maintainer documentation with the release process and branch policy.
8. Update user-facing documentation to remove the "without a version bump"
   warning.
9. Add `CHANGELOG.toml` entry for `0.4.0`.
10. Configure GitHub branch and tag protection.
