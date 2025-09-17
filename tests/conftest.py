from __future__ import annotations

import os
import shutil
from pathlib import Path
import sys
import pytest


def copy_test_files(remote_root: Path, local_root: Path) -> None:
    """
    Copy all files and subdirectories from remote_root to local_root.
    Skips existing files with the same size & mtime for efficiency.
    """
    local_root.mkdir(parents=True, exist_ok=True)

    for root, _dirs, files in os.walk(remote_root):
        rel_path = Path(root).relative_to(remote_root)
        dest_dir = local_root / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            src_file = Path(root) / f
            dst_file = dest_dir / f

            if dst_file.exists():
                # Compare size and modified time to skip identical files
                try:
                    if (
                        src_file.stat().st_size == dst_file.stat().st_size
                        and int(src_file.stat().st_mtime) == int(dst_file.stat().st_mtime)
                    ):
                        continue
                except FileNotFoundError:
                    # If stat fails (race), fall through to copy
                    pass

            shutil.copy2(src_file, dst_file)


def _get_remote_root(pytestconfig: pytest.Config | None = None) -> Path:
    # Prefer env var if provided
    env = os.getenv("LIMND2_TEST_DATA_ROOT")
    if env:
        return Path(env)
    # Default path on the server
    return Path(r"\\server\home\lukas.jirusek\limnd2_test_files")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Pre-copy test data before test collection so module-level discovery works.

    Some test modules compute ND2 file lists at import time; copying here ensures
    `tests/test_files` exists before collection/import happens.
    """
    # Ensure src/ is importable when using a src layout
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    remote_root = _get_remote_root(session.config)
    local_root = Path(__file__).parent / "test_files"
    try:
        if remote_root.exists():
            copy_test_files(remote_root, local_root)
        else:
            # Do not skip here; individual tests will skip as needed
            pass
    except Exception:
        # Avoid hard failure during session start; tests will handle absence
        pass


@pytest.fixture(scope="session")
def test_data_root(pytestconfig: pytest.Config) -> Path:
    """
    Ensures test data from a remote location is mirrored under tests/test_files.
    """
    remote_root = _get_remote_root(pytestconfig)

    local_root = Path(__file__).parent / "test_files"

    if not remote_root.exists():
        pytest.skip(
            f"Test data not found at {remote_root}. "
        )

    copy_test_files(remote_root, local_root)
    return local_root


@pytest.fixture(scope="session", autouse=True)
def prepare_conversion_output_dir(test_data_root: Path) -> Path:
    """Ensure conversion output dir under tests/test_files is clean for the session."""
    out_dir = test_data_root / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
