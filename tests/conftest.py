from __future__ import annotations
"""Test configuration helpers for integration tests.

Fixtures (name -> provides):
- test_data_root -> local cache under tests/test_files
- nd2_base_dir -> directory of sample ND2 files under tests/test_files/nd2_files
- nd2_files -> list of all detected .nd2 files; skips when directory is empty
- nd2_with_result_path -> path to ND2 sample that includes matching result .h5
- prepare_conversion_output_dir -> cleaned output directory reused by conversion tests

Other responsibilities:
- Mirror ND2 sample data from REMOTE_ROOT into the local cache
- Download from S3 if network share unavailable
- Make src/ importable before collection starts
- Parametrize tests requesting nd2_path with every discovered .nd2 file (auto-skip when none exist)
"""

import os
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pytest

LOCAL_ROOT = Path(__file__).parent / "test_files"
REMOTE_ROOT = Path(r"\\server\home\lukas.jirusek\limnd2_test_files")
S3_TEST_DATA_URL = "https://lim-public-af010c85-0d3e-4156-9378-5adc1bbef7b3.s3.eu-central-1.amazonaws.com/LimNd2TestFiles/nd2_test_images_from_talley.7z"


def copy_test_files(remote_root: Path, local_root: Path) -> None:
    """Copy files and directories from the remote root into the local cache."""
    local_root.mkdir(parents=True, exist_ok=True)

    for root, _dirs, files in os.walk(remote_root):
        rel_path = Path(root).relative_to(remote_root)
        dest_dir = local_root / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            src_file = Path(root) / name
            dst_file = dest_dir / name

            if dst_file.exists():
                try:
                    if (
                        src_file.stat().st_size == dst_file.stat().st_size
                        and int(src_file.stat().st_mtime) == int(dst_file.stat().st_mtime)
                    ):
                        continue
                except FileNotFoundError:
                    pass

            shutil.copy2(src_file, dst_file)


def _download_and_extract_from_s3(local_root: Path) -> bool:
    """Download test files from S3 and extract them.

    Returns True if successful, False otherwise.
    """
    try:
        import py7zr
    except ImportError:
        print("py7zr not installed. Skipping S3 download. Install with: pip install py7zr")
        return False

    archive_path = local_root / "nd2_test_images.7z"

    try:
        # Download the 7z archive
        print(f"Downloading test data from S3: {S3_TEST_DATA_URL}")
        urlretrieve(S3_TEST_DATA_URL, archive_path)

        # Extract to nd2_files directory
        extract_dir = local_root / "nd2_files"
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting test data to {extract_dir}")
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=extract_dir)

        # Clean up archive
        archive_path.unlink()

        print("Test data successfully downloaded and extracted from S3")
        return True

    except Exception as e:
        print(f"Failed to download/extract from S3: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return False


def _get_remote_root(pytestconfig: pytest.Config | None = None) -> Path:
    env_value = os.getenv("LIMND2_TEST_DATA_ROOT")
    if env_value:
        return Path(env_value)
    return REMOTE_ROOT


def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Function called before tests are run

    It ensures that the src/ directory is in sys.path and
    that test files are copied from the remote location if needed.

    Test data acquisition strategy (in order):
    1. Try network share (REMOTE_ROOT or LIMND2_TEST_DATA_ROOT env variable)
    2. Try downloading from S3 if network share unavailable
    3. Skip tests if both fail (no test data available)

    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    local_root = LOCAL_ROOT
    nd2_files_dir = local_root / "nd2_files"
    remote_root = _get_remote_root(session.config)

    # Check if we already have test files
    if nd2_files_dir.exists() and list(nd2_files_dir.rglob("*.nd2")):
        if remote_root.exists():
            print(f"Syncing test data from network share: {remote_root}")
            copy_test_files(remote_root, local_root)
        else:
            print(f"Test data already present in {nd2_files_dir}")
        return

    # Strategy 1: Try network share
    if remote_root.exists():
        print(f"Copying test data from network share: {remote_root}")
        copy_test_files(remote_root, local_root)
        return

    # Strategy 2: Try S3 download
    print(f"Network share not accessible: {remote_root}")
    print("Attempting to download test data from S3...")
    if _download_and_extract_from_s3(local_root):
        return

    # Strategy 3: No test data available - tests will be skipped
    print("No test data available. Tests requiring .nd2 files will be skipped.")
    print("To run these tests, either:")
    print(f"  1. Ensure access to network share: {remote_root}")
    print(f"  2. Install py7zr (pip install py7zr) for S3 download")
    print(f"  3. Set LIMND2_TEST_DATA_ROOT environment variable to a directory with .nd2 files")


def _list_local_nd2_files(base: Path) -> tuple[Path, ...]:
    if not base.exists():
        return ()
    return tuple(sorted(base.rglob("*.nd2")))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "nd2_path" not in metafunc.fixturenames:
        return

    base_dir = LOCAL_ROOT / "nd2_files"
    files = list(_list_local_nd2_files(base_dir))
    if not files:
        metafunc.parametrize(
            "nd2_path",
            [pytest.param(None, marks=pytest.mark.skip(reason=f"No .nd2 files found under {base_dir}"))],
        )
    else:
        metafunc.parametrize(
            "nd2_path",
            files,
            ids=[path.name for path in files],
        )

@pytest.fixture(scope="session")
def test_data_root(pytestconfig: pytest.Config) -> Path:
    return LOCAL_ROOT


@pytest.fixture(scope="session")
def nd2_base_dir(test_data_root: Path) -> Path:
    """Directory that contains sample ND2 files for tests."""
    return test_data_root / "nd2_files"


@pytest.fixture(scope="session")
def nd2_files(nd2_base_dir: Path) -> list[Path]:
    files = list(_list_local_nd2_files(nd2_base_dir)) if nd2_base_dir.exists() else []
    if not files:
        pytest.skip(f"No .nd2 files found under {nd2_base_dir}")
    return files


@pytest.fixture(scope="session")
def nd2_with_result_path(test_data_root: Path) -> Path:
    """Path to ND2 file that includes precomputed results."""
    path = test_data_root / "nd2_with_result" / "nd2_with_result.nd2"
    if not path.exists():
        pytest.skip(f"Expected ND2-with-result file missing at {path}")
    return path


@pytest.fixture(scope="session", autouse=True)
def prepare_conversion_output_dir(test_data_root: Path) -> Path:
    """Ensure conversion output dir under tests/test_files is clean for the session."""
    out_dir = test_data_root / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir





