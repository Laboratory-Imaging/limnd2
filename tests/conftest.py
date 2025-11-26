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
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import pytest

LOCAL_ROOT = Path(__file__).parent / "test_files"
REMOTE_ROOT = Path(r"\\teak\devel\limnd2stk\limnd2_test_files")
S3_TEST_DATA_TALLEY_URL = (
    "https://lim-public-af010c85-0d3e-4156-9378-5adc1bbef7b3.s3.eu-central-1.amazonaws.com/"
    "LimNd2TestFiles/nd2_test_images_from_talley.7z"
)
# Legacy Talley dataset (unused but kept for reference)
S3_TEST_DATA_ZIP_URL = "https://lim-public-af010c85-0d3e-4156-9378-5adc1bbef7b3.s3.eu-central-1.amazonaws.com/LimNd2TestFiles/limnd2_test_files.zip"
ZIP_ROOT_DIRNAME = "limnd2_test_files"


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


def _local_test_data_present(local_root: Path) -> bool:
    nd2_dir = local_root / "nd2_files"
    has_nd2 = nd2_dir.exists() and any(nd2_dir.rglob("*.nd2"))

    result_file = local_root / "nd2_with_result" / "nd2_with_result.nd2"
    has_results = result_file.exists()

    conv_dir = local_root / "conversion"
    has_conversion = conv_dir.exists() and any(conv_dir.iterdir())

    return has_nd2 and has_results and has_conversion


def _download_with_progress(url: str, destination: Path, *, description: str = "Downloading") -> None:
    """Download a file from URL showing a simple progress percentage."""
    print(f"{description}: {url}")
    last_percent = {"value": -1}

    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        percent = int(block_count * block_size * 100 / total_size)
        if percent != last_percent["value"]:
            sys.stdout.write(f"\r  {percent}%")
            sys.stdout.flush()
            last_percent["value"] = percent

    urlretrieve(url, destination, reporthook=_hook)
    sys.stdout.write("\r  100%\n")


def _locate_extracted_zip_root(extracted_root: Path) -> Path | None:
    preferred = extracted_root / ZIP_ROOT_DIRNAME
    if preferred.exists():
        return preferred

    dirs = [p for p in extracted_root.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]

    return extracted_root if any(extracted_root.iterdir()) else None


def _download_and_extract_zip_archive(
    local_root: Path,
    *,
    source_zip: Path | None = None,
    url: str | None = None,
) -> bool:
    """Download (or reuse) a ZIP archive containing ND2 fixtures and copy them into local_root."""
    try:
        if source_zip is not None:
            archive_path = Path(source_zip)
            if not archive_path.exists():
                print(f"Provided ZIP archive not found: {archive_path}")
                return False
            print(f"Using existing ZIP archive: {archive_path}")
            with TemporaryDirectory() as tmpdir:
                extract_dir = Path(tmpdir) / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive_path, "r") as archive:
                    archive.extractall(extract_dir)
                src_root = _locate_extracted_zip_root(extract_dir)
                if src_root is None:
                    print("Failed to locate extracted root directory inside ZIP archive.")
                    return False
                copy_test_files(src_root, local_root)
            return True

        download_url = url or S3_TEST_DATA_ZIP_URL
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            archive_path = tmpdir_path / "limnd2_test_files.zip"
            _download_with_progress(download_url, archive_path, description="Downloading test data ZIP from S3")

            extract_dir = tmpdir_path / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting ZIP archive to {extract_dir}")
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_dir)

            src_root = _locate_extracted_zip_root(extract_dir)
            if src_root is None:
                print("Failed to locate extracted root directory inside ZIP archive.")
                return False

            print(f"Copying extracted test data from {src_root} to {local_root}")
            copy_test_files(src_root, local_root)
            return True

    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to download or extract ZIP archive: {exc}")
        return False


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
        print(f"Downloading test data from S3: {S3_TEST_DATA_TALLEY_URL}")
        urlretrieve(S3_TEST_DATA_TALLEY_URL, archive_path)

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
    remote_root = _get_remote_root(session.config)
    has_local_data = _local_test_data_present(local_root)

    # Strategy 1: Try network share
    if remote_root.exists():
        action = "Syncing" if has_local_data else "Copying"
        print(f"{action} test data from network share: {remote_root}")
        copy_test_files(remote_root, local_root)
        return

    print(f"Network share not accessible: {remote_root}")
    if has_local_data:
        print(f"Local test data already present at {local_root}")
        return

    # Strategy 2: Download ZIP archive from S3 (only when data missing)
    print("Attempting to download ZIP test data from S3...")
    if _download_and_extract_zip_archive(local_root, url=S3_TEST_DATA_ZIP_URL):
        return

    # Strategy 3: No test data available - tests will be skipped
    print("No test data available. Tests requiring .nd2 files will be skipped.")
    print("To run these tests, either:")
    print(f"  1. Ensure access to network share: {remote_root}")
    print(f"  2. Download limnd2_test_files.zip from: {S3_TEST_DATA_ZIP_URL}")
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





