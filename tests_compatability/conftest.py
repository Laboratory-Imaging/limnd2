from pathlib import Path
import sys
import zipfile
from urllib.request import urlretrieve

import pytest

# Ensure repo src/ is used for limnd2
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Ensure compat shim package is used instead of any installed nd2
SHIM_ROOT = Path(__file__).parent
if str(SHIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SHIM_ROOT))

# If nd2/limnd2 were imported before we adjusted sys.path, drop them to
# guarantee tests use the local shim + repo sources.
for _mod in list(sys.modules):
    if _mod == "nd2" or _mod.startswith("nd2.") or _mod == "limnd2" or _mod.startswith("limnd2."):
        sys.modules.pop(_mod, None)

from nd2._util import is_new_format

DATA = Path(__file__).parent / "data"
ND2_DROPBOX_URL = "https://www.dropbox.com/scl/fi/behxmt6ps2s5lp3k5qpjp/nd2_test_data.zip?rlkey=u8ra0s99xxuyan73669jwoq7f&dl=1"
S3_TALLEY_7Z_URL = "https://lim-public-af010c85-0d3e-4156-9378-5adc1bbef7b3.s3.eu-central-1.amazonaws.com/LimNd2TestFiles/nd2_test_images_from_talley.7z"


def _has_nd2_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.nd2"))


def _download_zip_to_data(url: str, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "nd2_test_data.zip"
    try:
        urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(dest_dir)
        archive_path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _download_7z_to_data(url: str, dest_dir: Path) -> bool:
    try:
        import py7zr  # type: ignore
    except ImportError:
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "nd2_test_data.7z"
    try:
        urlretrieve(url, archive_path)
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=dest_dir)
        archive_path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _ensure_data() -> None:
    if _has_nd2_files(DATA):
        return
    # Try Dropbox (nd2 official test data)
    if _download_zip_to_data(ND2_DROPBOX_URL, DATA) and _has_nd2_files(DATA):
        return
    # Try AWS S3 Talley dataset if available
    if _download_7z_to_data(S3_TALLEY_7Z_URL, DATA) and _has_nd2_files(DATA):
        return


_ensure_data()

_DATA_AVAILABLE = _has_nd2_files(DATA)
_SKIP_REASON = (
    f"No ND2 test data available under {DATA}. "
    "Tried Dropbox and S3 downloads."
)


def _skip_param():
    return pytest.param(
        DATA / "MISSING.nd2",
        marks=pytest.mark.skip(reason=_SKIP_REASON),
        id="no-nd2-data",
    )


def pytest_collection_modifyitems(config, items):
    if _DATA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        item.add_marker(skip)
MAX_FILES = None
if _DATA_AVAILABLE:
    ALL = sorted(
        (x for x in DATA.glob("*.nd2") if not x.name.startswith(".")),
        key=lambda x: x.stat().st_size,
    )[:MAX_FILES]
    NEW: list[Path] = []
    OLD: list[Path] = []
    for x in ALL:
        NEW.append(x) if is_new_format(str(x)) else OLD.append(x)
    SMALL_ND2S_PARAMS = [*ALL[:40], DATA / "jonas_control002.nd2"]
else:
    ALL = [_skip_param()]
    NEW = [_skip_param()]
    OLD = [_skip_param()]
    SMALL_ND2S_PARAMS = [_skip_param()]

SINGLE = DATA / "dims_t3c2y32x32.nd2"


@pytest.fixture()
def single_nd2():
    return SINGLE


@pytest.fixture(params=SMALL_ND2S_PARAMS, ids=lambda x: x.name)
def small_nd2s(request) -> Path:
    return request.param


@pytest.fixture(params=ALL, ids=lambda x: x.name)
def any_nd2(request):
    return request.param


@pytest.fixture(params=NEW, ids=lambda x: f"{x.name}")
def new_nd2(request):
    return request.param


@pytest.fixture(params=OLD, ids=lambda x: x.name)
def old_nd2(request):
    return request.param


@pytest.fixture(autouse=True)
def _assert_no_files_left_open():
    try:
        import psutil
    except Exception:
        yield
        return
    files_before = {p for p in psutil.Process().open_files() if p.path.endswith("nd2")}
    yield
    files_after = {p for p in psutil.Process().open_files() if p.path.endswith("nd2")}
    assert files_before == files_after == set()
