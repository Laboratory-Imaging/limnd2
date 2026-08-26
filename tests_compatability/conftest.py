from pathlib import Path
from types import ModuleType
import sys
import zipfile
from urllib.request import urlretrieve

import pytest

# Ensure repo src/ is used for limnd2 compat sources
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# If nd2/limnd2 were imported before we adjusted sys.path, drop them to
# guarantee tests use the local repo sources.
for _mod in list(sys.modules):
    if _mod == "nd2" or _mod.startswith("nd2.") or _mod == "limnd2" or _mod.startswith("limnd2."):
        sys.modules.pop(_mod, None)

from importlib.metadata import PackageNotFoundError, version

from limnd2.nd2_compatability import _parse as compat_parse_pkg
from limnd2.nd2_compatability import _readers as compat_readers_pkg
from limnd2.nd2_compatability._parse import _chunk_decode as compat_chunk_decode
from limnd2.nd2_compatability._parse import _clx_lite as compat_clx_lite
from limnd2.nd2_compatability._parse import _clx_xml as compat_clx_xml
from limnd2.nd2_compatability._parse import _legacy_xml as compat_legacy_xml
from limnd2.nd2_compatability._parse import _parse as compat_parse_mod
from limnd2.nd2_compatability.nd2file import AXIS, ND2File
from limnd2.nd2_compatability import nd2file_types as compat_structures


def _install_nd2_runtime_shim() -> None:
    nd2_mod = ModuleType("nd2")
    nd2_mod.__path__ = []  # type: ignore[attr-defined]
    nd2_mod.AXIS = AXIS
    nd2_mod.ND2File = ND2File
    try:
        nd2_mod.__version__ = version("limnd2")
    except PackageNotFoundError:  # pragma: no cover
        nd2_mod.__version__ = "unknown"

    def imread(path, *args, **kwargs):
        with ND2File(path, *args, **kwargs) as f:
            return f.asarray()

    def nd2_to_tiff(
        source,
        dest,
        *,
        include_unstructured_metadata: bool = True,
        progress: bool = False,
        on_frame=None,
        modify_ome=None,
    ) -> None:
        import tifffile as tf

        dest_path = Path(dest).expanduser().resolve()
        output_ome = ".ome." in dest_path.name

        close_when_done = False
        if isinstance(source, (str, Path)):
            nd2f = ND2File(source)
            close_when_done = True
        else:
            nd2f = source
            if close_when_done := nd2f.closed:
                nd2f.open()

        try:
            sizes = dict(nd2f.sizes)
            n_positions = sizes.pop(AXIS.POSITION, 1)
            axes, shape = zip(*sizes.items())
            metadata = {"axes": "".join(axes).upper().replace(AXIS.UNKNOWN, "Q")}

            ome_xml: bytes | None = None
            if output_ome and not nd2f.is_legacy:
                ome = nd2f.ome_metadata()
                if modify_ome:
                    modify_ome(ome)
                ome_xml = ome.to_xml(exclude_unset=True).encode("utf-8")

            total_frames = nd2f._frame_count
            p_groups: dict[int, list[tuple[int, dict[str, int]]]] = {}
            for f_num, f_index in enumerate(nd2f.loop_indices):
                p_groups.setdefault(f_index.get(AXIS.POSITION, 0), []).append((f_num, f_index))

            def position_iter(position_index: int):
                for f_num, f_index in p_groups[position_index]:
                    if on_frame is not None:
                        on_frame(f_num, total_frames, f_index)
                    yield nd2f.read_frame(f_num)

            tf_ome = False if ome_xml else None
            pixel_size = nd2f.voxel_size().x
            photometric = tf.PHOTOMETRIC.RGB if nd2f.is_rgb else tf.PHOTOMETRIC.MINISBLACK
            with tf.TiffWriter(dest_path, bigtiff=True, ome=tf_ome) as tif:
                for position_index in range(n_positions):
                    tif.write(
                        iter(position_iter(position_index)),
                        shape=shape,
                        dtype=nd2f.dtype,
                        resolution=(1 / pixel_size, 1 / pixel_size),
                        resolutionunit=tf.RESUNIT.MICROMETER,
                        photometric=photometric,
                        metadata=metadata,
                        description=ome_xml,
                    )
        finally:
            if close_when_done:
                nd2f.close()

    util_mod = ModuleType("nd2._util")
    util_mod.AXIS = AXIS
    util_mod.is_supported_file = ND2File.is_supported_file
    util_mod.is_new_format = lambda path: Path(path).read_bytes()[:4] == b"\xda\xce\xbe\n"
    util_mod.is_legacy = lambda path: Path(path).read_bytes()[:4] == b"\x00\x00\x00\x0c"

    nd2_mod.imread = imread
    nd2_mod.nd2_to_tiff = nd2_to_tiff
    nd2_mod.is_supported_file = util_mod.is_supported_file
    nd2_mod.is_new_format = util_mod.is_new_format
    nd2_mod.is_legacy = util_mod.is_legacy
    nd2_mod.structures = compat_structures
    nd2_mod.rescue_nd2 = getattr(compat_chunk_decode, "rescue_nd2", None)

    sys.modules["nd2"] = nd2_mod
    sys.modules["nd2._util"] = util_mod
    sys.modules["nd2.structures"] = compat_structures
    sys.modules["nd2._parse"] = compat_parse_pkg
    sys.modules["nd2._parse._chunk_decode"] = compat_chunk_decode
    sys.modules["nd2._parse._clx_lite"] = compat_clx_lite
    sys.modules["nd2._parse._clx_xml"] = compat_clx_xml
    sys.modules["nd2._parse._legacy_xml"] = compat_legacy_xml
    sys.modules["nd2._parse._parse"] = compat_parse_mod
    sys.modules["nd2._readers"] = compat_readers_pkg


_install_nd2_runtime_shim()

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
