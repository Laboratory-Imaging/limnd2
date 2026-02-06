from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import limnd2
from limnd2.nd2file import ND2File
from limnd2.nd2file_types import Attributes, FrameMetadata, Metadata


DATA = Path(__file__).parent / "data"


def _find_sample(name: str) -> Path | None:
    matches = list(DATA.rglob(name)) if DATA.exists() else []
    return matches[0] if matches else None


@pytest.fixture(scope="session")
def nd2_samples() -> dict[str, Path]:
    names = [
        "dims_t3c2y32x32.nd2",
        "dims_c2y32x32.nd2",
        "dims_p4z5t3c2y32x32.nd2",
        "dims_rgb_t3p2c2z3x64y64.nd2",
        "cluster.nd2",
        "with_binary_and_rois.nd2",
    ]
    out: dict[str, Path] = {}
    for name in names:
        path = _find_sample(name)
        if path is not None:
            out[name] = path
    return out


@pytest.fixture(scope="session")
def small_nd2(nd2_samples: dict[str, Path]) -> Path:
    if "dims_t3c2y32x32.nd2" in nd2_samples:
        return nd2_samples["dims_t3c2y32x32.nd2"]
    if DATA.exists():
        candidates = sorted(DATA.glob("*.nd2"), key=lambda x: x.stat().st_size)
        if candidates:
            return candidates[0]
    pytest.skip("No ND2 files available for compatibility tests")


def test_basic_properties_and_sizes(small_nd2: Path) -> None:
    with ND2File(small_nd2) as f:
        assert isinstance(f.attributes, Attributes)
        assert isinstance(f.sizes, Mapping)
        assert isinstance(f.shape, tuple)
        assert len(f.shape) == len(f.sizes)
        assert f.ndim == len(f.shape)
        assert f.size == int(np.prod(f.shape))
        assert f.nbytes == f.size * f.dtype.itemsize


def test_dtype_matches_limnd2(small_nd2: Path) -> None:
    with limnd2.Nd2Reader(small_nd2) as r, ND2File(small_nd2) as f:
        assert f.dtype == np.dtype(r.imageAttributes.dtype)


def test_components_and_rgb_flags(small_nd2: Path) -> None:
    with ND2File(small_nd2) as f:
        assert isinstance(f.components_per_channel, int)
        assert f.components_per_channel >= 1
        assert f.is_rgb == (f.components_per_channel in (3, 4))


def test_read_frame_and_asarray(small_nd2: Path) -> None:
    with ND2File(small_nd2) as f:
        frame = f.read_frame(0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == f._frame_shape
        arr = f.asarray()
        assert arr.shape == f.shape


def test_to_dask(small_nd2: Path) -> None:
    da = pytest.importorskip("dask.array")
    with ND2File(small_nd2) as f:
        darr = f.to_dask()
        assert isinstance(darr, da.Array)
        assert darr.shape == f.shape
        sample = np.asarray(darr[(0,) * (len(f.shape) - 2)])
        assert sample.shape == f.shape[-2:]


def test_to_xarray(small_nd2: Path) -> None:
    xr = pytest.importorskip("xarray")
    with ND2File(small_nd2) as f:
        xarr = f.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert xarr.shape == f.shape


def test_metadata_smoke(small_nd2: Path) -> None:
    with ND2File(small_nd2) as f:
        assert isinstance(f.metadata, Metadata)
        assert isinstance(f.frame_metadata(0), FrameMetadata)
        assert isinstance(f.experiment, list)
        assert isinstance(f.text_info, dict)
        voxel = f.voxel_size()
        assert voxel.x >= 0
        assert voxel.y >= 0
        assert voxel.z >= 0


@pytest.mark.parametrize("orient", ["records", "dict", "list"])
def test_events_orientations(small_nd2: Path, orient: str) -> None:
    with ND2File(small_nd2) as f:
        events = f.events(orient=orient)
        assert isinstance(events, list if orient == "records" else dict)


def test_binary_data_smoke(nd2_samples: dict[str, Path]) -> None:
    path = nd2_samples.get("with_binary_and_rois.nd2")
    if path is None:
        pytest.skip("Binary data sample not available")
    with ND2File(path) as f:
        layers = f.binary_data
        assert layers is not None
        assert len(layers) > 0
        arr = np.asarray(layers)
        assert arr.ndim >= 3


def test_ome_metadata_smoke(small_nd2: Path) -> None:
    ome_types = pytest.importorskip("ome_types")
    with ND2File(small_nd2) as f:
        if f.is_legacy:
            pytest.xfail("OME metadata not supported for legacy files")
        meta = f.ome_metadata()
        assert isinstance(meta, ome_types.OME)


def test_write_tiff_smoke(tmp_path: Path, small_nd2: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    with ND2File(small_nd2) as f:
        dest = tmp_path / "out.ome.tif"
        f.write_tiff(dest)
        assert dest.exists()
        with tifffile.TiffFile(dest) as tif:
            assert tif.series


def test_write_ome_zarr_not_supported(small_nd2: Path, tmp_path: Path) -> None:
    with ND2File(small_nd2) as f:
        assert hasattr(f, "write_ome_zarr"), "write_ome_zarr is not implemented"
        f.write_ome_zarr(tmp_path / "out.zarr")
