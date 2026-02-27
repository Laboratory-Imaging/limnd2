from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import pytest
from nd2 import ND2File, structures

try:
    import dask.array as da
except ImportError:  # pragma: no cover - optional dependency
    da = None  # type: ignore

try:
    import xarray as xr
except ImportError:
    xr = None


DATA = Path(__file__).parent / "data"


def test_metadata_integrity_smoke(new_nd2: Path) -> None:
    """Compatibility smoke test for metadata access."""
    with ND2File(new_nd2) as nd:
        assert isinstance(nd.metadata, structures.Metadata)


def test_decode_all_chunks(new_nd2: Path) -> None:
    assert ND2File.is_supported_file(new_nd2)
    with ND2File(new_nd2) as _nd:
        assert _nd.unstructured_metadata()


def test_metadata_extraction(new_nd2: Path) -> None:
    assert ND2File.is_supported_file(new_nd2)
    with ND2File(new_nd2) as nd:
        assert repr(nd)
        assert nd.path == str(new_nd2)
        assert not nd.closed

        assert isinstance(nd.attributes, structures.Attributes)

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata, structures.Metadata)
        for i in range(min(3, nd.attributes.sequenceCount)):
            assert isinstance(nd.frame_metadata(i), structures.FrameMetadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.loop_indices, tuple)
        assert all(isinstance(x, dict) for x in nd.loop_indices)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd.sizes, MappingProxyType)
        assert isinstance(nd.custom_data, dict)
        assert isinstance(nd.shape, tuple)
        assert isinstance(nd.size, int)
        assert isinstance(nd.closed, bool)
        assert isinstance(nd.ndim, int)
        _bd = nd.binary_data
        assert isinstance(nd.is_rgb, bool)
        assert isinstance(nd.nbytes, int)

        assert isinstance(nd.unstructured_metadata(), dict)
        assert isinstance(nd.events(), list)

    assert nd.closed


def test_metadata_extraction_legacy(old_nd2: Path) -> None:
    assert ND2File.is_supported_file(old_nd2)
    with ND2File(old_nd2) as nd:
        assert repr(nd)
        assert nd.path == str(old_nd2)
        assert not nd.closed

        assert isinstance(nd.attributes, structures.Attributes)

        # # TODO: deal with typing when metadata is completely missing
        # assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd.metadata, structures.Metadata)
        if xr is not None:
            if da is None:
                pytest.skip("dask not installed")
            xarr = nd.to_xarray()
            assert isinstance(xarr, xr.DataArray)
            assert isinstance(xarr.data, da.Array)

        _ = nd.events()

    assert nd.closed


def test_events() -> None:
    with ND2File(DATA / "cluster.nd2") as f:
        rd = f.events(orient="list")
        assert isinstance(rd, dict)


@pytest.mark.parametrize("orient", ["records", "dict", "list"])
def test_events2(new_nd2: Path, orient: Literal["records", "dict", "list"]) -> None:
    with ND2File(new_nd2) as f:
        events = f.events(orient=orient)

    assert isinstance(events, list if orient == "records" else dict)
    if events and isinstance(events, dict):
        assert len(events) > 0

    pd = pytest.importorskip("pandas")
    print(pd.DataFrame(events))


def test_compressed_metadata() -> None:
    with ND2File(DATA / "cluster.nd2") as _nd:
        assert _nd.unstructured_metadata()


def test_cached_decoded_chunks() -> None:
    # test fix to https://github.com/tlambert03/nd2/issues/255
    with ND2File(DATA / "dims_p2z5t3-2c4y32x32.nd2") as f:
        _meta = f.unstructured_metadata()
        assert f.sizes

    with ND2File(DATA / "dims_p2z5t3-2c4y32x32.nd2") as f:
        assert f.sizes
        _meta = f.unstructured_metadata()
