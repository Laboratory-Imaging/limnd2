from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import limnd2
from limnd2.export_ome_zarr import (
    _cleanup_output_root,
    _frame_lookup,
    _is_s3_path,
    _open_output_root,
    _resolve_use_dask,
    _validate_s3_path,
)


FORBIDDEN_ATTRS = ("limnd2", "limnd2_position")
TCZYX_AXES = ["t", "c", "z", "y", "x"]
TZYX_AXES = ["t", "z", "y", "x"]
EXPECTED_EXPORT_XFAILS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _require_ome_zarr_dependencies() -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("ome_zarr")


def _zarr() -> Any:
    return pytest.importorskip("zarr")


def _sample_path(nd2_base_dir: Path, name: str) -> Path:
    path = nd2_base_dir / name
    if not path.exists():
        pytest.skip(f"Required ND2 sample not found: {path}")
    return path


@pytest.fixture()
def rgb_nd2_path(nd2_base_dir: Path) -> Path:
    return _sample_path(nd2_base_dir, "multipage.nd2")


@pytest.fixture()
def multipoint_nd2_path(nd2_base_dir: Path) -> Path:
    return _sample_path(nd2_base_dir, "md2.nd2")


@pytest.fixture()
def float_nd2_path(nd2_base_dir: Path) -> Path:
    return _sample_path(nd2_base_dir, "convallaria_FLIM_crop.nd2")


@pytest.fixture()
def chunk_shard_nd2_path(nd2_base_dir: Path) -> Path:
    return _sample_path(nd2_base_dir, "cont-crop.nd2")


@pytest.fixture()
def binary_mask_nd2_path(nd2_base_dir: Path) -> Path:
    return _sample_path(nd2_base_dir, "underwater_bmx_generated_by_NIS.nd2")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _color_tuple_to_ome(color: tuple[float, float, float]) -> str:
    vals = [
        max(0, min(255, int(round(component * 255 if component <= 1 else component))))
        for component in color
    ]
    return f"{vals[0]:02X}{vals[1]:02X}{vals[2]:02X}"


def _walk_groups(group: Any, current_path: str = "") -> list[tuple[str, Any]]:
    zarr = _zarr()
    groups = [(current_path, group)]
    for name, child in group.members():
        if isinstance(child, zarr.Group):
            child_path = f"{current_path}/{name}".strip("/")
            groups.extend(_walk_groups(child, child_path))
    return groups


def _assert_no_custom_attrs(zarr_path: Path) -> None:
    zarr = _zarr()
    root = zarr.open_group(str(zarr_path), mode="r")
    for group_path, group in _walk_groups(root):
        attrs = group.attrs.asdict()
        for attr_name in FORBIDDEN_ATTRS:
            assert attr_name not in attrs, f"Unexpected attr '{attr_name}' in /{group_path}"


def _image_group_meta(group_path: Path) -> tuple[dict[str, Any], str]:
    meta = _load_json(group_path / "zarr.json")
    ome = meta["attributes"]["ome"]
    multiscale = ome["multiscales"][0]
    axes = [axis["name"] for axis in multiscale["axes"]]
    assert axes == TCZYX_AXES
    return ome, multiscale["datasets"][0]["path"]


def _level0_array(group_path: Path) -> Any:
    zarr = _zarr()
    _ome, level0_path = _image_group_meta(group_path)
    return zarr.open_array(str(group_path / level0_path), mode="r")


def _label_group_meta(group_path: Path) -> tuple[dict[str, Any], str]:
    meta = _load_json(group_path / "zarr.json")
    ome = meta["attributes"]["ome"]
    multiscale = ome["multiscales"][0]
    axes = [axis["name"] for axis in multiscale["axes"]]
    assert axes == TZYX_AXES
    return ome, multiscale["datasets"][0]["path"]


def _dataset_paths(group_path: Path) -> list[str]:
    meta = _load_json(group_path / "zarr.json")
    return [
        str(dataset["path"])
        for dataset in meta["attributes"]["ome"]["multiscales"][0]["datasets"]
    ]


def _assert_numeric_level_paths(group_path: Path) -> None:
    paths = _dataset_paths(group_path)
    assert paths == [str(index) for index in range(len(paths))]


def _level0_label_array(image_group_path: Path, label_name: str) -> Any:
    zarr = _zarr()
    label_group = image_group_path / "labels" / label_name
    _ome, level0_path = _label_group_meta(label_group)
    return zarr.open_array(str(label_group / level0_path), mode="r")


def _label_group_json(image_group_path: Path, label_name: str) -> dict[str, Any]:
    return _load_json(image_group_path / "labels" / label_name / "zarr.json")


def _image_groups_for_export(zarr_path: Path) -> list[Path]:
    root_meta = _load_json(zarr_path / "zarr.json")
    ome = root_meta["attributes"].get("ome", {})
    if "multiscales" in ome:
        return [zarr_path]
    if "plate" in ome:
        image_groups: list[Path] = []
        for well in ome["plate"]["wells"]:
            well_path = zarr_path / str(well["path"])
            well_meta = _load_json(well_path / "zarr.json")
            for image in well_meta["attributes"]["ome"]["well"]["images"]:
                image_groups.append(well_path / str(image["path"]))
        return image_groups

    ome_meta = _load_json(zarr_path / "OME" / "zarr.json")
    series_paths = ome_meta["attributes"]["ome"]["series"]
    return [zarr_path / str(path) for path in series_paths]


def _label_names_for_image_group(group_path: Path) -> list[str]:
    meta = _load_json(group_path / "labels" / "zarr.json")
    return list(meta["attributes"]["ome"].get("labels", []))


def _expected_plane(
    nd2_reader: limnd2.Nd2Reader, *, position_index: int
) -> np.ndarray:
    nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
    assert nt >= 1 and nz >= 1 and nm >= position_index + 1
    frame_index = _frame_lookup(nd2_reader)[(0, position_index, 0)]
    frame = np.asarray(nd2_reader.image(frame_index))
    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]
    assert frame.shape[:2] == (ny, nx)
    return np.moveaxis(frame[:, :, :nc], -1, 0)


def _assert_roundtrip_plane(
    nd2_reader: limnd2.Nd2Reader,
    group_path: Path,
    *,
    position_index: int,
) -> None:
    arr = _level0_array(group_path)
    expected = _expected_plane(nd2_reader, position_index=position_index)
    got = np.asarray(arr[0, :, 0, :, :])
    np.testing.assert_array_equal(got, expected)


def _assert_roundtrip_label_plane(
    nd2_reader: limnd2.Nd2Reader,
    image_group_path: Path,
    *,
    position_index: int,
    layer_index: int = 0,
    seq_index: int = 0,
    z_index: int = 0,
) -> None:
    label_names = _label_names_for_image_group(image_group_path)
    assert len(label_names) > layer_index
    label_meta = list(nd2_reader.binaryRasterMetadata)
    assert len(label_meta) > layer_index

    arr = _level0_label_array(image_group_path, label_names[layer_index])
    expected = np.asarray(nd2_reader.binaryRasterData(label_meta[layer_index].id, seq_index))
    got = np.asarray(arr[seq_index, z_index, :, :])
    np.testing.assert_array_equal(got, expected)


def test_to_ome_zarr_single_position_rgb_roundtrip(
    rgb_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "rgb_single.ome.zarr"

    with limnd2.Nd2Reader(rgb_nd2_path) as nd2_reader:
        assert nd2_reader.isRgb
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        assert nm == 1

        nd2_reader.to_ome_zarr(dest)

        root_meta = _load_json(dest / "zarr.json")
        ome = root_meta["attributes"]["ome"]
        assert "multiscales" in ome
        assert not (dest / "OME").exists()
        _assert_numeric_level_paths(dest)

        arr = _level0_array(dest)
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)
        assert np.dtype(arr.dtype) == np.dtype(nd2_reader.imageAttributes.dtype)
        assert arr.shards is None

        channels = ome["omero"]["channels"]
        assert [channel["label"] for channel in channels] == list(
            nd2_reader.pictureMetadata.componentNames
        )
        assert [channel["color"] for channel in channels] == [
            _color_tuple_to_ome(color)
            for color in nd2_reader.pictureMetadata.componentColors
        ]

        _assert_roundtrip_plane(nd2_reader, dest, position_index=0)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_multi_position_layout_and_pixels(
    multipoint_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "multipoint.ome.zarr"

    with limnd2.Nd2Reader(multipoint_nd2_path) as nd2_reader:
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        assert nm > 1

        nd2_reader.to_ome_zarr(dest)

        root_meta = _load_json(dest / "zarr.json")
        assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

        ome_meta = _load_json(dest / "OME" / "zarr.json")
        series_paths = ome_meta["attributes"]["ome"]["series"]
        child_groups = sorted(
            item.name
            for item in dest.iterdir()
            if item.is_dir() and item.name != "OME"
        )
        assert series_paths == child_groups
        assert len(child_groups) == nm

        first_group = dest / child_groups[0]
        _assert_numeric_level_paths(first_group)
        arr = _level0_array(first_group)
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)
        assert np.dtype(arr.dtype) == np.dtype(nd2_reader.imageAttributes.dtype)

        _assert_roundtrip_plane(nd2_reader, first_group, position_index=0)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_selected_position_at_root(
    multipoint_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "multipoint_position_3.ome.zarr"

    with limnd2.Nd2Reader(multipoint_nd2_path) as nd2_reader:
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        assert nm > 1

        nd2_reader.to_ome_zarr(dest, position=3)

        root_meta = _load_json(dest / "zarr.json")
        ome = root_meta["attributes"]["ome"]
        assert "multiscales" in ome
        assert "plate" not in ome
        assert not (dest / "OME").exists()

        arr = _level0_array(dest)
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)
        assert np.dtype(arr.dtype) == np.dtype(nd2_reader.imageAttributes.dtype)

        _assert_roundtrip_plane(nd2_reader, dest, position_index=3)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_selected_position_at_root_with_dask(
    multipoint_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "multipoint_position_3_dask.ome.zarr"

    with limnd2.Nd2Reader(multipoint_nd2_path) as nd2_reader:
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        assert nm > 1

        nd2_reader.to_ome_zarr(dest, position=3, use_dask=True)

        root_meta = _load_json(dest / "zarr.json")
        ome = root_meta["attributes"]["ome"]
        assert "multiscales" in ome
        assert "plate" not in ome
        assert not (dest / "OME").exists()

        arr = _level0_array(dest)
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)
        assert np.dtype(arr.dtype) == np.dtype(nd2_reader.imageAttributes.dtype)

        _assert_roundtrip_plane(nd2_reader, dest, position_index=3)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_use_dask_auto_threshold() -> None:
    small = _resolve_use_dask(
        use_dask=None,
        shape=(1, 1, 1, 512, 512),
        dtype=np.dtype(np.uint16),
    )
    large = _resolve_use_dask(
        use_dask=None,
        shape=(1, 1, 235, 6000, 6000),
        dtype=np.dtype(np.uint16),
    )
    assert small is False
    assert large is True
    assert _resolve_use_dask(
        use_dask=True,
        shape=(1, 1, 1, 1, 1),
        dtype=np.dtype(np.uint16),
    )
    assert not _resolve_use_dask(
        use_dask=False,
        shape=(1, 1, 235, 6000, 6000),
        dtype=np.dtype(np.uint16),
    )


def test_s3_path_detection_and_validation() -> None:
    assert _is_s3_path("s3://bucket/prefix/out.ome.zarr")
    assert not _is_s3_path(Path(r"C:\tmp\out.ome.zarr"))

    _validate_s3_path("s3://bucket/prefix/out.ome.zarr")
    with pytest.raises(ValueError):
        _validate_s3_path("s3://bucket")
    with pytest.raises(ValueError):
        _validate_s3_path("s3://bucket/")


class _FakeFs:
    def __init__(self, exists_result: bool) -> None:
        self.exists_result = exists_result
        self.exists_calls: list[str] = []
        self.rm_calls: list[tuple[str, bool]] = []

    def exists(self, path: str) -> bool:
        self.exists_calls.append(path)
        return self.exists_result

    def rm(self, path: str, recursive: bool = False) -> None:
        self.rm_calls.append((path, recursive))


class _FakeStore:
    def __init__(self, path: str, fs: _FakeFs) -> None:
        self.path = path
        self.fs = fs


def test_open_output_root_local_path_uses_filesystem(tmp_path: Path) -> None:
    created_group = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def open_group(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return created_group

    fake_zarr = SimpleNamespace(open_group=open_group)
    out = tmp_path / "local.ome.zarr"

    root, returned_path = _open_output_root(
        zarr=fake_zarr,
        path=out,
        overwrite=False,
    )

    assert root is created_group
    assert returned_path == out
    assert calls == [(((str(out),), {"mode": "w", "zarr_format": 3}))]


def test_open_output_root_s3_existing_without_overwrite_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs = _FakeFs(exists_result=True)
    store = _FakeStore("/bucket/prefix/out.ome.zarr", fs)

    class _FakeFsspecStore:
        @staticmethod
        def from_url(url: str, read_only: bool = False) -> _FakeStore:
            assert url == "s3://bucket/prefix/out.ome.zarr"
            assert read_only is False
            return store

    def open_group(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("open_group should not be called when overwrite is False")

    fake_zarr = SimpleNamespace(
        open_group=open_group,
        storage=SimpleNamespace(FsspecStore=_FakeFsspecStore),
    )
    monkeypatch.setattr(
        "fsspec.core.url_to_fs",
        lambda url: (fs, "/bucket/prefix/out.ome.zarr"),
    )

    with pytest.raises(FileExistsError):
        _open_output_root(
            zarr=fake_zarr,
            path="s3://bucket/prefix/out.ome.zarr",
            overwrite=False,
        )

    assert fs.exists_calls == ["/bucket/prefix/out.ome.zarr"]
    assert fs.rm_calls == []


def test_open_output_root_s3_overwrite_deletes_prefix_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs = _FakeFs(exists_result=True)
    store = _FakeStore("/bucket/prefix/out.ome.zarr", fs)
    created_group = object()
    open_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _FakeFsspecStore:
        @staticmethod
        def from_url(url: str, read_only: bool = False) -> _FakeStore:
            assert url == "s3://bucket/prefix/out.ome.zarr"
            assert read_only is False
            return store

    def open_group(*args: Any, **kwargs: Any) -> Any:
        open_calls.append((args, kwargs))
        return created_group

    fake_zarr = SimpleNamespace(
        open_group=open_group,
        storage=SimpleNamespace(FsspecStore=_FakeFsspecStore),
    )
    monkeypatch.setattr(
        "fsspec.core.url_to_fs",
        lambda url: (fs, "/bucket/prefix/out.ome.zarr"),
    )

    root, returned_path = _open_output_root(
        zarr=fake_zarr,
        path="s3://bucket/prefix/out.ome.zarr",
        overwrite=True,
    )

    assert root is created_group
    assert returned_path == "s3://bucket/prefix/out.ome.zarr"
    assert fs.exists_calls == ["/bucket/prefix/out.ome.zarr"]
    assert fs.rm_calls == [("/bucket/prefix/out.ome.zarr", True)]
    assert open_calls == [(
        tuple(),
        {"store": store, "mode": "w", "zarr_format": 3},
    )]


def test_cleanup_output_root_s3_removes_existing_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs = _FakeFs(exists_result=True)
    monkeypatch.setattr(
        "fsspec.core.url_to_fs",
        lambda url: (fs, "/bucket/prefix/out.ome.zarr"),
    )

    _cleanup_output_root("s3://bucket/prefix/out.ome.zarr")

    assert fs.exists_calls == ["/bucket/prefix/out.ome.zarr"]
    assert fs.rm_calls == [("/bucket/prefix/out.ome.zarr", True)]


def test_to_ome_zarr_failure_cleans_partial_local_output(
    rgb_nd2_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "failed.ome.zarr"

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("limnd2.export_ome_zarr._write_image_numeric_levels", fail_write)

    with limnd2.Nd2Reader(rgb_nd2_path) as nd2_reader:
        with pytest.raises(RuntimeError, match="boom"):
            nd2_reader.to_ome_zarr(dest)

    assert not dest.exists()


def test_to_ome_zarr_progress_callback_includes_image_and_label_phases(
    binary_mask_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "progress_callback.ome.zarr"
    events: list[tuple[int, int, str]] = []

    def progress_callback(current: int, total: int, phase: str) -> None:
        events.append((current, total, phase))

    with limnd2.Nd2Reader(binary_mask_nd2_path) as nd2_reader:
        nd2_reader.to_ome_zarr(
            dest,
            include_binaries=True,
            progress_callback=progress_callback,
        )

    assert events
    currents = [current for current, _total, _phase in events]
    totals = [total for _current, total, _phase in events]
    phases = {phase for _current, _total, phase in events}
    assert currents == sorted(currents)
    assert len(set(totals)) == 1
    assert events[-1][0] == events[-1][1]
    assert "read-image-frame" in phases
    assert "write-image-group" in phases
    assert "read-label-frame" in phases
    assert "write-label-group" in phases


def test_to_ome_zarr_progress_callback_is_monotonic_with_dask(
    rgb_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "progress_callback_dask.ome.zarr"
    events: list[tuple[int, int, str]] = []

    def progress_callback(current: int, total: int, phase: str) -> None:
        events.append((current, total, phase))

    with limnd2.Nd2Reader(rgb_nd2_path) as nd2_reader:
        nd2_reader.to_ome_zarr(
            dest,
            use_dask=True,
            progress_callback=progress_callback,
        )

    assert events
    currents = [current for current, _total, _phase in events]
    assert currents == sorted(currents)
    assert events[-1][0] == events[-1][1]


def test_to_ome_zarr_ignores_progress_callback_errors(
    rgb_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "progress_callback_error.ome.zarr"
    calls = 0

    def progress_callback(current: int, total: int, phase: str) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("progress callback failure")

    with limnd2.Nd2Reader(rgb_nd2_path) as nd2_reader:
        nd2_reader.to_ome_zarr(
            dest,
            progress_callback=progress_callback,
        )

        arr = _level0_array(dest)
        nt, _nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)

    assert calls >= 1


def test_to_ome_zarr_float_dtype_and_pixels(
    float_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "float_export.ome.zarr"

    with limnd2.Nd2Reader(float_nd2_path) as nd2_reader:
        assert np.issubdtype(np.dtype(nd2_reader.imageAttributes.dtype), np.floating)
        _nt, nm, _nz, _ny, _nx, _nc = nd2_reader.imageDataShape

        nd2_reader.to_ome_zarr(dest)

        if nm == 1:
            image_group = dest
        else:
            child_groups = sorted(
                item.name
                for item in dest.iterdir()
                if item.is_dir() and item.name != "OME"
            )
            image_group = dest / child_groups[0]

        arr = _level0_array(image_group)
        assert np.issubdtype(np.dtype(arr.dtype), np.floating)
        _assert_roundtrip_plane(nd2_reader, image_group, position_index=0)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_custom_chunks_and_shards(
    chunk_shard_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "chunked_sharded.ome.zarr"
    chunks = (1, 1, 1, 64, 64)
    shard_shape = (1, 1, 1, 128, 128)

    with limnd2.Nd2Reader(chunk_shard_nd2_path) as nd2_reader:
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        assert nm == 1

        nd2_reader.to_ome_zarr(dest, chunks=chunks, shard_shape=shard_shape)

        arr = _level0_array(dest)
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)
        assert tuple(arr.chunks) == chunks
        assert tuple(arr.shards) == shard_shape

        _assert_roundtrip_plane(nd2_reader, dest, position_index=0)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_binary_mask_labels(
    binary_mask_nd2_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "binary_mask_labels.ome.zarr"

    with limnd2.Nd2Reader(binary_mask_nd2_path) as nd2_reader:
        nd2_reader.to_ome_zarr(dest, include_binaries=True)

        label_names = _label_names_for_image_group(dest)
        assert len(label_names) == len(nd2_reader.binaryRasterMetadata)
        _assert_numeric_level_paths(dest / "labels" / label_names[0])

        arr = _level0_label_array(dest, label_names[0])
        values = set(np.unique(np.asarray(arr)).tolist())
        assert values == {0, 1}
        label_meta = _label_group_json(dest, label_names[0])
        image_label = label_meta["attributes"]["ome"]["image-label"]
        assert "colors" not in image_label

        _assert_roundtrip_label_plane(nd2_reader, dest, position_index=0)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_plate_layout_from_wellplate_metadata(
    multipoint_nd2_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "plate_export.ome.zarr"

    factory = limnd2.WellplateFactory(name="DemoPlate", rows=1, columns=2)
    factory.addWell("A1", seqStart=0, frameCount=5)
    factory.addWell("A2", seqStart=5, frameCount=5)
    desc, frame_info = factory.create()
    monkeypatch.setattr(limnd2.Nd2Reader, "wellplateDesc", property(lambda self: desc))
    monkeypatch.setattr(
        limnd2.Nd2Reader, "wellplateFrameInfo", property(lambda self: frame_info)
    )

    with limnd2.Nd2Reader(multipoint_nd2_path) as nd2_reader:
        _nt, nm, _nz, _ny, _nx, _nc = nd2_reader.imageDataShape
        assert nm == 10
        nd2_reader.to_ome_zarr(dest)

        root_meta = _load_json(dest / "zarr.json")
        ome = root_meta["attributes"]["ome"]
        assert "plate" in ome
        assert ome["plate"]["name"] == "DemoPlate"
        assert ome["plate"]["field_count"] == 5
        assert [well["path"] for well in ome["plate"]["wells"]] == ["A/1", "A/2"]
        assert not (dest / "OME").exists()

        well_meta = _load_json(dest / "A" / "1" / "zarr.json")
        assert [image["path"] for image in well_meta["attributes"]["ome"]["well"]["images"]] == [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]

        image_groups = _image_groups_for_export(dest)
        assert len(image_groups) == nm
        _assert_roundtrip_plane(nd2_reader, dest / "A" / "1" / "0", position_index=0)
        _assert_roundtrip_plane(nd2_reader, dest / "A" / "2" / "4", position_index=9)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_integer_labels_from_results(
    nd2_with_result_path: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "result_labels.ome.zarr"

    with limnd2.Nd2Reader(nd2_with_result_path) as nd2_reader:
        nd2_reader.to_ome_zarr(dest, include_binaries=True)

        label_names = _label_names_for_image_group(dest)
        assert len(label_names) == len(nd2_reader.binaryRasterMetadata)

        arr = _level0_label_array(dest, label_names[0])
        values = set(np.unique(np.asarray(arr[0, 0, :, :])).tolist())
        assert any(value > 1 for value in values)
        label_meta = _label_group_json(dest, label_names[0])
        image_label = label_meta["attributes"]["ome"]["image-label"]
        assert "colors" not in image_label

        _assert_roundtrip_label_plane(nd2_reader, dest, position_index=0)

    _assert_no_custom_attrs(dest)


def test_to_ome_zarr_smoke_all_nd2_fixtures(
    nd2_path: Path | None, tmp_path: Path
) -> None:
    if nd2_path is None:
        pytest.skip("No ND2 fixtures available.")

    xfail_reason = EXPECTED_EXPORT_XFAILS.get(nd2_path.name)
    if xfail_reason is not None:
        pytest.xfail(xfail_reason)

    dest = tmp_path / f"{nd2_path.stem}.ome.zarr"

    with limnd2.Nd2Reader(nd2_path) as nd2_reader:
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape

        nd2_reader.to_ome_zarr(dest)

        image_groups = _image_groups_for_export(dest)
        assert len(image_groups) == nm

        first_group = image_groups[0]
        arr = _level0_array(first_group)
        assert tuple(arr.shape) == (nt, nc, nz, ny, nx)
        assert np.dtype(arr.dtype) == np.dtype(nd2_reader.imageAttributes.dtype)

        _assert_roundtrip_plane(nd2_reader, first_group, position_index=0)

        if nd2_reader.isRgb:
            ome, _level0_path = _image_group_meta(first_group)
            channels = ome["omero"]["channels"]
            assert [channel["label"] for channel in channels] == list(
                nd2_reader.pictureMetadata.componentNames
            )

    _assert_no_custom_attrs(dest)
