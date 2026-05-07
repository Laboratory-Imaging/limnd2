from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import limnd2
import numpy as np
import zarr

DEFAULT_ND2_FOLDER = Path(r"D:\files\nd2_files")
TEST_CHUNKS = (1, 1, 1, 16, 16)
TEST_SHARDS = (1, 1, 1, 32, 32)
FORBIDDEN_ATTRS = ("limnd2", "limnd2_position")
TCZYX_AXES = ["t", "c", "z", "y", "x"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test limnd2 OME-Zarr export.")
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=DEFAULT_ND2_FOLDER,
        help="Folder containing .nd2 files to export and validate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of .nd2 files to process.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clamp_chunks(
    chunks: tuple[int, int, int, int, int], shape: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int, int]:
    return tuple(max(1, min(chunk, dim)) for chunk, dim in zip(chunks, shape))


def _is_group(obj: Any) -> bool:
    return isinstance(obj, zarr.Group)


def _is_array(obj: Any) -> bool:
    return isinstance(obj, zarr.Array)


def _walk_groups(group: zarr.Group, current_path: str = "") -> list[tuple[str, zarr.Group]]:
    groups = [(current_path, group)]
    for name, child in group.members():
        if _is_group(child):
            child_path = f"{current_path}/{name}".strip("/")
            groups.extend(_walk_groups(child, child_path))
    return groups


def _assert_no_custom_attrs(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="r")
    for group_path, group in _walk_groups(root):
        attrs = group.attrs.asdict()
        for attr_name in FORBIDDEN_ATTRS:
            if attr_name in attrs:
                where = "/" if not group_path else f"/{group_path}"
                raise AssertionError(f"Found forbidden attr '{attr_name}' in group {where}.")


def _assert_image_group(
    group_path: Path,
    *,
    expected_shape: tuple[int, int, int, int, int],
    expected_dtype: np.dtype[Any],
    expected_chunks: tuple[int, int, int, int, int] | None,
    expect_shards: bool,
    expect_rgb: bool,
) -> None:
    group_meta = _load_json(group_path / "zarr.json")
    ome = group_meta["attributes"]["ome"]
    assert "multiscales" in ome, f"{group_path} does not contain ome.multiscales."

    multiscale = ome["multiscales"][0]
    axes = [axis["name"] for axis in multiscale["axes"]]
    assert axes == TCZYX_AXES, f"{group_path} axes mismatch: {axes}"

    level0_path = multiscale["datasets"][0]["path"]
    arr = zarr.open_array(str(group_path / level0_path), mode="r")
    assert tuple(arr.shape) == expected_shape, (
        f"{group_path} shape mismatch: {tuple(arr.shape)} != {expected_shape}"
    )
    assert np.dtype(arr.dtype) == expected_dtype, (
        f"{group_path} dtype mismatch: {arr.dtype} != {expected_dtype}"
    )

    if expected_chunks is not None:
        assert tuple(arr.chunks) == expected_chunks, (
            f"{group_path} chunks mismatch: {arr.chunks} != {expected_chunks}"
        )

    actual_shards = getattr(arr, "shards", None)
    if expect_shards:
        assert actual_shards is not None, f"{group_path} should be sharded."
        assert all(
            shard_dim >= chunk_dim and shard_dim % chunk_dim == 0
            for shard_dim, chunk_dim in zip(actual_shards, arr.chunks)
        ), f"{group_path} invalid shard shape: {actual_shards} for chunks {arr.chunks}"
    else:
        assert actual_shards is None, f"{group_path} should not be sharded."

    omero = ome.get("omero")
    assert omero is not None, f"{group_path} missing OMERO metadata."
    if expect_rgb:
        labels = [channel["label"] for channel in omero["channels"]]
        colors = [channel["color"] for channel in omero["channels"]]
        assert labels == ["Red", "Green", "Blue"], (
            f"{group_path} RGB labels mismatch: {labels}"
        )
        assert colors == ["FF0000", "00FF00", "0000FF"], (
            f"{group_path} RGB colors mismatch: {colors}"
        )


def _validate_export(
    nd2_path: Path,
    zarr_path: Path,
    *,
    expected_shape: tuple[int, int, int, int, int],
    expected_dtype: np.dtype[Any],
    expected_positions: int,
    expected_chunks: tuple[int, int, int, int, int] | None,
    expect_shards: bool,
    expect_rgb: bool,
) -> None:
    assert zarr_path.exists(), f"Missing export output: {zarr_path}"
    _assert_no_custom_attrs(zarr_path)

    root_meta = _load_json(zarr_path / "zarr.json")
    root_ome = root_meta["attributes"]["ome"]

    if expected_positions == 1:
        assert "multiscales" in root_ome, f"{nd2_path.name}: single image should be at root."
        assert not (zarr_path / "OME").exists(), f"{nd2_path.name}: single image should not create OME/."
        _assert_image_group(
            zarr_path,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            expected_chunks=expected_chunks,
            expect_shards=expect_shards,
            expect_rgb=expect_rgb,
        )
        return

    assert root_ome.get("bioformats2raw.layout") == 3, (
        f"{nd2_path.name}: multiposition root missing bioformats2raw layout."
    )

    ome_meta = _load_json(zarr_path / "OME" / "zarr.json")
    series_paths = ome_meta["attributes"]["ome"]["series"]
    child_groups = sorted(
        item.name
        for item in zarr_path.iterdir()
        if item.is_dir() and item.name != "OME"
    )
    assert series_paths == child_groups, (
        f"{nd2_path.name}: series list mismatch: {series_paths} != {child_groups}"
    )
    assert len(child_groups) == expected_positions, (
        f"{nd2_path.name}: position count mismatch: {len(child_groups)} != {expected_positions}"
    )

    for child_name in child_groups:
        _assert_image_group(
            zarr_path / child_name,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            expected_chunks=expected_chunks,
            expect_shards=expect_shards,
            expect_rgb=expect_rgb,
        )


def _has_mixed_rgb_channels(nd2_reader: limnd2.Nd2Reader) -> bool:
    planes = list(getattr(nd2_reader.pictureMetadata, "channels", []))
    return (not nd2_reader.isRgb) and any(
        int(getattr(plane, "uiCompCount", 1)) > 1 for plane in planes
    )


def _run_case(
    nd2_reader: limnd2.Nd2Reader,
    source_path: Path,
    *,
    suffix: str,
    chunks: tuple[int, int, int, int, int] | None,
    shard_shape: tuple[int, int, int, int, int] | None,
) -> None:
    output = source_path.with_name(source_path.name + suffix)
    kwargs: dict[str, Any] = {"overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if shard_shape is not None:
        kwargs["shard_shape"] = shard_shape

    nd2_reader.to_ome_zarr(output, **kwargs)

    nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
    expected_shape = (nt, nc, nz, ny, nx)
    expected_chunks = _clamp_chunks(chunks, expected_shape) if chunks is not None else None
    _validate_export(
        source_path,
        output,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(nd2_reader.imageAttributes.dtype),
        expected_positions=nm,
        expected_chunks=expected_chunks,
        expect_shards=shard_shape is not None,
        expect_rgb=nd2_reader.isRgb,
    )


def main() -> int:
    args = _parse_args()
    nd2_files = sorted(args.folder.glob("*.nd2"))
    if args.limit is not None:
        nd2_files = nd2_files[: args.limit]

    if not nd2_files:
        print(f"No .nd2 files found in {args.folder}")
        return 1

    failures: list[tuple[str, str]] = []

    for source_path in nd2_files:
        print(f"\nTesting {source_path.name}")
        try:
            with limnd2.Nd2Reader(source_path) as nd2_reader:
                if _has_mixed_rgb_channels(nd2_reader):
                    output = source_path.with_name(source_path.name + ".mixed_rgb_should_fail.ome.zarr")
                    try:
                        nd2_reader.to_ome_zarr(output, overwrite=True)
                    except ValueError as exc:
                        print(f"  mixed RGB/channels: expected failure: {exc}")
                    else:
                        raise AssertionError(
                            "Expected mixed RGB/channel export to fail, but it succeeded."
                        )
                    continue

                _run_case(
                    nd2_reader,
                    source_path,
                    suffix=".ome.zarr",
                    chunks=None,
                    shard_shape=None,
                )
                print("  default export: OK")

            with limnd2.Nd2Reader(source_path) as nd2_reader:
                _run_case(
                    nd2_reader,
                    source_path,
                    suffix=".chunked_sharded.ome.zarr",
                    chunks=TEST_CHUNKS,
                    shard_shape=TEST_SHARDS,
                )
                print("  chunked/sharded export: OK")
        except Exception as exc:
            failures.append((source_path.name, str(exc)))
            print(f"  FAILED: {exc}")

    if failures:
        print("\nFailures:")
        for name, message in failures:
            print(f"  - {name}: {message}")
        return 1

    print(f"\nAll OME-Zarr checks passed for {len(nd2_files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
