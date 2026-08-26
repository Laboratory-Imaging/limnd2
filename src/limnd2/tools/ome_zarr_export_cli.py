from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PureWindowsPath
from time import perf_counter

import numpy as np

import limnd2
from limnd2.export_ome_zarr import ensure_ome_zarr_dependencies
from limnd2.tools.ome_zarr_exporter import (
    DEFAULT_S3_PREFIX,
    _aws_cli,
    _parse_int_tuple,
    _safe_segment,
    _s3_write_check,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _normalize_output_name(value: str | None, source: Path) -> str:
    if not value:
        return f"{_safe_segment(source.stem)}.ome.zarr"
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("Output name cannot be empty.")
    if trimmed.endswith(".ome.zarr"):
        return trimmed
    if trimmed.endswith(".zarr"):
        return trimmed[:-5] + ".ome.zarr"
    return f"{trimmed}.ome.zarr"


def _normalize_output_folder(value: Path | None, source: Path) -> Path:
    if value is None:
        return source.parent

    text = str(value).strip()
    if "\\" in text:
        windows_path = PureWindowsPath(text)
        if windows_path.is_absolute():
            return Path(*windows_path.parts).expanduser().resolve()
        return Path(*windows_path.parts).expanduser().resolve()

    return value.expanduser().resolve()


def _upload_local_ome_zarr(local_path: Path, dest_uri: str) -> None:
    aws = _aws_cli()
    subprocess.run(
        [aws, "s3", "sync", str(local_path), dest_uri, "--delete"],
        check=True,
    )


def _progress_logger() -> callable:
    state = {"last_bucket": -1, "last_phase": ""}

    def callback(
        current: int, total: int, file: str | Path | None, message: str
    ) -> None:
        if total <= 0:
            return
        percent = (current * 100.0) / total
        bucket = min(100, int(percent // 10) * 10)
        if current == total:
            bucket = 100
        if bucket == state["last_bucket"] and message == state["last_phase"] and current != total:
            return
        state["last_bucket"] = bucket
        state["last_phase"] = message
        _log(f"Progress: {percent:5.1f}% ({current}/{total}) file={file} message={message}")

    return callback


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one ND2 file to OME-Zarr locally and/or to S3."
    )
    parser.add_argument("input_nd2", type=Path, help="Path to the input .nd2 file.")
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Local output folder. Defaults to the ND2 file folder. Supports relative paths.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output store name. Defaults to <input-stem>.ome.zarr.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        const=DEFAULT_S3_PREFIX,
        nargs="?",
        help=(
            "Export to S3. If provided alone, export directly to S3. "
            f"If used together with --output-folder, export locally first and then upload. "
            f"If no value is given, defaults to {DEFAULT_S3_PREFIX}."
        ),
    )
    parser.add_argument(
        "--include-binaries",
        action="store_true",
        help="Include ND2 binary layers as OME-Zarr labels.",
    )
    dask_group = parser.add_mutually_exclusive_group()
    dask_group.add_argument(
        "--use-dask",
        dest="use_dask",
        action="store_true",
        help="Force Dask-backed export.",
    )
    dask_group.add_argument(
        "--no-dask",
        dest="use_dask",
        action="store_false",
        help="Force eager export without Dask.",
    )
    parser.set_defaults(use_dask=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing local folder or S3 prefix.",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="1,1,1,512,512",
        help="Chunk shape as T,C,Z,Y,X. Default: 1,1,1,512,512",
    )
    parser.add_argument(
        "--shard-shape",
        type=str,
        default="",
        help="Optional shard shape as T,C,Z,Y,X.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source = args.input_nd2.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input ND2 not found: {source}")

    output_name = _normalize_output_name(args.output_name, source)
    chunks = _parse_int_tuple(args.chunks, 5, allow_empty=False)
    shard_shape = _parse_int_tuple(args.shard_shape, 5, allow_empty=True)

    local_folder = _normalize_output_folder(args.output_folder, source)
    local_dest = local_folder / output_name

    s3_prefix = args.s3_prefix.rstrip("/") if args.s3_prefix else None
    s3_dest = f"{s3_prefix}/{output_name}" if s3_prefix else None
    direct_s3 = bool(s3_dest and args.output_folder is None)

    _log(f"Source: {source}")
    _log(f"Local destination: {local_dest}")
    _log("Checking OME-Zarr dependencies...")
    ensure_ome_zarr_dependencies(
        require_s3=direct_s3,
        require_dask=True,
    )
    _log("OME-Zarr dependencies look available.")
    if s3_dest:
        _log(f"S3 destination: {s3_dest}")
        _log(f"S3 mode: {'direct export' if direct_s3 else 'local export then upload'}")
        _log("Checking S3 write access...")
        _s3_write_check(s3_prefix)
        _log("S3 write check succeeded.")

    total_start = perf_counter()
    if not direct_s3:
        local_folder.mkdir(parents=True, exist_ok=True)

    with limnd2.Nd2Reader(source) as reader:
        nt, nm, nz, ny, nx, nc = reader.imageDataShape
        dtype = np.dtype(reader.imageAttributes.dtype)
        approx_gib = (nt * nm * nz * ny * nx * nc * dtype.itemsize) / (1024**3)
        binary_count = len(list(reader.binaryRasterMetadata))
        _log(
            f"shape={(nt, nm, nz, ny, nx, nc)} dtype={dtype} "
            f"approx={approx_gib:.2f} GiB binaries={binary_count}"
        )
        _log(f"chunks={chunks}")
        _log(f"shard_shape={shard_shape}")

        export_target: str | Path = s3_dest if direct_s3 and s3_dest else local_dest
        export_start = perf_counter()
        result = reader.to_ome_zarr(
            export_target,
            overwrite=args.overwrite,
            use_dask=args.use_dask,
            chunks=chunks,
            shard_shape=shard_shape,
            include_binaries=args.include_binaries,
            progress_callback=_progress_logger(),
        )
        export_elapsed = perf_counter() - export_start
        _log(f"Export finished in {export_elapsed:.2f}s")

    if s3_dest and not direct_s3:
        upload_start = perf_counter()
        _log("Uploading local OME-Zarr to S3...")
        _upload_local_ome_zarr(local_dest, s3_dest)
        upload_elapsed = perf_counter() - upload_start
        _log(f"S3 upload finished in {upload_elapsed:.2f}s")

    total_elapsed = perf_counter() - total_start
    _log(f"Total runtime: {total_elapsed:.2f}s")
    _log(f"Result: {result}")
