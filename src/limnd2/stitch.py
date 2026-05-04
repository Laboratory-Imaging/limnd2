from __future__ import annotations

import inspect
import datetime as dt
import time
from pathlib import Path
import site
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from .nd2 import Nd2Reader, Nd2Writer
from .experiment import ExperimentLoopType, ExperimentXYPosLoop, find_zstack

if TYPE_CHECKING:
    from .base import FileLikeObject

_STITCH_HINT = (
    "Install optional stitching dependencies with "
    "`python -m pip install multiview-stitcher dask`."
)
# Default to tiled ND2 write for safety.
_SIMPLE_ND2_TILED_THRESHOLD_BYTES = 0


def _missing_stitch_dependency(package: str) -> ImportError:
    return ImportError(
        f'Missing optional dependency "{package}" required for stitching. {_STITCH_HINT}'
    )


def _retry_with_user_site() -> None:
    """
    Ensure user-site is on sys.path, then allow a second import attempt.
    This helps when packages were installed with `pip --user`.
    """
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if isinstance(user_site, str) and user_site and user_site not in sys.path:
        sys.path.append(user_site)


def _require_multiview_stitcher():
    try:
        from multiview_stitcher import fusion, msi_utils, registration, spatial_image_utils
    except ImportError as exc:
        _retry_with_user_site()
        try:
            from multiview_stitcher import (
                fusion,
                msi_utils,
                registration,
                spatial_image_utils,
            )
        except ImportError as exc2:
            raise _missing_stitch_dependency("multiview-stitcher") from exc2
    return fusion, msi_utils, registration, spatial_image_utils


def _require_dask():
    try:
        import dask.array as da
        from dask.delayed import delayed
    except ImportError as exc:
        _retry_with_user_site()
        try:
            import dask.array as da
            from dask.delayed import delayed
        except ImportError as exc2:
            raise _missing_stitch_dependency("dask") from exc2
    return da, delayed


def _fused_to_array(fused: Any) -> np.ndarray:
    arr = fused.data if hasattr(fused, "data") else fused
    if hasattr(arr, "compute"):
        arr = arr.compute()
    return np.asarray(arr)


def _save_fused_output(fused: Any, output: Path) -> None:
    suffix = output.suffix.lower()
    name = output.name.lower()
    if name.endswith(".ome.zarr") or suffix == ".zarr":
        if hasattr(fused, "to_zarr"):
            fused.to_zarr(str(output), mode="w")
            return
        data = fused.data if hasattr(fused, "data") else fused
        if hasattr(data, "to_zarr"):
            data.to_zarr(str(output), overwrite=True)  # type: ignore[call-arg]
            return
        raise ValueError("Fused object does not support zarr serialization.")

    if suffix == ".npy":
        np.save(output, _fused_to_array(fused))
        return

    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile  # type: ignore
        except ImportError as exc:
            raise ImportError(
                'Missing optional dependency "tifffile" required for TIFF output.'
            ) from exc
        tifffile.imwrite(str(output), _fused_to_array(fused))
        return

    raise ValueError(
        f"Unsupported output extension '{output.suffix}'. "
        "Supported: .zarr, .ome.zarr, .npy, .tif, .tiff."
    )


def _simple_log(verbose: bool, message: str) -> None:
    """Print a stitch_simple progress line when verbose logging is enabled."""
    if verbose:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [limnd2.stitch_simple] {message}", flush=True)


def _simple_inverse_2x2(
    a11: float, a12: float, a21: float, a22: float
) -> tuple[float, float, float, float]:
    """Return the inverse of a 2x2 matrix as `(i11, i12, i21, i22)`."""
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        raise ValueError("dStgLgCT matrix is singular; cannot invert.")
    return (a22 / det, -a12 / det, -a21 / det, a11 / det)


def _simple_open_reader(nd2: Nd2Reader | FileLikeObject) -> tuple[Nd2Reader, bool]:
    """Return an ND2 reader and whether this function created it (thus must close it)."""
    if isinstance(nd2, Nd2Reader):
        return nd2, False
    return Nd2Reader(nd2), True


def _simple_get_multipoint_loop(reader: Nd2Reader) -> tuple[Any, ExperimentXYPosLoop]:
    """Extract experiment metadata and the multipoint loop, or raise when unavailable."""
    exp = reader.experiment
    if exp is None:
        raise ValueError("ND2 file does not contain experiment metadata.")

    mp_level = exp.findLevel(ExperimentLoopType.eEtXYPosLoop)
    if mp_level is None or not isinstance(mp_level.uLoopPars, ExperimentXYPosLoop):
        raise ValueError("stitch_simple supports only ND2 files with multipoint experiment.")
    mp = mp_level.uLoopPars
    if not mp.Points:
        raise ValueError("Multipoint loop has no parsed points.")
    return exp, mp


def _simple_validate_dims(reader: Nd2Reader, mp: ExperimentXYPosLoop) -> tuple[int, int, int]:
    """Validate stitch_simple dimensions and return `(t_count, m_count, z_count)`."""
    dims = reader.dimensionSizes(skipSpectralLoop=True)
    t_count = int(dims.get("t", 1))
    m_count = int(dims.get("m", mp.uiCount))
    z_count = int(dims.get("z", 1))
    if len(mp.Points) < m_count:
        raise ValueError(
            f"Multipoint loop expects {m_count} points but only {len(mp.Points)} parsed."
        )
    return t_count, m_count, z_count


def _simple_resolve_output_path(
    reader: Nd2Reader, output_filename: str | Path | None
) -> Path:
    """Resolve stitch_simple output path, using `<input>_stitched_simple.ome.zarr` by default."""
    if output_filename is not None:
        return Path(output_filename)

    filename = getattr(reader.store, "filename", None)
    if filename is None:
        raise ValueError("output_filename is required when input ND2 has no filename on disk.")
    src = Path(filename)
    return src.with_name(f"{src.stem}_stitched_simple.ome.zarr")


def _simple_build_seq_lookup(exp: Any) -> dict[int, int]:
    """Build a lookup from multipoint index `m` to sequence index for t=0, z=0."""
    seq_lookup: dict[int, int] = {}
    for seq_index, idx in enumerate(exp.generateLoopIndexes(named=True)):
        m_idx = int(idx.get("m", 0))
        if int(idx.get("t", 0)) == 0 and int(idx.get("z", 0)) == 0:
            seq_lookup[m_idx] = seq_index
    return seq_lookup


def _simple_frame_layout(reader: Nd2Reader) -> tuple[int, int, int, np.dtype[Any]]:
    """Return `(y_size, x_size, c_size, dtype)` from image attributes."""
    attrs = reader.imageAttributes
    y_size, x_size, c_size = attrs.shape
    dtype = np.dtype(attrs.dtype)
    return y_size, x_size, c_size, dtype


def _simple_read_frame_cyx(reader: Nd2Reader, seq_index: int) -> np.ndarray:
    """Read one frame and return channel-first array with shape `(c, y, x)`."""
    arr = np.asarray(reader.image(seq_index))
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return np.moveaxis(arr, -1, 0)


def _simple_xy_spacing(reader: Nd2Reader) -> float:
    """Return calibrated XY spacing in microns/pixel or `1.0` when missing."""
    if (
        reader.pictureMetadata is not None
        and reader.pictureMetadata.bCalibrated
        and reader.pictureMetadata.dCalibration > 0
    ):
        return float(reader.pictureMetadata.dCalibration)
    return 1.0


def _simple_inverse_stage_matrix(reader: Nd2Reader) -> tuple[float, float, float, float]:
    """Return inverse dStgLgCT matrix from picture metadata."""
    pm = reader.pictureMetadata
    m11 = float(pm.dStgLgCT11)
    m12 = float(pm.dStgLgCT12)
    m21 = float(pm.dStgLgCT21)
    m22 = float(pm.dStgLgCT22)
    return _simple_inverse_2x2(m11, m12, m21, m22)


def _simple_build_msims(
    *,
    reader: Nd2Reader,
    m_count: int,
    mp: ExperimentXYPosLoop,
    seq_lookup: dict[int, int],
    y_size: int,
    x_size: int,
    c_size: int,
    dtype: np.dtype[Any],
    xy_spacing: float,
    inverse_stage: tuple[float, float, float, float],
    da: Any,
    delayed: Callable[..., Any],
    si_utils: Any,
    msi_utils: Any,
    verbose: bool,
) -> list[Any]:
    """Build multiscale spatial images from ND2 tiles using inverse stage XY mapping."""
    i11, i12, i21, i22 = inverse_stage
    msims: list[Any] = []
    _simple_log(verbose, f"Preparing tiles ({m_count} total).")
    for m_index in range(m_count):
        if m_index not in seq_lookup:
            raise ValueError(f"Missing frame for multipoint index m={m_index}.")
        seq_idx = seq_lookup[m_index]
        frame = da.from_delayed(
            delayed(_simple_read_frame_cyx)(reader, seq_idx),
            shape=(c_size, y_size, x_size),
            dtype=dtype,
        )
        point = mp.Points[m_index]
        tx = i11 * float(point.dPosX) + i12 * float(point.dPosY)
        ty = i21 * float(point.dPosX) + i22 * float(point.dPosY)
        sim = si_utils.get_sim_from_array(
            frame,
            dims=["c", "y", "x"],
            scale={"y": xy_spacing, "x": xy_spacing},
            translation={"y": ty, "x": tx},
            transform_key="stage_metadata",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
        if m_count >= 100:
            if (m_index + 1) % 100 == 0 or m_index == m_count - 1:
                _simple_log(verbose, f"Prepared tiles {m_index + 1}/{m_count}")
    if m_count < 100:
        _simple_log(verbose, f"Prepared tiles {m_count}/{m_count}")
    return msims


def _simple_build_fuse_kwargs(out_path: Path) -> dict[str, Any]:
    """Return hardcoded fusion kwargs used by stitch_simple."""
    fuse_kwargs: dict[str, Any] = {
        "transform_key": "stage_metadata",
        "output_chunksize": {"y": 8192, "x": 8192},
        # Balanced preset: smoother seams, slightly slower.
        "overlap_in_pixels": 64,
        "blending_widths": {"y": 24.0, "x": 24.0},
        "interpolation_order": 1,
        "batch_options": {"n_batch": 8},
    }
    if out_path.name.lower().endswith(".ome.zarr") or out_path.suffix.lower() == ".zarr":
        fuse_kwargs["output_zarr_url"] = str(out_path)
        fuse_kwargs["zarr_options"] = {"ome_zarr": False}
    return fuse_kwargs


def _simple_estimate_output_bytes_2d(
    *,
    mp: ExperimentXYPosLoop,
    m_count: int,
    y_size: int,
    x_size: int,
    c_size: int,
    dtype: np.dtype[Any],
    xy_spacing: float,
    inverse_stage: tuple[float, float, float, float],
) -> int:
    """Estimate fused 2D output memory size in bytes from stage-transformed tile bounds."""
    i11, i12, i21, i22 = inverse_stage
    px_positions_x: list[float] = []
    px_positions_y: list[float] = []
    spacing = max(float(xy_spacing), 1e-9)
    for m_index in range(m_count):
        point = mp.Points[m_index]
        tx = i11 * float(point.dPosX) + i12 * float(point.dPosY)
        ty = i21 * float(point.dPosX) + i22 * float(point.dPosY)
        px_positions_x.append(tx / spacing)
        px_positions_y.append(ty / spacing)

    if not px_positions_x or not px_positions_y:
        return int(y_size * x_size * c_size * dtype.itemsize)

    min_x = min(px_positions_x)
    max_x = max(px_positions_x)
    min_y = min(px_positions_y)
    max_y = max(px_positions_y)
    out_w = int(np.ceil((max_x - min_x) + x_size))
    out_h = int(np.ceil((max_y - min_y) + y_size))
    out_w = max(1, out_w)
    out_h = max(1, out_h)
    return int(out_h * out_w * c_size * dtype.itemsize)


def _simple_write_fused_nd2_full(
    *,
    fused: Any,
    output_path: Path,
    source_reader: Nd2Reader | None,
    verbose: bool = False,
) -> Path:
    """Materialize fused image in memory and write it as a single-frame ND2."""
    from .attributes import ImageAttributes

    if output_path.exists():
        output_path.unlink()

    if hasattr(fused, "dims") and hasattr(fused, "transpose"):
        sim = fused
        dims = tuple(getattr(sim, "dims", ()))
        if "t" in dims:
            sim = sim.isel(t=0)
            dims = tuple(getattr(sim, "dims", ()))
        if "z" in dims:
            sim = sim.isel(z=0)
            dims = tuple(getattr(sim, "dims", ()))
        if "c" not in dims:
            sim = sim.expand_dims(dim={"c": [0]})
        sim = sim.transpose("y", "x", "c")
        arr = sim.data
        if hasattr(arr, "compute"):
            arr = arr.compute()
        img_yxc = np.asarray(arr)
    else:
        arr = _fused_to_array(fused)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            img_yxc = arr[..., np.newaxis]
        elif arr.ndim == 3:
            # Heuristic: assume C,Y,X when first axis is likely channel axis.
            if arr.shape[0] <= 8 and arr.shape[-1] > 8:
                img_yxc = np.moveaxis(arr, 0, -1)
            else:
                img_yxc = arr
        else:
            raise ValueError(f"Unsupported fused array ndim={arr.ndim} for ND2 output.")

    if img_yxc.ndim != 3:
        raise ValueError(f"Expected fused image shape (y,x,c), got {img_yxc.shape}.")
    y_size, x_size, c_size = map(int, img_yxc.shape)

    dtype = np.dtype(img_yxc.dtype)
    if dtype == np.uint8:
        bits = 8
    elif dtype == np.uint16:
        bits = 16
    else:
        bits = max(8, int(dtype.itemsize * 8))

    attrs = ImageAttributes.create(
        width=x_size,
        height=y_size,
        component_count=c_size,
        bits=bits,
        sequence_count=1,
    )
    _simple_log(
        verbose,
        f"ND2 full write: materialized fused numpy shape={img_yxc.shape}, dtype={img_yxc.dtype}.",
    )

    with Nd2Writer(output_path) as writer:
        writer.imageAttributes = attrs
        if source_reader is not None:
            try:
                writer.pictureMetadata = source_reader.pictureMetadata
            except Exception:
                pass
        writer.setImage(0, img_yxc)
    return output_path


def _simple_write_fused_nd2_tiled(
    *,
    fused: Any,
    output_path: Path,
    source_reader: Nd2Reader | None,
    tile_size: tuple[int, int] = (1024, 1024),
    verbose: bool = False,
) -> Path:
    """Write fused image to ND2 by computing and writing one tile at a time."""
    from .attributes import ImageAttributes

    if output_path.exists():
        output_path.unlink()

    if hasattr(fused, "dims") and hasattr(fused, "transpose"):
        sim = fused
        dims = tuple(getattr(sim, "dims", ()))
        if "t" in dims:
            sim = sim.isel(t=0)
            dims = tuple(getattr(sim, "dims", ()))
        if "z" in dims:
            sim = sim.isel(z=0)
            dims = tuple(getattr(sim, "dims", ()))
        if "c" not in dims:
            sim = sim.expand_dims(dim={"c": [0]})
        sim = sim.transpose("y", "x", "c")
        data = sim.data
        sizes = getattr(sim, "sizes", None)
        if sizes is None:
            raise TypeError("Fused object missing `sizes` needed for tiled ND2 writing.")
        y_size = int(sizes["y"])
        x_size = int(sizes["x"])
        c_size = int(sizes["c"])
        dtype = np.dtype(getattr(data, "dtype", np.uint16))
    else:
        arr = np.asarray(_fused_to_array(fused))
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        elif arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[-1] > 8:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported fused array shape for tiled ND2: {arr.shape}")
        data = arr
        y_size, x_size, c_size = map(int, arr.shape)
        dtype = np.dtype(arr.dtype)

    if dtype == np.uint8:
        bits = 8
    elif dtype == np.uint16:
        bits = 16
    else:
        bits = max(8, int(dtype.itemsize * 8))

    tile_w = max(1, int(tile_size[0]))
    tile_h = max(1, int(tile_size[1]))
    attrs = ImageAttributes.create(
        width=x_size,
        height=y_size,
        component_count=c_size,
        bits=bits,
        sequence_count=1,
    )
    # Ensure ND2 attributes know the intended tile size to avoid gaps.
    attrs = replace(attrs, uiTileWidth=tile_w, uiTileHeight=tile_h)

    with Nd2Writer(output_path) as writer:
        writer.imageAttributes = attrs
        if source_reader is not None:
            try:
                writer.pictureMetadata = source_reader.pictureMetadata
            except Exception:
                pass

        total_tiles_x = (x_size + tile_w - 1) // tile_w
        total_tiles_y = (y_size + tile_h - 1) // tile_h
        total_tiles = total_tiles_x * total_tiles_y
        _simple_log(
            verbose,
            "ND2 tiled write: materializing fused image in numpy tiles "
            f"tile_size=({tile_w},{tile_h}), tile_grid=({total_tiles_y},{total_tiles_x}), total_tiles={total_tiles}.",
        )
        done = 0
        for y0 in range(0, y_size, tile_h):
            y1 = min(y_size, y0 + tile_h)
            for x0 in range(0, x_size, tile_w):
                x1 = min(x_size, x0 + tile_w)
                tile = data[y0:y1, x0:x1, :]
                if hasattr(tile, "compute"):
                    tile = tile.compute()
                writer.chunker.setImageTile(0, x0, y0, np.asarray(tile))
                done += 1
                if verbose:
                    _simple_log(
                        verbose,
                        f"Wrote tile {done}/{total_tiles}: x=[{x0}:{x1}) y=[{y0}:{y1})",
                    )
    return output_path


def _simple_build_seq_lookup_tmz(exp: Any) -> dict[tuple[int, int, int], int]:
    """Build `(t, m, z) -> seq_index` lookup from experiment loop indexes."""
    seq_lookup: dict[tuple[int, int, int], int] = {}
    for seq_index, idx in enumerate(exp.generateLoopIndexes(named=True)):
        seq_lookup[(int(idx.get("t", 0)), int(idx.get("m", 0)), int(idx.get("z", 0)))] = seq_index
    return seq_lookup


def _simple_z_spacing(exp: Any, z_count: int) -> float:
    """Return z spacing from ND2 z-loop metadata when available."""
    z_spacing = 1.0
    z_loop = find_zstack(exp)
    if z_count > 1 and z_loop is not None and z_loop.step is not None and z_loop.step > 0:
        z_spacing = float(z_loop.step)
    return z_spacing


def _simple_build_msims_for_time(
    *,
    reader: Nd2Reader,
    mp: ExperimentXYPosLoop,
    seq_lookup_tmz: dict[tuple[int, int, int], int],
    time_index: int,
    m_count: int,
    z_count: int,
    y_size: int,
    x_size: int,
    c_size: int,
    dtype: np.dtype[Any],
    xy_spacing: float,
    z_spacing: float,
    inverse_stage: tuple[float, float, float, float],
    da: Any,
    delayed: Callable[..., Any],
    si_utils: Any,
    msi_utils: Any,
    verbose: bool,
) -> list[Any]:
    """Build per-multipoint msims for a single timepoint, preserving z when present."""
    i11, i12, i21, i22 = inverse_stage
    msims: list[Any] = []
    _simple_log(verbose, f"[t={time_index}] Preparing tiles ({m_count} total).")
    for m_index in range(m_count):
        point = mp.Points[m_index]
        frames: list[Any] = []
        for z_index in range(z_count):
            key = (time_index, m_index, z_index)
            if key not in seq_lookup_tmz:
                raise ValueError(f"Missing frame for loop indexes {key}.")
            seq_idx = seq_lookup_tmz[key]
            frm = da.from_delayed(
                delayed(_simple_read_frame_cyx)(reader, seq_idx),
                shape=(c_size, y_size, x_size),
                dtype=dtype,
            )
            frames.append(frm)

        tile_array = frames[0] if z_count == 1 else da.stack(frames, axis=1)
        tx = i11 * float(point.dPosX) + i12 * float(point.dPosY)
        ty = i21 * float(point.dPosX) + i22 * float(point.dPosY)
        dims = ["c", "y", "x"] if z_count == 1 else ["c", "z", "y", "x"]
        scale = {"y": xy_spacing, "x": xy_spacing}
        if z_count > 1:
            scale = {"z": z_spacing, **scale}
        translation: dict[str, float] = {"y": ty, "x": tx}
        if z_count > 1:
            translation["z"] = float(point.dPosZ)

        sim = si_utils.get_sim_from_array(
            tile_array,
            dims=dims,
            scale=scale,
            translation=translation,
            transform_key="stage_metadata",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
        if m_count >= 100:
            if (m_index + 1) % 100 == 0 or m_index == m_count - 1:
                _simple_log(verbose, f"[t={time_index}] Prepared tiles {m_index + 1}/{m_count}")
    if m_count < 100:
        _simple_log(verbose, f"[t={time_index}] Prepared tiles {m_count}/{m_count}")
    return msims


def _simple_to_zyxc(fused: Any) -> Any:
    """Normalize fused output to xarray-like dims `('z', 'y', 'x', 'c')`."""
    sim = fused
    if not hasattr(sim, "dims"):
        arr = np.asarray(_fused_to_array(fused))
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ..., np.newaxis]  # z,y,x,c
        elif arr.ndim == 3:
            # assume y,x,c
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 4 and arr.shape[0] <= 8 and arr.shape[-1] > 8:
            # c,z,y,x -> z,y,x,c
            arr = np.moveaxis(arr, 0, -1)
        return arr

    dims = tuple(getattr(sim, "dims", ()))
    if "t" in dims:
        sim = sim.isel(t=0)
        dims = tuple(getattr(sim, "dims", ()))
    if "c" not in dims:
        sim = sim.expand_dims(dim={"c": [0]})
    if "z" not in dims:
        sim = sim.expand_dims(dim={"z": [0]})
    return sim.transpose("z", "y", "x", "c")


def _simple_write_zyxc_to_nd2_tiles(
    *,
    writer: Nd2Writer,
    zyxc: Any,
    t_index: int,
    z_count_out: int,
    tile_size: tuple[int, int],
    verbose: bool,
) -> None:
    """Write one timepoint fused stack (`z,y,x,c`) to ND2 tiles."""
    z_size = int(getattr(zyxc, "sizes", {}).get("z", zyxc.shape[0]))
    y_size = int(getattr(zyxc, "sizes", {}).get("y", zyxc.shape[1]))
    x_size = int(getattr(zyxc, "sizes", {}).get("x", zyxc.shape[2]))
    tile_w = max(1, int(tile_size[0]))
    tile_h = max(1, int(tile_size[1]))

    for z_index in range(z_size):
        plane = zyxc.isel(z=z_index) if hasattr(zyxc, "isel") else zyxc[z_index]
        seqindex = t_index * z_count_out + z_index
        done = 0
        total_tiles = ((x_size + tile_w - 1) // tile_w) * ((y_size + tile_h - 1) // tile_h)
        for y0 in range(0, y_size, tile_h):
            y1 = min(y_size, y0 + tile_h)
            for x0 in range(0, x_size, tile_w):
                x1 = min(x_size, x0 + tile_w)
                tile = plane.isel(y=slice(y0, y1), x=slice(x0, x1)) if hasattr(plane, "isel") else plane[y0:y1, x0:x1, :]
                if hasattr(tile, "data"):
                    tile = tile.data
                if hasattr(tile, "compute"):
                    tile = tile.compute()
                writer.chunker.setImageTile(seqindex, x0, y0, np.asarray(tile))
                done += 1
                if verbose and (done == total_tiles or done % 50 == 0):
                    _simple_log(verbose, f"[t={t_index}, z={z_index}] Wrote tiles {done}/{total_tiles}")


def _simple_stitch_multidim_to_nd2(
    *,
    reader: Nd2Reader,
    exp: Any,
    mp: ExperimentXYPosLoop,
    out_path: Path,
    t_count: int,
    m_count: int,
    z_count: int,
    y_size: int,
    x_size: int,
    c_size: int,
    dtype: np.dtype[Any],
    xy_spacing: float,
    inverse_stage: tuple[float, float, float, float],
    fusion: Any,
    msi_utils: Any,
    si_utils: Any,
    da: Any,
    delayed: Callable[..., Any],
    verbose: bool,
) -> Path:
    """Stitch all `T` and preserve `Z` into ND2, collapsing multipoint `M`."""
    from .attributes import ImageAttributes
    from .experiment_factory import ExperimentFactory

    seq_lookup_tmz = _simple_build_seq_lookup_tmz(exp)
    z_spacing = _simple_z_spacing(exp, z_count)
    fuse_kwargs = _simple_build_fuse_kwargs(out_path)
    # ND2 output path is not a fuse target.
    fuse_kwargs.pop("output_zarr_url", None)
    fuse_kwargs.pop("zarr_options", None)
    if z_count > 1:
        chunks = dict(fuse_kwargs.get("output_chunksize", {}))
        chunks.setdefault("z", int(max(1, min(z_count, 8))))
        fuse_kwargs["output_chunksize"] = chunks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with Nd2Writer(out_path) as writer:
        writer.imageAttributes = replace(
            ImageAttributes.create(
            width=1, height=1, component_count=max(1, c_size),
            bits=max(8, int(dtype.itemsize * 8)), sequence_count=max(1, t_count * z_count)
            ),
            uiTileWidth=1024,
            uiTileHeight=1024,
        )
        try:
            writer.pictureMetadata = reader.pictureMetadata
        except Exception:
            pass

        initialized = False
        for t_index in range(t_count):
            _simple_log(verbose, f"[t={t_index}] Building lazy tile inputs.")
            msims = _simple_build_msims_for_time(
                reader=reader,
                mp=mp,
                seq_lookup_tmz=seq_lookup_tmz,
                time_index=t_index,
                m_count=m_count,
                z_count=z_count,
                y_size=y_size,
                x_size=x_size,
                c_size=c_size,
                dtype=dtype,
                xy_spacing=xy_spacing,
                z_spacing=z_spacing,
                inverse_stage=inverse_stage,
                da=da,
                delayed=delayed,
                si_utils=si_utils,
                msi_utils=msi_utils,
                verbose=verbose,
            )
            _simple_log(verbose, f"[t={t_index}] Starting fusion.")
            fused = fusion.fuse([msi_utils.get_sim_from_msim(msim) for msim in msims], **fuse_kwargs)
            _simple_log(verbose, f"[t={t_index}] Fusion finished.")

            zyxc = _simple_to_zyxc(fused)
            if hasattr(zyxc, "sizes"):
                out_z = int(zyxc.sizes["z"])
                out_y = int(zyxc.sizes["y"])
                out_x = int(zyxc.sizes["x"])
                out_c = int(zyxc.sizes["c"])
                out_dtype = np.dtype(getattr(getattr(zyxc, "data", None), "dtype", dtype))
            else:
                out_z, out_y, out_x, out_c = map(int, zyxc.shape)
                out_dtype = np.dtype(zyxc.dtype)

            if not initialized:
                bits = 8 if out_dtype == np.uint8 else (16 if out_dtype == np.uint16 else max(8, int(out_dtype.itemsize * 8)))
                writer.imageAttributes = replace(
                    ImageAttributes.create(
                    width=out_x,
                    height=out_y,
                    component_count=out_c,
                    bits=bits,
                    sequence_count=t_count * out_z,
                    ),
                    uiTileWidth=1024,
                    uiTileHeight=1024,
                )
                ef = ExperimentFactory()
                if t_count > 1:
                    ef.t.count = t_count
                if out_z > 1:
                    ef.z.count = out_z
                    ef.z.step = z_spacing
                writer.experiment = ef.createExperiment()
                z_count = out_z
                initialized = True
                _simple_log(verbose, f"Initialized ND2 output shape: t={t_count}, z={z_count}, y={out_y}, x={out_x}, c={out_c}.")

            _simple_write_zyxc_to_nd2_tiles(
                writer=writer,
                zyxc=zyxc,
                t_index=t_index,
                z_count_out=z_count,
                tile_size=(1024, 1024),
                verbose=verbose,
            )
    return out_path


def stitch_simple(
    nd2: Nd2Reader | FileLikeObject,
    output_filename: str | Path | None = None,
    *,
    verbose: bool = False,
) -> Any:
    """
    Simple multipoint-only stitch convenience wrapper.

    This function is a minimal multipoint stitch convenience wrapper with
    hardcoded defaults.

    - For 2D inputs (`t=1`, `z=1`), it fuses one stitched image.
    - For ND2 output with multidimensional inputs (`t>1` and/or `z>1`), it
      stitches each timepoint and preserves `T/Z` in the output ND2
      (collapsing only multipoint `M`).

    Hardcoded stitching settings:
    - ``time_index=0``
    - ``register=False``
    - ``stage_transform_mode="inverse"``
    - ``output_chunksize={"y": 8192, "x": 8192}``
    - ``fusion_batch_size=8``
    - ``dask_scheduler="processes"``
    - ``dask_num_workers=8``
    - ``overlap_in_pixels=64``
    - ``blending_widths=24``
    - ``interpolation_order=1``
    - ``zarr_options={"ome_zarr": False}``
    - ND2 auto writer mode threshold: 4 GiB per stitched frame

    Output behavior:
    - If `output_filename` is omitted, output defaults to `<input>_stitched_simple.nd2`.
    - For `.nd2` output, this function chooses:
      - full in-memory write for smaller estimated mosaics
      - tiled streaming ND2 write for larger estimated mosaics
    - For multidimensional ND2 input (`t/z`), output ND2 is streamed tiled and
      keeps `T/Z` axes.
    """
    t_import0 = time.perf_counter()
    fusion, msi_utils, _, si_utils = _require_multiview_stitcher()
    t_import1 = time.perf_counter()
    da, delayed = _require_dask()
    t_import2 = time.perf_counter()
    _simple_log(
        verbose,
        f"Import timing: multiview-stitcher={(t_import1 - t_import0):.3f}s, dask={(t_import2 - t_import1):.3f}s, total={(t_import2 - t_import0):.3f}s.",
    )

    reader, close_reader = _simple_open_reader(nd2)

    try:
        exp, mp = _simple_get_multipoint_loop(reader)
        t_count, m_count, z_count = _simple_validate_dims(reader, mp)
        source_name = getattr(reader.store, "filename", None)
        if output_filename is None:
            if source_name is None:
                raise ValueError("output_filename is required when input ND2 has no filename on disk.")
            src = Path(source_name)
            out_path = src.with_name(f"{src.stem}_stitched_simple.nd2")
        else:
            out_path = Path(output_filename)

        _simple_log(verbose, f"Input ND2: {source_name if source_name is not None else '<in-memory>'}")
        _simple_log(verbose, f"Output: {out_path}")
        _simple_log(verbose, "Pipeline: ND2 -> lazy dask tiles -> multiview-stitcher fuse -> output writer.")
        _simple_log(verbose, f"Dimensions: t={t_count}, m={m_count}, z={z_count}.")

        # For multidimensional datasets, keep T/Z in output ND2 and collapse only M via stitching.
        # Example: T20 Z10 M4 -> stitched ND2 with T20 Z10.
        if out_path.suffix.lower() == ".nd2" and (t_count > 1 or z_count > 1):
            _simple_log(
                verbose,
                "Multidimensional input detected. Stitching all T and preserving Z in output ND2.",
            )
            return _simple_stitch_multidim_to_nd2(
                reader=reader,
                exp=exp,
                mp=mp,
                out_path=out_path,
                t_count=t_count,
                m_count=m_count,
                z_count=z_count,
                y_size=y_size,
                x_size=x_size,
                c_size=c_size,
                dtype=dtype,
                xy_spacing=xy_spacing,
                inverse_stage=inverse_stage,
                fusion=fusion,
                msi_utils=msi_utils,
                si_utils=si_utils,
                da=da,
                delayed=delayed,
                verbose=verbose,
            )

        seq_lookup = _simple_build_seq_lookup(exp)
        y_size, x_size, c_size, dtype = _simple_frame_layout(reader)
        xy_spacing = _simple_xy_spacing(reader)
        inverse_stage = _simple_inverse_stage_matrix(reader)
        _simple_log(
            verbose,
            f"Input tile shape=(y={y_size}, x={x_size}, c={c_size}), dtype={dtype}, multipoints={m_count}.",
        )

        use_tiled_nd2 = False
        if out_path.suffix.lower() == ".nd2":
            est_bytes = _simple_estimate_output_bytes_2d(
                mp=mp,
                m_count=m_count,
                y_size=y_size,
                x_size=x_size,
                c_size=c_size,
                dtype=dtype,
                xy_spacing=xy_spacing,
                inverse_stage=inverse_stage,
            )
            _simple_log(
                verbose,
                f"ND2 output requested. Estimated fused size: {est_bytes / (1024**3):.2f} GiB.",
            )
            if est_bytes > _SIMPLE_ND2_TILED_THRESHOLD_BYTES:
                _simple_log(
                    verbose,
                    f"Using tiled ND2 writer (estimated frame > {_SIMPLE_ND2_TILED_THRESHOLD_BYTES / (1024**3):.2f} GiB.).",
                )
                use_tiled_nd2 = True
            else:
                _simple_log(verbose, "Using full in-memory ND2 writer (small output estimate).")

        _simple_log(verbose, "Building lazy dask tile inputs (frames are not read/decoded all at once).")
        msims = _simple_build_msims(
            reader=reader,
            m_count=m_count,
            mp=mp,
            seq_lookup=seq_lookup,
            y_size=y_size,
            x_size=x_size,
            c_size=c_size,
            dtype=dtype,
            xy_spacing=xy_spacing,
            inverse_stage=inverse_stage,
            da=da,
            delayed=delayed,
            si_utils=si_utils,
            msi_utils=msi_utils,
            verbose=verbose,
        )
        fuse_kwargs = _simple_build_fuse_kwargs(out_path)
        if out_path.suffix.lower() == ".nd2" and use_tiled_nd2:
            # Keep fusion chunk size aligned with tiled writes to avoid large temp arrays.
            fuse_kwargs["output_chunksize"] = {"y": 1024, "x": 1024}
            fuse_kwargs["batch_options"] = {"n_batch": 1}
            _simple_log(
                verbose,
                "Tiled ND2 mode: using fusion chunks 1024x1024 with n_batch=1 to limit RAM.",
            )
        # No special chunk override for full ND2 write path.
        _simple_log(verbose, f"Fuser kwargs: {fuse_kwargs}")

        _simple_log(verbose, "Starting fusion.")
        progress_ctx = nullcontext()
        if verbose:
            try:
                from dask.diagnostics import ProgressBar  # type: ignore
                progress_ctx = ProgressBar()
            except Exception:
                pass
        with progress_ctx:
            fused = fusion.fuse(
                [msi_utils.get_sim_from_msim(msim) for msim in msims],
                **fuse_kwargs,
            )
        _simple_log(verbose, "Fusion finished.")

        if out_path.suffix.lower() == ".nd2":
            if use_tiled_nd2:
                _simple_log(verbose, "Starting ND2 tiled write.")
                _simple_write_fused_nd2_tiled(
                    fused=fused,
                    output_path=out_path,
                    source_reader=reader,
                    tile_size=(1024, 1024),
                    verbose=verbose,
                )
            else:
                _simple_log(verbose, "Fusion done. Starting ND2 full write.")
                _simple_write_fused_nd2_full(
                    fused=fused,
                    output_path=out_path,
                    source_reader=reader,
                    verbose=verbose,
                )
        elif not (
            out_path.name.lower().endswith(".ome.zarr")
            or out_path.suffix.lower() == ".zarr"
        ):
            _save_fused_output(fused, out_path)

        _simple_log(verbose, f"Finished. Output written to: {out_path}")
        return fused
    finally:
        if close_reader:
            reader.finalize()

'''
Optional/old helper APIs (currently not needed) kept commented for reference:
- _simple_default_ome_zarr_path
- _strip_ome_zarr_suffix
- stitch_simple_ome_zarr
- ome_zarr_to_nd2

def _simple_default_ome_zarr_path(reader: Nd2Reader) -> Path:
    """Return default OME-Zarr output path for stitch_simple_ome_zarr."""
    filename = getattr(reader.store, "filename", None)
    if filename is None:
        raise ValueError("output_filename is required when input ND2 has no filename on disk.")
    src = Path(filename)
    return src.with_name(f"{src.stem}_stitched_simple.ome.zarr")


def _strip_ome_zarr_suffix(path: Path) -> str:
    """Return stem-like name that removes `.ome.zarr` or `.zarr` suffixes."""
    name = path.name
    lower = name.lower()
    if lower.endswith(".ome.zarr"):
        return name[: -len(".ome.zarr")]
    if lower.endswith(".zarr"):
        return name[: -len(".zarr")]
    return path.stem


def stitch_simple_ome_zarr(
    nd2: Nd2Reader | FileLikeObject,
    output_filename: str | Path | None = None,
    *,
    verbose: bool = False,
) -> Any:
    """
    Simple multipoint stitch that always writes fused output to OME-Zarr.

    This keeps the same multipoint-only assumptions as `stitch_simple`
    (single timepoint and single z-plane).
    """
    t_import0 = time.perf_counter()
    fusion, msi_utils, _, si_utils = _require_multiview_stitcher()
    t_import1 = time.perf_counter()
    da, delayed = _require_dask()
    t_import2 = time.perf_counter()
    _simple_log(
        verbose,
        f"Import timing: multiview-stitcher={(t_import1 - t_import0):.3f}s, dask={(t_import2 - t_import1):.3f}s, total={(t_import2 - t_import0):.3f}s.",
    )

    reader, close_reader = _simple_open_reader(nd2)
    try:
        exp, mp = _simple_get_multipoint_loop(reader)
        m_count = _simple_validate_dims(reader, mp)
        out_path = Path(output_filename) if output_filename is not None else _simple_default_ome_zarr_path(reader)

        seq_lookup = _simple_build_seq_lookup(exp)
        y_size, x_size, c_size, dtype = _simple_frame_layout(reader)
        xy_spacing = _simple_xy_spacing(reader)
        inverse_stage = _simple_inverse_stage_matrix(reader)
        _simple_log(verbose, f"Input ND2: {getattr(reader.store, 'filename', None)}")
        _simple_log(verbose, f"Output OME-Zarr: {out_path}")
        _simple_log(verbose, "Building lazy dask tile inputs (frames are not read/decoded all at once).")

        msims = _simple_build_msims(
            reader=reader,
            m_count=m_count,
            mp=mp,
            seq_lookup=seq_lookup,
            y_size=y_size,
            x_size=x_size,
            c_size=c_size,
            dtype=dtype,
            xy_spacing=xy_spacing,
            inverse_stage=inverse_stage,
            da=da,
            delayed=delayed,
            si_utils=si_utils,
            msi_utils=msi_utils,
            verbose=verbose,
        )
        fuse_kwargs: dict[str, Any] = {
            "transform_key": "stage_metadata",
            "output_chunksize": {"y": 8192, "x": 8192},
            "overlap_in_pixels": 64,
            "blending_widths": {"y": 24.0, "x": 24.0},
            "interpolation_order": 1,
            "batch_options": {"n_batch": 8},
            "output_zarr_url": str(out_path),
            "zarr_options": {"ome_zarr": True, "overwrite": True},
        }
        _simple_log(verbose, f"Fuser kwargs: {fuse_kwargs}")
        _simple_log(verbose, "Starting fusion to OME-Zarr.")
        fused = fusion.fuse([msi_utils.get_sim_from_msim(msim) for msim in msims], **fuse_kwargs)
        _simple_log(verbose, f"Finished. Output written to: {out_path}")
        return fused
    finally:
        if close_reader:
            reader.finalize()


def ome_zarr_to_nd2(
    ome_zarr_path: str | Path,
    output_nd2_filename: str | Path | None = None,
    *,
    tile_size: tuple[int, int] = (1024, 1024),
    verbose: bool = False,
    source_nd2: Nd2Reader | FileLikeObject | None = None,
) -> Path:
    """
    Convert OME-Zarr fused image to ND2 by streaming tiles into ND2 writer.

    Only the first timepoint and first z-plane are written when present.
    """
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:
        raise _missing_stitch_dependency("xarray") from exc

    in_path = Path(ome_zarr_path)
    if output_nd2_filename is None:
        base = _strip_ome_zarr_suffix(in_path)
        out_path = in_path.with_name(f"{base}.nd2")
    else:
        out_path = Path(output_nd2_filename)

    _simple_log(verbose, f"Input OME-Zarr: {in_path}")
    _simple_log(verbose, f"Output ND2: {out_path}")
    ds = xr.open_zarr(str(in_path), consolidated=False)
    if hasattr(ds, "data_vars"):
        data_vars = list(ds.data_vars.values())
        if not data_vars:
            raise ValueError("OME-Zarr dataset has no data variables.")
        sim = data_vars[0]
    else:
        sim = ds

    rename_map: dict[str, str] = {}
    dims_now = set(getattr(sim, "dims", ()))
    if "channel" in dims_now and "c" not in dims_now:
        rename_map["channel"] = "c"
    if "time" in dims_now and "t" not in dims_now:
        rename_map["time"] = "t"
    if rename_map:
        sim = sim.rename(rename_map)
    if "t" in sim.dims:
        _simple_log(verbose, f"Selecting first timepoint from OME-Zarr t-size={int(sim.sizes['t'])}.")
        sim = sim.isel(t=0)
    if "z" in sim.dims:
        _simple_log(verbose, f"Selecting first z-plane from OME-Zarr z-size={int(sim.sizes['z'])}.")
        sim = sim.isel(z=0)
    if "c" not in sim.dims:
        sim = sim.expand_dims(dim={"c": [0]})
    if "y" not in sim.dims or "x" not in sim.dims:
        raise ValueError(f"OME-Zarr image must contain 'y' and 'x' dims. Got: {tuple(sim.dims)}")
    sim = sim.transpose("y", "x", "c")

    src_reader: Nd2Reader | None = None
    close_src = False
    if source_nd2 is not None:
        src_reader, close_src = _simple_open_reader(source_nd2)
    try:
        _simple_write_fused_nd2_tiled(
            fused=sim,
            output_path=out_path,
            source_reader=src_reader,
            tile_size=tile_size,
            verbose=verbose,
        )
    finally:
        if close_src and src_reader is not None:
            src_reader.finalize()

    _simple_log(verbose, f"Finished OME-Zarr -> ND2 conversion: {out_path}")
    return out_path
'''

'''
"""
Legacy advanced stitch implementation below.

Top-level public API (`limnd2.stitch`) is intentionally mapped to the simple
workflow in `__init__.py`:
    from .stitch import stitch_simple as stitch

This advanced function is kept for internal compatibility
(`stitch_to_nd2_tiled` and power users importing from `limnd2.stitch`).
"""
def stitch(
    nd2: Nd2Reader | FileLikeObject,
    output_filename: str | Path | None = None,
    *,
    time_index: int = 0,
    register: bool = False,
    reg_channel: str | int | None = None,
    stage_transform_key: str = "stage_metadata",
    registered_transform_key: str = "translation_registered",
    verbose: bool = False,
    log_every_tiles: int = 100,
    show_dask_progress: bool = False,
    downsample_level: int = 0,
    output_chunksize: dict[str, int] | None = None,
    fusion_batch_size: int | None = None,
    stage_transform_mode: str = "none",
    overlap_in_pixels: int | None = None,
    blending_widths: int | dict[str, int] | None = None,
    interpolation_order: int | None = None,
    dask_scheduler: str | None = None,
    dask_num_workers: int | None = None,
    use_ray_batches: bool = False,
    ray_num_cpus: int | None = None,
    zarr_options: dict[str, Any] | None = None,
    batch_options: dict[str, Any] | None = None,
) -> Any:
    """
    Stitch ND2 multipoint data using ``multiview-stitcher``.

    Parameters
    ----------
    nd2:
        Open ``Nd2Reader`` instance or input ND2 filename/path.
    output_filename:
        Optional output path. Supported output extensions are
        ``.zarr``, ``.ome.zarr``, ``.npy``, ``.tif``, ``.tiff``.
    time_index:
        Timepoint index to stitch for time-lapse data.
    register:
        If ``True``, run multiview-stitcher registration before fusion.
        If ``False``, fuse directly from stage metadata positions.
    reg_channel:
        Registration channel identifier (passed to multiview-stitcher).
        If omitted, multiview-stitcher default is used.
    stage_transform_key:
        Input transform key for stage metadata.
    registered_transform_key:
        Output transform key when ``register=True``.
    verbose:
        Print progress logs to stdout.
    log_every_tiles:
        Log cadence while constructing tiles.
    show_dask_progress:
        If ``True``, enable ``dask.diagnostics.ProgressBar`` during heavy steps.
    downsample_level:
        ND2 downsample level for frame reads. Each level downsamples XY by ``2**level``.
    output_chunksize:
        Optional output chunk shape for fusion (for APIs that support it), e.g.
        ``{"y": 4096, "x": 4096}``.
    fusion_batch_size:
        Optional fusion batch size (for APIs that support it).
    stage_transform_mode:
        How to apply ND2 stage/camera 2x2 matrix (`dStgLgCT`) to tile translations.
        Supported values:
        - ``"none"``: use raw stage coordinates (default).
        - ``"direct"``: apply matrix directly.
        - ``"inverse"``: apply inverse matrix (often useful for stage->image frame conversion).
        - ``"stitcher_direct"``: legacy alias of ``"direct"``.
        - ``"stitcher_inverse"``: legacy alias of ``"inverse"``.
    overlap_in_pixels:
        Optional overlap value forwarded to ``fusion.fuse`` when supported.
    blending_widths:
        Optional blending widths forwarded to ``fusion.fuse`` when supported.
    interpolation_order:
        Optional interpolation order forwarded to ``fusion.fuse`` when supported.
    dask_scheduler:
        Optional dask scheduler override, e.g. ``"threads"``.
    dask_num_workers:
        Optional dask worker count hint for threaded/process scheduler.
    use_ray_batches:
        If ``True`` and supported by the installed multiview-stitcher version,
        use Ray-based batch execution during fusion.
    ray_num_cpus:
        Optional CPU count passed to Ray batch execution.
    zarr_options:
        Optional ``zarr_options`` passthrough to ``fusion.fuse`` (when supported),
        e.g. ``{"ome_zarr": True}``.
    batch_options:
        Optional explicit ``batch_options`` passthrough to ``fusion.fuse``.
        If provided, this has priority over ``fusion_batch_size``/``use_ray_batches``.

    Returns
    -------
    Any
        The fused spatial image object returned by ``multiview_stitcher.fusion.fuse``.
    """
    def _log(message: str) -> None:
        if verbose:
            print(f"[limnd2.stitch] {message}", flush=True)

    def _inverse_2x2(a11: float, a12: float, a21: float, a22: float) -> tuple[float, float, float, float]:
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-12:
            raise ValueError("dStgLgCT matrix is singular; cannot invert.")
        return (a22 / det, -a12 / det, -a21 / det, a11 / det)

    def _transform_xy(
        x: float, y: float, m11: float, m12: float, m21: float, m22: float
    ) -> tuple[float, float]:
        if stage_transform_mode == "none":
            return x, y
        if stage_transform_mode in {"direct", "stitcher_direct"}:
            return (m11 * x + m12 * y, m21 * x + m22 * y)
        if stage_transform_mode in {"inverse", "stitcher_inverse"}:
            i11, i12, i21, i22 = _inverse_2x2(m11, m12, m21, m22)
            return (
                i11 * x + i12 * y,
                i21 * x + i22 * y,
            )
        raise ValueError(
            f"Invalid stage_transform_mode={stage_transform_mode!r}. "
            "Supported: 'none', 'direct', 'inverse', 'stitcher_direct', 'stitcher_inverse'."
        )

    fusion, msi_utils, registration, si_utils = _require_multiview_stitcher()
    da, delayed = _require_dask()
    progress_ctx = nullcontext()
    if show_dask_progress:
        try:
            from dask.diagnostics import ProgressBar  # type: ignore
            progress_ctx = ProgressBar()
        except Exception:
            _log("dask ProgressBar unavailable; continuing without it.")
    dask_ctx = nullcontext()
    if dask_scheduler is not None or dask_num_workers is not None:
        try:
            import dask  # type: ignore
            cfg: dict[str, Any] = {}
            if dask_scheduler is not None:
                cfg["scheduler"] = dask_scheduler
            if dask_num_workers is not None:
                cfg["num_workers"] = int(dask_num_workers)
            dask_ctx = dask.config.set(**cfg)
            _log(f"Using dask config override: {cfg}")
        except Exception:
            _log("Failed to apply dask config override; continuing with defaults.")

    close_reader = False
    reader: Nd2Reader
    if isinstance(nd2, Nd2Reader):
        reader = nd2
    else:
        reader = Nd2Reader(nd2)
        close_reader = True

    try:
        _log("Preparing stitch context.")
        exp = reader.experiment
        if exp is None:
            raise ValueError("ND2 file does not contain experiment metadata.")

        mp_level = exp.findLevel(ExperimentLoopType.eEtXYPosLoop)
        if mp_level is None or not isinstance(mp_level.uLoopPars, ExperimentXYPosLoop):
            raise ValueError("ND2 file does not contain a multipoint experiment.")
        mp = mp_level.uLoopPars
        if not mp.Points:
            raise ValueError("Multipoint loop has no parsed points.")

        dims = reader.dimensionSizes(skipSpectralLoop=True)
        t_count = int(dims.get("t", 1))
        m_count = int(dims.get("m", mp.uiCount))
        z_count = int(dims.get("z", 1))

        if t_count <= time_index:
            raise IndexError(f"time_index {time_index} out of range [0, {t_count - 1}]")
        if len(mp.Points) < m_count:
            raise ValueError(
                f"Multipoint loop expects {m_count} points but only {len(mp.Points)} parsed."
            )
        _log(f"Dimensions resolved: t={t_count}, m={m_count}, z={z_count}.")

        seq_lookup: dict[tuple[int, int, int], int] = {}
        for seq_index, idx in enumerate(exp.generateLoopIndexes(named=True)):
            seq_lookup[(idx.get("t", 0), idx.get("m", 0), idx.get("z", 0))] = seq_index
        _log(f"Sequence lookup built: {len(seq_lookup)} entries.")
        pm = reader.pictureMetadata
        m11 = float(pm.dStgLgCT11)
        m12 = float(pm.dStgLgCT12)
        m21 = float(pm.dStgLgCT21)
        m22 = float(pm.dStgLgCT22)
        _log(
            "Using dStgLgCT="
            f"[[{m11:.6f}, {m12:.6f}], [{m21:.6f}, {m22:.6f}]] "
            f"with stage_transform_mode='{stage_transform_mode}'."
        )
        if stage_transform_mode in {"stitcher_direct", "stitcher_inverse"}:
            _log(
                f"stage_transform_mode={stage_transform_mode!r} is a legacy alias; "
                "using coordinate-transform behavior."
            )

        attrs = reader.imageAttributes
        assert isinstance(downsample_level, int) and 0 <= downsample_level
        if downsample_level > 0:
            probe = np.asarray(reader.image(0, downsample_level=downsample_level))
            if probe.ndim == 2:
                probe = probe[..., np.newaxis]
            y_size, x_size, c_size = cast(tuple[int, int, int], probe.shape)
        else:
            y_size, x_size, c_size = attrs.shape
        dtype = np.dtype(attrs.dtype)

        def _read_frame_cyx(seq_index: int) -> np.ndarray:
            arr = np.asarray(reader.image(seq_index, downsample_level=downsample_level))
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            return np.moveaxis(arr, -1, 0)

        xy_spacing = 1.0
        if (
            reader.pictureMetadata is not None
            and reader.pictureMetadata.bCalibrated
            and reader.pictureMetadata.dCalibration > 0
        ):
            xy_spacing = float(reader.pictureMetadata.dCalibration)
        if downsample_level > 0:
            xy_spacing *= float(2**downsample_level)

        z_spacing = 1.0
        z_loop = find_zstack(exp)
        if z_count > 1 and z_loop is not None and z_loop.step is not None and z_loop.step > 0:
            z_spacing = float(z_loop.step)

        dims_list = ["c", "y", "x"] if z_count == 1 else ["c", "z", "y", "x"]
        scale = {"y": xy_spacing, "x": xy_spacing}
        if z_count > 1:
            scale = {"z": z_spacing, **scale}

        msims: list[Any] = []
        for m_index in range(m_count):
            point = mp.Points[m_index]
            frames: list[Any] = []
            for z_index in range(z_count):
                key = (time_index, m_index, z_index)
                if key not in seq_lookup:
                    raise ValueError(f"Missing frame for loop indexes {key}.")
                seq_idx = seq_lookup[key]
                frm = da.from_delayed(
                    delayed(_read_frame_cyx)(seq_idx),
                    shape=(c_size, y_size, x_size),
                    dtype=dtype,
                )
                frames.append(frm)

            tile_array = frames[0] if z_count == 1 else da.stack(frames, axis=1)
            tx, ty = _transform_xy(float(point.dPosX), float(point.dPosY), m11, m12, m21, m22)
            translation: dict[str, float] = {
                "y": ty,
                "x": tx,
            }
            affine: np.ndarray | None = None
            if z_count > 1:
                translation["z"] = float(point.dPosZ)

            sim = si_utils.get_sim_from_array(
                tile_array,
                dims=dims_list,
                scale=scale,
                translation=translation,
                affine=affine,
                transform_key=stage_transform_key,
            )
            msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
            if verbose and ((m_index + 1) % max(1, log_every_tiles) == 0 or m_index == m_count - 1):
                _log(f"Prepared msims: {m_index + 1}/{m_count}")

        transform_key = stage_transform_key
        if register:
            _log("Starting registration.")
            reg_kwargs: dict[str, Any] = {
                "transform_key": stage_transform_key,
                "new_transform_key": registered_transform_key,
                "plot_summary": False,
            }
            sig = inspect.signature(registration.register)
            if "pre_registration_pruning_method" in sig.parameters:
                reg_kwargs["pre_registration_pruning_method"] = None
            if reg_channel is not None and "reg_channel" in sig.parameters:
                reg_kwargs["reg_channel"] = reg_channel
            with progress_ctx:
                registration.register(msims, **reg_kwargs)
            transform_key = registered_transform_key
            _log("Registration finished.")

        fuse_kwargs: dict[str, Any] = {"transform_key": transform_key}
        if output_filename is not None:
            out = Path(output_filename)
            sig = inspect.signature(fusion.fuse)
            if out.name.lower().endswith(".ome.zarr") or out.suffix.lower() == ".zarr":
                if "output_zarr_url" in sig.parameters:
                    fuse_kwargs["output_zarr_url"] = str(out)
            if zarr_options is not None and "zarr_options" in sig.parameters:
                fuse_kwargs["zarr_options"] = dict(zarr_options)
            if overlap_in_pixels is not None and "overlap_in_pixels" in sig.parameters:
                fuse_kwargs["overlap_in_pixels"] = int(overlap_in_pixels)
            if blending_widths is not None and "blending_widths" in sig.parameters:
                sdims = (["z", "y", "x"] if z_count > 1 else ["y", "x"])
                if isinstance(blending_widths, (int, float)):
                    bw_norm: dict[str, float] = {
                        dim: float(blending_widths) for dim in sdims
                    }
                else:
                    bw_norm = {
                        str(k): float(v) for k, v in dict(blending_widths).items()
                    }
                    # Fill missing spatial dims with reasonable fallback.
                    fallback = (
                        bw_norm.get("y")
                        if "y" in bw_norm
                        else (bw_norm.get("x") if "x" in bw_norm else 0.0)
                    )
                    for dim in sdims:
                        bw_norm.setdefault(dim, float(fallback))
                fuse_kwargs["blending_widths"] = bw_norm
            if interpolation_order is not None and "interpolation_order" in sig.parameters:
                fuse_kwargs["interpolation_order"] = int(interpolation_order)
            if output_chunksize is not None and "output_chunksize" in sig.parameters:
                normalized_chunks = dict(output_chunksize)
                if "y" not in normalized_chunks:
                    normalized_chunks["y"] = int(max(1, min(y_size, 4096)))
                if "x" not in normalized_chunks:
                    normalized_chunks["x"] = int(max(1, min(x_size, 4096)))
                if z_count > 1 and "z" not in normalized_chunks:
                    # For 3D fusion, multiview-stitcher expects explicit chunk size for 'z'.
                    normalized_chunks["z"] = int(max(1, min(z_count, 8)))
                fuse_kwargs["output_chunksize"] = normalized_chunks
            if batch_options is not None and "batch_options" in sig.parameters:
                fuse_kwargs["batch_options"] = dict(batch_options)
            elif fusion_batch_size is not None:
                for key in ("batch_size", "blocks_per_batch", "n_blocks_per_batch"):
                    if key in sig.parameters:
                        fuse_kwargs[key] = int(fusion_batch_size)
                        break
                else:
                    if "batch_options" in sig.parameters:
                        batch_options: dict[str, Any] = {"n_batch": int(fusion_batch_size)}
                        if use_ray_batches:
                            try:
                                from multiview_stitcher import misc_utils  # type: ignore
                                batch_options["batch_func"] = misc_utils.process_batch_using_ray
                                batch_func_kwargs: dict[str, Any] = {}
                                if ray_num_cpus is not None:
                                    batch_func_kwargs["num_cpus"] = int(ray_num_cpus)
                                if batch_func_kwargs:
                                    batch_options["batch_func_kwargs"] = batch_func_kwargs
                            except Exception:
                                _log("Ray batch setup unavailable; using default batch function.")
                        fuse_kwargs["batch_options"] = batch_options
            if verbose:
                _log(f"fusion.fuse supports params: {sorted(sig.parameters.keys())}")
                _log(f"fusion.fuse kwargs used: {fuse_kwargs}")

        _log("Starting fusion.")
        with ExitStack() as stack:
            stack.enter_context(progress_ctx)
            stack.enter_context(dask_ctx)
            fused = fusion.fuse(
                [msi_utils.get_sim_from_msim(msim) for msim in msims], **fuse_kwargs
            )
        _log("Fusion finished.")

        if output_filename is not None:
            out = Path(output_filename)
            if not (
                out.name.lower().endswith(".ome.zarr")
                or out.suffix.lower() == ".zarr"
                and "output_zarr_url" in fuse_kwargs
            ):
                _log(f"Saving fused output to {out}.")
                _save_fused_output(fused, out)
                _log("Save finished.")

        return fused
    finally:
        if close_reader:
            reader.finalize()


def stitch_to_nd2_tiled(
    nd2: Nd2Reader | FileLikeObject,
    output_nd2_filename: str | Path,
    *,
    tile_size: tuple[int, int] = (1024, 1024),
    stream_all_timepoints: bool = False,
    time_indices: list[int] | tuple[int, ...] | None = None,
    verbose: bool = False,
    **stitch_kwargs: Any,
) -> Path:
    """
    Fuse multipoint data and write fused mosaic into a new ND2 using tile writes.

    Notes
    -----
    - Keeps memory bounded by computing and writing one tile at a time.
    - By default this stitches a single timepoint (`time_index` from `stitch_kwargs`,
      default 0) into Z sequence frames.
    - Use `stream_all_timepoints=True` (or `time_indices=[...]`) to stream multiple
      timepoints into one output ND2 with sequence index mapping `(t, z)`.
    """
    from .attributes import ImageAttributes
    from .experiment_factory import ExperimentFactory

    def _log(message: str) -> None:
        if verbose:
            print(f"[limnd2.stitch_to_nd2_tiled] {message}", flush=True)

    out_path = Path(output_nd2_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    close_reader = False
    reader: Nd2Reader
    if isinstance(nd2, Nd2Reader):
        reader = nd2
    else:
        reader = Nd2Reader(nd2)
        close_reader = True

    try:
        dims_src = reader.dimensionSizes(skipSpectralLoop=True)
        t_total = int(dims_src.get("t", 1))
        if time_indices is not None:
            selected_t = [int(t) for t in time_indices]
        elif stream_all_timepoints:
            selected_t = list(range(t_total))
        else:
            selected_t = [int(stitch_kwargs.get("time_index", 0))]

        if len(selected_t) == 0:
            raise ValueError("No time indices selected for stitched ND2 output.")
        for t in selected_t:
            if t < 0 or t >= t_total:
                raise IndexError(f"time index {t} out of range [0, {t_total - 1}]")

        # Avoid passing a stale/single time index into each iteration below.
        stitch_base_kwargs = dict(stitch_kwargs)
        stitch_base_kwargs.pop("time_index", None)

        _log(
            f"Preparing tiled ND2 stream for {len(selected_t)} timepoint(s): {selected_t[:5]}"
            + (" ..." if len(selected_t) > 5 else "")
        )

        def _normalize_fused_for_writing(fused_obj: Any) -> Any:
            sim_obj = fused_obj
            dims_obj = tuple(getattr(sim_obj, "dims", ()))
            sizes_obj = getattr(sim_obj, "sizes", None)
            if not dims_obj or sizes_obj is None:
                raise TypeError(
                    "Expected fused output with xarray-like `dims`/`sizes` attributes."
                )
            if "t" in dims_obj:
                sim_obj = sim_obj.isel(t=0)
                dims_obj = tuple(getattr(sim_obj, "dims", ()))
            if "c" not in dims_obj:
                sim_obj = sim_obj.expand_dims(dim={"c": [0]})
                dims_obj = tuple(getattr(sim_obj, "dims", ()))
            if "z" not in dims_obj:
                sim_obj = sim_obj.expand_dims(dim={"z": [0]})
            return sim_obj.transpose("z", "y", "x", "c")

        _log(f"Running stitch() for t={selected_t[0]}.")
        first_fused = stitch(
            reader,
            output_filename=None,
            verbose=verbose,
            time_index=selected_t[0],
            **stitch_base_kwargs,
        )
        first_sim = _normalize_fused_for_writing(first_fused)
        first_sizes = first_sim.sizes
        z_count = int(first_sizes["z"])
        height = int(first_sizes["y"])
        width = int(first_sizes["x"])
        c_count = int(first_sizes["c"])
        t_count_out = len(selected_t)

        bits = int(reader.imageAttributes.uiBpcSignificant)
        if bits not in (8, 16, 32):
            bits = max(8, int(np.dtype(reader.imageAttributes.dtype).itemsize * 8))

        _log(
            "Preparing output ND2 attrs: "
            f"shape(t,z,y,x,c)=({t_count_out},{z_count},{height},{width},{c_count}), bits={bits}."
        )
        attrs = ImageAttributes.create(
            width=width,
            height=height,
            component_count=c_count,
            bits=bits,
            sequence_count=t_count_out * z_count,
        )

        if len(tile_size) != 2:
            raise ValueError("tile_size must be (tile_width, tile_height).")
        tile_w = max(1, int(tile_size[0]))
        tile_h = max(1, int(tile_size[1]))

        with Nd2Writer(out_path) as writer:
            writer.imageAttributes = attrs

            z_step = 1.0
            z_loop = find_zstack(reader.experiment)
            if z_loop is not None and z_loop.step is not None and z_loop.step > 0:
                z_step = float(z_loop.step)
            t_step = 0.0
            t_level = (
                reader.experiment.findLevel(ExperimentLoopType.eEtTimeLoop)
                if reader.experiment is not None
                else None
            )
            if t_level is not None and t_level.uLoopPars.step is not None:
                t_step = float(t_level.uLoopPars.step)

            exp_kwargs: dict[str, Any] = {}
            if t_count_out > 1:
                exp_kwargs["t"] = {"count": t_count_out, "step": t_step}
            if z_count > 1:
                exp_kwargs["z"] = {"count": z_count, "step": z_step}
            if exp_kwargs:
                fac = ExperimentFactory(**exp_kwargs)
                writer.experiment = fac.createExperiment()

            if reader.pictureMetadata is not None:
                try:
                    writer.pictureMetadata = reader.pictureMetadata
                except Exception:
                    _log("Skipping pictureMetadata copy (incompatible with fused shape).")

            for out_t_index, src_t_index in enumerate(selected_t):
                sim = first_sim if out_t_index == 0 else _normalize_fused_for_writing(
                    stitch(
                        reader,
                        output_filename=None,
                        verbose=verbose,
                        time_index=src_t_index,
                        **stitch_base_kwargs,
                    )
                )
                sim_sizes = sim.sizes
                if (
                    int(sim_sizes["z"]) != z_count
                    or int(sim_sizes["y"]) != height
                    or int(sim_sizes["x"]) != width
                    or int(sim_sizes["c"]) != c_count
                ):
                    raise ValueError(
                        "Inconsistent fused shape across timepoints; cannot write one ND2 output."
                    )

                _log(
                    f"Writing timepoint {out_t_index + 1}/{t_count_out} (src t={src_t_index})."
                )
                for z_index in range(z_count):
                    frame = sim.isel(z=z_index)
                    seqindex = out_t_index * z_count + z_index
                    for y0 in range(0, height, tile_h):
                        y1 = min(y0 + tile_h, height)
                        for x0 in range(0, width, tile_w):
                            x1 = min(x0 + tile_w, width)
                            tile_da = frame.isel(y=slice(y0, y1), x=slice(x0, x1))
                            tile_data = tile_da.data
                            tile_np = (
                                np.asarray(tile_data.compute())
                                if hasattr(tile_data, "compute")
                                else np.asarray(tile_data)
                            )
                            if c_count == 1 and tile_np.ndim == 3 and tile_np.shape[-1] == 1:
                                tile_np = tile_np[..., 0]
                            writer.chunker.setImageTile(seqindex, x0, y0, tile_np)

        _log(f"Finished writing tiled fused ND2: {out_path}")
        return out_path
    finally:
        if close_reader:
            reader.finalize()

'''
