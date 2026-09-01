from __future__ import annotations

import importlib
import inspect
import re
import shutil
import threading
from copy import deepcopy
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import (
    ND2_CHUNK_FORMAT_ImageDataSeq_1p,
    NameNotInChunkmapError,
    UnexpectedCallError,
)
from .export import ExportProgressCallback, ExportProgressReporter

if TYPE_CHECKING:
    from .nd2 import Nd2Reader


TCZYX_AXES: tuple[str, str, str, str, str] = ("t", "c", "z", "y", "x")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_S3_URL_RE = re.compile(r"^s3://([^/]+)/(.+)$")
_DASK_AUTO_THRESHOLD_BYTES = 4 * 1024**3
_OME_ZARR_EXTRA_HINT = (
    'Install the optional OME-Zarr support with `pip install "limnd2[ome-zarr]"`.'
)


class _ProgressTracker:
    def __init__(
        self,
        callback: ExportProgressCallback | None,
        total: int,
        output_target: str | Path,
    ) -> None:
        self._reporter = ExportProgressReporter(callback)
        self._total = total
        self._completed = 0
        self._output_target = output_target
        self._lock = threading.Lock()

    def start(self, message: str | None = None) -> None:
        self._reporter.emit(
            0,
            self._total,
            self._output_target,
            message if message is not None else _phase_message("start-export"),
            phase="start-export",
        )

    def advance(self, phase: str, step: int = 1, message: str | None = None) -> None:
        with self._lock:
            self._completed += step
            current = self._completed
            total = self._total
        self._reporter.emit(
            current,
            total,
            self._output_target,
            message if message is not None else _phase_message(phase),
            phase=phase,
        )


def _phase_message(phase: str) -> str:
    messages = {
        "start-export": "Starting OME-Zarr export",
        "read-image-frame": "Reading an image frame for OME-Zarr export",
        "write-image-group": "Finished writing an OME-Zarr image group",
        "read-label-frame": "Reading a label frame for OME-Zarr export",
        "write-label-group": "Finished writing an OME-Zarr label group",
        "finalize-export": "Finished exporting OME-Zarr",
    }
    return messages.get(phase, phase)


def _missing_ome_zarr_dependency(package: str, *, context: str) -> ImportError:
    return ImportError(
        f'Missing optional dependency "{package}" required for {context}. '
        f"{_OME_ZARR_EXTRA_HINT}"
    )


def ensure_ome_zarr_dependencies(
    *,
    require_s3: bool = False,
    require_dask: bool = False,
) -> None:
    try:
        zarr = importlib.import_module("zarr")
    except ImportError as exc:
        raise _missing_ome_zarr_dependency("zarr", context="OME-Zarr export") from exc

    version = str(getattr(zarr, "__version__", "0"))
    if int(version.split(".", maxsplit=1)[0]) < 3:
        raise ImportError(
            f"OME-Zarr export requires zarr>=3; found zarr {version}. "
            f"{_OME_ZARR_EXTRA_HINT}"
        )

    try:
        importlib.import_module("ome_zarr")
    except ImportError as exc:
        raise _missing_ome_zarr_dependency("ome-zarr", context="OME-Zarr export") from exc

    if require_dask:
        try:
            importlib.import_module("dask.array")
        except ImportError as exc:
            raise _missing_ome_zarr_dependency(
                "dask", context="Dask-backed OME-Zarr export"
            ) from exc

    if require_s3:
        try:
            importlib.import_module("fsspec")
        except ImportError as exc:
            raise _missing_ome_zarr_dependency(
                "fsspec", context="direct S3 OME-Zarr export"
            ) from exc
        try:
            importlib.import_module("s3fs")
        except ImportError as exc:
            raise _missing_ome_zarr_dependency(
                "s3fs", context="direct S3 OME-Zarr export"
            ) from exc


def _progress_total(
    *,
    nt: int,
    nz: int,
    image_group_count: int,
    label_count: int,
    include_binaries: bool,
) -> int:
    image_frame_reads = image_group_count * nt * nz
    image_group_writes = image_group_count
    if not include_binaries or label_count <= 0:
        return image_frame_reads + image_group_writes
    label_frame_reads = image_group_count * label_count * nt * nz
    label_group_writes = image_group_count * label_count
    return image_frame_reads + image_group_writes + label_frame_reads + label_group_writes


def to_ome_zarr(
    nd2_reader: "Nd2Reader",
    path: str | Path,
    *,
    min_layer_size: int = 1024,
    chunks: tuple[int, int, int, int, int] = (1, 1, 1, 512, 512),
    shard_shape: tuple[int, int, int, int, int] | None = None,
    position: int | None = None,
    use_dask: bool | None = None,
    progress_callback: ExportProgressCallback | None = None,
    include_binaries: bool = False,
    include_well_info: bool = True,
    overwrite: bool = False,
    # include_ome_xml: bool = False,
) -> str | Path:
    """
    Export an ND2 file to OME-Zarr 0.5 / Zarr v3 groups.

    Each XY/multipoint position is written as its own image subgroup. Pixel
    arrays inside each subgroup use the conventional OME-NGFF axis order
    ``t, c, z, y, x``.
    """
    _validate_chunks(chunks)
    _validate_shard_shape(shard_shape, chunks)
    ensure_ome_zarr_dependencies(
        require_s3=_is_s3_path(path),
        require_dask=True,
    )

    try:
        import zarr  # type: ignore
        from ome_zarr.format import FormatV05  # type: ignore
        from ome_zarr.writer import (  # type: ignore
            write_image,
            write_labels,
            write_plate_metadata,
            write_well_metadata,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "OME-Zarr export dependencies were preflight-checked but imports still "
            f"failed. {_OME_ZARR_EXTRA_HINT}"
        ) from exc

    root, out_path = _open_output_root(
        zarr=zarr,
        path=path,
        overwrite=overwrite,
    )
    try:
        nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
        all_positions = _position_infos(nd2_reader, nm)
        if position is not None:
            if not (0 <= position < len(all_positions)):
                raise IndexError(
                    f"Position {position} out of range. File has {len(all_positions)} positions."
                )
            positions = [all_positions[position]]
        else:
            positions = all_positions
        is_single_position = len(positions) == 1
        wellplate_layout = (
            _wellplate_layout_info(nd2_reader, positions)
            if include_well_info and position is None
            else None
        )
        if wellplate_layout is None and not is_single_position:
            _write_root_metadata(root, series_paths=[pos.name for pos in positions])

        _validate_supported_channel_layout(nd2_reader, channel_count=nc)
        scale = _scale_values(nd2_reader)
        axes = _axes_metadata()
        label_axes = _label_axes_metadata()
        channel_metadata = _omero_channels(nd2_reader, nc)
        auto_window_channels = _default_window_channels(channel_metadata)
        max_layer = _max_layer_count(nx, ny, min_layer_size)
        scale_factors = _xy_scale_factors(max_layer)
        supports_scale_factors = "scale_factors" in inspect.signature(write_image).parameters
        fmt = FormatV05()
        resolved_use_dask = _resolve_use_dask(
            use_dask=use_dask,
            shape=(nt, nc, nz, ny, nx),
            dtype=np.dtype(nd2_reader.imageAttributes.dtype),
        )
        label_infos = _label_infos(nd2_reader) if include_binaries else []
        progress = _ProgressTracker(
            progress_callback,
            _progress_total(
                nt=nt,
                nz=nz,
                image_group_count=(
                    len(positions)
                    if wellplate_layout is None
                    else len(wellplate_layout.fields)
                ),
                label_count=len(label_infos),
                include_binaries=include_binaries,
            ),
            out_path,
        )
        progress.start(message=f"Starting OME-Zarr export to {out_path}")
        storage_options = _storage_options_for_levels(
            shape=(nt, nc, nz, ny, nx),
            chunks=chunks,
            shard_shape=shard_shape,
            levels=max_layer + 1,
        )

        frame_lookup = _frame_lookup(nd2_reader)
        if wellplate_layout is not None:
            _write_plate_layout(
                write_image=write_image,
                write_labels=write_labels,
                write_plate_metadata=write_plate_metadata,
                write_well_metadata=write_well_metadata,
                nd2_reader=nd2_reader,
                root=root,
                layout=wellplate_layout,
                frame_lookup=frame_lookup,
                image_shape=(nt, nc, nz, ny, nx),
                fmt=fmt,
                axes=axes,
                label_axes=label_axes,
                scale=scale,
                storage_options=storage_options,
                scale_factors=scale_factors,
                max_layer=max_layer,
                chunks=chunks,
                shard_shape=shard_shape,
                channel_metadata=channel_metadata,
                auto_window_channels=auto_window_channels,
                supports_scale_factors=supports_scale_factors,
                include_binaries=include_binaries,
                use_dask=resolved_use_dask,
                progress=progress,
                label_infos=label_infos,
            )
        elif is_single_position:
            pos = positions[0]
            data = _position_image_data(
                nd2_reader=nd2_reader,
                position_index=pos.index,
                shape=(nt, nc, nz, ny, nx),
                frame_lookup=frame_lookup,
                use_dask=resolved_use_dask,
                progress=progress,
            )
            coordinate_transformations = _coordinate_transformations(
                scale=scale,
                translation=(0.0, 0.0, pos.stage_z_um, pos.stage_y_um, pos.stage_x_um),
                levels=max_layer + 1,
            )
            omero_metadata = {
                "name": pos.label,
                "version": "0.4",
                "channels": deepcopy(channel_metadata),
                "rdefs": {
                    "model": "color" if nc > 1 else "greyscale",
                    "defaultT": 0,
                    "defaultZ": 0,
                },
            }
            _write_image_compat(
                write_image=write_image,
                data=data,
                group=root,
                fmt=fmt,
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                storage_options=storage_options,
                scale_factors=scale_factors,
                max_layer=max_layer,
                name=_output_name(out_path) or pos.label,
                omero_metadata=omero_metadata,
                supports_scale_factors=supports_scale_factors,
                progress=progress,
                auto_window_channels=auto_window_channels,
            )
            if include_binaries:
                _write_position_labels(
                    write_labels=write_labels,
                    nd2_reader=nd2_reader,
                    group=root,
                    position=pos,
                    frame_lookup=frame_lookup,
                    image_shape=(nt, nc, nz, ny, nx),
                    fmt=fmt,
                    axes=label_axes,
                    scale=scale,
                    chunks=chunks,
                    shard_shape=shard_shape,
                    scale_factors=scale_factors,
                    progress=progress,
                    label_infos=label_infos,
                )
        else:
            for pos in positions:
                data = _position_image_data(
                    nd2_reader=nd2_reader,
                    position_index=pos.index,
                    shape=(nt, nc, nz, ny, nx),
                    frame_lookup=frame_lookup,
                    use_dask=resolved_use_dask,
                    progress=progress,
                )
                coordinate_transformations = _coordinate_transformations(
                    scale=scale,
                    translation=(0.0, 0.0, pos.stage_z_um, pos.stage_y_um, pos.stage_x_um),
                    levels=max_layer + 1,
                )

                group = root.create_group(pos.name)
                omero_metadata = {
                    "name": pos.label,
                    "version": "0.4",
                    "channels": deepcopy(channel_metadata),
                    "rdefs": {
                        "model": "color" if nc > 1 else "greyscale",
                        "defaultT": 0,
                        "defaultZ": 0,
                    },
                }
                _write_image_compat(
                    write_image=write_image,
                    data=data,
                    group=group,
                    fmt=fmt,
                    axes=axes,
                    coordinate_transformations=coordinate_transformations,
                    storage_options=storage_options,
                    scale_factors=scale_factors,
                    max_layer=max_layer,
                    name=pos.label,
                    omero_metadata=omero_metadata,
                    supports_scale_factors=supports_scale_factors,
                    progress=progress,
                    auto_window_channels=auto_window_channels,
                )
                if include_binaries:
                    _write_position_labels(
                        write_labels=write_labels,
                        nd2_reader=nd2_reader,
                        group=group,
                        position=pos,
                        frame_lookup=frame_lookup,
                        image_shape=(nt, nc, nz, ny, nx),
                        fmt=fmt,
                        axes=label_axes,
                        scale=scale,
                        chunks=chunks,
                        shard_shape=shard_shape,
                        scale_factors=scale_factors,
                        progress=progress,
                        label_infos=label_infos,
                    )
        progress.advance(
            "finalize-export",
            step=0,
            message=f"Finished exporting OME-Zarr to {out_path}",
        )
        return out_path
    except Exception:
        with suppress(Exception):
            _cleanup_output_root(out_path)
        raise


def _is_s3_path(path: str | Path) -> bool:
    return str(path).startswith("s3://")


def _validate_s3_path(path: str) -> None:
    if _S3_URL_RE.match(path) is None:
        raise ValueError(
            "S3 OME-Zarr output must include a bucket and non-empty object prefix, "
            f"got: {path!r}"
        )


def _output_name(path: str | Path) -> str:
    raw = str(path).rstrip("/")
    if _is_s3_path(raw):
        return raw.rsplit("/", maxsplit=1)[-1]
    return Path(raw).stem


def _open_output_root(
    *,
    zarr: Any,
    path: str | Path,
    overwrite: bool,
) -> tuple[Any, str | Path]:
    if _is_s3_path(path):
        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise _missing_ome_zarr_dependency(
                "fsspec", context="direct S3 OME-Zarr export"
            ) from exc

        out_path = str(path)
        _validate_s3_path(out_path)
        store = zarr.storage.FsspecStore.from_url(out_path, read_only=False)
        sync_fs, sync_path = fsspec.core.url_to_fs(out_path)
        if sync_fs.exists(sync_path):
            if not overwrite:
                raise FileExistsError(f"OME-Zarr output already exists: {out_path}")
            sync_fs.rm(sync_path, recursive=True)
        root = zarr.open_group(store=store, mode="w", zarr_format=3)
        return root, out_path

    out_path = Path(path)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"OME-Zarr output already exists: {out_path}")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    root = zarr.open_group(str(out_path), mode="w", zarr_format=3)
    return root, out_path


def _cleanup_output_root(path: str | Path) -> None:
    if _is_s3_path(path):
        import fsspec

        out_path = str(path)
        sync_fs, sync_path = fsspec.core.url_to_fs(out_path)
        if sync_fs.exists(sync_path):
            sync_fs.rm(sync_path, recursive=True)
        return

    out_path = Path(path)
    if not out_path.exists():
        return
    if out_path.is_dir():
        shutil.rmtree(out_path)
    else:
        out_path.unlink(missing_ok=True)


def _write_root_metadata(
    root: Any,
    *,
    series_paths: list[str],
) -> None:
    root.attrs.update(
        {
            "ome": {"bioformats2raw.layout": 3},
        }
    )
    ome_group = root.create_group("OME")
    ome_group.attrs["ome"] = {"version": "0.5", "series": series_paths}


def _write_plate_layout(
    *,
    write_image: Any,
    write_labels: Any,
    write_plate_metadata: Any,
    write_well_metadata: Any,
    nd2_reader: "Nd2Reader",
    root: Any,
    layout: _WellplateLayoutInfo,
    frame_lookup: dict[tuple[int, int, int], int],
    image_shape: tuple[int, int, int, int, int],
    fmt: Any,
    axes: list[dict[str, str]],
    label_axes: list[dict[str, str]],
    scale: tuple[float, float, float, float, float],
    storage_options: list[dict[str, tuple[int, int, int, int, int]]],
    scale_factors: list[dict[str, int]],
    max_layer: int,
    chunks: tuple[int, int, int, int, int],
    shard_shape: tuple[int, int, int, int, int] | None,
    channel_metadata: list[dict[str, Any]],
    auto_window_channels: list[bool],
    supports_scale_factors: bool,
    include_binaries: bool,
    use_dask: bool,
    progress: _ProgressTracker,
    label_infos: list[_LabelInfo],
) -> None:
    write_plate_metadata(
        root,
        rows=layout.rows,
        columns=layout.columns,
        wells=sorted({f"{field.row}/{field.col}" for field in layout.fields}),
        fmt=fmt,
        field_count=layout.field_count,
        name=layout.plate_name,
    )

    fields_by_well: dict[tuple[str, str], list[_WellFieldInfo]] = {}
    for field in layout.fields:
        fields_by_well.setdefault((field.row, field.col), []).append(field)

    nt, nc, nz, ny, nx = image_shape
    for (row, col), fields in sorted(fields_by_well.items()):
        fields.sort(key=lambda field: int(field.field))
        row_group = root.require_group(row)
        well_group = row_group.require_group(col)
        write_well_metadata(
            well_group,
            images=[field.field for field in fields],
            fmt=fmt,
        )

        for field in fields:
            pos = field.position
            data = _position_image_data(
                nd2_reader=nd2_reader,
                position_index=pos.index,
                shape=(nt, nc, nz, ny, nx),
                frame_lookup=frame_lookup,
                use_dask=use_dask,
                progress=progress,
            )
            coordinate_transformations = _coordinate_transformations(
                scale=scale,
                translation=(0.0, 0.0, pos.stage_z_um, pos.stage_y_um, pos.stage_x_um),
                levels=max_layer + 1,
            )
            image_group = well_group.create_group(field.field)
            display_name = pos.label or field.well_name or f"{row}{col}"
            omero_metadata = {
                "name": display_name,
                "version": "0.4",
                "channels": deepcopy(channel_metadata),
                "rdefs": {
                    "model": "color" if nc > 1 else "greyscale",
                    "defaultT": 0,
                    "defaultZ": 0,
                },
            }
            _write_image_compat(
                write_image=write_image,
                data=data,
                group=image_group,
                fmt=fmt,
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                storage_options=storage_options,
                scale_factors=scale_factors,
                max_layer=max_layer,
                name=display_name,
                omero_metadata=omero_metadata,
                supports_scale_factors=supports_scale_factors,
                progress=progress,
                auto_window_channels=auto_window_channels,
            )
            if include_binaries:
                _write_position_labels(
                    write_labels=write_labels,
                    nd2_reader=nd2_reader,
                    group=image_group,
                    position=pos,
                    frame_lookup=frame_lookup,
                    image_shape=(nt, nc, nz, ny, nx),
                    fmt=fmt,
                    axes=label_axes,
                    scale=scale,
                    chunks=chunks,
                    shard_shape=shard_shape,
                    scale_factors=scale_factors,
                    progress=progress,
                    label_infos=label_infos,
                )


def _write_image_compat(
    *,
    write_image: Any,
    data: Any,
    group: Any,
    fmt: Any,
    axes: list[dict[str, str]],
    coordinate_transformations: list[list[dict[str, list[float] | str]]],
    storage_options: list[dict[str, tuple[int, int, int, int, int]]],
    scale_factors: list[dict[str, int]],
    max_layer: int,
    name: str,
    omero_metadata: dict[str, Any],
    supports_scale_factors: bool,
    progress: _ProgressTracker,
    auto_window_channels: list[bool] | None = None,
) -> None:
    _write_image_numeric_levels(
        data=data,
        group=group,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        scale_factors=scale_factors,
        max_layer=max_layer,
        name=name,
        omero_metadata=omero_metadata,
        supports_scale_factors=supports_scale_factors,
    )
    if auto_window_channels and any(auto_window_channels):
        _apply_sampled_display_windows(
            group=group,
            omero_metadata=omero_metadata,
            auto_window_channels=auto_window_channels,
        )
    progress.advance("write-image-group")


class _PositionInfo:
    def __init__(
        self,
        *,
        index: int,
        name: str,
        label: str,
        stage_x_um: float,
        stage_y_um: float,
        stage_z_um: float,
    ) -> None:
        self.index = index
        self.name = name
        self.label = label
        self.stage_x_um = stage_x_um
        self.stage_y_um = stage_y_um
        self.stage_z_um = stage_z_um


class _LabelInfo:
    def __init__(
        self,
        *,
        layer_id: int,
        name: str,
        label: str,
        color: str | None,
    ) -> None:
        self.layer_id = layer_id
        self.name = name
        self.label = label
        self.color = color


class _WellFieldInfo:
    def __init__(
        self,
        *,
        position: _PositionInfo,
        row: str,
        col: str,
        field: str,
        well_name: str,
    ) -> None:
        self.position = position
        self.row = row
        self.col = col
        self.field = field
        self.well_name = well_name


class _WellplateLayoutInfo:
    def __init__(
        self,
        *,
        rows: list[str],
        columns: list[str],
        fields: list[_WellFieldInfo],
        plate_name: str | None,
        field_count: int,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.fields = fields
        self.plate_name = plate_name
        self.field_count = field_count


def _position_infos(nd2_reader: "Nd2Reader", count: int) -> list[_PositionInfo]:
    xy_loop = None
    if nd2_reader.experiment is not None:
        from .experiment import ExperimentLoopType, ExperimentXYPosLoop

        level = nd2_reader.experiment.findLevel(ExperimentLoopType.eEtXYPosLoop)
        if level is not None and isinstance(level.uLoopPars, ExperimentXYPosLoop):
            xy_loop = level.uLoopPars

    well_items_by_seq: dict[int, Any] = {}
    if nd2_reader.wellplateFrameInfo is not None:
        for item in nd2_reader.wellplateFrameInfo:
            well_items_by_seq.setdefault(item.seqIndex, item)

    positions: list[_PositionInfo] = []
    for index in range(max(1, count)):
        point = (
            xy_loop.Points[index]
            if xy_loop is not None and xy_loop.Points and index < len(xy_loop.Points)
            else None
        )
        well_item = well_items_by_seq.get(index)

        label = ""
        if well_item is not None and getattr(well_item, "wellName", ""):
            label = str(well_item.wellName)
        elif point is not None and point.dPosName:
            label = str(point.dPosName)
        else:
            label = str(index)

        positions.append(
            _PositionInfo(
                index=index,
                name=str(index),
                label=label,
                stage_x_um=float(point.dPosX if point is not None else 0.0),
                stage_y_um=float(point.dPosY if point is not None else 0.0),
                stage_z_um=float(point.dPosZ if point is not None else 0.0),
            )
        )
    return positions


def _wellplate_layout_info(
    nd2_reader: "Nd2Reader", positions: list[_PositionInfo]
) -> _WellplateLayoutInfo | None:
    wellplate_desc = nd2_reader.wellplateDesc
    wellplate_info = nd2_reader.wellplateFrameInfo
    if wellplate_desc is None or wellplate_info is None or not len(wellplate_info):
        return None
    nwells = wellplate_info.nwells
    if nwells <= 0 or not positions or len(positions) % nwells != 0:
        return None

    representative_by_well: dict[int, Any] = {}
    for item in wellplate_info:
        prev = representative_by_well.get(item.wellIndex)
        if prev is None or int(item.seqIndex) < int(prev.seqIndex):
            representative_by_well[item.wellIndex] = item
    if len(representative_by_well) != nwells:
        return None

    ordered_wells = sorted(
        representative_by_well.values(),
        key=lambda item: (
            int(item.seqIndex),
            int(item.wellRowIndex),
            int(item.wellColIndex),
            int(item.wellIndex),
        ),
    )
    positions_per_well = len(positions) // nwells
    fields: list[_WellFieldInfo] = []
    for pos in positions:
        well_ordinal = pos.index // positions_per_well
        field_ordinal = pos.index % positions_per_well
        well_item = ordered_wells[well_ordinal]
        fields.append(
            _WellFieldInfo(
                position=pos,
                row=_row_label_from_index(int(well_item.wellRowIndex)),
                col=str(int(well_item.wellColIndex) + 1),
                field=str(field_ordinal),
                well_name=str(getattr(well_item, "wellName", "") or ""),
            )
        )

    row_labels = sorted({field.row for field in fields}, key=_row_sort_key)
    col_labels = sorted({field.col for field in fields}, key=lambda value: int(value))
    plate_name = getattr(wellplate_desc, "name", "") or None
    return _WellplateLayoutInfo(
        rows=row_labels,
        columns=col_labels,
        fields=fields,
        plate_name=str(plate_name) if plate_name else None,
        field_count=positions_per_well,
    )


def _row_label_from_index(index: int) -> str:
    if index < 0:
        raise ValueError("Well row index must be non-negative.")
    value = index + 1
    chars: list[str] = []
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _row_sort_key(label: str) -> tuple[int, ...]:
    return tuple(ord(char) for char in label)


def _safe_name(value: str) -> str:
    clean = _SAFE_NAME_RE.sub("_", value.strip()).strip("._")
    return clean or "0"


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    index = 1
    while f"{base}_{index}" in used:
        index += 1
    name = f"{base}_{index}"
    used.add(name)
    return name


def _frame_lookup(nd2_reader: "Nd2Reader") -> dict[tuple[int, int, int], int]:
    lookup: dict[tuple[int, int, int], int] = {}
    indices = nd2_reader.generateLoopIndexes(named=True)
    if not indices:
        return {(0, 0, 0): 0}
    for frame_index, coords in enumerate(indices):
        position_index = _position_index_from_coords(nd2_reader, coords)
        lookup[
            (
                int(coords.get("t", 0)),
                position_index,
                int(coords.get("z", 0)),
            )
        ] = frame_index
    return lookup


def _label_infos(nd2_reader: "Nd2Reader") -> list[_LabelInfo]:
    metadata = list(nd2_reader.binaryRasterMetadata)
    used: set[str] = {"labels"}
    labels: list[_LabelInfo] = []
    for index, item in enumerate(metadata):
        base_label = item.name or f"Label_{index}"
        safe_name = _unique_name(_safe_name(base_label), used)
        color = _binary_color_hex(getattr(item, "color", None))
        labels.append(
            _LabelInfo(
                layer_id=int(item.id),
                name=safe_name,
                label=base_label,
                color=color,
            )
        )
    return labels


def _position_index_from_coords(
    nd2_reader: "Nd2Reader",
    coords: dict[str, int],
) -> int:
    if "w" in coords and "m" in coords:
        wellplate = nd2_reader.wellplateFrameInfo
        if wellplate is None or wellplate.nwells <= 0:
            raise ValueError(
                "Loop coordinates contain both well and multipoint indices, "
                "but wellplate metadata is unavailable."
            )
        total_positions = nd2_reader.imageDataShape[1]
        if total_positions % wellplate.nwells != 0:
            raise ValueError(
                "Cannot reconstruct combined position index for wellplate export: "
                f"{total_positions} positions are not evenly divisible by "
                f"{wellplate.nwells} wells."
            )
        positions_per_well = total_positions // wellplate.nwells
        return int(coords["w"]) * positions_per_well + int(coords["m"])
    return int(coords.get("m", coords.get("w", 0)))


def _frame_chunk_present(nd2_reader: "Nd2Reader", frame_index: int) -> bool:
    chunk_name = ND2_CHUNK_FORMAT_ImageDataSeq_1p % frame_index
    chunker = getattr(nd2_reader, "_chunker", None)
    if chunker is not None and hasattr(chunker, "_chunk_pos"):
        try:
            chunker._chunk_pos(chunk_name)
            return True
        except NameNotInChunkmapError:
            return False
    try:
        return nd2_reader.chunk(chunk_name) is not None
    except (UnexpectedCallError, AttributeError):
        return True


def _read_position_tczyx(
    *,
    nd2_reader: "Nd2Reader",
    position_index: int,
    shape: tuple[int, int, int, int, int],
    frame_lookup: dict[tuple[int, int, int], int],
    progress: _ProgressTracker,
) -> np.ndarray:
    nt, nc, nz, ny, nx = shape
    data = np.zeros(shape, dtype=nd2_reader.imageAttributes.dtype)
    frame_indices = _position_frame_indices(
        nd2_reader=nd2_reader,
        position_index=position_index,
        shape=shape,
        frame_lookup=frame_lookup,
    )
    for t, z_frames in enumerate(frame_indices):
        for z, frame_index in enumerate(z_frames):
            data[t, :, z, :, :] = _read_frame_cyx(
                nd2_reader,
                frame_index,
                ny=ny,
                nx=nx,
                nc=nc,
                progress=progress,
            )
    return data


def _position_image_data(
    *,
    nd2_reader: "Nd2Reader",
    position_index: int,
    shape: tuple[int, int, int, int, int],
    frame_lookup: dict[tuple[int, int, int], int],
    use_dask: bool,
    progress: _ProgressTracker,
) -> Any:
    if use_dask:
        return _delayed_position_tczyx(
            nd2_reader=nd2_reader,
            position_index=position_index,
            shape=shape,
            frame_lookup=frame_lookup,
            progress=progress,
        )
    return _read_position_tczyx(
        nd2_reader=nd2_reader,
        position_index=position_index,
        shape=shape,
        frame_lookup=frame_lookup,
        progress=progress,
    )


def _delayed_position_tczyx(
    *,
    nd2_reader: "Nd2Reader",
    position_index: int,
    shape: tuple[int, int, int, int, int],
    frame_lookup: dict[tuple[int, int, int], int],
    progress: _ProgressTracker,
) -> Any:
    try:
        import dask.array as da  # type: ignore
        from dask.delayed import delayed  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            'Dask-backed OME-Zarr export requires the "dask" package.'
        ) from exc

    nt, nc, nz, ny, nx = shape
    frame_indices = _position_frame_indices(
        nd2_reader=nd2_reader,
        position_index=position_index,
        shape=shape,
        frame_lookup=frame_lookup,
    )

    t_stacks = []
    for z_indices in frame_indices:
        z_stacks = []
        for frame_index in z_indices:
            delayed_frame = delayed(_read_frame_cyx)(
                nd2_reader,
                frame_index,
                ny=ny,
                nx=nx,
                nc=nc,
                progress=progress,
            )
            z_stacks.append(
                da.from_delayed(
                    delayed_frame,
                    shape=(nc, ny, nx),
                    dtype=nd2_reader.imageAttributes.dtype,
                )
            )
        t_stacks.append(da.stack(z_stacks, axis=1))
    return da.stack(t_stacks, axis=0).reshape((nt, nc, nz, ny, nx))


def _position_frame_indices(
    *,
    nd2_reader: "Nd2Reader",
    position_index: int,
    shape: tuple[int, int, int, int, int],
    frame_lookup: dict[tuple[int, int, int], int],
) -> list[list[int]]:
    nt, _nc, nz, _ny, _nx = shape
    frame_indices: list[list[int]] = []
    missing_frames: list[tuple[int, int, int]] = []
    for t in range(nt):
        z_frames: list[int] = []
        for z in range(nz):
            frame_index = frame_lookup.get((t, position_index, z))
            if frame_index is None or not _frame_chunk_present(nd2_reader, frame_index):
                missing_frames.append((t, position_index, z))
                continue
            z_frames.append(frame_index)
        frame_indices.append(z_frames)
    if missing_frames:
        preview = ", ".join(
            f"(t={t}, m={m}, z={z})" for t, m, z in missing_frames[:5]
        )
        remainder = len(missing_frames) - min(len(missing_frames), 5)
        if remainder > 0:
            preview = f"{preview}, ... (+{remainder} more)"
        raise ValueError(
            "Missing frame data for exported OME-Zarr position "
            f"{position_index}: {preview}."
        )
    return frame_indices


def _read_frame_cyx(
    nd2_reader: "Nd2Reader",
    frame_index: int,
    *,
    ny: int,
    nx: int,
    nc: int,
    progress: _ProgressTracker | None = None,
) -> np.ndarray:
    frame = np.asarray(nd2_reader.image(frame_index))
    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]
    if frame.shape[0] != ny or frame.shape[1] != nx:
        raise ValueError(
            f"Unexpected frame shape for frame {frame_index}: "
            f"{frame.shape}, expected Y/X {ny, nx}."
        )
    if frame.shape[2] < nc:
        raise ValueError(
            f"Unexpected component count for frame {frame_index}: "
            f"{frame.shape[2]}, expected at least {nc}."
        )
    result = np.moveaxis(frame[:, :, :nc], -1, 0)
    if progress is not None:
        progress.advance("read-image-frame")
    return result


def _read_position_tzyx_binary(
    *,
    nd2_reader: "Nd2Reader",
    layer_id: int,
    position_index: int,
    shape: tuple[int, int, int, int],
    frame_lookup: dict[tuple[int, int, int], int],
    progress: _ProgressTracker,
) -> np.ndarray:
    nt, nz, ny, nx = shape
    data = np.zeros(shape, dtype=np.uint32)
    for t in range(nt):
        for z in range(nz):
            frame_index = frame_lookup.get((t, position_index, z))
            if frame_index is None:
                continue
            frame = np.asarray(nd2_reader.binaryRasterData(layer_id, frame_index))
            if frame.shape != (ny, nx):
                raise ValueError(
                    f"Unexpected binary frame shape for layer {layer_id}, frame "
                    f"{frame_index}: {frame.shape}, expected {(ny, nx)}."
                )
            data[t, z, :, :] = frame
            progress.advance("read-label-frame")
    return data


def _scale_values(nd2_reader: "Nd2Reader") -> tuple[float, float, float, float, float]:
    t_ms, _, z_um, y_um, x_um, _ = nd2_reader.imageDataCalibration
    return (
        float(t_ms or 1.0),
        1.0,
        float(z_um or 1.0),
        float(y_um or 1.0),
        float(x_um or 1.0),
    )


def _resolve_use_dask(
    *,
    use_dask: bool | None,
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype[Any],
) -> bool:
    if use_dask is not None:
        return use_dask
    return _estimated_array_bytes(shape=shape, dtype=dtype) >= _DASK_AUTO_THRESHOLD_BYTES


def _estimated_array_bytes(
    *,
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype[Any],
) -> int:
    itemsize = int(dtype.itemsize)
    total = itemsize
    for dim in shape:
        total *= int(dim)
    return total


def _validate_supported_channel_layout(
    nd2_reader: "Nd2Reader", *, channel_count: int
) -> None:
    planes = list(getattr(nd2_reader.pictureMetadata, "channels", []))
    multi_component_planes = [
        plane for plane in planes if int(getattr(plane, "uiCompCount", 1)) > 1
    ]

    if nd2_reader.isRgb:
        if channel_count != 3:
            raise ValueError(
                "Pure RGB OME-Zarr export requires exactly 3 components; "
                f"found {channel_count}."
            )
        if len(planes) != 1 or len(multi_component_planes) != 1:
            raise ValueError(
                "Pure RGB OME-Zarr export requires a single RGB picture plane "
                "with no additional optical channels."
            )
        return

    if multi_component_planes:
        raise ValueError(
            "OME-NGFF does not support files with both RGB samples and multiple "
            "optical channels. This ND2 file contains multi-component picture "
            "planes alongside additional optical channels, which cannot be "
            "represented in the current limnd2 OME-Zarr export path."
        )


def _axes_metadata() -> list[dict[str, str]]:
    return [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def _label_axes_metadata() -> list[dict[str, str]]:
    return [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def _coordinate_transformations(
    *,
    scale: tuple[float, float, float, float, float],
    translation: tuple[float, float, float, float, float],
    levels: int,
) -> list[list[dict[str, list[float] | str]]]:
    transforms: list[list[dict[str, list[float] | str]]] = []
    for level in range(levels):
        factor = float(2**level)
        transforms.append(
            [
                {
                    "type": "scale",
                    "scale": [
                        scale[0],
                        scale[1],
                        scale[2],
                        scale[3] * factor,
                        scale[4] * factor,
                    ],
                },
                {"type": "translation", "translation": list(translation)},
            ]
        )
    return transforms


def _label_coordinate_transformations(
    *,
    scale: tuple[float, float, float, float],
    translation: tuple[float, float, float, float],
    levels: int,
) -> list[list[dict[str, list[float] | str]]]:
    transforms: list[list[dict[str, list[float] | str]]] = []
    for level in range(levels):
        factor = float(2**level)
        transforms.append(
            [
                {
                    "type": "scale",
                    "scale": [
                        scale[0],
                        scale[1],
                        scale[2] * factor,
                        scale[3] * factor,
                    ],
                },
                {"type": "translation", "translation": list(translation)},
            ]
        )
    return transforms


def _xy_scale_factors(level_count: int) -> list[dict[str, int]]:
    return [
        {"t": 1, "c": 1, "z": 1, "y": 2**level, "x": 2**level}
        for level in range(1, level_count + 1)
    ]


def _storage_options_for_levels(
    *,
    shape: tuple[int, int, int, int, int],
    chunks: tuple[int, int, int, int, int],
    shard_shape: tuple[int, int, int, int, int] | None,
    levels: int,
) -> list[dict[str, tuple[int, int, int, int, int]]]:
    options: list[dict[str, tuple[int, int, int, int, int]]] = []
    for level_shape in _pyramid_level_shapes(shape, levels):
        level_chunks = _chunks_for_shape(chunks, level_shape)
        level_options: dict[str, tuple[int, int, int, int, int]] = {
            "chunks": level_chunks
        }
        if shard_shape is not None:
            level_options["shards"] = _shards_for_level(level_chunks, shard_shape)
        options.append(level_options)
    return options


def _storage_options_for_label_levels(
    *,
    shape: tuple[int, int, int, int],
    chunks: tuple[int, int, int, int],
    shard_shape: tuple[int, int, int, int] | None,
    levels: int,
) -> list[dict[str, tuple[int, ...]]]:
    options: list[dict[str, tuple[int, ...]]] = []
    for level_shape in _label_pyramid_level_shapes(shape, levels):
        level_chunks = _label_chunks_for_shape(chunks, level_shape)
        level_options: dict[str, tuple[int, ...]] = {"chunks": level_chunks}
        if shard_shape is not None:
            level_options["shards"] = _label_shards_for_level(level_chunks, shard_shape)
        options.append(level_options)
    return options


def _pyramid_level_shapes(
    shape: tuple[int, int, int, int, int], levels: int
) -> list[tuple[int, int, int, int, int]]:
    if levels <= 0:
        return []
    nt, nc, nz, ny, nx = shape
    shapes = []
    cur_y, cur_x = ny, nx
    for _ in range(levels):
        shapes.append((nt, nc, nz, cur_y, cur_x))
        cur_y = max(1, (cur_y + 1) // 2)
        cur_x = max(1, (cur_x + 1) // 2)
    return shapes


def _label_pyramid_level_shapes(
    shape: tuple[int, int, int, int], levels: int
) -> list[tuple[int, int, int, int]]:
    if levels <= 0:
        return []
    nt, nz, ny, nx = shape
    shapes = []
    cur_y, cur_x = ny, nx
    for _ in range(levels):
        shapes.append((nt, nz, cur_y, cur_x))
        cur_y = max(1, (cur_y + 1) // 2)
        cur_x = max(1, (cur_x + 1) // 2)
    return shapes


def _shards_for_level(
    chunks: tuple[int, int, int, int, int],
    shard_shape: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    result = []
    for chunk_dim, shard_dim in zip(chunks, shard_shape):
        shard_dim = max(chunk_dim, shard_dim)
        shard_dim -= shard_dim % chunk_dim
        if shard_dim == 0:
            shard_dim = chunk_dim
        result.append(shard_dim)
    return tuple(result)


def _label_shards_for_level(
    chunks: tuple[int, int, int, int],
    shard_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    result = []
    for chunk_dim, shard_dim in zip(chunks, shard_shape):
        shard_dim = max(chunk_dim, shard_dim)
        shard_dim -= shard_dim % chunk_dim
        if shard_dim == 0:
            shard_dim = chunk_dim
        result.append(shard_dim)
    return tuple(result)


def _omero_channels(nd2_reader: "Nd2Reader", channel_count: int) -> list[dict[str, Any]]:
    names = _channel_names(nd2_reader, channel_count)
    colors = _channel_colors(nd2_reader, channel_count)
    ranges = _channel_ranges(nd2_reader, channel_count)
    channels = []
    for index in range(channel_count):
        start, end, min_value, max_value = ranges[index]
        channels.append(
            {
                "label": names[index],
                "color": colors[index].lstrip("#").upper(),
                "active": True,
                "window": {
                    "start": float(start),
                    "end": float(end),
                    "min": float(min_value),
                    "max": float(max_value),
                },
            }
        )
    return channels


def _channel_names(nd2_reader: "Nd2Reader", channel_count: int) -> list[str]:
    names: list[str] = []
    with suppress(Exception):
        names = [str(name) for name in nd2_reader.pictureMetadata.componentNames]
    if len(names) < channel_count:
        if nd2_reader.isRgb:
            fallback = ["Red", "Green", "Blue"]
            names.extend(fallback[i] for i in range(len(names), min(channel_count, len(fallback))))
        if len(names) < channel_count:
            names.extend(f"Channel_{i + 1}" for i in range(len(names), channel_count))
    return names[:channel_count]


def _channel_colors(nd2_reader: "Nd2Reader", channel_count: int) -> list[str]:
    colors: list[str] = []
    with suppress(Exception):
        colors = [
            _tuple_to_hex(color)
            for color in nd2_reader.pictureMetadata.componentColors
        ]
    if len(colors) < channel_count:
        if nd2_reader.isRgb:
            fallback = ["#ff0000", "#00ff00", "#0000ff"]
            colors.extend(
                fallback[i] for i in range(len(colors), min(channel_count, len(fallback)))
            )
        if len(colors) < channel_count:
            fallback = ["#ffffff", "#00ffff", "#ff00ff", "#ffff00", "#ff8000"]
            colors.extend(
                fallback[i % len(fallback)] for i in range(channel_count - len(colors))
            )
    return colors[:channel_count]


def _tuple_to_hex(color: tuple[float, float, float]) -> str:
    vals = [
        max(0, min(255, int(round(component * 255 if component <= 1 else component))))
        for component in color
    ]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _channel_ranges(
    nd2_reader: "Nd2Reader", channel_count: int
) -> list[tuple[float, float, float, float]]:
    min_value, max_value = nd2_reader.imageDataRange
    ranges = []
    with suppress(Exception):
        comp_range = np.asarray(nd2_reader.compRange)
        for index in range(min(channel_count, comp_range.shape[0])):
            ranges.append(
                (
                    float(comp_range[index, 0]),
                    float(comp_range[index, 1]),
                    float(min_value),
                    float(max_value),
                )
            )
    while len(ranges) < channel_count:
        ranges.append((float(min_value), float(max_value), float(min_value), float(max_value)))
    return ranges[:channel_count]


def _default_window_channels(channels: list[dict[str, Any]]) -> list[bool]:
    """Identify channels whose ND2 display range fell back to the dtype range."""
    result = []
    for channel in channels:
        window = channel.get("window", {})
        result.append(
            float(window.get("start", 0.0)) == float(window.get("min", 0.0))
            and float(window.get("end", 0.0)) == float(window.get("max", 0.0))
        )
    return result


def _apply_sampled_display_windows(
    *,
    group: Any,
    omero_metadata: dict[str, Any],
    auto_window_channels: list[bool],
    max_frames: int = 64,
    samples_per_frame: int = 4096,
) -> None:
    """Replace missing ND2 display windows with sampled data percentiles.

    The image has already been written at this point. Sampling level 0 avoids
    rereading the ND2 while bounding the additional I/O and memory use.
    """
    array = group["0"]
    nt, nc, nz, ny, nx = array.shape
    frame_count = max(1, nt * nz)
    frame_step = max(1, int(np.ceil(np.sqrt(frame_count / max_frames))))
    spatial_step = max(1, int(np.ceil(np.sqrt((ny * nx) / samples_per_frame))))
    channels = omero_metadata.get("channels", [])

    for channel_index, needs_window in enumerate(auto_window_channels[:nc]):
        if not needs_window or channel_index >= len(channels):
            continue
        samples = []
        for t in range(0, nt, frame_step):
            for z in range(0, nz, frame_step):
                values = np.asarray(
                    array[t, channel_index, z, ::spatial_step, ::spatial_step]
                ).reshape(-1)
                if np.issubdtype(values.dtype, np.floating):
                    values = values[np.isfinite(values)]
                if values.size:
                    samples.append(values)
        if not samples:
            continue
        values = np.concatenate(samples)
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        start, end = np.percentile(values, (0.01, 99.9))
        if start >= end:
            start, end = observed_min, observed_max
        window = channels[channel_index].setdefault("window", {})
        window.update(
            {
                "start": float(start),
                "end": float(end),
                "min": observed_min,
                "max": observed_max,
            }
        )

    ome = dict(group.attrs.get("ome", {}))
    ome["omero"] = omero_metadata
    group.attrs["ome"] = ome


def _write_image_numeric_levels(
    *,
    data: Any,
    group: Any,
    fmt: Any,
    axes: list[dict[str, str]],
    coordinate_transformations: list[list[dict[str, list[float] | str]]],
    storage_options: list[dict[str, tuple[int, int, int, int, int]]],
    scale_factors: list[dict[str, int]],
    max_layer: int,
    name: str,
    omero_metadata: dict[str, Any],
    supports_scale_factors: bool,
) -> None:
    import dask.array as da  # type: ignore
    from ome_zarr import writer  # type: ignore
    from ome_zarr.scale import Methods  # type: ignore
    from ome_zarr.scale import Scaler, _build_pyramid  # type: ignore

    image = data if isinstance(data, da.Array) else da.from_array(data)
    first_level_chunks = writer._resolve_storage_options(storage_options, 0).get("chunks")
    if first_level_chunks and first_level_chunks != "auto":
        image = image.rechunk(_chunks_for_shape(tuple(first_level_chunks), image.shape))
    dims = writer._extract_dims_from_axes(axes)

    if supports_scale_factors:
        pyramid = _build_pyramid(
            image,
            scale_factors,
            dims=dims,
            method=Methods.LOCAL_MEAN,
        )
    else:
        pyramid = _build_pyramid(
            image,
            [
                {d: 2**i if d in ("y", "x") else 1 for d in dims}
                for i in range(1, max_layer + 1)
            ],
            dims=dims,
            method=Methods.LOCAL_MEAN,
        )

    _write_pyramid_numeric_levels(
        pyramid=pyramid,
        group=group,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        name=name,
        metadata={"omero": omero_metadata},
    )


def _write_pyramid_numeric_levels(
    *,
    pyramid: list[Any],
    group: Any,
    fmt: Any,
    axes: list[dict[str, str]] | None,
    coordinate_transformations: list[list[dict[str, Any]]] | None,
    storage_options: list[dict[str, Any]] | dict[str, Any] | None,
    name: str | None,
    metadata: dict[str, Any],
) -> None:
    import dask.array as da  # type: ignore
    from ome_zarr import writer  # type: ignore

    group, fmt = writer.check_group_fmt(group, fmt)

    zarr_array_kwargs: dict[str, Any] = {}
    zarr_format = zarr_array_kwargs["zarr_format"] = fmt.zarr_format
    if axes is not None and zarr_format != 2:
        zarr_array_kwargs["dimension_names"] = [
            a["name"] for a in axes if isinstance(a, dict)
        ]

    shapes = []
    datasets: list[dict[str, Any]] = []
    delayed = []

    for idx, level in enumerate(pyramid):
        zarr_array_kwargs_copy = zarr_array_kwargs.copy()
        options = writer._resolve_storage_options(storage_options, idx)
        if writer.USE_DASK_ARRAY_KWARGS:
            options.pop("compressor", None)
        else:
            zarr_array_kwargs_copy["compressor"] = options.pop("compressor", None)

        if "compressors" not in zarr_array_kwargs_copy and writer.USE_DASK_ARRAY_KWARGS:
            zarr_array_kwargs_copy["compressors"] = options.pop("compressors", "auto")

        chunks_opt = options.get("chunks", None)
        shards_opt = options.get("shards", None)

        if chunks_opt and not isinstance(chunks_opt, str) and not shards_opt:
            chunks_opt = writer._retuple(chunks_opt, level.shape)
            level_image = da.array(level).rechunk(chunks=chunks_opt)
        elif shards_opt is not None:
            if chunks_opt and chunks_opt != "auto":
                chunks_opt = writer._retuple(chunks_opt, level.shape)
            else:
                chunks_opt = level.chunksize
            chunks_opt = writer._retuple(chunks_opt, level.shape)
            shards_opt = writer._retuple(shards_opt, level.shape)
            level_image = da.array(level).rechunk(shards_opt)
        else:
            chunks_opt = level.chunksize
            level_image = level

        shapes.append(level_image.shape)
        zarr_array_kwargs_copy["shards"] = shards_opt
        zarr_array_kwargs_copy["chunks"] = chunks_opt
        for key, value in options.items():
            if key not in zarr_array_kwargs_copy:
                zarr_array_kwargs_copy[key] = value

        if not writer.USE_DASK_ARRAY_KWARGS:
            if shards_opt is not None:
                create_kwargs: dict[str, Any] = {}
                for key in ("dimension_names", "chunks", "shards", "serializer"):
                    if key in zarr_array_kwargs_copy:
                        create_kwargs[key] = zarr_array_kwargs_copy[key]
                target = group.create_array(
                    str(idx),
                    shape=level_image.shape,
                    dtype=np.dtype(level_image.dtype),
                    **create_kwargs,
                )
                delayed.append(
                    da.store(
                        level_image,
                        target,
                        lock=True,
                        compute=False,
                    )
                )
                datasets.append({"path": str(idx)})
                continue
            if "chunks" in zarr_array_kwargs_copy:
                level_image = level_image.rechunk(zarr_array_kwargs_copy["chunks"])
                del zarr_array_kwargs_copy["chunks"]
            if zarr_format != 2:
                zarr_array_kwargs_copy["compressor"] = "auto"

            zarr_array_kwargs_copy.pop("compressors", None)
            zarr_array_kwargs_copy.pop("shards", None)
            zarr_array_kwargs_copy.pop("serializer", None)

        delayed.append(
            da.to_zarr(
                arr=level_image,
                url=group.store,
                component=str(Path(group.path, str(idx))),
                compute=False,
                **zarr_array_kwargs_copy,
            )
        )
        datasets.append({"path": str(idx)})

    da.compute(*delayed)

    if coordinate_transformations is None:
        coordinate_transformations = fmt.generate_coordinate_transformations(shapes)
    fmt.validate_coordinate_transformations(
        len(pyramid[0].shape), len(datasets), coordinate_transformations
    )
    for dataset, transform in zip(datasets, coordinate_transformations):
        dataset["coordinateTransformations"] = transform

    writer.write_multiscales_metadata(
        group,
        datasets,
        fmt=fmt,
        axes=axes,
        name=name,
        metadata=metadata,
    )


def _max_layer_count(width: int, height: int, min_layer_size: int) -> int:
    if min_layer_size <= 0:
        raise ValueError("min_layer_size must be positive.")
    max_dim = max(width, height)
    levels = 0
    while max_dim > min_layer_size:
        max_dim = max(1, max_dim // 2)
        levels += 1
    return levels


def _chunks_for_shape(
    chunks: tuple[int, int, int, int, int], shape: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int, int]:
    return tuple(max(1, min(chunk, dim)) for chunk, dim in zip(chunks, shape))


def _label_chunks_for_shape(
    chunks: tuple[int, int, int, int], shape: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return tuple(max(1, min(chunk, dim)) for chunk, dim in zip(chunks, shape))


def _validate_chunks(chunks: tuple[int, int, int, int, int]) -> None:
    if len(chunks) != 5 or any(chunk <= 0 for chunk in chunks):
        raise ValueError("chunks must be a 5-item tuple of positive integers.")


def _validate_shard_shape(
    shard_shape: tuple[int, int, int, int, int] | None,
    chunks: tuple[int, int, int, int, int],
) -> None:
    if shard_shape is None:
        return
    if len(shard_shape) != 5 or any(dim <= 0 for dim in shard_shape):
        raise ValueError("shard_shape must be a 5-item tuple of positive integers.")
    for shard_dim, chunk_dim in zip(shard_shape, chunks):
        if shard_dim < chunk_dim:
            raise ValueError(
                "Each shard dimension must be greater than or equal to the "
                "corresponding chunk dimension."
            )
        if shard_dim % chunk_dim != 0:
            raise ValueError(
                "Each shard dimension must be divisible by the corresponding "
                "chunk dimension."
            )


def _label_chunk_shape(
    chunks: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int]:
    return (chunks[0], chunks[2], chunks[3], chunks[4])


def _label_shard_shape(
    shard_shape: tuple[int, int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if shard_shape is None:
        return None
    return (shard_shape[0], shard_shape[2], shard_shape[3], shard_shape[4])


def _label_scale_values(
    scale: tuple[float, float, float, float, float]
) -> tuple[float, float, float, float]:
    return (scale[0], scale[2], scale[3], scale[4])


def _binary_color_hex(color: Any) -> str | None:
    if color is None:
        return None
    try:
        return _tuple_to_hex(color)
    except Exception:
        return None


def _write_position_labels(
    *,
    write_labels: Any,
    nd2_reader: "Nd2Reader",
    group: Any,
    position: _PositionInfo,
    frame_lookup: dict[tuple[int, int, int], int],
    image_shape: tuple[int, int, int, int, int],
    fmt: Any,
    axes: list[dict[str, str]],
    scale: tuple[float, float, float, float, float],
    chunks: tuple[int, int, int, int, int],
    shard_shape: tuple[int, int, int, int, int] | None,
    scale_factors: list[dict[str, int]],
    progress: _ProgressTracker,
    label_infos: list[_LabelInfo],
) -> None:
    nt, _nc, nz, ny, nx = image_shape
    if not label_infos:
        return

    label_scale = _label_scale_values(scale)
    label_chunks = _label_chunk_shape(chunks)
    label_shards = _label_shard_shape(shard_shape)
    storage_options = _storage_options_for_label_levels(
        shape=(nt, nz, ny, nx),
        chunks=label_chunks,
        shard_shape=label_shards,
        levels=len(scale_factors) + 1,
    )
    coordinate_transformations = _label_coordinate_transformations(
        scale=label_scale,
        translation=(0.0, position.stage_z_um, position.stage_y_um, position.stage_x_um),
        levels=len(scale_factors) + 1,
    )
    label_scale_factors = [
        {"t": 1, "z": 1, "y": factor["y"], "x": factor["x"]}
        for factor in scale_factors
    ]

    for label in label_infos:
        label_data = _read_position_tzyx_binary(
            nd2_reader=nd2_reader,
            layer_id=label.layer_id,
            position_index=position.index,
            shape=(nt, nz, ny, nx),
            frame_lookup=frame_lookup,
            progress=progress,
        )
        if not np.any(label_data):
            progress.advance("write-label-group")
            continue

        # Keep label metadata minimal. The integer label array itself is the
        # authoritative data; incomplete per-value styling metadata can confuse
        # viewers, especially for instance labels with many object ids.
        label_metadata: dict[str, Any] = {}

        _write_labels_numeric_levels(
            labels=label_data,
            group=group,
            name=label.name,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            scale_factors=label_scale_factors,
            label_metadata=label_metadata,
        )
        progress.advance("write-label-group")


def _hex_to_rgba(color: str) -> list[int]:
    cleaned = color.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid RGB color string: {color}")
    return [
        int(cleaned[0:2], 16),
        int(cleaned[2:4], 16),
        int(cleaned[4:6], 16),
        255,
    ]


def _write_labels_numeric_levels(
    *,
    labels: Any,
    group: Any,
    name: str,
    fmt: Any,
    axes: list[dict[str, str]],
    coordinate_transformations: list[list[dict[str, Any]]],
    storage_options: list[dict[str, Any]] | dict[str, Any] | None,
    scale_factors: list[dict[str, int]],
    label_metadata: dict[str, Any],
) -> None:
    import dask.array as da  # type: ignore
    from ome_zarr import writer  # type: ignore
    from ome_zarr.scale import Methods, _build_pyramid  # type: ignore

    group, fmt = writer.check_group_fmt(group, fmt)
    sub_group = group.require_group(f"labels/{name}")

    labels_arr = labels if isinstance(labels, da.Array) else da.from_array(labels)
    first_level_chunks = writer._resolve_storage_options(storage_options, 0).get("chunks")
    if first_level_chunks and first_level_chunks != "auto":
        labels_arr = labels_arr.rechunk(
            _label_chunks_for_shape(tuple(first_level_chunks), labels_arr.shape)
        )
    dims = writer._extract_dims_from_axes(axes)
    pyramid = _build_pyramid(
        labels_arr,
        scale_factors,
        dims=dims,
        method=Methods.NEAREST,
    )

    _write_pyramid_numeric_levels(
        pyramid=pyramid,
        group=sub_group,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        name=name,
        metadata={},
    )

    writer.write_label_metadata(
        group=group["labels"],
        name=name,
        fmt=fmt,
        **label_metadata,
    )


def _write_legacy_ome_xml(
    path: Path,
    *,
    nd2_reader: "Nd2Reader",
    positions: list[_PositionInfo],
    shape: tuple[int, int, int, int, int],
    scale: tuple[float, float, float, float, float],
    channel_metadata: list[dict[str, Any]],
) -> None:
    import ome_types.model as m  # type: ignore

    nt, nc, nz, ny, nx = shape
    images = []
    dtype = str(np.dtype(nd2_reader.imageAttributes.dtype))
    for pos in positions:
        channels = [
            m.Channel(
                id=f"Channel:{pos.index}:{index}",
                name=str(channel.get("label", f"Channel_{index + 1}")),
            )
            for index, channel in enumerate(channel_metadata)
        ]
        pixels_kwargs: dict[str, Any] = {
            "id": f"Pixels:{pos.index}",
            "dimension_order": m.Pixels_DimensionOrder.XYZCT,
            "type": dtype,
            "size_x": nx,
            "size_y": ny,
            "size_z": nz,
            "size_c": nc,
            "size_t": nt,
            "significant_bits": nd2_reader.imageAttributes.uiBpcSignificant,
            "physical_size_x": scale[4],
            "physical_size_y": scale[3],
            "physical_size_x_unit": m.UnitsLength.MICROMETER,
            "physical_size_y_unit": m.UnitsLength.MICROMETER,
            "channels": channels,
        }
        if nz > 1:
            pixels_kwargs["physical_size_z"] = scale[2]
            pixels_kwargs["physical_size_z_unit"] = m.UnitsLength.MICROMETER
        pixels = m.Pixels(**pixels_kwargs)
        images.append(m.Image(id=f"Image:{pos.index}", name=pos.label, pixels=pixels))

    ome = m.OME(images=images, creator="limnd2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ome.to_xml(exclude_unset=True), encoding="utf-8")
