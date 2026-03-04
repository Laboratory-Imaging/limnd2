
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
import math
import os
import re

import numpy as np

import limnd2
from limnd2.experiment_factory import ExperimentFactory

from limnd2.tools.conversion.LimConvertUtils import ConvertSequenceArgs, ProgressPrinter, logprint
from contextlib import ExitStack, contextmanager

from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from limnd2.tools.conversion.LimImageSource import LimImageSource


def _flatten_debug_enabled() -> bool:
    return os.getenv("LIMND2_DEBUG_FLATTEN") == "1"


_SAFE_READ_DEBUG_SAMPLES_REMAINING = 40


def _effective_multiprocessing_for_source(sample_file: "LimImageSource", requested: bool) -> bool:
    """
    Decide whether threaded write path should be used for a source type.

    CZI currently loads full data with ``czi.asarray()`` per read call, so
    running frame writes in parallel can multiply peak memory significantly.
    """
    if not requested:
        return False

    if sample_file.__class__.__name__ == "LimImageSourceCzi":
        logprint(
            "CZI input detected: disabling multiprocessing to reduce peak memory usage.",
            type="warning",
        )
        return False

    return True


def convert_to_nd2(sources: list[LimImageSource], sample_file: LimImageSource, parsed_args: ConvertSequenceArgs, dimensions: dict):
    """Convert a list of LimImageSource to ND2 format."""

    sources, dimensions = sample_file.parse_additional_dimensions(sources, dimensions, parsed_args.unknown_dim)
    ConvertUtils.convert_mx_my_to_m(sources, dimensions)
    sources, dimensions = ConvertUtils.reorder_experiments(sources, dimensions)
    grouped_files = ConvertUtils.group_by_channel(sources, parsed_args, dimensions)


    sample_file.parse_additional_metadata(parsed_args)
    nd2_attributes_base = sample_file.nd2_attributes(sequence_count=len(grouped_files))

    if "channel" in dimensions:
        comp_count = dimensions["channel"]
    else:
        comp_count = nd2_attributes_base.uiComp

    nd2_attributes = limnd2.ImageAttributes.create(height = nd2_attributes_base.height,
                                            width = nd2_attributes_base.width,
                                            component_count = comp_count,
                                            bits = nd2_attributes_base.uiBpcSignificant,
                                            sequence_count = len(grouped_files))

    for plane in parsed_args.channels.values():
        parsed_args.metadata.addPlane(plane)

    nd2_metadata = parsed_args.metadata.createMetadata(number_of_channels_fallback = nd2_attributes.componentCount, is_rgb_fallback=sample_file.is_rgb)

    # get image experiments
    nd2_experiment = LIMND2Utils.create_experiment(dimensions, parsed_args.time_step, parsed_args.z_step)
    outfile = Path(parsed_args.output_dir) / parsed_args.nd2_output
    use_multiprocessing = _effective_multiprocessing_for_source(sample_file, parsed_args.multiprocessing)
    return LIMND2Utils.write_files_to_nd2(
        outfile,
        grouped_files,
        nd2_attributes,
        nd2_experiment,
        nd2_metadata,
        use_multiprocessing,
    )



class LIMND2Utils:
    STACKED_CHANNELS_DTYPE_MESSAGE_PRINTED = False
    STACKED_CHANNELS_COLLAPSE_MESSAGE_PRINTED = False
    COMPONENT_MISMATCH_MESSAGE_PRINTED = False
    TILE_WRITE_THRESHOLD_BYTES = 256 * 1024 * 1024  # 256 MB
    TILE_WRITE_TILE_SIZE = 2048  # default tile edge

    @staticmethod
    def _coerce_array_components(array: np.ndarray, attrs) -> np.ndarray:
        arr = np.asarray(array)
        expected_components = int(getattr(attrs, "componentCount", getattr(attrs, "uiComp", 1)))
        expected_components = max(expected_components, 1)

        if expected_components == 1:
            if arr.ndim == 3 and arr.shape[-1] > 1:
                if not LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED:
                    LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED = True
                    logprint(
                        f"Input frame has {arr.shape[-1]} components but ND2 expects 1. "
                        "Keeping first component only.",
                        type="warning",
                    )
                return np.asarray(arr[..., 0])
            return arr

        if arr.ndim == 2:
            out = np.zeros((arr.shape[0], arr.shape[1], expected_components), dtype=arr.dtype)
            out[..., 0] = arr
            if not LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED:
                LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED = True
                logprint(
                    f"Input frame is single-channel but ND2 expects {expected_components} components. "
                    "Padding missing components with zeros.",
                    type="warning",
                )
            return out

        if arr.ndim != 3:
            return arr

        current_components = int(arr.shape[-1])
        if current_components == expected_components:
            return arr

        if current_components > expected_components:
            if not LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED:
                LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED = True
                logprint(
                    f"Input frame has {current_components} components but ND2 expects {expected_components}. "
                    "Truncating extra components.",
                    type="warning",
                )
            return np.asarray(arr[..., :expected_components])

        # current_components < expected_components
        out = np.zeros((arr.shape[0], arr.shape[1], expected_components), dtype=arr.dtype)
        out[..., :current_components] = arr
        if not LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED:
            LIMND2Utils.COMPONENT_MISMATCH_MESSAGE_PRINTED = True
            logprint(
                f"Input frame has {current_components} components but ND2 expects {expected_components}. "
                "Padding missing components with zeros.",
                type="warning",
            )
        return out

    @staticmethod
    def _frame_work_units(frame_sources, attrs) -> int:
        """
        Estimate progress units for one frame: tile count when tiling, otherwise channel count.
        """
        frame_bytes = LIMND2Utils._frame_bytes(attrs)
        can_tile_all = all(getattr(src, "supports_tile_read", False) for src in frame_sources)
        if can_tile_all and frame_bytes >= LIMND2Utils.TILE_WRITE_THRESHOLD_BYTES:
            tile_size = LIMND2Utils.TILE_WRITE_TILE_SIZE
            H = int(getattr(attrs, "height", getattr(attrs, "uiHeight")))
            W = int(getattr(attrs, "width", getattr(attrs, "uiWidth")))
            tiles_y = math.ceil(H / tile_size)
            tiles_x = math.ceil(W / tile_size)
            return tiles_y * tiles_x
        return len(frame_sources)

    @staticmethod
    def total_work_units(grouped_files: list[list["LimImageSource"]], attrs) -> int:
        return sum(LIMND2Utils._frame_work_units(frame, attrs) for frame in grouped_files)

    @staticmethod
    def create_experiment(dims: dict[str, int], tstep, zstep):
        exp = ExperimentFactory()
        for loop, size in dims.items():
            if loop == "timeloop":
                resolved_tstep = 100.0 if tstep is None else float(tstep)
                if tstep is None:
                    logprint("Timeloop detected but step was not provided. Using default 100.0 ms.", type="warning")
                exp.t.count = size
                exp.t.step = resolved_tstep
            if loop == "multipoint":
                exp.m.count = size
            if loop == "zstack":
                resolved_zstep = 100.0 if zstep is None else float(zstep)
                if zstep is None:
                    logprint("Z-stack detected but step was not provided. Using default 100.0 um.", type="warning")
                exp.z.count = size
                exp.z.step = resolved_zstep
        return exp.createExperiment()

    @staticmethod
    def _frame_bytes(attrs) -> int:
        """
        Prefer widthBytes * height if available (typical for imaging formats).
        Fallback to width * height * comps * dtype.itemsize.
        """
        h = int(getattr(attrs, "height", getattr(attrs, "uiHeight", 0)))
        w = int(getattr(attrs, "width", getattr(attrs, "uiWidth", 0)))

        width_bytes = getattr(attrs, "widthBytes", None)
        if width_bytes is None:
            width_bytes = getattr(attrs, "uiWidthBytes", None)

        if width_bytes is not None and h > 0:
            return int(width_bytes) * h

        # last resort
        dtype = getattr(attrs, "dtype", None)
        itemsize = np.dtype(dtype).itemsize if dtype is not None else 1
        comps = int(getattr(attrs, "components", getattr(attrs, "uiComp", 1)))
        return w * h * comps * itemsize

    @staticmethod
    def store_frame_or_frames(nd2, image_seq_index, frame_sources, progress=None, nd2_file_lock=None):
        attrs = nd2.imageAttributes

        frame_bytes = LIMND2Utils._frame_bytes(attrs)
        can_tile_all = all(getattr(src, "supports_tile_read", False) for src in frame_sources)
        do_tile = can_tile_all and frame_bytes >= LIMND2Utils.TILE_WRITE_THRESHOLD_BYTES

        tile_size = LIMND2Utils.TILE_WRITE_TILE_SIZE

        # --- Tile path ---
        if do_tile:
            H = int(getattr(attrs, "height", getattr(attrs, "uiHeight")))
            W = int(getattr(attrs, "width", getattr(attrs, "uiWidth")))
            out_dtype = np.dtype(getattr(attrs, "dtype", np.uint16))

            tiles_y = math.ceil(H / tile_size)
            tiles_x = math.ceil(W / tile_size)
            total_tiles = tiles_y * tiles_x

            logprint(
                f"Tiling frame {image_seq_index}: {W}x{H}px "
                f"({frame_bytes/1e6:.1f} MB) using {tile_size}px tiles "
                f"({total_tiles} tiles)"
            )

            with ExitStack() as stack:
                readers = [stack.enter_context(src.open_tile_reader()) for src in frame_sources]
                nsrc = len(readers)

                tile_idx = 0
                for y in range(0, H, tile_size):
                    tile_h = min(tile_size, H - y)
                    for x in range(0, W, tile_size):
                        tile_w = min(tile_size, W - x)

                        if nsrc == 1:
                            tile = readers[0](x, y, tile_w, tile_h)
                            if tile.size == 0:
                                continue
                            if tile.dtype != out_dtype:
                                tile = tile.astype(out_dtype, copy=False)

                        else:
                            # Preallocate to avoid np.stack allocations
                            # Expected per-source tile shape: (tile_h, tile_w)
                            tile = np.empty((tile_h, tile_w, nsrc), dtype=out_dtype)
                            for c, r in enumerate(readers):
                                t = r(x, y, tile_w, tile_h)
                                if t.size == 0:
                                    # If one channel is missing, keep it zero-filled for this tile
                                    continue
                                if t.ndim == 3:
                                    if t.shape[-1] == nsrc:
                                        if _flatten_debug_enabled() and not LIMND2Utils.STACKED_CHANNELS_COLLAPSE_MESSAGE_PRINTED:
                                            LIMND2Utils.STACKED_CHANNELS_COLLAPSE_MESSAGE_PRINTED = True
                                            logprint(
                                                "Flatten debug: collapsing multi-component tile inputs to per-slot planes.",
                                                type="warning",
                                            )
                                        t = t[..., c]
                                    elif t.shape[-1] == 1:
                                        t = t[..., 0]
                                if t.ndim != 2:
                                    raise ValueError(
                                        f"Multi-source stacking expects 2D tiles per source; got shape {t.shape}"
                                    )
                                if t.dtype != out_dtype:
                                    t = t.astype(out_dtype, copy=False)
                                # guard for edge cases where reader clamps (should match tile_h/tile_w)
                                th, tw = t.shape[:2]
                                tile[:th, :tw, c] = t

                        tile_idx += 1

                        # Write (lock only around writer call)
                        if nd2_file_lock:
                            with nd2_file_lock:
                                nd2.chunker.setImageTile(image_seq_index, x, y, tile)
                        else:
                            nd2.chunker.setImageTile(image_seq_index, x, y, tile)

                        if progress:
                            progress.increase(1)

            return

        # --- Non-tile path ---
        if len(frame_sources) > 1:
            arrays = tuple(src.read() for src in frame_sources)
            # Optional: sanity check to avoid massive stack blowups
            base_shape = arrays[0].shape
            if any(a.shape != base_shape for a in arrays[1:]):
                raise ValueError(f"Cannot stack channels; shapes differ: {[a.shape for a in arrays]}")
            if arrays and arrays[0].ndim == 3:
                if arrays[0].shape[-1] == len(arrays):
                    if _flatten_debug_enabled() and not LIMND2Utils.STACKED_CHANNELS_COLLAPSE_MESSAGE_PRINTED:
                        LIMND2Utils.STACKED_CHANNELS_COLLAPSE_MESSAGE_PRINTED = True
                        logprint(
                            "Flatten debug: collapsing multi-component frame inputs to per-slot planes.",
                            type="warning",
                        )
                    arrays = tuple(np.asarray(a)[..., idx] for idx, a in enumerate(arrays))
                elif arrays[0].shape[-1] == 1:
                    arrays = tuple(np.asarray(a)[..., 0] for a in arrays)
            array = np.stack(arrays, axis=-1)
        else:
            array = frame_sources[0].read()

        array = LIMND2Utils._coerce_array_components(array, attrs)

        if nd2_file_lock:
            with nd2_file_lock:
                nd2.setImage(image_seq_index, array)
        else:
            nd2.setImage(image_seq_index, array)

        if progress:
            progress.increase(len(frame_sources))

    @staticmethod
    def write_files_to_nd2(
        nd2_path: Path,
        grouped_files: list[list["LimImageSource"]],
        attr,
        exp,
        metadata,
        multiprocessing: bool = True,
    ) -> bool:

        if nd2_path.is_file():
            try:
                nd2_path.unlink()
            except PermissionError:
                raise PermissionError(
                    f"ND2 file {nd2_path} is open in this or another program. Please close it and try again."
                ) from None

        with limnd2.Nd2Writer(nd2_path) as nd2:
            nd2.imageAttributes = attr
            nd2.experiment = exp
            nd2.pictureMetadata = metadata

            # Avoid assuming grouped_files[0] exists
            total = LIMND2Utils.total_work_units(grouped_files, attr) if grouped_files else 0
            expected_size_bytes = (attr.imageBytes + 4096) * len(grouped_files) + 512 * 1024 if grouped_files else None
            progress = ProgressPrinter(nd2_path, total, expected_size_bytes=expected_size_bytes)
            if multiprocessing:
                nd2_file_lock = Lock()
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for image_seq_index, frame in enumerate(grouped_files):
                        futures.append(
                            executor.submit(
                                LIMND2Utils.store_frame_or_frames,
                                nd2,
                                image_seq_index,
                                frame,
                                progress,
                                nd2_file_lock,
                            )
                        )
                    wait(futures)
                    errors: list[tuple[int, Exception]] = []
                    for future_idx, future in enumerate(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            errors.append((future_idx, exc))
                    if errors:
                        first_idx, first_exc = errors[0]
                        raise RuntimeError(
                            f"Failed to write {len(errors)} frame(s) to ND2. "
                            f"First failure at frame index {first_idx}: "
                            f"{type(first_exc).__name__}: {first_exc}"
                        ) from first_exc
            else:
                for image_seq_index, frame in enumerate(grouped_files):
                    LIMND2Utils.store_frame_or_frames(
                        nd2,
                        image_seq_index,
                        frame,
                        progress,
                        nd2_file_lock=None,
                    )

        return True


class ConvertUtils:
    @staticmethod
    def convert_mx_my_to_m(files, experiments_count):
        if "multipoint_x" not in experiments_count or "multipoint_y" not in experiments_count:
            return
        keys = list(experiments_count.keys())
        x_index = keys.index("multipoint_x")
        y_index = keys.index("multipoint_y")
        insert_index = min(x_index, y_index)

        target_name = "multipoint"
        if target_name in experiments_count:
            suffix = 2
            while f"multipoint__dup{suffix}" in experiments_count:
                suffix += 1
            target_name = f"multipoint__dup{suffix}"

        for file in files:
            values = list(files[file])
            x = values[x_index]
            y = values[y_index]
            for idx in sorted((x_index, y_index), reverse=True):
                values.pop(idx)
            values.insert(insert_index, (x, y))
            files[file] = values

        x_count = int(experiments_count["multipoint_x"])
        y_count = int(experiments_count["multipoint_y"])
        merged_size = x_count * y_count

        new_keys = [k for k in keys if k not in ("multipoint_x", "multipoint_y")]
        insert_index = min(insert_index, len(new_keys))
        new_keys.insert(insert_index, target_name)

        original_sizes = dict(experiments_count)
        experiments_count.clear()
        for key in new_keys:
            if key == target_name:
                experiments_count[key] = merged_size
            else:
                experiments_count[key] = int(original_sizes[key])

    @staticmethod
    def reorder_experiments(files, exp_count):
        reordered_files = {file: [] for file in files}
        reordered_experiments = {}

        for experiment in ["timeloop", "multipoint", "zstack", "channel"]:
            if experiment not in exp_count:
                continue
            exp_index = list(exp_count.keys()).index(experiment)
            for file in files:
                reordered_files[file].append(files[file][exp_index])
            reordered_experiments[experiment] = exp_count[experiment]

        return reordered_files, reordered_experiments

    @staticmethod
    def group_by_channel(files, arguments, exp_count: dict[str, int]):
        if "channel" not in exp_count:
            sorted_paths = [file for file in sorted(files, key=lambda k: files[k])]
            frames = [[file] for file in sorted_paths]
        else:
            # last item in a tuple is channel name, group results by all but last items (channel is ALWAYS last) in the list
            grouped_files = {}
            for file, lst in files.items():
                group = tuple(lst[:-1])
                channel = lst[-1]
                if group not in grouped_files:
                    grouped_files[group] = {}
                grouped_files[group][channel] = file

            frames = []
            for key in sorted(grouped_files):
                if arguments.channels:
                    # if channels were provided, sort files in group based on custom order
                    channel_order = list(arguments.channels.keys())
                    try:
                        group_files = [file[1] for file in sorted(grouped_files[key].items(), key=lambda x: channel_order.index(x[0]))]
                    except ValueError as e:
                        # if the channel is not in the provided channels, just use the default order
                        group_files = [file[1] for file in sorted(grouped_files[key].items(), key=lambda x: x[0])]
                else:
                    # if channels were not provided, sort files in group based on channel name
                    group = grouped_files[key]
                    group_files = []
                    for group_key in sorted(group):
                        file = group[group_key]
                        group_files.append(file)
                frames.append(group_files)
        return frames


DEFAULT_CHANNEL_COLOR_CYCLE = [
    "#FF0000",  # red
    "#00FF00",  # green
    "#0000FF",  # blue
    "#FFFF00",  # yellow
    "#FF00FF",  # magenta
    "#00FFFF",  # cyan
    "#FFA500",  # orange
    "#FFFFFF",  # white
]


class ZeroChannelSource:
    """
    Virtual source returning all-black channel data.
    Used for padded channels in flattened spectral conversion.
    """

    supports_tile_read: bool = True

    def __init__(
        self,
        height: int,
        width: int,
        dtype: np.dtype | str | type[np.generic],
        components: int = 1,
    ):
        self.height = int(height)
        self.width = int(width)
        self.dtype = np.dtype(dtype)
        self.components = int(components)

    def read(self) -> np.ndarray:
        if self.components > 1:
            return np.zeros((self.height, self.width, self.components), dtype=self.dtype)
        return np.zeros((self.height, self.width), dtype=self.dtype)

    def read_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, x0 + int(w))
        y1 = min(self.height, y0 + int(h))
        tile_w = max(0, x1 - x0)
        tile_h = max(0, y1 - y0)
        if self.components > 1:
            return np.zeros((tile_h, tile_w, self.components), dtype=self.dtype)
        return np.zeros((tile_h, tile_w), dtype=self.dtype)

    @contextmanager
    def open_tile_reader(self):
        def _read_tile(x: int, y: int, w: int, h: int) -> np.ndarray:
            return self.read_tile(x, y, w, h)

        yield _read_tile


class SafeReadSource:
    """
    Wraps a source and converts read failures to black image data.
    Useful when per-file internal dimensions are uneven (missing pages/channels).
    """

    def __init__(
        self,
        source: Any,
        height: int,
        width: int,
        dtype: np.dtype | str | type[np.generic],
        fallback_components: int = 1,
        forced_component_index: int | None = None,
        allow_missing_files: bool = False,
    ):
        self._source = source
        self.height = int(height)
        self.width = int(width)
        self.dtype = np.dtype(dtype)
        self.fallback_components = int(fallback_components)
        self.forced_component_index = (
            int(forced_component_index) if forced_component_index is not None else None
        )
        self.allow_missing_files = bool(allow_missing_files)
        self.supports_tile_read = bool(getattr(source, "supports_tile_read", False))
        self._warning_printed = False
        self._sample_logged = False

    def _zero_frame(self) -> np.ndarray:
        if self.fallback_components > 1:
            return np.zeros((self.height, self.width, self.fallback_components), dtype=self.dtype)
        return np.zeros((self.height, self.width), dtype=self.dtype)

    def _zero_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, x0 + int(w))
        y1 = min(self.height, y0 + int(h))
        tile_w = max(0, x1 - x0)
        tile_h = max(0, y1 - y0)
        if self.fallback_components > 1:
            return np.zeros((tile_h, tile_w, self.fallback_components), dtype=self.dtype)
        return np.zeros((tile_h, tile_w), dtype=self.dtype)

    def _warn_once(self, exc: Exception):
        if not self._warning_printed:
            self._warning_printed = True
            if self.allow_missing_files:
                return
            logprint(
                f"Failed to read source {self._source} ({type(exc).__name__}: {exc}). "
                "Using black fallback channel. [allow_missing_files=False (strict mode)]",
                type="warning",
            )

    def _apply_forced_component(self, arr: np.ndarray) -> np.ndarray:
        idx = self.forced_component_index
        if idx is None:
            return arr
        if arr.ndim < 3:
            return arr
        channel_count = int(arr.shape[-1])
        if channel_count <= 0:
            return arr
        clamped = min(max(idx, 0), channel_count - 1)
        return arr[..., clamped]

    def read(self) -> np.ndarray:
        try:
            arr = self._source.read()
            arr = self._apply_forced_component(np.asarray(arr))
            if _flatten_debug_enabled() and not self._sample_logged:
                global _SAFE_READ_DEBUG_SAMPLES_REMAINING
                if _SAFE_READ_DEBUG_SAMPLES_REMAINING > 0:
                    _SAFE_READ_DEBUG_SAMPLES_REMAINING -= 1
                    self._sample_logged = True
                    try:
                        arr_np = np.asarray(arr)
                        if arr_np.size > 0:
                            min_val = arr_np.min()
                            max_val = arr_np.max()
                        else:
                            min_val = "empty"
                            max_val = "empty"
                        logprint(
                            "Flatten debug read sample: "
                            f"source={self._source}, shape={arr_np.shape}, dtype={arr_np.dtype}, "
                            f"min={min_val}, max={max_val}",
                            type="warning",
                        )
                    except Exception as stat_exc:
                        logprint(
                            f"Flatten debug read sample failed for {self._source}: {type(stat_exc).__name__}: {stat_exc}",
                            type="warning",
                        )
            return arr
        except Exception as exc:
            self._warn_once(exc)
            return self._zero_frame()

    def read_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        if not hasattr(self._source, "read_tile"):
            self._warn_once(RuntimeError("Source does not implement read_tile"))
            return self._zero_tile(x, y, w, h)
        try:
            arr = self._source.read_tile(x, y, w, h)
            arr = self._apply_forced_component(np.asarray(arr))
            if _flatten_debug_enabled() and not self._sample_logged:
                global _SAFE_READ_DEBUG_SAMPLES_REMAINING
                if _SAFE_READ_DEBUG_SAMPLES_REMAINING > 0:
                    _SAFE_READ_DEBUG_SAMPLES_REMAINING -= 1
                    self._sample_logged = True
                    try:
                        arr_np = np.asarray(arr)
                        if arr_np.size > 0:
                            min_val = arr_np.min()
                            max_val = arr_np.max()
                        else:
                            min_val = "empty"
                            max_val = "empty"
                        logprint(
                            "Flatten debug tile sample: "
                            f"source={self._source}, shape={arr_np.shape}, dtype={arr_np.dtype}, "
                            f"min={min_val}, max={max_val}",
                            type="warning",
                        )
                    except Exception as stat_exc:
                        logprint(
                            f"Flatten debug tile sample failed for {self._source}: {type(stat_exc).__name__}: {stat_exc}",
                            type="warning",
                        )
            return arr
        except Exception as exc:
            self._warn_once(exc)
            return self._zero_tile(x, y, w, h)

    @contextmanager
    def open_tile_reader(self):
        if not self.supports_tile_read or not hasattr(self._source, "open_tile_reader"):
            self._warn_once(RuntimeError("Source does not support open_tile_reader"))
            def _reader(x: int, y: int, w: int, h: int) -> np.ndarray:
                return self._zero_tile(x, y, w, h)

            yield _reader
            return

        try:
            with self._source.open_tile_reader() as reader:
                def _reader(x: int, y: int, w: int, h: int) -> np.ndarray:
                    try:
                        arr = reader(x, y, w, h)
                        arr = self._apply_forced_component(np.asarray(arr))
                        if _flatten_debug_enabled() and not self._sample_logged:
                            global _SAFE_READ_DEBUG_SAMPLES_REMAINING
                            if _SAFE_READ_DEBUG_SAMPLES_REMAINING > 0:
                                _SAFE_READ_DEBUG_SAMPLES_REMAINING -= 1
                                self._sample_logged = True
                                try:
                                    arr_np = np.asarray(arr)
                                    if arr_np.size > 0:
                                        min_val = arr_np.min()
                                        max_val = arr_np.max()
                                    else:
                                        min_val = "empty"
                                        max_val = "empty"
                                    logprint(
                                        "Flatten debug tile sample: "
                                        f"source={self._source}, shape={arr_np.shape}, dtype={arr_np.dtype}, "
                                        f"min={min_val}, max={max_val}",
                                        type="warning",
                                    )
                                except Exception as stat_exc:
                                    logprint(
                                        f"Flatten debug tile sample failed for {self._source}: {type(stat_exc).__name__}: {stat_exc}",
                                        type="warning",
                                    )
                        return arr
                    except Exception as exc:
                        self._warn_once(exc)
                        return self._zero_tile(x, y, w, h)

                yield _reader
        except Exception as exc:
            self._warn_once(exc)

            def _reader(x: int, y: int, w: int, h: int) -> np.ndarray:
                return self._zero_tile(x, y, w, h)

            yield _reader


def _unwrap_source(source: Any) -> Any:
    current = source
    visited = set()
    while hasattr(current, "_source"):
        next_source = getattr(current, "_source")
        if id(next_source) in visited:
            break
        visited.add(id(next_source))
        current = next_source
    return current


def _filename_token_from_stem(stem: str) -> str | None:
    # Prefer explicit sequence-like tokens and use only their numeric part.
    seq_match = re.search(r"(?i)(?:^|[^A-Za-z0-9])seq(?:uence)?[_-]?(\d+)(?:$|[^A-Za-z0-9])", stem)
    if seq_match:
        return seq_match.group(1)

    # Other explicit index-like tokens.
    for pattern in (
        r"(?i)(?:^|[^A-Za-z0-9])frame[_-]?(\d+)(?:$|[^A-Za-z0-9])",
        r"(?i)(?:^|[^A-Za-z0-9])file[_-]?(\d+)(?:$|[^A-Za-z0-9])",
        r"(?i)(?:^|[^A-Za-z0-9])img(?:age)?[_-]?(\d+)(?:$|[^A-Za-z0-9])",
        r"(?i)(?:^|[^A-Za-z0-9])index[_-]?(\d+)(?:$|[^A-Za-z0-9])",
        r"(?i)(?:^|[^A-Za-z0-9])idx[_-]?(\d+)(?:$|[^A-Za-z0-9])",
    ):
        m = re.search(pattern, stem)
        if m:
            return m.group(1)

    parts = [part for part in re.split(r"[_\-. ]+", stem) if part]
    if not parts:
        return stem if stem else None

    alpha_numeric = [part for part in parts if any(ch.isalpha() for ch in part) and any(ch.isdigit() for ch in part)]
    if alpha_numeric:
        time_like = [part for part in alpha_numeric if part.lower().startswith("time")]
        if time_like:
            digits = re.search(r"(\d+)$", time_like[0])
            if digits:
                return digits.group(1)
            return time_like[0]
        trailing_digits = []
        for part in alpha_numeric:
            m = re.search(r"(\d+)$", part)
            if m:
                trailing_digits.append(m.group(1))
        if trailing_digits:
            return trailing_digits[-1]
        return alpha_numeric[-1]

    with_digits = [part for part in parts if any(ch.isdigit() for ch in part)]
    if with_digits:
        return with_digits[0]

    return parts[0]


def _infer_file_token_from_source(inner: Any) -> str | None:
    filename = getattr(inner, "filename", None)
    if filename is None:
        return None

    try:
        stem = Path(filename).stem
    except Exception:
        return None
    return _filename_token_from_stem(stem)


def _nd2_channel_name_from_source(inner: Any, nd2_channel_cache: dict[Path, list[str]]) -> str | None:
    filename = getattr(inner, "filename", None)
    channel_index = getattr(inner, "channel_index", None)
    if filename is None or channel_index is None:
        return None

    try:
        file_path = Path(filename)
    except Exception:
        return None

    if file_path.suffix.lower() != ".nd2":
        return None

    if file_path not in nd2_channel_cache:
        names: list[str] = []
        try:
            with limnd2.Nd2Reader(file_path) as nd2:
                planes = getattr(nd2.pictureMetadata, "channels", []) or []
                for idx, plane in enumerate(planes):
                    name = getattr(plane, "sDescription", None) or f"Channel {idx + 1}"
                    names.append(str(name).strip() or f"Channel {idx + 1}")
                if not names:
                    comp_count = max(int(nd2.imageAttributes.componentCount), 1)
                    names = [f"Channel {i + 1}" for i in range(comp_count)]
        except Exception:
            nd2_channel_cache[file_path] = []
        else:
            nd2_channel_cache[file_path] = names

    names = nd2_channel_cache.get(file_path, [])
    try:
        idx = int(channel_index)
    except Exception:
        return None
    if 0 <= idx < len(names):
        return names[idx]
    return None


def _tiff_channel_name_from_source(inner: Any, tiff_channel_cache: dict[Path, list[str]]) -> str | None:
    filename = getattr(inner, "filename", None)
    channel_index = getattr(inner, "channel_index", None)
    if filename is None or channel_index is None:
        return None

    try:
        file_path = Path(filename)
    except Exception:
        return None

    if file_path.suffix.lower() not in {".tif", ".tiff", ".btf", ".lsm", ".czi", ".oib", ".oif"}:
        return None

    if file_path not in tiff_channel_cache:
        names: list[str] = []
        try:
            from .LimImageSource import LimImageSource

            source = LimImageSource.open(file_path)
            settings = source.metadata_as_pattern_settings() if hasattr(source, "metadata_as_pattern_settings") else {}
            rows = settings.get("channels", []) if isinstance(settings, dict) else []
            for idx, row in enumerate(rows):
                if isinstance(row, list) and len(row) > 0 and str(row[0]).strip():
                    names.append(str(row[0]).strip())
                else:
                    names.append(f"Channel {idx + 1}")
        except Exception:
            tiff_channel_cache[file_path] = []
        else:
            tiff_channel_cache[file_path] = names

    names = tiff_channel_cache.get(file_path, [])
    try:
        idx = int(channel_index)
    except Exception:
        return None
    if 0 <= idx < len(names):
        return names[idx]
    return None


def _nd2_channel_template_from_source(inner: Any, nd2_template_cache: dict[Path, list[dict[str, Any]]]) -> dict[str, Any] | None:
    filename = getattr(inner, "filename", None)
    channel_index = getattr(inner, "channel_index", None)
    if filename is None or channel_index is None:
        return None

    try:
        file_path = Path(filename)
    except Exception:
        return None

    if file_path.suffix.lower() != ".nd2":
        return None

    if file_path not in nd2_template_cache:
        templates: list[dict[str, Any]] = []
        try:
            with limnd2.Nd2Reader(file_path) as nd2:
                meta = nd2.pictureMetadata
                planes = getattr(meta, "channels", []) or []
                for idx, plane in enumerate(planes):
                    ex = getattr(plane, "excitationWavelengthNm", 0.0) or 0.0
                    em = getattr(plane, "emissionWavelengthNm", 0.0) or 0.0
                    plane_settings: dict[str, Any] = {
                        "name": getattr(plane, "sDescription", None) or f"Channel {idx + 1}",
                        "modality": plane.uiModalityMask,
                        "excitation_wavelength": int(round(ex)) if ex > 0 else 0,
                        "emission_wavelength": int(round(em)) if em > 0 else 0,
                        "color": plane.colorAsTuple,
                    }
                    if plane.dPinholeDiameter > 0:
                        plane_settings["pinhole_diameter"] = plane.dPinholeDiameter

                    per_obj_mag = meta.objectiveMagnification(idx)
                    if per_obj_mag > 0:
                        plane_settings["objective_magnification"] = per_obj_mag
                    per_obj_na = meta.objectiveNumericAperture(idx)
                    if per_obj_na > 0:
                        plane_settings["objective_numerical_aperture"] = per_obj_na
                    per_refr = meta.refractiveIndex(idx)
                    if per_refr > 0:
                        plane_settings["immersion_refractive_index"] = per_refr
                    cam = meta.cameraName(idx)
                    if cam:
                        plane_settings["camera_name"] = cam
                    scope = meta.microscopeName(idx)
                    if scope:
                        plane_settings["microscope_name"] = scope

                    templates.append(plane_settings)
        except Exception:
            nd2_template_cache[file_path] = []
        else:
            nd2_template_cache[file_path] = templates

    templates = nd2_template_cache.get(file_path, [])
    try:
        idx = int(channel_index)
    except Exception:
        return None
    if 0 <= idx < len(templates):
        return dict(templates[idx])
    return None


def _infer_channel_label_parts_from_source(
    source: Any,
    nd2_channel_cache: dict[Path, list[str]],
    tiff_channel_cache: dict[Path, list[str]],
) -> tuple[str | None, str | None]:
    inner = _unwrap_source(source)
    if isinstance(inner, ZeroChannelSource):
        return None, None

    file_token = _infer_file_token_from_source(inner)
    channel_name = _nd2_channel_name_from_source(inner, nd2_channel_cache)
    if channel_name is None:
        channel_name = _tiff_channel_name_from_source(inner, tiff_channel_cache)
    if channel_name is None:
        channel_index = getattr(inner, "channel_index", None)
        try:
            if channel_index is not None and int(channel_index) >= 0:
                channel_name = f"Channel{int(channel_index) + 1}"
        except Exception:
            pass

    return channel_name, file_token


def _derive_channel_labels_and_templates(
    grouped_files: list[list[Any]],
    component_count: int,
) -> tuple[list[str], list[dict[str, Any] | None]]:
    channel_names: list[str | None] = [None] * component_count
    file_tokens: list[str | None] = [None] * component_count
    labels: list[str | None] = [None] * component_count
    nd2_channel_cache: dict[Path, list[str]] = {}
    tiff_channel_cache: dict[Path, list[str]] = {}
    templates: list[dict[str, Any] | None] = [None] * component_count
    nd2_template_cache: dict[Path, list[dict[str, Any]]] = {}

    for frame in grouped_files:
        for idx, source in enumerate(frame[:component_count]):
            if labels[idx] is not None:
                if templates[idx] is None:
                    inner = _unwrap_source(source)
                    template = _nd2_channel_template_from_source(inner, nd2_template_cache)
                    if template:
                        templates[idx] = template
                continue
            channel_name, file_token = _infer_channel_label_parts_from_source(
                source,
                nd2_channel_cache,
                tiff_channel_cache,
            )
            if channel_name:
                channel_names[idx] = channel_name
                labels[idx] = channel_name
            elif file_token:
                file_tokens[idx] = file_token
                labels[idx] = file_token
            if templates[idx] is None:
                inner = _unwrap_source(source)
                template = _nd2_channel_template_from_source(inner, nd2_template_cache)
                if template:
                    templates[idx] = template
            if file_tokens[idx] is None and file_token is not None:
                file_tokens[idx] = file_token

        if all(label is not None for label in labels):
            break

    name_counts: dict[str, int] = {}
    for name in channel_names:
        if not name:
            continue
        key = str(name)
        name_counts[key] = name_counts.get(key, 0) + 1

    has_duplicate_channel_names = any(count > 1 for count in name_counts.values())
    distinct_file_tokens = {str(token) for token in file_tokens if token}
    use_file_channel_labels = has_duplicate_channel_names and len(distinct_file_tokens) > 1
    file_token_counts: dict[str, int] = {}
    for token in file_tokens:
        if not token:
            continue
        key = str(token)
        file_token_counts[key] = file_token_counts.get(key, 0) + 1

    used: set[str] = set()
    resolved: list[str] = []
    per_file_channel_index: dict[str, int] = {}
    has_multiple_file_tokens = len(distinct_file_tokens) > 1

    for idx, raw in enumerate(labels):
        channel_name = channel_names[idx]
        file_token = file_tokens[idx]

        if use_file_channel_labels and channel_name and file_token:
            base = f"{file_token}-{channel_name}"
        elif (
            channel_name is None
            and file_token
            and has_multiple_file_tokens
            and file_token_counts.get(str(file_token), 0) > 1
        ):
            token_key = str(file_token)
            per_file_channel_index[token_key] = per_file_channel_index.get(token_key, 0) + 1
            base = f"{token_key}-Channel{per_file_channel_index[token_key]}"
        elif channel_name is None:
            base = f"Channel{idx + 1}"
        else:
            base = raw if raw else f"Missing_channel_{idx + 1}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        resolved.append(candidate)

    return resolved, templates


def _derive_channel_labels(grouped_files: list[list[Any]], component_count: int) -> list[str]:
    labels, _ = _derive_channel_labels_and_templates(grouped_files, component_count)
    return labels


def _token_natural_sort_key(token: Any) -> tuple[Any, ...]:
    text = str(token).strip()
    if text == "":
        return (3, "")

    if _is_int_like(text):
        try:
            return (0, _to_int_like(text))
        except Exception:
            pass

    trailing_digits = re.match(r"^(.*?)(\d+)$", text)
    if trailing_digits:
        prefix = trailing_digits.group(1).strip().casefold()
        suffix = int(trailing_digits.group(2))
        return (1, prefix, suffix)

    return (2, text.casefold())


def _channel_name_natural_sort_key(name: Any) -> tuple[Any, ...]:
    text = str(name).strip()
    if text == "":
        return (4, "")

    channel_match = re.match(r"(?i)^channel[_ ]?(\d+)(?:x)?$", text)
    if channel_match:
        return (0, int(channel_match.group(1)))

    if _is_int_like(text):
        try:
            return (0, _to_int_like(text))
        except Exception:
            pass

    first_number = re.search(r"(\d+)", text)
    if first_number:
        return (1, int(first_number.group(1)), text.casefold())

    return (2, text.casefold())


def _auto_channel_label_sort_key(label: Any) -> tuple[Any, ...]:
    text = str(label).strip()
    if "-" in text:
        token, channel_name = text.split("-", 1)
        return (0, _token_natural_sort_key(token), _channel_name_natural_sort_key(channel_name))
    return (1, _channel_name_natural_sort_key(text))


def reorder_grouped_files_by_auto_channel_labels(
    grouped_files: list[list[Any]],
    component_count: int,
) -> tuple[list[list[Any]], list[int] | None]:
    if component_count <= 1 or not grouped_files:
        return grouped_files, None

    channel_labels, _ = _derive_channel_labels_and_templates(grouped_files, component_count)
    if len(channel_labels) != component_count:
        return grouped_files, None

    current_order = list(range(component_count))
    new_order = sorted(
        current_order,
        key=lambda idx: (_auto_channel_label_sort_key(channel_labels[idx]), idx),
    )
    if new_order == current_order:
        return grouped_files, None

    reordered_files: list[list[Any]] = []
    for frame in grouped_files:
        if len(frame) < component_count:
            return grouped_files, None
        reordered_files.append([frame[idx] for idx in new_order])

    return reordered_files, new_order


def _channel_label_color_key(label: str | None) -> str | None:
    if not label:
        return None
    text = str(label).strip()
    if not text:
        return None
    if "-" in text:
        _, suffix = text.split("-", 1)
        text = suffix.strip()
    return text.casefold()


def _is_generic_channel_label(label: Any) -> bool:
    text = str(label).strip()
    if not text:
        return True
    if text.isdigit():
        return True
    return bool(re.match(r"(?i)^channel(?:[_ ]?\d+)?$", text))


def _dimension_base_name(name: str) -> str:
    return name.split("__dup", 1)[0]


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return float(value).is_integer()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return False
        try:
            return float(stripped).is_integer()
        except ValueError:
            return False
    return False


def _to_int_like(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(float(value))
    if isinstance(value, str):
        return int(float(value.strip()))
    raise ValueError(f"Value '{value}' can not be converted to integer index.")


def _sort_values(values: set[Any]) -> list[Any]:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda x: str(x))


def _build_linear_map(values: list[Any], declared_size: int) -> dict[Any, int]:
    unique_values = _sort_values(set(values))

    # Keep gaps when values already represent explicit indexes (e.g. channels 0,2 without 1).
    if all(_is_int_like(v) for v in unique_values):
        mapped = {_value: _to_int_like(_value) for _value in unique_values}
        if all(0 <= idx < declared_size for idx in mapped.values()):
            return mapped

    if len(unique_values) > declared_size:
        raise ValueError(
            f"Found {len(unique_values)} distinct values for a dimension with declared size {declared_size}."
        )

    return {value: index for index, value in enumerate(unique_values)}


def flatten_duplicate_dimensions(
    sources: dict["LimImageSource", list[Any]],
    dimensions: dict[str, int],
) -> tuple[dict["LimImageSource", list[Any]], dict[str, int]]:
    dim_names = list(dimensions.keys())
    if not dim_names:
        return sources, dimensions

    families: dict[str, list[str]] = {}
    for name in dim_names:
        base = _dimension_base_name(name)
        families.setdefault(base, []).append(name)

    duplicates = {base: names for base, names in families.items() if len(names) > 1}
    if not duplicates:
        return sources, dimensions

    duplicates_str = ", ".join(f"{base}: {names}" for base, names in duplicates.items())
    logprint(f"Flattening duplicate dimensions: {duplicates_str}")

    dim_index = {name: idx for idx, name in enumerate(dim_names)}
    value_maps: dict[str, dict[Any, int]] = {}
    flattened_index_maps: dict[str, dict[tuple[int, ...], int]] = {}

    for names in duplicates.values():
        for name in names:
            idx = dim_index[name]
            declared_size = int(dimensions[name])
            values = [dimension_values[idx] for dimension_values in sources.values()]
            value_maps[name] = _build_linear_map(values, declared_size)

    for base, names in duplicates.items():
        indexes = [dim_index[name] for name in names]
        observed_combinations: set[tuple[int, ...]] = set()
        for dimension_values in sources.values():
            combo: list[int] = []
            for name, idx in zip(names, indexes):
                value = dimension_values[idx]
                mapped = value_maps[name].get(value)
                if mapped is None:
                    raise ValueError(
                        f"Could not map value '{value}' for flattened dimension '{name}'."
                    )
                combo.append(int(mapped))
            observed_combinations.add(tuple(combo))

        ordered_combinations = sorted(observed_combinations)
        flattened_index_maps[base] = {
            combo: index for index, combo in enumerate(ordered_combinations)
        }

    flattened_sources: dict["LimImageSource", list[Any]] = {}
    for source, dimension_values in sources.items():
        values = list(dimension_values)
        replacements: dict[int, int] = {}
        removed_indexes: set[int] = set()

        for base, names in duplicates.items():
            indexes = [dim_index[name] for name in names]
            combo_indices: list[int] = []
            for name, idx in zip(names, indexes):
                value = values[idx]
                mapped = value_maps[name].get(value)
                if mapped is None:
                    raise ValueError(f"Could not map value '{value}' for flattened dimension '{name}'.")
                if mapped < 0 or mapped >= int(dimensions[name]):
                    raise ValueError(
                        f"Mapped value '{mapped}' is outside bounds for dimension '{name}' (size={dimensions[name]})."
                    )
                combo_indices.append(int(mapped))

            combo_key = tuple(combo_indices)
            flattened_index = flattened_index_maps[base].get(combo_key)
            if flattened_index is None:
                raise ValueError(
                    f"Could not flatten value combination '{combo_key}' for duplicated dimension family '{base}'."
                )
            replacements[indexes[0]] = flattened_index
            for idx in indexes[1:]:
                removed_indexes.add(idx)

        new_values: list[Any] = []
        for idx, value in enumerate(values):
            if idx in removed_indexes:
                continue
            new_values.append(replacements.get(idx, value))

        flattened_sources[source] = new_values

    removed_dimension_names: set[str] = set()
    for names in duplicates.values():
        removed_dimension_names.update(names[1:])

    flattened_dimensions: dict[str, int] = {}
    for name in dim_names:
        if name in removed_dimension_names:
            continue

        base = _dimension_base_name(name)
        if base in duplicates and name == duplicates[base][0]:
            flattened_dimensions[base] = len(flattened_index_maps[base])
        else:
            flattened_dimensions[name] = int(dimensions[name])

    return flattened_sources, flattened_dimensions


def _normalize_channel_key(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return value
        if _is_int_like(stripped):
            return _to_int_like(stripped)
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


def _is_numeric_channel_set(values: set[Any]) -> bool:
    if not values:
        return False
    return all(_is_int_like(value) for value in values)


def group_by_channel_with_padding(
    files: dict["LimImageSource", list[Any]],
    arguments: ConvertSequenceArgs,
    exp_count: dict[str, int],
    channel_count: int,
    zero_source_factory,
    source_wrapper_factory,
    allow_missing_files: bool = False,
) -> list[list[Any]]:
    debug = _flatten_debug_enabled()
    dim_names = list(exp_count.keys())
    has_channel = "channel" in exp_count
    non_channel_names = [name for name in dim_names if name != "channel"]

    axis_index = {name: idx for idx, name in enumerate(dim_names)}

    non_channel_maps: dict[str, dict[Any, int]] = {}
    for name in non_channel_names:
        idx = axis_index[name]
        observed_values = [values[idx] for values in files.values()]
        declared_size = int(exp_count[name])
        non_channel_maps[name] = _build_linear_map(observed_values, declared_size)

    grouped_files: dict[tuple[int, ...], dict[int, Any]] = {}
    all_channel_values: set[Any] = set()
    channel_idx = axis_index["channel"] if has_channel else None

    for values in files.values():
        if has_channel and channel_idx is not None:
            all_channel_values.add(values[channel_idx])

    if has_channel:
        if arguments.channels:
            provided_keys = list(arguments.channels.keys())
            channel_order = [_normalize_channel_key(key) for key in provided_keys]

            # QML/JSON pipelines can pass numeric channel keys as lexicographically sorted strings
            # (0,1,10,11,...) which would scramble channel slots. Detect that exact pattern and
            # normalize to natural numeric order.
            if (
                provided_keys
                and all(isinstance(key, str) and _is_int_like(key) for key in provided_keys)
            ):
                provided_numeric = [_to_int_like(key) for key in provided_keys]
                if (
                    len(set(provided_numeric)) == len(provided_numeric) == channel_count
                    and set(provided_numeric) == set(range(channel_count))
                ):
                    numeric_sorted = list(range(channel_count))
                    lexicographic_numeric = [_to_int_like(key) for key in sorted(provided_keys)]
                    if provided_numeric == lexicographic_numeric and provided_numeric != numeric_sorted:
                        channel_order = numeric_sorted
        elif _is_numeric_channel_set(all_channel_values):
            channel_order = list(range(channel_count))
        else:
            channel_order = _sort_values(all_channel_values)

        if len(channel_order) < channel_count:
            if _is_numeric_channel_set(set(channel_order)):
                existing = {_to_int_like(ch) for ch in channel_order}
                for candidate in range(channel_count):
                    if candidate not in existing:
                        channel_order.append(candidate)
                        if len(channel_order) == channel_count:
                            break

            while len(channel_order) < channel_count:
                channel_order.append(f"__missing_channel_{len(channel_order)}")

        if len(channel_order) > channel_count:
            raise ValueError(
                f"Resolved channel order has {len(channel_order)} channels but expected {channel_count}."
            )

        channel_to_slot = {channel: idx for idx, channel in enumerate(channel_order)}
        linear_channel_map = _build_linear_map(list(all_channel_values), channel_count) if all_channel_values else {}
    else:
        channel_to_slot = {}
        linear_channel_map = {}

    for file, values in files.items():
        non_channel_index_values: list[int] = []
        for name in non_channel_names:
            idx = axis_index[name]
            mapped = non_channel_maps[name].get(values[idx])
            if mapped is None:
                raise ValueError(f"Could not map value '{values[idx]}' for dimension '{name}'.")
            non_channel_index_values.append(mapped)

        group = tuple(non_channel_index_values)

        if has_channel and channel_idx is not None:
            channel_value = values[channel_idx]
            if channel_value in channel_to_slot:
                slot = channel_to_slot[channel_value]
            elif _is_int_like(channel_value):
                candidate = _to_int_like(channel_value)
                if 0 <= candidate < channel_count:
                    slot = candidate
                else:
                    raise ValueError(
                        f"Channel value '{channel_value}' maps outside channel range <0, {channel_count - 1}>."
                    )
            elif channel_value in linear_channel_map:
                slot = linear_channel_map[channel_value]
            else:
                raise ValueError(f"Could not map channel value '{channel_value}' to output channel index.")
        else:
            slot = 0

        if group not in grouped_files:
            grouped_files[group] = {}
        grouped_files[group][slot] = source_wrapper_factory(file, slot)

    if allow_missing_files:
        non_channel_sizes = [int(exp_count[name]) for name in non_channel_names]
        expected_groups = list(np.ndindex(*non_channel_sizes)) if non_channel_sizes else [()]
    else:
        expected_groups = sorted(grouped_files.keys())

    frames: list[list[Any]] = []
    missing_groups_count = 0
    missing_channels_count = 0
    channel_slots = range(channel_count) if has_channel else range(1)

    for group in expected_groups:
        channels_in_group = grouped_files.get(group)
        if channels_in_group is None:
            missing_groups_count += 1
            frames.append([zero_source_factory() for _ in channel_slots])
            continue

        frame_sources: list[Any] = []
        for slot in channel_slots:
            source = channels_in_group.get(slot)
            if source is None:
                missing_channels_count += 1
                source = zero_source_factory()
            frame_sources.append(source)
        frames.append(frame_sources)

    if allow_missing_files and missing_groups_count > 0:
        logprint(
            f"Inserted {missing_groups_count} black frame group(s) for missing file combinations.",
            type="warning",
        )
    if allow_missing_files and missing_channels_count > 0:
        logprint(
            f"Inserted {missing_channels_count} black channel image(s) for missing channel combinations.",
            type="warning",
        )

    if debug:
        logprint(
            "Flatten debug: group_by_channel_with_padding "
            f"has_channel={has_channel}, channel_count={channel_count}, "
            f"input_sources={len(files)}, grouped_keys={len(grouped_files)}, "
            f"expected_groups={len(expected_groups)}, output_frames={len(frames)}, "
            f"missing_groups={missing_groups_count}, missing_channels={missing_channels_count}",
            type="warning",
        )

    return frames


def _ensure_metadata_plane_count(
    parsed_args: ConvertSequenceArgs,
    component_count: int,
    is_rgb: bool,
    channel_labels: list[str] | None = None,
    rename_existing_with_labels: bool = False,
    channel_templates: list[dict[str, Any] | None] | None = None,
    apply_templates_to_existing: bool = False,
) -> None:
    if is_rgb:
        return

    for plane in parsed_args.channels.values():
        parsed_args.metadata.addPlane(plane)

    existing_count = len(parsed_args.metadata.planes)
    if existing_count > component_count:
        raise ValueError(
            f"Metadata contains {existing_count} channels but output image has {component_count} components."
        )

    if apply_templates_to_existing and channel_templates:
        update_count = min(existing_count, len(channel_templates))
        for index in range(update_count):
            template = channel_templates[index]
            if not template:
                continue
            plane = parsed_args.metadata.planes[index]
            for key, value in template.items():
                if hasattr(plane, key):
                    setattr(plane, key, value)

    if rename_existing_with_labels and channel_labels:
        rename_count = min(existing_count, len(channel_labels))
        for index in range(rename_count):
            new_name = channel_labels[index]
            current_name = getattr(parsed_args.metadata.planes[index], "name", None)
            # Keep richer names parsed from file metadata (e.g. EGFP, TexasRed)
            # when inferred labels are only generic placeholders.
            if _is_generic_channel_label(new_name) and not _is_generic_channel_label(current_name):
                continue
            parsed_args.metadata.planes[index].name = new_name

    propagated_colors: dict[str, Any] = {}
    propagated_plane_settings: dict[str, dict[str, Any]] = {}
    if channel_labels:
        mapped_count = min(existing_count, len(channel_labels))
        for index in range(mapped_count):
            key = _channel_label_color_key(channel_labels[index])
            if key is None:
                continue

            if key not in propagated_plane_settings:
                plane = parsed_args.metadata.planes[index]
                template: dict[str, Any] = {}
                for attr in (
                    "modality",
                    "excitation_wavelength",
                    "emission_wavelength",
                    "pinhole_diameter",
                    "objective_magnification",
                    "objective_numerical_aperture",
                    "immersion_refractive_index",
                    "camera_name",
                    "microscope_name",
                    "color",
                ):
                    if not hasattr(plane, attr):
                        continue
                    value = getattr(plane, attr)
                    if value is None:
                        continue
                    if isinstance(value, str) and value.strip() == "":
                        continue
                    template[attr] = value
                if template:
                    propagated_plane_settings[key] = template

            if key in propagated_colors:
                continue
            color = getattr(parsed_args.metadata.planes[index], "color", None)
            if color is not None:
                propagated_colors[key] = color

    for index in range(existing_count, component_count):
        default_name = channel_labels[index] if channel_labels and index < len(channel_labels) else f"Channel {index + 1}"
        plane_settings: dict[str, Any] = {}
        if channel_templates and index < len(channel_templates) and channel_templates[index]:
            plane_settings.update(channel_templates[index])

        key = _channel_label_color_key(default_name)
        if key is not None and key in propagated_plane_settings:
            for attr, value in propagated_plane_settings[key].items():
                if attr not in plane_settings:
                    plane_settings[attr] = value

        color = plane_settings.get("color")
        if color is None:
            color = DEFAULT_CHANNEL_COLOR_CYCLE[index % len(DEFAULT_CHANNEL_COLOR_CYCLE)]
        if key is not None and key in propagated_colors and "color" not in plane_settings:
            color = propagated_colors[key]
        plane_settings["name"] = default_name
        plane_settings["color"] = color
        if default_name.startswith("Missing_channel_"):
            plane_settings["color"] = "black"
        parsed_args.metadata.addPlane(plane_settings)


def convert_to_nd2_flatten(
    sources: dict["LimImageSource", list[Any]],
    sample_file: "LimImageSource",
    parsed_args: ConvertSequenceArgs,
    dimensions: dict[str, int],
    flatten_duplicates: bool = True,
    allow_missing_files: bool = False,
) -> bool:
    """
    Convert sequence to ND2 using flexible flatten/padding workflow.

    The pipeline:
    1. Expands per-file internal dimensions.
    2. Optionally flattens duplicate logical dimensions (``*_dupN`` families).
    3. Reorders dimensions to ND2 experiment order.
    4. Groups frame sources by non-channel coordinates and channel slots.
    5. Optionally inserts black fallback frames/channels for missing combinations.

    Parameters
    ----------
    flatten_duplicates:
        If ``True``, duplicate logical dimensions are merged into a single axis.
    allow_missing_files:
        If ``True``, missing file/channel combinations are padded with black data.
        If ``False``, sequence-grid holes fail earlier in strict mode.
    """
    debug = _flatten_debug_enabled()

    channels_were_user_provided = bool(parsed_args.channels)

    sources, dimensions = sample_file.parse_additional_dimensions(
        sources,
        dimensions,
        parsed_args.unknown_dim,
        preserve_duplicate_dimension_names=flatten_duplicates,
        respect_per_file_channel_count=True,
    )
    if debug:
        logprint(
            f"Flatten debug: after parse_additional_dimensions -> sources={len(sources)}, dimensions={dimensions}",
            type="warning",
        )

    ConvertUtils.convert_mx_my_to_m(sources, dimensions)
    if flatten_duplicates:
        sources, dimensions = flatten_duplicate_dimensions(sources, dimensions)
        if debug:
            logprint(
                f"Flatten debug: after flatten_duplicate_dimensions -> sources={len(sources)}, dimensions={dimensions}",
                type="warning",
            )
    sources, dimensions = ConvertUtils.reorder_experiments(sources, dimensions)
    if debug:
        logprint(
            f"Flatten debug: after reorder_experiments -> dimensions={dimensions}",
            type="warning",
        )

    sample_file.parse_additional_metadata(parsed_args)
    nd2_attributes_base = sample_file.nd2_attributes(sequence_count=1)

    if "channel" in dimensions:
        component_count = int(dimensions["channel"])
    else:
        component_count = int(nd2_attributes_base.uiComp)

    if sample_file.is_rgb and "channel" in dimensions:
        raise ValueError("Can not use channel dimension with RGB image.")

    height = int(nd2_attributes_base.height)
    width = int(nd2_attributes_base.width)
    out_dtype = np.dtype(getattr(nd2_attributes_base, "dtype", np.uint16))

    fallback_components = 1 if "channel" in dimensions else component_count

    def _source_wrapper_factory(source: Any, slot: int) -> SafeReadSource:
        force_component = None
        if "channel" in dimensions and getattr(source, "channel_index", None) is None:
            force_component = int(slot)
        return SafeReadSource(
            source=source,
            height=height,
            width=width,
            dtype=out_dtype,
            fallback_components=fallback_components,
            forced_component_index=force_component,
            allow_missing_files=allow_missing_files,
        )

    def _zero_source_factory() -> ZeroChannelSource:
        return ZeroChannelSource(
            height=height,
            width=width,
            dtype=out_dtype,
            components=fallback_components,
        )

    grouped_files = group_by_channel_with_padding(
        sources,
        parsed_args,
        dimensions,
        component_count,
        _zero_source_factory,
        _source_wrapper_factory,
        allow_missing_files=allow_missing_files,
    )
    if debug:
        logprint(
            f"Flatten debug: grouped_files frames={len(grouped_files)}, component_count={component_count}",
            type="warning",
        )
    if not channels_were_user_provided and "channel" in dimensions and component_count > 1:
        grouped_files, reordered = reorder_grouped_files_by_auto_channel_labels(grouped_files, component_count)
        if debug and reordered is not None:
            logprint(
                f"Flatten debug: reordered channel slots by inferred labels -> order={reordered}",
                type="warning",
            )
    channel_labels, channel_templates = _derive_channel_labels_and_templates(grouped_files, component_count)

    nd2_attributes = limnd2.ImageAttributes.create(
        height=height,
        width=width,
        component_count=component_count,
        bits=nd2_attributes_base.uiBpcSignificant,
        sequence_count=len(grouped_files),
    )

    _ensure_metadata_plane_count(
        parsed_args,
        nd2_attributes.componentCount,
        sample_file.is_rgb,
        channel_labels,
        rename_existing_with_labels=not channels_were_user_provided,
        channel_templates=channel_templates,
        apply_templates_to_existing=not channels_were_user_provided,
    )
    nd2_metadata = parsed_args.metadata.createMetadata(
        number_of_channels_fallback=nd2_attributes.componentCount,
        is_rgb_fallback=sample_file.is_rgb,
    )

    nd2_experiment = LIMND2Utils.create_experiment(dimensions, parsed_args.time_step, parsed_args.z_step)
    outfile = Path(parsed_args.output_dir) / parsed_args.nd2_output

    use_multiprocessing = _effective_multiprocessing_for_source(sample_file, parsed_args.multiprocessing)
    return LIMND2Utils.write_files_to_nd2(
        outfile,
        grouped_files,
        nd2_attributes,
        nd2_experiment,
        nd2_metadata,
        use_multiprocessing,
    )
