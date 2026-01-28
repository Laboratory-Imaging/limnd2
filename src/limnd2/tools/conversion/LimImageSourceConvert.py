
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
import math

import numpy as np

import limnd2
from limnd2.experiment_factory import ExperimentFactory

from limnd2.tools.conversion.LimConvertUtils import ConvertSequenceArgs, ProgressPrinter, logprint
from contextlib import ExitStack

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from limnd2.tools.conversion.LimImageSource import LimImageSource


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
    elif sample_file.is_rgb:
        comp_count = 3
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
    return LIMND2Utils.write_files_to_nd2(outfile, grouped_files, nd2_attributes, nd2_experiment, nd2_metadata, parsed_args.multiprocessing)



class LIMND2Utils:
    STACKED_CHANNELS_DTYPE_MESSAGE_PRINTED = False
    TILE_WRITE_THRESHOLD_BYTES = 256 * 1024 * 1024  # 256 MB
    TILE_WRITE_TILE_SIZE = 2048  # default tile edge

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
                exp.t.count = size
                exp.t.step = tstep
            if loop == "multipoint":
                exp.m.count = size
            if loop == "zstack":
                exp.z.count = size
                exp.z.step = zstep
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
            array = np.stack(arrays, axis=-1)
        else:
            array = frame_sources[0].read()

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
        x_index = list(experiments_count.keys()).index("multipoint_x")
        y_index = list(experiments_count.keys()).index("multipoint_y")

        for file in files:
            x = files[file][x_index]
            y = files[file][y_index]

            files[file].pop(max(x_index, y_index))
            files[file].pop(min(x_index, y_index))
            files[file].append((x, y))

        x_count = experiments_count.pop("multipoint_x")
        y_count = experiments_count.pop("multipoint_y")
        experiments_count["multipoint"] = x_count * y_count

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
