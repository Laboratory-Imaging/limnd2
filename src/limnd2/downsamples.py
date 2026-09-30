"""Helpers for generating stored color-image pyramids in ND2 files."""

from __future__ import annotations

import shutil
import os
import uuid
import math
from pathlib import Path

import numpy as np

from .attributes import ImageAttributesCompression
from .base import ND2_CHUNK_FORMAT_DownsampledColorData_2p, _downsample_2x_linear
from .export import ExportProgressCallback, ExportProgressReporter
from .nd2 import Nd2Writer
from .nd2 import Nd2Reader


def downsample_info(
    input_file: str | Path, *, include_frames: bool = False
) -> dict:
    """Return JSON-serializable color-image pyramid information for an ND2 file.

    This is a convenience wrapper for :meth:`Nd2Reader.downsampleInfo` which
    opens and closes ``input_file`` automatically.
    """
    with Nd2Reader(input_file) as reader:
        return reader.downsampleInfo(include_frames=include_frames)


def generate_downsamples(
    input_file: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    overwrite_output: bool = False,
    tile_height: int | None = 512,
    progress_callback: ExportProgressCallback | None = None,
) -> Path:
    """Generate stored color-image downsample chunks for an ND2 file.

    With no ``output``, missing chunks are added to ``input_file`` in place.
    With ``output``, the input file is copied first and the copy is modified.
    Existing downsample chunks are preserved unless ``overwrite`` is true.
    An existing output file is rejected unless ``overwrite_output`` is true.
    ``tile_height`` bounds memory use by generating all pyramid levels from one
    source stripe before moving to the next. It defaults to 512 level-1 output
    rows; pass ``None`` to use the legacy full-frame implementation. The value
    is rounded down to a pyramid-aligned stripe height. ``progress_callback``
    receives progress after each stripe and once more after finalization.

    Binary raster pyramids are intentionally not handled by this helper.
    """
    source = Path(input_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input ND2 file does not exist: {source}")
    if source.suffix.lower() != ".nd2":
        raise ValueError(f"Input file must have a .nd2 suffix: {source}")

    if output is None:
        if overwrite_output:
            raise ValueError("overwrite_output requires an output path")
        destination = source
    else:
        destination = Path(output).expanduser().resolve()
        if destination == source:
            raise ValueError("output must be different from the input file")
        if destination.exists() and not overwrite_output:
            raise FileExistsError(f"Output file already exists: {destination}")
        if destination.exists():
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    if tile_height is not None and tile_height <= 0:
        raise ValueError("tile_height must be a positive integer or None")

    reporter = ExportProgressReporter(progress_callback)
    writer = Nd2Writer(destination)
    try:
        chunker = writer.chunker
        if chunker.is_readonly:
            raise NotImplementedError(
                "Generating downsample chunks requires a writable modern ND2 file."
            )
        frame_count = writer.imageAttributes.frameCount
        level_count = len(writer.imageAttributes.downsampleLevels)
        effective_tile_height = (
            _aligned_tile_height(writer.imageAttributes, tile_height)
            if tile_height is not None
            else None
        )
        tiles_per_frame = (
            math.ceil(
                writer.imageAttributes.makeDownsampled(1).height
                / effective_tile_height
            )
            if effective_tile_height is not None and level_count
            else 1
        )
        total_work = max(1, frame_count * tiles_per_frame)
        completed_work = 0
        for seqindex in range(frame_count):
            can_tile = (
                tile_height is not None
                and writer.imageAttributes.eCompression == ImageAttributesCompression.ictNone
                and callable(getattr(chunker, "setDownsampledImageTile", None))
            )
            if can_tile:
                for y, stripe_height in _generate_downsampled_frame_tiled(
                    chunker, seqindex, effective_tile_height, overwrite=overwrite
                ):
                    completed_work += 1
                    reporter.emit(
                        completed_work,
                        total_work,
                        destination,
                        f"All {level_count} levels, frame {seqindex + 1}/{frame_count}: "
                        f"level-1 rows {y}-{y + stripe_height} of "
                        f"{writer.imageAttributes.makeDownsampled(1).height}",
                    )
            else:
                image = chunker.image(seqindex)
                chunker.generateAndSetDownsampledImages(
                    seqindex, image, overwrite=overwrite
                )
                completed_work += 1
                reporter.emit(
                    completed_work,
                    total_work,
                    destination,
                    f"Processed frame {seqindex + 1} of {frame_count} for downsample generation",
                )
        writer.finalize()
    except Exception:
        try:
            writer.chunker.rollback()
        except Exception:
            pass
        raise
    reporter.emit(
        total_work,
        total_work,
        destination,
        f"Finished generating downsamples in {destination}",
    )
    return destination


def _generate_downsampled_frame_tiled(
    chunker,
    seqindex: int,
    tile_height: int,
    *,
    overwrite: bool,
):
    """Yield level-1 stripes while generating every pyramid level from each one."""
    attrs = chunker.imageAttributes
    chunk_names = set(chunker.chunk_names)
    write_tile = chunker.setDownsampledImageTile
    level_attrs = attrs.makeDownsampled(1)
    write_levels = {
        level: overwrite
        or ND2_CHUNK_FORMAT_DownsampledColorData_2p
        % (attrs.makeDownsampled(level).powSize, seqindex)
        not in chunk_names
        for level in attrs.downsampleLevels
    }
    for y in range(0, level_attrs.height, tile_height):
        stripe_height = min(tile_height, level_attrs.height - y)
        current = chunker.image(
            seqindex,
            rect=(0, 2 * y, 2 * level_attrs.width, 2 * stripe_height),
        )
        for level in attrs.downsampleLevels:
            current_height, current_width = current.shape[:2]
            if current_height < 2 or current_width < 2:
                break
            current_attrs = attrs.makeDownsampled(level)
            downsampled = np.zeros(
                (current_height // 2, current_width // 2, current_attrs.componentCount),
                dtype=current_attrs.safe_dtype,
            )
            _downsample_2x_linear(downsampled, current)
            if write_levels[level]:
                level_y = y // (2 ** (level - 1))
                write_tile(
                    seqindex,
                    0,
                    level_y,
                    downsampled.astype(current_attrs.dtype),
                    downsample_level=level,
                )
            current = downsampled
        yield y, stripe_height


def _aligned_tile_height(attrs, requested_tile_height: int) -> int:
    """Return a level-1 stripe height aligned for every stored pyramid level."""
    alignment = 2 ** max(0, len(attrs.downsampleLevels) - 1)
    return max(alignment, requested_tile_height // alignment * alignment)


def remove_downsamples(
    input_file: str | Path,
    *,
    output: str | Path | None = None,
    overwrite_output: bool = False,
    progress_callback: ExportProgressCallback | None = None,
) -> Path:
    """Remove stored color-image pyramid chunks and compact an ND2 file.

    The file is rebuilt so removed chunks actually release disk space. With no
    ``output``, the rebuilt file atomically replaces ``input_file`` only after
    successful finalization. Binary-raster pyramids are not removed.
    """
    source = Path(input_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input ND2 file does not exist: {source}")
    if source.suffix.lower() != ".nd2":
        raise ValueError(f"Input file must have a .nd2 suffix: {source}")

    replace_source = output is None
    if replace_source:
        destination = source.with_name(f".{source.stem}.{uuid.uuid4().hex}.nd2")
    else:
        destination = Path(output).expanduser().resolve()
        if destination == source:
            raise ValueError("output must be different from the input file")
        if destination.exists() and not overwrite_output:
            raise FileExistsError(f"Output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    reporter = ExportProgressReporter(progress_callback)
    reader = Nd2Reader(source)
    writer = Nd2Writer(destination)
    try:
        source_chunker = reader.chunker
        destination_chunker = writer.chunker
        if destination_chunker.is_readonly:
            raise NotImplementedError("Compacting downsample chunks requires modern ND2 support.")
        retained = [
            (name, position)
            for name, position in source_chunker._chunkmap.items()
            if source_chunker.isDownsampledImageChunk(name) is None
        ]
        total = len(retained)
        for current, (name, (position, _size)) in enumerate(retained, start=1):
            data = source_chunker._read_chunk(position)
            destination_chunker._update_chunkmap(
                name, destination_chunker._write_chunk(name, data)
            )
            reporter.emit(current, total, destination, f"Copied chunk {current} of {total}")
        writer.finalize()
        reader.finalize()
    except Exception:
        try:
            writer.chunker.rollback()
        except Exception:
            pass
        reader.finalize()
        if destination.exists():
            destination.unlink()
        raise

    if replace_source:
        try:
            os.replace(destination, source)
        except PermissionError as error:
            raise PermissionError(
                f"Cannot replace {source}; it is locked by another process. "
                f"The compacted ND2 was kept at {destination}. Close the process "
                "using the source file, then replace it manually, or rerun with "
                "an explicit output path."
            ) from error
        destination = source
    reporter.emit(total, total, destination, f"Finished removing downsamples from {destination}")
    return destination
