"""Helpers for generating stored color-image pyramids in ND2 files."""

from __future__ import annotations

import shutil
import os
import uuid
from pathlib import Path

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
    progress_callback: ExportProgressCallback | None = None,
) -> Path:
    """Generate stored color-image downsample chunks for an ND2 file.

    With no ``output``, missing chunks are added to ``input_file`` in place.
    With ``output``, the input file is copied first and the copy is modified.
    Existing downsample chunks are preserved unless ``overwrite`` is true.
    An existing output file is rejected unless ``overwrite_output`` is true.
    ``progress_callback`` receives ``(current, total, file, message)`` after
    each processed frame and once more after the file is finalized.

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

    reporter = ExportProgressReporter(progress_callback)
    writer = Nd2Writer(destination)
    try:
        chunker = writer.chunker
        if chunker.is_readonly:
            raise NotImplementedError(
                "Generating downsample chunks requires a writable modern ND2 file."
            )
        frame_count = writer.imageAttributes.frameCount
        for seqindex in range(frame_count):
            image = chunker.image(seqindex)
            chunker.generateAndSetDownsampledImages(
                seqindex, image, overwrite=overwrite
            )
            reporter.emit(
                seqindex + 1,
                frame_count,
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
        frame_count,
        frame_count,
        destination,
        f"Finished generating downsamples in {destination}",
    )
    return destination


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
