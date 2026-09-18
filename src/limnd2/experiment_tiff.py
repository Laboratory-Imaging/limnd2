"""Read native NIS Elements experiment metadata embedded in TIFF files.

NIS TIFF exports keep the original ``SLxExperiment`` LiteVariant inside the
``ExperimentTiffV1_0`` entry of private tag 65330.  This module decodes that
small custom-data table before handing the entry to :class:`ExperimentLevel`.
It deliberately keeps :mod:`tifffile` optional: callers with already-read tag
bytes can use :func:`custom_data_from_tiff_tag` without it.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping

from .experiment import ExperimentLevel
from .lite_variant import decode_lv

NIS_CUSTOM_DATA_TAG = 65330
"""Private TIFF tag containing NIS non-sequenced custom-data entries."""

EXPERIMENT_TIFF_KEY = "ExperimentTiffV1_0"
"""Name of the NIS custom-data entry containing ``SLxExperiment``."""


def custom_data_from_tiff_tag(data: bytes | bytearray | memoryview) -> dict[str, bytes]:
    """Return named NIS custom-data entries stored in a tag-65330 payload.

    The binary table starts with a little-endian entry count.  Each entry has
    a UTF-16LE name, byte length, and offset relative to the start of the
    complete payload.  Malformed entries are ignored so an unrelated private
    TIFF tag cannot make metadata inspection fail.
    """
    payload = memoryview(data).cast("B")
    if len(payload) < 4:
        return {}
    count = struct.unpack_from("<I", payload, 0)[0]
    position = 4
    result: dict[str, bytes] = {}
    for _ in range(count):
        if position + 4 > len(payload):
            return result
        name_length = struct.unpack_from("<I", payload, position)[0]
        position += 4
        name_end = position + name_length * 2
        if name_end + 8 > len(payload):
            return result
        try:
            name = bytes(payload[position:name_end]).decode("utf-16le")
        except UnicodeDecodeError:
            return result
        length, offset = struct.unpack_from("<II", payload, name_end)
        position = name_end + 8
        end = offset + length
        if offset > len(payload) or end > len(payload):
            continue
        result[name] = bytes(payload[offset:end])
    return result


def experiment_from_tiff_tag(data: bytes | bytearray | memoryview) -> ExperimentLevel | None:
    """Decode ``ExperimentTiffV1_0`` from a NIS custom-data TIFF tag.

    ``None`` means the tag did not contain a valid native experiment record;
    it is not an error for ordinary TIFF or OME-TIFF files.
    """
    payload = custom_data_from_tiff_tag(data).get(EXPERIMENT_TIFF_KEY)
    if payload is None:
        return None
    try:
        return ExperimentLevel.from_lv(payload)
    except Exception:
        return None


def recorded_time_step_ms_from_tiff_tag(
    data: bytes | bytearray | memoryview,
) -> float | None:
    """Return the native Recorded Data sampling interval in milliseconds.

    NIS stores this as ``SLxExperiment.pRecordedData.dValTime``.  It remains
    useful even when TIFF exports retain only one shared picture-metadata
    record instead of one timestamped record per frame.
    """
    payload = custom_data_from_tiff_tag(data).get(EXPERIMENT_TIFF_KEY)
    if payload is None:
        return None
    try:
        value = decode_lv(payload).get("SLxExperiment", {}).get("pRecordedData", {}).get("dValTime")
        return float(value) if value is not None and float(value) > 0 else None
    except (TypeError, ValueError):
        return None


def experiment_from_tiff_file(path: str | Path) -> ExperimentLevel | None:
    """Read and decode native experiment metadata from a TIFF file.

    This convenience wrapper imports optional :mod:`tifffile` only when it is
    called and reads the final page, where NIS Elements writes private tags.
    """
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            'Missing optional dependency "tifffile"; use '
            "experiment_from_tiff_tag() when tag bytes are available."
        ) from exc
    try:
        with tifffile.TiffFile(path) as tif:
            tag = _private_tiff_tag(tif, NIS_CUSTOM_DATA_TAG)
            if tag is None:
                return None
            return experiment_from_tiff_tag(tag.value)
    except (OSError, ValueError):
        return None


def _private_tiff_tag(tif: object, code: int) -> object | None:
    """Find a private tag, materializing lightweight OME frames if needed."""
    page_count = len(tif.pages)
    page_order = () if page_count == 0 else ((0,) if page_count == 1 else (0, page_count - 1, *range(1, page_count - 1)))
    for index in page_order:
        page = tif.pages[index]
        if not hasattr(page, "tags") and hasattr(page, "aspage"):
            page = page.aspage()
        tags = getattr(page, "tags", None)
        if tags is not None and (tag := tags.get(code)) is not None:
            return tag
    return None
