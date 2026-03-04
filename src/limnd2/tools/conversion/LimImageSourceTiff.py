from __future__ import annotations

import os
from pathlib import Path

from .LimConvertUtils import logprint
from .LimImageSourceTiff_base import LimImageSourceTiffBase
from .LimImageSourceTiff_meta import LimImageSourceTiffMeta
from .LimImageSourceTiff_ometiff import LimImageSourceTiffOmeTiff, OMEUtils


def _debug_dispatch_enabled() -> bool:
    return os.getenv("LIMND2_DEBUG_TIFF_DISPATCH") == "1"


def resolve_tiff_source_class(filename: str | Path):
    if LimImageSourceTiffOmeTiff.has_ome_metadata(filename):
        if _debug_dispatch_enabled():
            logprint(f"[TIFF-DISPATCH] {Path(filename).name} -> LimImageSourceTiffOmeTiff")
        return LimImageSourceTiffOmeTiff
    if LimImageSourceTiffMeta.has_meta_metadata(filename):
        if _debug_dispatch_enabled():
            logprint(f"[TIFF-DISPATCH] {Path(filename).name} -> LimImageSourceTiffMeta")
        return LimImageSourceTiffMeta
    if _debug_dispatch_enabled():
        logprint(f"[TIFF-DISPATCH] {Path(filename).name} -> LimImageSourceTiffBase")
    return LimImageSourceTiffBase


class LimImageSourceTiff(LimImageSourceTiffBase):
    """TIFF entry-point class dispatching to specialized TIFF source classes."""

    def __new__(cls, filename: str | Path, idf: int = 0, channel_index: int | None = None):
        if cls is not LimImageSourceTiff:
            return super().__new__(cls)

        source_cls = resolve_tiff_source_class(filename)
        return source_cls(filename, idf=idf, channel_index=channel_index)


__all__ = [
    "LimImageSourceTiff",
    "LimImageSourceTiffBase",
    "LimImageSourceTiffOmeTiff",
    "LimImageSourceTiffMeta",
    "OMEUtils",
    "resolve_tiff_source_class",
]
