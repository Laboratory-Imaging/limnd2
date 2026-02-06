from __future__ import annotations

from typing import Any

from limnd2.base import (
    ND2_CHUNK_FORMAT_ImageMetadataLV_1p,
    ND2_CHUNK_FORMAT_ImageMetadata_1p,
    ND2_CHUNK_NAME_ImageAttributes,
    ND2_CHUNK_NAME_ImageAttributesLV,
    ND2_CHUNK_NAME_ImageMetadata,
    ND2_CHUNK_NAME_ImageMetadataLV,
    ND2_CHUNK_NAME_ImageTextInfo,
    ND2_CHUNK_NAME_ImageTextInfoLV,
)
from limnd2.nd2 import Nd2Reader

from .._parse._clx_lite import json_from_clx_lite_variant
from .._parse._clx_xml import json_from_clx_variant


class ModernReader:
    def __init__(self, path, error_radius: int | None = None) -> None:
        self._path = path
        self._error_radius = error_radius
        self._limnd2 = Nd2Reader(path)
        self._cached_decoded_chunks: dict[tuple[bytes, bool], Any] = {}

        self._raw_attributes: dict[str, Any] | None = None
        self._raw_experiment: dict[str, Any] | None = None
        self._raw_text_info: dict[str, Any] | None = None
        self._raw_image_metadata: dict[str, Any] | None = None
        self._global_metadata: dict[str, Any] | None = None

    def __enter__(self) -> ModernReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._limnd2.finalize()

    def version(self) -> tuple[int, int]:
        return self._limnd2.version

    def _load_chunk(self, name: bytes) -> bytes:
        data = self._limnd2.chunker.chunk(name)
        if data is None:
            raise KeyError(f"Chunk {name!r} not found")
        if isinstance(data, memoryview):
            return data.tobytes()
        return data

    def _decode_chunk(self, name: bytes, strip_prefix: bool = True) -> Any:
        key = (name, strip_prefix)
        if key not in self._cached_decoded_chunks:
            data = self._load_chunk(name)
            if data.lstrip().startswith(b"<"):
                decoded = json_from_clx_variant(data, strip_prefix=strip_prefix)
            else:
                decoded = json_from_clx_lite_variant(data, strip_prefix=strip_prefix)
            self._cached_decoded_chunks[key] = decoded
        return self._cached_decoded_chunks[key]

    def _ensure_raw_metadata(self) -> None:
        if self._raw_attributes is None:
            key = (
                ND2_CHUNK_NAME_ImageAttributesLV
                if self.version() >= (3, 0)
                else ND2_CHUNK_NAME_ImageAttributes
            )
            attrs = self._decode_chunk(key, strip_prefix=False)
            if isinstance(attrs, dict) and "SLxImageAttributes" in attrs:
                attrs = attrs["SLxImageAttributes"]
            if isinstance(attrs, list) and len(attrs) == 1:
                attrs = attrs[0]
            self._raw_attributes = attrs

        if self._raw_experiment is None:
            key = (
                ND2_CHUNK_NAME_ImageMetadataLV
                if self.version() >= (3, 0)
                else ND2_CHUNK_NAME_ImageMetadata
            )
            exp = self._decode_chunk(key, strip_prefix=False)
            if isinstance(exp, dict) and "SLxExperiment" in exp:
                exp = exp["SLxExperiment"]
            if isinstance(exp, list) and len(exp) == 1:
                exp = exp[0]
            self._raw_experiment = exp

        if self._raw_text_info is None:
            key = (
                ND2_CHUNK_NAME_ImageTextInfoLV
                if self.version() >= (3, 0)
                else ND2_CHUNK_NAME_ImageTextInfo
            )
            try:
                info = self._decode_chunk(key, strip_prefix=False)
            except KeyError:
                info = {}
            if isinstance(info, dict) and "SLxImageTextInfo" in info:
                info = info["SLxImageTextInfo"]
            if isinstance(info, list) and len(info) == 1:
                info = info[0]
            self._raw_text_info = info

        if self._raw_image_metadata is None:
            key = (
                ND2_CHUNK_FORMAT_ImageMetadataLV_1p % 0
                if self.version() >= (3, 0)
                else ND2_CHUNK_FORMAT_ImageMetadata_1p % 0
            )
            meta = self._decode_chunk(key, strip_prefix=False)
            if isinstance(meta, dict) and "SLxPictureMetadata" in meta:
                meta = meta["SLxPictureMetadata"]
            if isinstance(meta, list) and len(meta) == 1:
                meta = meta[0]
            self._raw_image_metadata = meta

    def _cached_global_metadata(self) -> dict[str, Any]:
        self._ensure_raw_metadata()
        if self._global_metadata is None:
            self._global_metadata = {}
        return self._global_metadata
