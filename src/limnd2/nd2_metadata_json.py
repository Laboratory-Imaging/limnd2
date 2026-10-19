"""Versioned import and export of decoded ND2 metadata JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
import numpy as np

if TYPE_CHECKING:
    from .nd2 import Nd2Reader


SCHEMA_VERSION = 1


def _is_s3_url(path: str | Path) -> bool:
    return str(path).startswith("s3://")


def _metadata_json_path(chunk_name: bytes, output_dir: Path) -> Path:
    """Map an ND2 chunk key to a portable, chunk-family-oriented JSON path."""
    text = chunk_name.decode("utf-8", errors="surrogateescape").removesuffix("!")
    parts = [quote(part, safe="-_.") for part in text.split("|")]
    return output_dir.joinpath(*parts[:-1], f"{parts[-1]}.json")


def _metadata_json_value(chunk_name: bytes, data: bytes) -> tuple[str, Any | None]:
    """Decode a recognised ND2 metadata chunk to its public dict or list."""
    from .base import (
        ND2_CHUNK_NAME_BinaryMetadata_v1,
        ND2_CHUNK_NAME_BinaryMetadata_v2,
        ND2_CHUNK_NAME_ImageAttributes,
        ND2_CHUNK_NAME_ImageAttributesLV,
        ND2_CHUNK_NAME_ImageMetadata,
        ND2_CHUNK_NAME_ImageMetadataLV,
        ND2_CHUNK_NAME_ImageTextInfo,
        ND2_CHUNK_NAME_ImageTextInfoLV,
        ND2_CHUNK_RE_ImageMetadataLV_1p,
        ND2_CHUNK_RE_ImageMetadata_1p,
    )
    from .attributes import ImageAttributes
    from .binary import BinaryRasterMetadata, BinaryRleMetadata
    from .experiment import ExperimentLevel
    from .metadata import PictureMetadata
    from .textinfo import ImageTextInfo

    if chunk_name == ND2_CHUNK_NAME_ImageAttributesLV:
        return "imageAttributes", ImageAttributes.from_lv(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_ImageAttributes:
        return "imageAttributes", ImageAttributes.from_var(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_ImageMetadataLV:
        return "experiment", ExperimentLevel.from_lv(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_ImageMetadata:
        return "experiment", ExperimentLevel.from_var(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_ImageTextInfoLV:
        return "imageTextInfo", ImageTextInfo.from_lv(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_ImageTextInfo:
        return "imageTextInfo", ImageTextInfo.from_var(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_BinaryMetadata_v1:
        return "binaryRleMetadata", BinaryRleMetadata.from_var(data).toDict()
    if chunk_name == ND2_CHUNK_NAME_BinaryMetadata_v2:
        return "binaryRasterMetadata", BinaryRasterMetadata.from_json(data).toDict()
    if ND2_CHUNK_RE_ImageMetadataLV_1p.fullmatch(chunk_name):
        return "pictureMetadata", PictureMetadata.from_lv(data).toDict()
    if ND2_CHUNK_RE_ImageMetadata_1p.fullmatch(chunk_name):
        return "pictureMetadata", PictureMetadata.from_var(data).toDict()
    return "raw", None


def _metadata_from_json(kind: str, payload: Any) -> Any:
    """Import one version-1 payload through its matching metadata model."""
    from .attributes import ImageAttributes
    from .binary import BinaryRasterMetadata, BinaryRleMetadata
    from .experiment import ExperimentLevel
    from .metadata import PictureMetadata
    from .textinfo import ImageTextInfo
    from .custom_data import RecordedDataSidecar

    if kind == "wellplateMetadata":
        if not isinstance(payload, dict):
            raise ValueError("Wellplate metadata payload must be an object")
        return payload

    models = {
        "imageAttributes": ImageAttributes,
        "experiment": ExperimentLevel,
        "imageTextInfo": ImageTextInfo,
        "binaryRleMetadata": BinaryRleMetadata,
        "binaryRasterMetadata": BinaryRasterMetadata,
        "pictureMetadata": PictureMetadata,
        "recordedData": RecordedDataSidecar,
    }
    try:
        model = models[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported ND2 metadata kind: {kind!r}") from exc
    if kind == "recordedData":
        return model.from_dict(payload)
    return model.fromDict(payload)


def _local_or_s3_destination(output_dir: str | Path) -> tuple[bool, Any, str | Path]:
    """Return an fsspec filesystem/key or a local destination path."""
    if not _is_s3_url(output_dir):
        return False, None, Path(output_dir)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "S3 metadata export requires fsspec. "
            'Install it with `pip install "limnd2[ome-zarr]"`.'
        ) from exc
    fs, key = fsspec.core.url_to_fs(str(output_dir).rstrip("/"))
    return True, fs, key


def export_nd2_metadata_json(nd2_reader: "Nd2Reader", output_dir: str | Path) -> str | Path:
    """Export decoded metadata as the versioned ND2 JSON sidecar format."""
    from .base import (
        BaseChunker,
        ND2_CHUNK_NAME_AcqFramesCache,
        ND2_CHUNK_NAME_AcqTimes2Cache,
        ND2_CHUNK_NAME_AcqTimesCache,
        ND2_CHUNK_NAME_ImageMetadata,
        ND2_CHUNK_NAME_ImageMetadataLV,
    )
    from .custom_data import RecordedDataSidecar

    is_s3, fs, destination = _local_or_s3_destination(output_dir)
    if is_s3:
        if fs.exists(destination) and fs.ls(destination, detail=False):
            raise FileExistsError(f"Metadata export directory is not empty: {output_dir}")
        fs.makedirs(destination, exist_ok=True)
    else:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Metadata export directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)

    binary_rle_patterns = nd2_reader.binaryRleMetadata.dataChunkNameRegexDict
    records: list[dict[str, str]] = []
    for chunk_name in sorted(nd2_reader.chunker.chunk_names):
        if (
            BaseChunker.isSkipChunk(chunk_name)
            or BaseChunker.isImageChunk(chunk_name) is not None
            or BaseChunker.isDownsampledImageChunk(chunk_name) is not None
            or BaseChunker.isBinaryRleDataChunk(binary_rle_patterns, chunk_name) is not None
            or BaseChunker.isBinaryRasterData(chunk_name) is not None
            or BaseChunker.isDownsampledBinaryRasterData(chunk_name) is not None
        ):
            continue
        raw = nd2_reader.chunk(chunk_name)
        if raw is None:
            continue
        raw_bytes = raw.tobytes() if isinstance(raw, memoryview) else raw
        kind, payload = _metadata_json_value(chunk_name, raw_bytes)
        # Nd2Reader.experiment enriches the base experiment chunk with the
        # spectral plane/filter information exposed by the reader. Export that
        # public model instead of decoding the raw chunk in isolation, or the
        # spectral loop survives but its plane calibration and filter data do not.
        if chunk_name in (ND2_CHUNK_NAME_ImageMetadataLV, ND2_CHUNK_NAME_ImageMetadata):
            experiment = nd2_reader.experiment
            if experiment is not None:
                kind, payload = "experiment", experiment.toDict()
        if payload is None:
            continue
        local_destination = Path(".") if is_s3 else destination
        relative_path = _metadata_json_path(chunk_name, local_destination).relative_to(local_destination)
        contents = json.dumps(payload, ensure_ascii=False, indent=2)
        if is_s3:
            target_key = f"{str(destination).rstrip('/')}/{relative_path.as_posix()}"
            fs.makedirs(target_key.rsplit("/", maxsplit=1)[0], exist_ok=True)
            with fs.open(target_key, "wt", encoding="utf-8") as file:
                file.write(contents)
        else:
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
        records.append({
            "nd2ChunkName": chunk_name.decode("utf-8", errors="surrogateescape"),
            "path": relative_path.as_posix(),
            "kind": kind,
        })

    recorded = nd2_reader.recordedData
    if len(recorded) > 1:
        def cache_array(name: bytes, dtype: Any) -> Any:
            raw = nd2_reader.chunk(name)
            if raw is None:
                return None
            raw_bytes = raw.tobytes() if isinstance(raw, memoryview) else raw
            return np.frombuffer(raw_bytes, dtype=dtype).copy()

        payload = RecordedDataSidecar(
            columns=recorded,
            acq_times=cache_array(ND2_CHUNK_NAME_AcqTimesCache, np.float64),
            acq_times2=cache_array(ND2_CHUNK_NAME_AcqTimes2Cache, np.float64),
            acq_frames=cache_array(ND2_CHUNK_NAME_AcqFramesCache, np.uint32),
        ).to_dict()
        relative_path = "recordedData.json"
        contents = json.dumps(payload, ensure_ascii=False, indent=2)
        if is_s3:
            target_key = f"{str(destination).rstrip('/')}/{relative_path}"
            with fs.open(target_key, "wt", encoding="utf-8") as file:
                file.write(contents)
        else:
            (destination / relative_path).write_text(contents, encoding="utf-8")
        records.append({
            "nd2ChunkName": "limnd2:recordedData",
            "path": relative_path,
            "kind": "recordedData",
        })

    wellplate_desc = nd2_reader.wellplateDesc
    wellplate_frame_info = nd2_reader.wellplateFrameInfo
    if wellplate_desc is not None or wellplate_frame_info is not None:
        payload = {
            "desc": asdict(wellplate_desc) if wellplate_desc is not None else None,
            "frameInfo": [asdict(item) for item in wellplate_frame_info]
            if wellplate_frame_info is not None else None,
        }
        relative_path = "wellplate.json"
        contents = json.dumps(payload, ensure_ascii=False, indent=2)
        if is_s3:
            target_key = f"{str(destination).rstrip('/')}/{relative_path}"
            with fs.open(target_key, "wt", encoding="utf-8") as file:
                file.write(contents)
        else:
            (destination / relative_path).write_text(contents, encoding="utf-8")
        records.append({
            "nd2ChunkName": "limnd2:wellplate",
            "path": relative_path,
            "kind": "wellplateMetadata",
        })

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "sourceFormat": "nd2",
        "chunkCount": len(records),
        "chunks": records,
    }
    contents = json.dumps(manifest, ensure_ascii=False, indent=2)
    if is_s3:
        with fs.open(f"{str(destination).rstrip('/')}/manifest.json", "wt", encoding="utf-8") as file:
            file.write(contents)
    else:
        (destination / "manifest.json").write_text(contents, encoding="utf-8")
    return output_dir


def _import_v1(input_dir: str | Path) -> dict[bytes, Any]:
    is_s3, fs, source = _local_or_s3_destination(input_dir)
    if is_s3:
        def read_json(relative_path: str) -> Any:
            with fs.open(f"{str(source).rstrip('/')}/{relative_path}", encoding="utf-8") as file:
                return json.load(file)
    else:
        def read_json(relative_path: str) -> Any:
            return json.loads((source / relative_path).read_text(encoding="utf-8"))

    manifest = read_json("manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("ND2 metadata manifest must be a JSON object")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("ND2 metadata manifest has no chunks list")
    result: dict[bytes, Any] = {}
    for record in chunks:
        if not isinstance(record, dict):
            raise ValueError("ND2 metadata manifest contains an invalid chunk record")
        chunk_name = record.get("nd2ChunkName")
        relative_path = record.get("path")
        kind = record.get("kind")
        if not all(isinstance(value, str) for value in (chunk_name, relative_path, kind)):
            raise ValueError("ND2 metadata chunk records require name, path, and kind")
        key = chunk_name.encode("utf-8", errors="surrogateescape")
        result[key] = _metadata_from_json(kind, read_json(relative_path))
    return result


def import_nd2_metadata_json(input_dir: str | Path) -> dict[bytes, Any]:
    """Read a versioned ND2 metadata sidecar into metadata model objects."""
    is_s3, fs, source = _local_or_s3_destination(input_dir)
    if is_s3:
        with fs.open(f"{str(source).rstrip('/')}/manifest.json", encoding="utf-8") as file:
            manifest = json.load(file)
    else:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("ND2 metadata manifest must be a JSON object")
    version = manifest.get("schemaVersion")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ND2 metadata schema version: {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return _import_v1(input_dir)
