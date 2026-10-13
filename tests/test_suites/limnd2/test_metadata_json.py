from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from limnd2.attributes import ImageAttributes
from limnd2.binary import (
    BinaryRasterMetadata,
    BinaryRasterMetadataItem,
    BinaryRleMetadata,
)
from limnd2.experiment import ExperimentLevel, ExperimentLoopType, ExperimentTimeLoop
from limnd2.metadata import PictureMetadata
from limnd2.nd2_metadata_json import import_nd2_metadata_json
from limnd2.nd2 import Nd2Reader, Nd2Writer
from limnd2.textinfo import ImageTextInfo


def test_image_attributes_json_round_trip() -> None:
    attributes = ImageAttributes.create(
        height=32, width=64, component_count=3, bits=16, sequence_count=5
    )

    restored = ImageAttributes.fromJson(attributes.toJson())

    assert restored == attributes
    assert json.loads(attributes.toJson())["uiWidth"] == 64


def test_experiment_json_round_trip_preserves_nested_levels_and_bytes() -> None:
    experiment = ExperimentLevel(
        eType=ExperimentLoopType.eEtTimeLoop,
        uLoopPars=ExperimentTimeLoop(uiCount=3, dPeriod=1.5),
        pItemValid=b"\x01\x00\x01",
        ppNextLevelEx=[
            ExperimentLevel(
                eType=ExperimentLoopType.eEtTimeLoop,
                uLoopPars=ExperimentTimeLoop(uiCount=2, dPeriod=2.0),
            )
        ],
        uiNextLevelCount=1,
        vendorExtension={"enabled": True},
    )

    restored = ExperimentLevel.fromJson(experiment.toJson())

    assert restored.eType == ExperimentLoopType.eEtTimeLoop
    assert restored.uLoopPars.dPeriod == 1.5
    assert restored.pItemValid == b"\x01\x00\x01"
    assert restored.ppNextLevelEx is not None
    assert restored.ppNextLevelEx[0].uLoopPars.dPeriod == 2.0
    assert restored._unknown_fields == {"vendorExtension": {"enabled": True}}


def test_picture_metadata_json_round_trip_preserves_planes_and_bytes() -> None:
    metadata = PictureMetadata(baOpticalPathsCorrections=b"\x00\xff")
    metadata.makeValid(1)

    restored = PictureMetadata.fromJson(metadata.toJson())

    assert restored.baOpticalPathsCorrections == b"\x00\xff"
    assert restored.channelNames == ["Mono"]
    assert restored.sPicturePlanes.valid


def test_image_text_info_json_round_trip() -> None:
    text_info = ImageTextInfo(
        TextInfoItem_0="image-42",
        TextInfoItem_4="Ada Lovelace",
        TextInfoItem_5="Café sample",
    )

    restored = ImageTextInfo.fromJson(text_info.toJson())

    assert restored == text_info
    assert json.loads(text_info.toJson())["description"] == "Café sample"
    assert ImageTextInfo.from_lv(restored.to_lv()) == text_info


def test_binary_metadata_json_round_trip() -> None:
    raster = BinaryRasterMetadata([
        BinaryRasterMetadataItem(
            binWidth=64, binHeight=32, binLayerId=1, binName="Cells",
            binUuid="layer-1", binComp="Objects",
        )
    ])
    rle = BinaryRleMetadata([
        {
            "uiBinLayerID": 1,
            "strName": "Cells",
            "strFileTag": "RleZipBinarySequence_1_v1",
        }
    ])

    assert BinaryRasterMetadata.fromJson(raster.toJson()) == raster
    assert BinaryRleMetadata.fromJson(rle.toJson()) == rle


def test_export_metadata_json_mirrors_metadata_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.nd2"
    attributes = ImageAttributes.create(
        height=4, width=4, component_count=1, bits=8, sequence_count=1
    )
    metadata = PictureMetadata()
    metadata.makeValid(1)

    with Nd2Writer(source) as writer:
        writer.imageAttributes = attributes
        writer.pictureMetadata = metadata
        writer.setChunk(b"ImageTextInfoLV!", ImageTextInfo(TextInfoItem_0="image-1").to_lv())
        writer.setChunk(b"CustomData|OpaqueMetadata!", b"vendor bytes")
        writer.setImage(0, np.zeros((4, 4, 1), dtype=np.uint8))

    with Nd2Reader(source) as reader:
        output_dir = reader.exportMetadataJson(tmp_path / "nd2_chunks")

    imported = import_nd2_metadata_json(output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == 1
    chunk_paths = {chunk["nd2ChunkName"]: chunk["path"] for chunk in manifest["chunks"]}
    assert "ImageDataSeq|0!" not in chunk_paths
    assert "CustomData|OpaqueMetadata!" not in chunk_paths
    assert chunk_paths["ImageMetadataSeqLV|0!"] == "ImageMetadataSeqLV/0.json"

    attributes_document = json.loads((output_dir / "ImageAttributesLV.json").read_text(encoding="utf-8"))
    assert attributes_document["uiWidth"] == 4
    assert ImageAttributes.fromJson(json.dumps(attributes_document)).uiWidth == 4
    assert imported[b"ImageAttributesLV!"].width == 4
    assert imported[b"ImageMetadataSeqLV|0!"].channelNames == ["Mono"]
    assert not (output_dir / "CustomData" / "OpaqueMetadata.json").exists()


def test_export_metadata_json_writes_s3_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsspec = pytest.importorskip("fsspec")
    source = tmp_path / "source.nd2"
    attributes = ImageAttributes.create(
        height=4, width=4, component_count=1, bits=8, sequence_count=1
    )
    metadata = PictureMetadata()
    metadata.makeValid(1)
    with Nd2Writer(source) as writer:
        writer.imageAttributes = attributes
        writer.pictureMetadata = metadata
        writer.setImage(0, np.zeros((4, 4, 1), dtype=np.uint8))

    memory_fs = fsspec.filesystem("memory")
    destination = "s3://test-bucket/exports/source.ome.zarr/nd2_chunks"

    def url_to_fs(_: str) -> tuple[Any, str]:
        return memory_fs, "test-bucket/exports/source.ome.zarr/nd2_chunks"

    monkeypatch.setattr(fsspec.core, "url_to_fs", url_to_fs)
    with Nd2Reader(source) as reader:
        result = reader.exportMetadataJson(destination)

    imported = import_nd2_metadata_json(destination)

    assert result == destination
    manifest_path = "test-bucket/exports/source.ome.zarr/nd2_chunks/manifest.json"
    with memory_fs.open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["sourceFormat"] == "nd2"
    assert manifest["schemaVersion"] == 1
    assert imported[b"ImageAttributesLV!"].width == 4
    assert memory_fs.exists(
        "test-bucket/exports/source.ome.zarr/nd2_chunks/ImageAttributesLV.json"
    )


def test_import_metadata_json_rejects_unknown_schema_version(tmp_path: Path) -> None:
    sidecar = tmp_path / "nd2_chunks"
    sidecar.mkdir()
    (sidecar / "manifest.json").write_text(
        json.dumps({"schemaVersion": 2, "chunks": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported ND2 metadata schema version"):
        import_nd2_metadata_json(sidecar)
