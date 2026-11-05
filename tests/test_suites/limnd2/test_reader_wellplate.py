"""Reader and writer coverage using the acquired wells.nd2 sample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import limnd2
from limnd2.base import ND2_CHUNK_NAME_WellPlateDescJson


@pytest.fixture(scope="module")
def wells_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "test_files" / "nd2_files" / "wells.nd2"
    if not path.is_file():
        pytest.skip(f"Wellplate sample not available: {path}")
    return path


def test_real_wellplate_legacy_and_combined_readers(wells_path: Path):
    with limnd2.Nd2Reader(wells_path) as reader:
        desc = reader.wellplateDesc
        frames = reader.wellplateFrameInfo
        plate = reader.wellplate

        assert desc is not None and frames is not None and plate is not None
        assert (desc.name, desc.rows, desc.columns) == ("bbb", 8, 12)
        assert (desc.rowNaming, desc.columnNaming) == ("letter", "number")
        assert plate.description == desc
        assert list(plate.frames) == list(frames)
        assert len(frames) == reader.imageAttributes.uiSequenceCount == 6
        assert [item.seqIndex for item in frames] == list(range(6))
        assert [(item.wellRowIndex, item.wellColIndex) for item in frames] == [
            (2, 3), (2, 4), (2, 5), (3, 5), (3, 4), (3, 3)
        ]
        assert [item.wellIndex for item in frames] == [27, 28, 29, 41, 40, 39]
        assert all(item.plateIndex == 0 for item in frames)
        assert all(item.wellName == "" for item in frames)
        assert {item.plateUuid for item in frames} == {"5be8ff5ed5324b0ca6f017508db7c4ab"}

        definition = json.loads(plate.definition)
        assert definition["uuid"] == frames[0].plateUuid
        assert (definition["rows"], definition["columns"]) == (8, 12)
        assert plate.definition == bytes(reader.chunk(ND2_CHUNK_NAME_WellPlateDescJson))


def test_real_wellplate_metadata_roundtrip(wells_path: Path, tmp_path: Path):
    with limnd2.Nd2Reader(wells_path) as original:
        plate = original.wellplate
        assert plate is not None

    output = tmp_path / "wellplate_roundtrip.nd2"
    with limnd2.Nd2Writer(output) as writer:
        writer.wellplate = plate

    with limnd2.Nd2Reader(output) as reader:
        restored = reader.wellplate
        assert restored is not None
        assert restored.description == plate.description
        assert list(restored.frames) == list(plate.frames)
        assert restored.definition == plate.definition
        assert reader.wellplateDesc == plate.description
        assert list(reader.wellplateFrameInfo) == list(plate.frames)


def test_reader_returns_none_without_wellplate(tmp_path: Path):
    path = tmp_path / "no_plate.nd2"
    with limnd2.Nd2Writer(path) as writer:
        writer.setChunk("CustomData|Unrelated!", b"unrelated")
    with limnd2.Nd2Reader(path) as reader:
        assert reader.wellplateDesc is None
        assert reader.wellplateFrameInfo is None
        assert reader.wellplate is None


@pytest.mark.parametrize("part", ["description", "frames"])
def test_reader_retains_partial_wellplate_metadata(tmp_path: Path, part: str):
    factory = limnd2.WellplateFactory(rows=8, columns=12)
    factory.addWell("B2")
    plate = factory.create()
    path = tmp_path / f"{part}.nd2"
    with limnd2.Nd2Writer(path) as writer:
        if part == "description":
            writer.setWellplateDesc(plate.description)
        else:
            writer.setWellplateFrameInfo(plate.frames)
    with limnd2.Nd2Reader(path) as reader:
        restored = reader.wellplate
        assert restored is not None
        assert (restored.description is not None) == (part == "description")
        assert (restored.frames is not None) == (part == "frames")
