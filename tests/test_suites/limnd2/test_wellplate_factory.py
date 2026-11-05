from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import limnd2


def test_sparse_rectangle_and_implicit_cursor():
    fac = limnd2.WellplateFactory(rows=8, columns=12)
    fac.addWell(["B2", "B3"], frameCount=2)
    fac.addWellRect("C2", "C3", frameCount=3)
    plate = fac.create()
    assert isinstance(plate, limnd2.Wellplate)
    assert (plate.description.rowNaming, plate.description.columnNaming) == ("letter", "number")
    assert [item.seqIndex for item in plate.frames] == list(range(10))
    assert [item.wellIndex for item in plate.frames] == [13, 13, 14, 14, 25, 25, 25, 26, 26, 26]


def test_factory_rejects_invalid_mapping_atomically():
    fac = limnd2.WellplateFactory(rows=8, columns=12)
    with pytest.raises(ValueError):
        fac.addWell(["B2", "J2"])
    assert not fac.create().frames
    assert fac.addWell("B2")[0].seqIndex == 0
    with pytest.raises(ValueError):
        fac.addItem(seqIndex=1, well="B3", wellIndex=0)
    with pytest.raises(ValueError):
        fac.addItem(seqIndex=0, well="B3")


def test_implicit_cursor_after_explicit_index_and_failed_rectangle():
    fac = limnd2.WellplateFactory(rows=8, columns=12)
    fac.addWell("A1", seqStart=10)
    fac.addWell("A2", seqStart=2)
    with pytest.raises(ValueError):
        fac.addWellRect("H11", "I12")
    item = fac.addWell("B2")[0]
    assert (item.seqIndex, item.wellIndex) == (11, 13)
    assert [item.seqIndex for item in fac.create().frames] == [2, 10, 11]


@pytest.mark.parametrize("row_naming,column_naming,expected", [
    ("number", "letter", "2B"),
    ("letter", "letter", "BB"),
    ("number", "number", "22"),
])
def test_nonstandard_naming_with_coordinates(row_naming, column_naming, expected):
    fac = limnd2.WellplateFactory(
        rows=8, columns=12, rowNaming=row_naming, columnNaming=column_naming
    )
    item = fac.addItem(seqIndex=0, wellRowIndex=1, wellColIndex=1)
    assert (item.wellName, item.wellIndex) == (expected, 13)
    with pytest.raises(ValueError):
        fac.addWell("B2")


def test_combined_wellplate_roundtrip_and_definition(tmp_path):
    fac = limnd2.WellplateFactory(rows=8, columns=12)
    fac.addWell("B2")
    plate = fac.create()
    plate.definition = b'{"uuid":"example","unknown":{"value":1}}'
    path = tmp_path / "combined.nd2"
    with limnd2.Nd2Writer(path) as writer:
        writer.wellplate = plate
    with limnd2.Nd2Reader(path) as reader:
        assert reader.wellplate.description == plate.description
        assert reader.wellplate.frames[0].wellIndex == 13
        assert reader.wellplate.definition == plate.definition


def test_conversion_accepts_combined_wellplate(tmp_path):
    import tifffile
    from limnd2.tools.conversion.LimConvertSequence import convert_sequence_to_nd2

    image = tmp_path / "input.tif"
    tifffile.imwrite(image, np.zeros((6, 8), dtype=np.uint8))
    factory = limnd2.WellplateFactory(rows=8, columns=12)
    factory.addWell("B2")
    plate = factory.create()
    plate.definition = b'{"uuid":"preserved"}'
    output = tmp_path / "output.nd2"
    assert convert_sequence_to_nd2([image], output, wellplate=plate)
    with limnd2.Nd2Reader(output) as reader:
        assert reader.wellplate.frames[0].wellIndex == 13
        assert reader.wellplate.definition == plate.definition


def test_wellplate_factory_grid_with_multiple_frames_per_well():
    fac = limnd2.WellplateFactory(name="96 Well Plate", rows=8, columns=12, plateUuid="plate-1")
    fac.addWell(["A1", "A2", "B1"], seqStart=0, frameCount=2)

    desc, info = fac.create()
    assert desc.name == "96 Well Plate"
    assert desc.rows == 8
    assert desc.columns == 12

    assert len(info) == 6
    # two frames for A1, then two for A2, then two for B1
    assert [it.seqIndex for it in info] == [0, 1, 2, 3, 4, 5]
    assert [it.wellName for it in info] == ["A1", "A1", "A2", "A2", "B1", "B1"]
    assert [it.wellIndex for it in info] == [0, 0, 1, 1, 12, 12]
    assert info.nwells == 3


def test_wellplate_factory_sparse_plate_with_18_frames_per_selected_well():
    fac = limnd2.WellplateFactory(name="96 Well Plate", rows=8, columns=12, plateUuid="plate-96")
    fac.addWell(["B2", "B11", "G6"], seqStart=0, frameCount=18)

    desc, info = fac.create()

    assert desc.rows == 8
    assert desc.columns == 12
    assert len(info) == 54
    assert info.nwells == 3
    assert [it.seqIndex for it in info] == list(range(54))
    assert [it.wellName for it in info] == (["B2"] * 18 + ["B11"] * 18 + ["G6"] * 18)
    assert [it.wellIndex for it in info] == ([13] * 18 + [22] * 18 + [77] * 18)


def test_wellplate_factory_add_item_from_row_col_tuple():
    fac = limnd2.WellplateFactory(rows=3, columns=4)
    item = fac.addItem(seqIndex=7, well=(1, 2))
    desc, info = fac()

    assert desc.rows == 3 and desc.columns == 4
    assert item.wellName == "B3"
    assert item.wellRowIndex == 1
    assert item.wellColIndex == 2
    assert len(info) == 1
    assert info[0].seqIndex == 7


def test_wellplate_factory_roundtrip_with_writer(tmp_path: Path):
    out_path = tmp_path / "wellplate_factory_roundtrip.nd2"
    fac = limnd2.WellplateFactory(name="Custom Plate", rows=2, columns=3, plateUuid="plate-2")
    fac.addWell("A1", seqStart=0, frameCount=2)
    fac.addWell("B2", seqStart=2, frameCount=1)

    desc, info = fac.create()

    with limnd2.Nd2Writer(out_path) as w:
        attrs = limnd2.attributes.ImageAttributes.create(
            width=8,
            height=6,
            component_count=1,
            bits=8,
            sequence_count=3,
        )
        w.imageAttributes = attrs
        for i in range(3):
            w.setImage(i, np.zeros((6, 8, 1), dtype=np.uint8))
        w.setWellplate(desc=desc, frame_info=info)

    with limnd2.Nd2Reader(out_path) as r:
        loaded_desc = r.wellplateDesc
        loaded_info = r.wellplateFrameInfo
        assert loaded_desc is not None and loaded_info is not None
        assert loaded_desc.name == "Custom Plate"
        assert loaded_desc.rows == 2
        assert loaded_desc.columns == 3
        assert [it.wellName for it in loaded_info] == ["A1", "A1", "B2"]
        assert [it.seqIndex for it in loaded_info] == [0, 1, 2]
