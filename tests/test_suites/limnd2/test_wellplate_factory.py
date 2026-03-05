from __future__ import annotations

from pathlib import Path

import numpy as np

import limnd2


def test_wellplate_factory_grid_with_multiple_frames_per_well():
    fac = limnd2.WellplateFactory(name="96 Well Plate", rows=8, columns=12, plateUuid="plate-1")
    fac.addGrid(["A1", "A2", "B1"], seqStart=0, framesPerWell=2)

    desc, info = fac.create()
    assert desc.name == "96 Well Plate"
    assert desc.rows == 8
    assert desc.columns == 12

    assert len(info) == 6
    # two frames for A1, then two for A2, then two for B1
    assert [it.seqIndex for it in info] == [0, 1, 2, 3, 4, 5]
    assert [it.wellName for it in info] == ["A1", "A1", "A2", "A2", "B1", "B1"]
    assert [it.wellIndex for it in info] == [0, 0, 1, 1, 2, 2]
    assert info.nwells == 3


def test_wellplate_factory_sparse_plate_with_18_frames_per_selected_well():
    fac = limnd2.WellplateFactory(name="96 Well Plate", rows=8, columns=12, plateUuid="plate-96")
    fac.addGrid(["B2", "B11", "G6"], seqStart=0, framesPerWell=18)

    desc, info = fac.create()

    assert desc.rows == 8
    assert desc.columns == 12
    assert len(info) == 54
    assert info.nwells == 3
    assert [it.seqIndex for it in info] == list(range(54))
    assert [it.wellName for it in info] == (["B2"] * 18 + ["B11"] * 18 + ["G6"] * 18)
    assert [it.wellIndex for it in info] == ([0] * 18 + [1] * 18 + [2] * 18)


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
