from __future__ import annotations

import pytest
from limnd2.tools.conversion.LimConvertUtils import ConvertSequenceArgs
from limnd2.tools.conversion.LimImageSourceConvert import _build_auto_wellplate_chunks


class _DummySource:
    def __init__(self, name: str):
        self.name = name


class LimImageSourceTiffMeta:
    def __init__(self, geometry: tuple[int, int] | None = None):
        self._geometry = geometry

    def get_htd_plate_geometry(self) -> tuple[int, int] | None:
        return self._geometry


class _GenericSampleSource:
    pass


def test_wellplate_range_maps_multipoint_positions_across_time_and_z() -> None:
    sources = [_DummySource(str(i)) for i in range(8)]
    # Output order includes two time points and two Z planes per position.
    values = [(t, position, z) for t in range(2) for position in (10, 20) for z in range(2)]
    args = ConvertSequenceArgs(wellplate_start="B2", wellplate_end="B3",
                               wellplate_rows=8, wellplate_columns=12)
    plate = _build_auto_wellplate_chunks(_GenericSampleSource(), [[s] for s in sources],
                                         dict(zip(sources, values)), ["timeloop", "multipoint", "zstack"], args)
    desc, info = plate
    assert (desc.rows, desc.columns) == (8, 12)
    assert [x.wellName for x in info] == ["B2", "B2", "B3", "B3"] * 2
    assert [x.wellIndex for x in info] == [13, 13, 14, 14] * 2
    assert [x.seqIndex for x in info] == list(range(8))


@pytest.mark.parametrize("start,end,values,message", [
    ("B2", "B3", [1, 2, 3], "2 wells"),
    ("C3", "B2", [1], "must not precede"),
    ("B0", "B1", [1], "endpoints"),
])
def test_wellplate_range_rejects_invalid_input(start, end, values, message) -> None:
    sources = [_DummySource(str(v)) for v in values]
    args = ConvertSequenceArgs(wellplate_start=start, wellplate_end=end)
    with pytest.raises(ValueError, match=message):
        _build_auto_wellplate_chunks(_GenericSampleSource(), [[s] for s in sources],
                                    {s: [v] for s, v in zip(sources, values)},
                                    ["multipoint"], args)


def test_wellplate_range_rejects_plate_too_small() -> None:
    source = _DummySource("1")
    args = ConvertSequenceArgs(wellplate_start="B2", wellplate_end="B2", wellplate_columns=1)
    with pytest.raises(ValueError, match="beyond"):
        _build_auto_wellplate_chunks(_GenericSampleSource(), [[source]], {source: [1]},
                                    ["multipoint"], args)


def test_partially_filled_range_across_time_and_z() -> None:
    values = [(t, pos, z) for t in range(2) for pos in range(55) for z in range(2)]
    sources = [_DummySource(str(i)) for i in range(len(values))]
    args = ConvertSequenceArgs(wellplate_start="B2", wellplate_end="G11",
                               wellplate_rows=8, wellplate_columns=12)
    desc, frames = _build_auto_wellplate_chunks(
        _GenericSampleSource(), [[s] for s in sources], dict(zip(sources, values)),
        ["timeloop", "multipoint", "zstack"], args,
    )
    assert (desc.rows, desc.columns) == (8, 12)
    assert len(frames) == 220
    assert len({item.wellName for item in frames}) == 55
    assert frames[0].wellName == frames[1].wellName == "B2"
    assert frames[108].wellName == frames[109].wellName == "G6"
    assert [item.wellName for item in frames[:110]] == [item.wellName for item in frames[110:]]
    assert "G7" not in {item.wellName for item in frames}


@pytest.mark.parametrize("rows,columns", [(8, 6), (5, 12)])
def test_partially_filled_range_still_requires_whole_rectangle_to_fit(rows, columns) -> None:
    source = _DummySource("only-first-position")
    args = ConvertSequenceArgs(wellplate_start="B2", wellplate_end="G11",
                               wellplate_rows=rows, wellplate_columns=columns)
    with pytest.raises(ValueError, match="beyond"):
        _build_auto_wellplate_chunks(_GenericSampleSource(), [[source]], {source: [1]},
                                    ["multipoint"], args)


def test_cli_range_conversion_repeats_positions_across_time_and_z(tmp_path) -> None:
    import numpy as np
    import tifffile
    import limnd2
    from limnd2.tools.conversion.LimConvertSequence import convert_sequence_to_nd2_args

    for t in range(2):
        for position in range(2):
            for z in range(2):
                tifffile.imwrite(tmp_path / f"T{t}_P{position}_Z{z}.tif",
                                 np.full((4, 4), t * 100 + position * 10 + z, dtype=np.uint8))
    output = convert_sequence_to_nd2_args([
        str(tmp_path), r"T(\d+)_P(\d+)_Z(\d+)\.tif", "--extension", "tif",
        "-t", "1", "-m", "2", "-z", "3", "--wellplate-start", "B2",
        "--wellplate-end", "B3", "--wellplate-rows", "8", "--wellplate-columns", "12",
        "-n", "range.nd2", "-o", str(tmp_path),
    ])
    with limnd2.Nd2Reader(output) as reader:
        assert [x.wellName for x in reader.wellplate.frames] == ["B2", "B2", "B3", "B3"] * 2
        assert [int(reader.image(i)[0, 0, 0]) for i in range(8)] == [0, 1, 10, 11, 100, 101, 110, 111]


@pytest.mark.parametrize("options", [
    ["--wellplate-start", "B2"],
    ["--wellplate-end", "B3"],
    ["--wellplate-start", "B2", "--wellplate-end", "B3", "--wellplate-mode", "off"],
])
def test_cli_rejects_incomplete_or_disabled_range(options, tmp_path) -> None:
    from limnd2.tools.conversion.LimConvertSequenceArgparser import convert_sequence_parse

    with pytest.raises(SystemExit):
        convert_sequence_parse([str(tmp_path), r"p(\d+)\.tif", "--extension", "tif",
                                "-m", "1", "-n", "out.nd2", *options])


def test_auto_wellplate_single_multipoint_well_tokens() -> None:
    sample = LimImageSourceTiffMeta((8, 12))
    src1 = _DummySource("f1")
    src2 = _DummySource("f2")
    src3 = _DummySource("f3")

    grouped_files = [[src1], [src2], [src3]]
    source_dims = {
        src1: ["B3"],
        src2: ["D4"],
        src3: ["A5"],
    }
    dims = ["multipoint"]

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims)
    assert result is not None
    desc, frame_info = result

    assert desc.rows == 8
    assert desc.columns == 12
    assert desc.rowNaming == "letter"
    assert desc.columnNaming == "number"
    assert [item.seqIndex for item in frame_info] == [0, 1, 2]
    assert [item.wellName for item in frame_info] == ["B3", "D4", "A5"]
    assert [item.wellIndex for item in frame_info] == [14, 39, 4]


def test_auto_wellplate_preserves_zero_padded_display_labels() -> None:
    sample = LimImageSourceTiffMeta((8, 12))
    src1 = _DummySource("f1")
    src2 = _DummySource("f2")

    grouped_files = [[src1], [src2]]
    source_dims = {
        src1: ["B01"],
        src2: ["B02"],
    }
    dims = ["multipoint"]

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims)
    assert result is not None
    _desc, frame_info = result

    assert [item.wellName for item in frame_info] == ["B01", "B02"]
    assert [item.wellIndex for item in frame_info] == [12, 13]


def test_auto_wellplate_two_multipoint_dims_merge_row_and_col() -> None:
    sample = LimImageSourceTiffMeta()
    src1 = _DummySource("f1")
    src2 = _DummySource("f2")

    grouped_files = [[src1], [src2]]
    source_dims = {
        src1: ["B", 7],
        src2: ["D", 8],
    }
    dims = ["multipoint", "multipoint__dup2"]

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims)
    assert result is not None
    desc, frame_info = result

    assert desc.rows == 4
    assert desc.columns == 8
    assert desc.rowNaming == "letter"
    assert desc.columnNaming == "number"
    assert [item.wellName for item in frame_info] == ["B7", "D8"]
    assert [item.wellIndex for item in frame_info] == [14, 31]


def test_auto_wellplate_two_multipoint_dims_well_plus_site() -> None:
    sample = LimImageSourceTiffMeta()
    src1 = _DummySource("f1")
    src2 = _DummySource("f2")

    grouped_files = [[src1], [src2]]
    source_dims = {
        src1: ["B11", "site1"],
        src2: ["B11", "site2"],
    }
    dims = ["multipoint", "multipoint__dup2"]

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims)
    assert result is not None
    desc, frame_info = result

    assert desc.rows == 2
    assert desc.columns == 11
    assert desc.rowNaming == "letter"
    assert desc.columnNaming == "number"
    assert [item.wellName for item in frame_info] == ["B11", "B11"]
    assert frame_info.nwells == 1
    assert [item.wellIndex for item in frame_info] == [21, 21]


def test_auto_wellplate_respects_override_settings() -> None:
    sample = LimImageSourceTiffMeta((8, 12))
    src1 = _DummySource("f1")

    grouped_files = [[src1]]
    source_dims = {src1: ["B02"]}
    dims = ["multipoint"]
    args = ConvertSequenceArgs(
        wellplate_rows=16,
        wellplate_columns=24,
        wellplate_name="Custom Plate",
    )

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims, parsed_args=args)
    assert result is not None
    desc, frame_info = result

    assert desc.name == "Custom Plate"
    assert desc.rows == 16
    assert desc.columns == 24
    assert [item.wellName for item in frame_info] == ["B02"]


def test_auto_wellplate_can_be_disabled_with_mode_off() -> None:
    sample = LimImageSourceTiffMeta((8, 12))
    src1 = _DummySource("f1")
    grouped_files = [[src1]]
    source_dims = {src1: ["B02"]}
    dims = ["multipoint"]
    args = ConvertSequenceArgs(wellplate_mode="off")

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims, parsed_args=args)
    assert result is None


def test_auto_wellplate_infers_for_non_meta_tiff_sources() -> None:
    sample = _GenericSampleSource()
    src1 = _DummySource("f1")
    src2 = _DummySource("f2")

    grouped_files = [[src1], [src2]]
    source_dims = {
        src1: [("B", "10")],
        src2: [("C", "11")],
    }
    dims = ["multipoint"]

    result = _build_auto_wellplate_chunks(sample, grouped_files, source_dims, dims)
    assert result is not None
    desc, frame_info = result
    assert desc.rows >= 3
    assert desc.columns >= 11
    assert [item.wellName for item in frame_info] == ["B10", "C11"]
