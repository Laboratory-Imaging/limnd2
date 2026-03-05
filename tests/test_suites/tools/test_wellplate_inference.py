from __future__ import annotations

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
        wellplate_row_naming="letter",
        wellplate_column_naming="number",
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
