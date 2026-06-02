from __future__ import annotations

from pathlib import Path

import limnd2
from ome_types import from_xml
import tifffile


def _sample_path(nd2_base_dir: Path, name: str) -> Path:
    path = nd2_base_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required ND2 sample not found: {path}")
    return path


def test_to_ome_types_single_position_rgb(nd2_base_dir: Path) -> None:
    source = _sample_path(nd2_base_dir, "multipage.nd2")

    with limnd2.Nd2Reader(source) as reader:
        ome = reader.to_ome_types(tiff_file_name="multipage.ome.tif")
        nt, nm, nz, ny, nx, nc = reader.imageDataShape

    assert ome.creator is not None
    assert len(ome.images) == 1

    image = ome.images[0]
    pixels = image.pixels
    assert pixels.size_t == nt
    assert pixels.size_z == nz
    assert pixels.size_y == ny
    assert pixels.size_x == nx
    assert pixels.size_c == nc
    assert pixels.dimension_order.value == "XYZCT"
    assert pixels.type.value == "uint8"
    assert [channel.name for channel in pixels.channels] == ["RGB"]
    assert pixels.channels[0].samples_per_pixel == 3
    assert pixels.channels[0].color is not None
    assert tuple(pixels.channels[0].color.as_rgb_tuple()) == (255, 255, 255)
    assert len(pixels.planes) == nt * nz
    assert pixels.metadata_only is None
    assert len(pixels.tiff_data_blocks) == nt * nz
    assert all(block.uuid is not None for block in pixels.tiff_data_blocks)
    assert image.stage_label is not None

    assert ome.structured_annotations is not None
    annotation_ids = {annotation.id for annotation in ome.structured_annotations.map_annotations}
    assert "Annotation:ND2Attributes" in annotation_ids
    assert "Annotation:ND2PictureMetadata" in annotation_ids
    assert "Annotation:ND2Experiment" in annotation_ids
    assert "Annotation:ND2PictureMetadataFull" in annotation_ids
    assert "Annotation:ND2ExperimentFull" in annotation_ids
    assert "Annotation:ND2AppInfo" in annotation_ids


def test_to_ome_types_multipoint_becomes_multiple_images(nd2_base_dir: Path) -> None:
    source = _sample_path(nd2_base_dir, "md2.nd2")

    with limnd2.Nd2Reader(source) as reader:
        ome = reader.to_ome_types(include_unstructured=False)
        nt, nm, nz, ny, nx, nc = reader.imageDataShape

    assert len(ome.images) == nm
    first = ome.images[0]
    assert first.pixels.size_t == nt
    assert first.pixels.size_z == nz
    assert first.pixels.size_c == nc
    assert first.pixels.size_y == ny
    assert first.pixels.size_x == nx
    assert first.stage_label is not None
    assert first.pixels.metadata_only is not None
    assert any(annotation.id == "Annotation:ND2Position:0" for annotation in ome.structured_annotations.map_annotations)
    assert all(
        annotation.id != "Annotation:ND2PictureMetadataFull"
        for annotation in ome.structured_annotations.map_annotations
    )


def test_to_ome_xml_roundtrips_model(nd2_base_dir: Path) -> None:
    source = _sample_path(nd2_base_dir, "multipage.nd2")

    with limnd2.Nd2Reader(source) as reader:
        xml = reader.to_ome_xml(tiff_file_name="multipage.ome.tif")
        ome = from_xml(xml)

    assert xml.lstrip().startswith("<OME")
    assert "multipage.ome.tif" in xml
    assert len(ome.images) == 1
    assert ome.images[0].pixels.tiff_data_blocks


def test_module_to_ome_xml_uses_same_mapping(nd2_base_dir: Path) -> None:
    source = _sample_path(nd2_base_dir, "md2.nd2")

    with limnd2.Nd2Reader(source) as reader:
        xml = limnd2.to_ome_xml(reader, include_unstructured=False, indent=0)
        ome = from_xml(xml)

    assert len(ome.images) > 1
    assert ome.structured_annotations is not None


def test_to_ome_types_wellplate_builds_plate_model(nd2_base_dir: Path) -> None:
    source = _sample_path(nd2_base_dir, "wells.nd2")

    with limnd2.Nd2Reader(source) as reader:
        ome = reader.to_ome_types()
        nm = reader.imageDataShape[1]

    assert ome.plates is not None
    assert len(ome.plates) == 1
    plate = ome.plates[0]
    assert plate.rows == 8
    assert plate.columns == 12
    assert len(plate.wells) > 0
    assert sum(len(well.well_samples) for well in plate.wells) == nm
    sample_image_ids = {
        sample.image_ref.id
        for well in plate.wells
        for sample in well.well_samples
        if sample.image_ref is not None
    }
    assert sample_image_ids == {f"Image:{index}" for index in range(nm)}


def test_to_ome_tiff_single_position_rgb(nd2_base_dir: Path, tmp_path: Path) -> None:
    source = _sample_path(nd2_base_dir, "multipage.nd2")
    dest = tmp_path / "multipage.ome.tif"

    with limnd2.Nd2Reader(source) as reader:
        out = reader.to_ome_tiff(dest)
        nt, nm, nz, ny, nx, nc = reader.imageDataShape

    assert out == dest
    with tifffile.TiffFile(dest) as tif:
        assert len(tif.series) == 1
        assert tif.series[0].shape == (nt, nz, ny, nx, nc)
        assert tif.series[0].axes == "TZYXS"
        ome = from_xml(tif.ome_metadata)
        assert len(ome.images) == nm
        assert ome.images[0].pixels.size_c == nc
        assert len(ome.images[0].pixels.tiff_data_blocks) == nt * nz


def test_to_ome_tiff_multipoint_becomes_multiple_series(nd2_base_dir: Path, tmp_path: Path) -> None:
    source = _sample_path(nd2_base_dir, "md2.nd2")
    dest = tmp_path / "md2.ome.tif"

    with limnd2.Nd2Reader(source) as reader:
        out = limnd2.to_ome_tiff(reader, dest, include_unstructured=False)
        nt, nm, nz, ny, nx, nc = reader.imageDataShape

    assert out == dest
    with tifffile.TiffFile(dest) as tif:
        assert len(tif.series) == nm
        assert tif.series[0].shape == (nt, nz, nc, ny, nx)
        assert tif.series[0].axes == "TZCYX"
        ome = from_xml(tif.ome_metadata)
        assert len(ome.images) == nm
        all_ifds = [
            block.ifd
            for image in ome.images
            for block in image.pixels.tiff_data_blocks
        ]
        assert all_ifds == list(range(nt * nz * nc * nm))


def test_to_ome_tiff_progress_callback_finishes_only_after_writer_closes(
    nd2_base_dir: Path, tmp_path: Path
) -> None:
    source = _sample_path(nd2_base_dir, "md2.nd2")
    dest = tmp_path / "progress.ome.tif"
    events: list[tuple[int, int, str | Path | None, str]] = []

    with limnd2.Nd2Reader(source) as reader:
        _out = reader.to_ome_tiff(
            dest,
            include_unstructured=False,
            progress_callback=lambda current, total, file, message: events.append(
                (current, total, file, message)
            ),
        )
        nt, nm, nz, _ny, _nx, nc = reader.imageDataShape

    assert events
    planes_per_position = nt * nz * nc
    total_planes = planes_per_position * nm
    assert len(events) == total_planes + 1
    assert all(event[2] == dest for event in events)
    assert events[0][0] == 1
    assert events[0][1] == total_planes
    assert f"Wrote 1 of {total_planes} OME-TIFF planes" in events[0][3]
    assert events[-2][0] == events[-2][1] == total_planes
    assert events[-1][0] == events[-1][1] == total_planes
    assert "Finished exporting OME-TIFF" in events[-1][3]
