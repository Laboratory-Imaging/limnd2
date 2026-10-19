from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

import limnd2
from limnd2.textinfo import ImageTextInfo
from limnd2.tools.conversion.LimConvertFile import convert_file_to_nd2
from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
from limnd2.tools.conversion.LimImageSource import LimImageSource


def _write_image_group(root, name: str, data: np.ndarray) -> None:
    group = root if name == "" else root.create_group(name)
    group.create_array("0", data=data)
    group.attrs["ome"] = {
        "multiscales": [{
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }]
    }


def test_ome_zarr_source_accepts_store_folder(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "folder-input.ome.zarr"
    root = zarr.open_group(store, mode="w")
    data = np.arange(6, dtype=np.uint16).reshape(1, 1, 1, 2, 3)
    _write_image_group(root, "", data)

    source = LimImageSource.open(store)

    np.testing.assert_array_equal(source.read(), data[0, 0, 0])
    assert source.metadata_as_pattern_settings() == {}


def test_convert_file_resolver_accepts_ome_zarr_folder(tmp_path):
    from limnd2.tools.conversion.LimConvertFile import resolve_input

    store = tmp_path / "resolver.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}", encoding="utf-8")

    assert resolve_input(str(store)) == store


def test_convert_file_resolver_rejects_non_zarr_folder(tmp_path):
    from limnd2.tools.conversion.LimConvertFile import resolve_input

    with pytest.raises(ValueError, match="not a supported OME-Zarr store"):
        resolve_input(str(tmp_path))


def test_ome_zarr_source_expands_axes_and_image_groups(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "positions.ome.zarr"
    root = zarr.open_group(store, mode="w")
    first = np.arange(2 * 2 * 1 * 3 * 4, dtype=np.uint16).reshape(2, 2, 1, 3, 4)
    _write_image_group(root, "0", first)
    _write_image_group(root, "1", first + 100)
    root.attrs["ome"] = {"bioformats2raw.layout": 3}
    ome_group = root.create_group("OME")
    ome_group.attrs["ome"] = {"version": "0.5", "series": ["0", "1"]}

    # The root is a series container, as emitted for ND2 multipoint exports.
    source = LimImageSource.open(store / "zarr.json")
    assert source.get_file_dimensions() == {"timeloop": 2, "channel": 2, "multipoint": 2}

    expanded, dimensions = source.parse_additional_dimensions({source: []}, {})
    assert dimensions == {"multipoint": 2, "timeloop": 2, "channel": 2}
    assert len(expanded) == 8

    selected = next(
        item for item, values in expanded.items()
        if values == [1, 1, 1]
    )
    np.testing.assert_array_equal(selected.read(), first[1, 1, 0] + 100)


@pytest.mark.parametrize(
    ("second_shape", "second_dtype"),
    [((1, 3, 1, 3, 4), np.uint16), ((1, 4, 1, 3, 4), np.uint8)],
)
def test_ome_zarr_incompatible_series_imports_only_first(
    tmp_path, second_shape, second_dtype
):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "incompatible-series.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "0", np.zeros((1, 4, 1, 3, 4), dtype=np.uint16))
    _write_image_group(root, "1", np.zeros(second_shape, dtype=second_dtype))
    root.attrs["ome"] = {"bioformats2raw.layout": 3}
    ome_group = root.create_group("OME")
    ome_group.attrs["ome"] = {"version": "0.5", "series": ["0", "1"]}

    source = LimImageSource.open(store / "zarr.json")
    assert source.get_file_dimensions() == {"channel": 4}

    expanded, dimensions = source.parse_additional_dimensions({source: []}, {})

    assert dimensions == {"channel": 4}
    assert len(expanded) == 4
    assert {item.group_path for item in expanded} == {Path("0")}
    assert {item.axis_indexes["c"] for item in expanded} == {0, 1, 2, 3}


def test_ome_zarr_series_with_different_translations_remain_multipoint(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "translated-positions.ome.zarr"
    root = zarr.open_group(store, mode="w")
    data = np.zeros((1, 1, 1, 3, 4), dtype=np.uint16)
    for name, translation in (("0", [0, 0, 0, 10, 20]), ("1", [0, 0, 0, 30, 40])):
        _write_image_group(root, name, data)
        metadata = root[name].attrs["ome"]
        metadata["multiscales"][0]["datasets"][0]["coordinateTransformations"] = [
            {"type": "scale", "scale": [1, 1, 1, 0.5, 0.5]},
            {"type": "translation", "translation": translation},
        ]
        root[name].attrs["ome"] = metadata
    root.attrs["ome"] = {"bioformats2raw.layout": 3}
    ome_group = root.create_group("OME")
    ome_group.attrs["ome"] = {"version": "0.5", "series": ["0", "1"]}

    source = LimImageSource.open(store / "zarr.json")
    assert source.get_file_dimensions() == {"multipoint": 2}


def test_ome_zarr_incompatible_plate_fields_drop_plate_layout(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "incompatible-plate.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "A/1/0", np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    _write_image_group(root, "B/2/0", np.zeros((1, 1, 1, 3, 2), dtype=np.uint8))
    root.attrs["ome"] = {
        "plate": {
            "name": "DemoPlate",
            "rows": [{"name": "A"}, {"name": "B"}],
            "columns": [{"name": "1"}, {"name": "2"}],
            "wells": [{"path": "A/1"}, {"path": "B/2"}],
            "field_count": 1,
        }
    }
    root["A/1"].attrs["ome"] = {"well": {"images": [{"path": "0"}]}}
    root["B/2"].attrs["ome"] = {"well": {"images": [{"path": "0"}]}}

    source = LimImageSource.open(store / "zarr.json")
    expanded, dimensions = source.parse_additional_dimensions({source: []}, {})

    assert dimensions == {}
    assert len(expanded) == 1
    assert next(iter(expanded)).group_path == Path("A/1/0")
    assert source.ome_zarr_wellplate([list(expanded)]) is None


def test_ome_zarr_plate_paths_become_well_multipoints(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "plate.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "A/1/0", np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    _write_image_group(root, "B/2/0", np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    root.attrs["ome"] = {
        "plate": {
            "name": "DemoPlate",
            "rows": [{"name": "A"}, {"name": "B"}],
            "columns": [{"name": "1"}, {"name": "2"}],
            "wells": [{"path": "A/1"}, {"path": "B/2"}],
            "field_count": 1,
        }
    }
    root["A/1"].attrs["ome"] = {"well": {"images": [{"path": "0"}]}}
    root["B/2"].attrs["ome"] = {"well": {"images": [{"path": "0"}]}}

    source = LimImageSource.open(store / "zarr.json")
    expanded, dimensions = source.parse_additional_dimensions({source: []}, {})
    assert dimensions == {"multipoint": 2}
    assert {values[0] for values in expanded.values()} == {0, 1}


def test_ome_zarr_plate_labels_restore_as_nd2_binary_layers(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "labels.ome.zarr"
    root = zarr.open_group(store, mode="w")
    root.attrs["ome"] = {
        "plate": {
            "name": "DemoPlate",
            "rows": [{"name": "A"}],
            "columns": [{"name": "1"}],
            "wells": [{"path": "A/1"}],
            "field_count": 1,
        }
    }
    root.require_group("A/1").attrs["ome"] = {"well": {"images": [{"path": "0"}]}}
    _write_image_group(root, "A/1/0", np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    label = root.require_group("A/1/0/labels/Mask")
    label.create_array("0", data=np.array([[[[0, 1], [2, 0]]]], dtype=np.uint32))
    label.attrs["ome"] = {
        "multiscales": [{
            "axes": [{"name": "t"}, {"name": "z"}, {"name": "y"}, {"name": "x"}],
            "datasets": [{"path": "0"}],
        }],
        "image-label": {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},
                {"label-value": 1, "rgba": [71, 255, 30, 255]},
                {"label-value": 2, "rgba": [71, 255, 30, 255]},
            ]
        },
    }

    out = tmp_path / "labels.nd2"
    convert_file_to_nd2(store / "zarr.json", out)
    with limnd2.Nd2Reader(out) as reader:
        assert [item.name for item in reader.binaryRasterMetadata] == ["Mask"]
        np.testing.assert_array_equal(
            reader.binaryRasterData(1, 0),
            np.array([[0, 1], [2, 0]], dtype=np.uint32),
        )
        assert reader.binaryRasterMetadata[0].color == (71 / 255, 1.0, 30 / 255)


def test_ome_zarr_sidecar_restores_image_text_info(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "sidecar.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "", np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    text_info = ImageTextInfo(
        TextInfoItem_4="limnd2 test author",
        TextInfoItem_5="Retain this image description",
    )
    sidecar = store / "nd2_chunks"
    sidecar.mkdir()
    (sidecar / "ImageTextInfoLV.json").write_text(text_info.toJson(), encoding="utf-8")
    (sidecar / "manifest.json").write_text(
        json.dumps({
            "schemaVersion": 1,
            "chunks": [{
                "nd2ChunkName": "ImageTextInfoLV!",
                "path": "ImageTextInfoLV.json",
                "kind": "imageTextInfo",
            }],
        }),
        encoding="utf-8",
    )

    out = tmp_path / "sidecar.nd2"
    convert_file_to_nd2(store / "zarr.json", out)
    with limnd2.Nd2Reader(out) as reader:
        assert reader.imageTextInfo is not None
        assert reader.imageTextInfo.toDict() == text_info.toDict()


@pytest.mark.parametrize("nested", [False, True])
def test_ome_zarr_reads_current_and_legacy_omero_channel_metadata(tmp_path, nested):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / ("nested.ome.zarr" if nested else "legacy.ome.zarr")
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "", np.zeros((1, 2, 1, 2, 2), dtype=np.uint8))
    omero = {
        "channels": [
            {"label": "BRF", "color": "FFFFFF"},
            {"label": "Hoechst", "color": "2205FF"},
        ]
    }
    if nested:
        root.attrs["ome"] = {**root.attrs["ome"], "omero": omero}
    else:
        root.attrs["omero"] = omero

    source = LimImageSource.open(store / "zarr.json")
    settings = ConversionSettings()
    source.parse_additional_metadata(settings)

    planes = list(settings.channels.values())
    assert [plane.name for plane in planes] == ["BRF", "Hoechst"]
    assert [plane.color for plane in planes] == ["#FFFFFF", "#2205FF"]

    qml_settings = source.metadata_as_pattern_settings()
    assert [channel[:2] for channel in qml_settings["channels"]] == [
        ["BRF", "BRF"], ["Hoechst", "Hoechst"]
    ]
    assert [channel[5] for channel in qml_settings["channels"]] == [
        "#FFFFFF", "#2205FF"
    ]


def test_ome_zarr_metadata_probe_accepts_store_folder_and_reports_qml_settings(
    tmp_path, capsys
):
    zarr = pytest.importorskip("zarr")
    from limnd2.tools.conversion.LimImageSource import get_file_dimensions_as_json

    store = tmp_path / "metadata-probe.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "", np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    root.attrs["ome"] = {
        **root.attrs["ome"],
        "omero": {"channels": [{"label": "DAPI", "color": "0000FF"}]},
    }

    get_file_dimensions_as_json(store)
    probe = json.loads(capsys.readouterr().out)

    assert probe["error"] is False
    assert probe["qml_settings"]["channels"][0][0:2] == ["DAPI", "DAPI"]
    assert probe["qml_settings"]["channels"][0][5] == "#0000FF"


def test_ome_zarr_rgb_components_import_as_one_rgb_plane(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "rgb.ome.zarr"
    root = zarr.open_group(store, mode="w")
    pixels = np.arange(1 * 3 * 1 * 4 * 5, dtype=np.uint8).reshape(1, 3, 1, 4, 5)
    _write_image_group(root, "", pixels)
    ome = root.attrs["ome"]
    ome["omero"] = {
        "channels": [
            {"label": "Blue", "color": "0000FF"},
            {"label": "Green", "color": "00FF00"},
            {"label": "Red", "color": "FF0000"},
        ],
        "rdefs": {"model": "color"},
    }
    root.attrs["ome"] = ome

    source = LimImageSource.open(store / "zarr.json")
    assert source.is_rgb is True
    assert source.get_file_dimensions() == {}
    expected = pixels[0, :, 0].transpose(1, 2, 0)
    np.testing.assert_array_equal(source.read(), expected)
    np.testing.assert_array_equal(source.read_tile(1, 1, 3, 2), expected[1:3, 1:4])

    # OME-XML expresses packed RGB as one Channel with SamplesPerPixel=3.
    # This must override the ordinary one-plane interpretation of Name="RGB".
    xml_dir = store / "XML"
    xml_dir.mkdir()
    (xml_dir / "METADATA.ome.xml").write_text(
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
        'Type="uint8" SizeX="5" SizeY="4" SizeZ="1" SizeC="3" SizeT="1">'
        '<Channel ID="Channel:0" Name="RGB" SamplesPerPixel="3" Color="-1"/>'
        '</Pixels></Image></OME>',
        encoding="utf-8",
    )

    out = tmp_path / "rgb.nd2"
    convert_file_to_nd2(store / "zarr.json", out)
    with limnd2.Nd2Reader(out) as reader:
        assert reader.isRgb is True
        assert reader.imageDataShape == (1, 1, 1, 4, 5, 3)
        assert reader.pictureMetadata.sPicturePlanes.uiCount == 1
        assert reader.pictureMetadata.sPicturePlanes.uiCompCount == 3
        np.testing.assert_array_equal(reader.image(0), expected)


def test_ome_zarr_recognizes_rgb_from_ome_xml_samples_per_pixel(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "rgb-from-xml.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(
        root,
        "",
        np.zeros((1, 3, 1, 4, 5), dtype=np.uint8),
    )
    xml_dir = store / "XML"
    xml_dir.mkdir()
    (xml_dir / "METADATA.ome.xml").write_text(
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
        'Type="uint8" SizeX="5" SizeY="4" SizeZ="1" SizeC="3" SizeT="1">'
        '<Channel ID="Channel:0" Name="RGB" SamplesPerPixel="3"/>'
        '</Pixels></Image></OME>',
        encoding="utf-8",
    )

    source = LimImageSource.open(store / "zarr.json")

    assert source.is_rgb
    assert source.get_file_dimensions() == {}
    picture_metadata = next(
        value
        for value in source.native_nd2_metadata().values()
        if isinstance(value, limnd2.PictureMetadata)
    )
    assert picture_metadata.isRgb


def test_ome_zarr_imports_significant_bits_from_ome_xml(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "significant-bits.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "", np.array([[[[[0, 1], [4095, 2048]]]]], dtype=np.uint16))
    ome_dir = store / "XML"
    ome_dir.mkdir()
    (ome_dir / "METADATA.ome.xml").write_text(
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
        'Type="uint16" SizeX="2" SizeY="2" SizeZ="1" SizeC="1" SizeT="1" '
        'SignificantBits="12"><Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
        '</Pixels></Image></OME>',
        encoding="utf-8",
    )

    source = LimImageSource.open(store / "zarr.json")
    attributes = source.nd2_attributes()
    assert attributes.uiBpcInMemory == 16
    assert attributes.uiBpcSignificant == 12


def test_ome_xml_channel_metadata_overrides_ngff_display_metadata(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "channel-metadata.ome.zarr"
    root = zarr.open_group(store, mode="w")
    _write_image_group(root, "", np.zeros((1, 1, 1, 2, 2), dtype=np.uint16))
    group_ome = root.attrs["ome"]
    group_ome["omero"] = {"channels": [{"label": "NGFF label", "color": "00FF00"}]}
    root.attrs["ome"] = group_ome
    xml_dir = store / "XML"
    xml_dir.mkdir()
    (xml_dir / "METADATA.ome.xml").write_text(
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
        'Type="uint16" SizeX="2" SizeY="2" SizeZ="1" SizeC="1" SizeT="1">'
        '<Channel ID="Channel:0:0" Name="XML label" SamplesPerPixel="1" '
        'ExcitationWavelength="488" EmissionWavelength="520"/></Pixels></Image></OME>',
        encoding="utf-8",
    )

    source = LimImageSource.open(store / "zarr.json")
    settings = ConversionSettings()
    source.parse_additional_metadata(settings)
    channel = settings.channels["ome-zarr-0"]
    assert channel.name == "XML label"
    assert channel.excitation_wavelength == 488
    assert channel.emission_wavelength == 520
