from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
from limnd2.tools.conversion.LimImageSourceConvert import _derive_channel_labels_and_templates
from limnd2.tools.conversion.LimImageSourceTiff import LimImageSourceTiff
from limnd2.tools.conversion.LimImageSourceTiff_base import LimImageSourceTiffBase
from limnd2.tools.conversion.LimImageSourceTiff_meta import LimImageSourceTiffMeta
from limnd2.tools.conversion.LimImageSourceTiff_ometiff import LimImageSourceTiffOmeTiff


def _write_tiff(path: Path, *, description: str | None = None) -> None:
    tifffile = pytest.importorskip("tifffile")
    arr = np.zeros((8, 8), dtype=np.uint16)
    tifffile.imwrite(path, arr, description=description)


def test_tiff_dispatch_selects_base(tmp_path: Path) -> None:
    path = tmp_path / "plain.tif"
    _write_tiff(path, description="Just a plain TIFF")

    src = LimImageSourceTiff(path)
    assert isinstance(src, LimImageSourceTiffBase)
    assert not isinstance(src, LimImageSourceTiffMeta)
    assert not isinstance(src, LimImageSourceTiffOmeTiff)


def test_tiff_dispatch_selects_ome(tmp_path: Path) -> None:
    path = tmp_path / "ome.tif"
    desc = "<?xml version='1.0'?><OME xmlns='http://www.openmicroscopy.org/Schemas/OME/2016-06'></OME>"
    _write_tiff(path, description=desc)

    src = LimImageSourceTiff(path)
    assert isinstance(src, LimImageSourceTiffOmeTiff)


def test_tiff_dispatch_selects_meta_and_extracts_qml(tmp_path: Path) -> None:
    path = tmp_path / "meta.tif"
    htd = tmp_path / "plate.HTD"
    htd.write_text(
        "\n".join(
            [
                '"HTSInfoFile", Version 1.0',
                '"Waves", TRUE',
                '"NWavelengths", 3',
                '"XWells", 12',
                '"YWells", 8',
                '"WaveName1", "1. BLUE penta"',
                '"WaveName2", "4. GREEN penta"',
                '"WaveName3", "8. RED"',
                '"WaveCollect1", 1',
                '"WaveCollect2", 1',
                '"WaveCollect3", 1',
                '"EndFile"',
            ]
        ),
        encoding="utf-8",
    )

    desc = (
        "<MetaData>."
        "<prop id=\"ApplicationName\" type=\"string\" value=\"MetaMorph\"/>."
        "<prop id=\"ApplicationVersion\" type=\"string\" value=\"6.7.2.290\"/>."
        "<prop id=\"spatial-calibration-x\" type=\"float\" value=\"0.3375\"/>."
        "<prop id=\"spatial-calibration-units\" type=\"string\" value=\"um\"/>."
        "<prop id=\"_MagNA_\" type=\"float\" value=\"0.75\"/>."
        "<prop id=\"_MagRI_\" type=\"float\" value=\"1\"/>."
        "<prop id=\"_MagSetting_\" type=\"string\" value=\"20X Plan Apo Lambda\"/>."
        "<prop id=\"_IllumSetting_\" type=\"string\" value=\"4. GREEN penta\"/>."
        "<prop id=\"zoom-percent\" type=\"int\" value=\"20\"/>."
        "<prop id=\"wavelength\" type=\"float\" value=\"517\"/>."
        "<custom-prop id=\"Exposure Time\" type=\"string\" value=\"300 ms\"/>."
        "<custom-prop id=\"IXConfocal Module Disk\" type=\"string\" value=\"IN - 60 um pinhole - Running\"/>."
        "<PlaneInfo>."
        "</PlaneInfo>."
        "</MetaData>."
    )
    _write_tiff(path, description=desc)

    src = LimImageSourceTiff(path)
    assert isinstance(src, LimImageSourceTiffMeta)

    settings = src.metadata_as_pattern_settings()
    assert settings["pixel_calibration"] == "0.3375"
    assert settings["objective_numerical_aperture"] == "0.75"
    assert settings["objective_magnification"] == "20.0"
    assert settings["immersion_refractive_index"] == "1.0"
    assert settings["pinhole_diameter"] == "60.0"
    assert settings["zoom_magnification"] == "20.0"
    assert [row[0] for row in settings["channels"]] == ["4. GREEN penta"]
    assert settings["channels"][0][4] == "517"
    assert settings["channels"][0][5] == "#00B050"
    assert settings["application_name"] == "MetaMorph"
    assert settings["application_version"] == "6.7.2.290"
    assert settings["exposure_time"] == "300 ms"
    assert settings["plate_rows"] == "8"
    assert settings["plate_columns"] == "12"
    assert settings["plate_well_count"] == "96"
    assert src.get_htd_plate_geometry() == (8, 12)


def test_meta_parse_additional_metadata_updates_conversion_settings(tmp_path: Path) -> None:
    path = tmp_path / "meta_merge.tif"
    desc = (
        "<MetaData>."
        "<prop id=\"spatial-calibration-x\" type=\"float\" value=\"0.5\"/>."
        "<prop id=\"spatial-calibration-units\" type=\"string\" value=\"um\"/>."
        "<prop id=\"_MagNA_\" type=\"float\" value=\"1.2\"/>."
        "<prop id=\"_MagRI_\" type=\"float\" value=\"1.33\"/>."
        "<prop id=\"_MagSetting_\" type=\"string\" value=\"40X\"/>."
        "<prop id=\"_IllumSetting_\" type=\"string\" value=\"DAPI\"/>."
        "<prop id=\"wavelength\" type=\"float\" value=\"465\"/>."
        "<PlaneInfo>."
        "</PlaneInfo>."
        "</MetaData>."
    )
    _write_tiff(path, description=desc)

    src = LimImageSourceTiff(path)
    settings = ConversionSettings()
    src.parse_additional_metadata(settings)

    assert settings.metadata.pixel_calibration == pytest.approx(0.5)
    assert settings.metadata._other_settings["objective_numerical_aperture"] == pytest.approx(1.2)
    assert settings.metadata._other_settings["immersion_refractive_index"] == pytest.approx(1.33)
    assert settings.metadata._other_settings["objective_magnification"] == pytest.approx(40.0)
    assert 0 in settings.channels
    assert settings.channels[0].name == "DAPI"
    assert settings.channels[0].emission_wavelength == 465


def test_tiff_base_mixed_resolution_series_uses_primary_resolution(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")

    path = tmp_path / "mixed_series.tif"
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((2, 8, 8), dtype=np.uint16), photometric="minisblack")
        tw.write(np.zeros((4, 4), dtype=np.uint16), photometric="minisblack")

    src = LimImageSourceTiff(path)
    assert isinstance(src, LimImageSourceTiffBase)

    dims = src.get_file_dimensions()
    assert dims == {"unknown": 2}

    expanded_sources, expanded_dims = src.parse_additional_dimensions({src: []}, {})
    assert expanded_dims["unknown"] == 2
    assert len(expanded_sources) == 2

    sorted_sources = sorted(expanded_sources.keys(), key=lambda s: s.idf)
    frames = [s.read() for s in sorted_sources]
    assert all(frame.shape == (8, 8) for frame in frames)


def test_channel_templates_infer_from_slot_when_channel_index_missing(tmp_path: Path) -> None:
    class _FakeMetaSource:
        def __init__(self, filename: Path):
            self.filename = filename
            self.channel_index = None

        def metadata_as_pattern_settings(self) -> dict:
            return {
                "channels": [
                    ["1. BLUE penta", "1. BLUE penta", "Confocal, Fluo", "0", "462", "#4080FF"],
                    ["4. GREEN penta", "4. GREEN penta", "Confocal, Fluo", "0", "517", "#00B050"],
                    ["8. RED", "8. RED", "Confocal, Fluo", "0", "630", "#FF3030"],
                ]
            }

    file_a = tmp_path / "A.tif"
    file_b = tmp_path / "B.tif"
    file_a.write_bytes(b"")
    file_b.write_bytes(b"")

    grouped = [[_FakeMetaSource(file_a), _FakeMetaSource(file_b)]]
    labels, templates = _derive_channel_labels_and_templates(grouped, component_count=2)

    assert labels == ["1. BLUE penta", "4. GREEN penta"]
    assert templates[0] is not None
    assert templates[1] is not None
    assert templates[0]["name"] == "1. BLUE penta"
    assert templates[1]["name"] == "4. GREEN penta"
    assert templates[0]["emission_wavelength"] == 462
    assert templates[1]["emission_wavelength"] == 517
    assert templates[0]["color"] == "#4080FF"
    assert templates[1]["color"] == "#00B050"


def test_channel_templates_expand_meta_tiff_channels_from_htd_in_sequence(tmp_path: Path) -> None:
    path = tmp_path / "meta_seq.tif"
    htd = tmp_path / "plate.HTD"
    htd.write_text(
        "\n".join(
            [
                '"HTSInfoFile", Version 1.0',
                '"Waves", TRUE',
                '"NWavelengths", 3',
                '"WaveName1", "1. BLUE penta"',
                '"WaveName2", "4. GREEN penta"',
                '"WaveName3", "8. RED"',
                '"WaveCollect1", 1',
                '"WaveCollect2", 1',
                '"WaveCollect3", 1',
                '"EndFile"',
            ]
        ),
        encoding="utf-8",
    )

    desc = (
        "<MetaData>."
        "<prop id=\"_IllumSetting_\" type=\"string\" value=\"4. GREEN penta\"/>."
        "<prop id=\"wavelength\" type=\"float\" value=\"517\"/>."
        "<custom-prop id=\"IXConfocal Module Disk\" type=\"string\" value=\"IN - 60 um pinhole - Running\"/>."
        "<PlaneInfo>."
        "</PlaneInfo>."
        "</MetaData>."
    )
    _write_tiff(path, description=desc)

    grouped = [
        [
            LimImageSourceTiff(path, channel_index=0),
            LimImageSourceTiff(path, channel_index=1),
            LimImageSourceTiff(path, channel_index=2),
        ]
    ]
    labels, templates = _derive_channel_labels_and_templates(grouped, component_count=3)

    assert labels == ["1. BLUE penta", "4. GREEN penta", "8. RED"]
    assert templates[1] is not None
    assert templates[1]["name"] == "4. GREEN penta"
    assert templates[1]["emission_wavelength"] == 517


def test_meta_tiff_channel_name_color_overrides_threshold_color(tmp_path: Path) -> None:
    path = tmp_path / "meta_color_priority.tif"
    desc = (
        "<MetaData>."
        "<prop id=\"_IllumSetting_\" type=\"string\" value=\"8. RED\"/>."
        "<prop id=\"wavelength\" type=\"float\" value=\"639\"/>."
        "<prop id=\"threshold-color\" type=\"string\" value=\"#4080FF\"/>."
        "<PlaneInfo>."
        "</PlaneInfo>."
        "</MetaData>."
    )
    _write_tiff(path, description=desc)

    src = LimImageSourceTiff(path)
    settings = src.metadata_as_pattern_settings()
    assert settings["channels"][0][0] == "8. RED"
    assert settings["channels"][0][5] == "#FF3030"
