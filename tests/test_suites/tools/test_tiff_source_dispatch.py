from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
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
    assert [row[0] for row in settings["channels"]] == ["1. BLUE penta", "4. GREEN penta", "8. RED"]
    assert settings["channels"][1][4] == "517"
    assert settings["channels"][0][5] == "#4080FF"
    assert settings["channels"][1][5] == "#00B050"
    assert settings["channels"][2][5] == "#FF3030"
    assert settings["application_name"] == "MetaMorph"
    assert settings["application_version"] == "6.7.2.290"
    assert settings["exposure_time"] == "300 ms"


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
