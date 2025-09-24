from __future__ import annotations

from pathlib import Path

import pytest

import limnd2
from limnd2.metadata import (
    calculateColor,
    PicturePlaneModality,
    PicturePlaneModalityFlags,
    PictureMetadata,
)


def test_calculate_color_and_modality_helpers():
    # color name and hex
    assert calculateColor("Red") == 0x0000FF
    assert calculateColor("#00FF00") == 0x00FF00
    # tuple
    assert calculateColor((1, 2, 3)) == 0x030201
    # invalid hex
    with pytest.raises(ValueError):
        calculateColor("#GG0000")

    # modality conversions
    bf = PicturePlaneModalityFlags.from_modality(PicturePlaneModality.eModBrightfield)
    assert bf & PicturePlaneModalityFlags.modBrightfield
    lst = PicturePlaneModalityFlags.to_str_list(bf)
    assert "Brightfield" in lst

    dic = PicturePlaneModalityFlags.from_modality_string("DIC")
    assert dic & PicturePlaneModalityFlags.modDIContrast


def test_picture_metadata_roundtrip_and_properties(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as nd2:
        md: PictureMetadata = nd2.pictureMetadata

        # Basic channel/component properties
        ch_names = md.channelNames
        comp_names = md.componentNames
        comp_colors = md.componentColors
        assert isinstance(ch_names, list) and all(isinstance(x, str) for x in ch_names)
        assert isinstance(comp_names, list) and len(comp_names) == len(comp_colors)
        for c in comp_colors:
            assert isinstance(c, tuple) and len(c) == 3
            assert all(0.0 <= v <= 1.0 for v in c)

        # Access sample settings for each plane (if present)
        planes = md.sPicturePlanes.sPlaneNew
        for plane in planes:
            ss = md.sampleSettings(plane)
            if ss is None:
                continue
            # Validate selected fields exist with correct types
            assert isinstance(md.cameraName(plane), str)
            assert isinstance(md.microscopeName(plane), str)
            assert isinstance(md.refractiveIndex(plane), float)
            assert isinstance(md.objectiveName(plane), str)
            assert isinstance(md.objectiveMagnification(plane), float)
            assert isinstance(md.objectiveNumericAperture(plane), float)
            assert isinstance(md.opticalConfigurations(plane), list)
            # spectral ints are numbers or 0
            assert isinstance(plane.emissionWavelengthNm, (int, float))
            assert isinstance(plane.excitationWavelengthNm, (int, float))

        # to_lv / from_lv roundtrip
        blob = md.to_lv()
        md2 = PictureMetadata.from_lv(blob)
        assert isinstance(md2, PictureMetadata)
        assert md2.channelNames == md.channelNames
        assert len(md2.sPicturePlanes.sPlaneNew) == len(md.sPicturePlanes.sPlaneNew)

