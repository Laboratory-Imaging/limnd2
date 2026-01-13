from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import limnd2


def test_general_info_and_basic_open(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as nd2:
        # Basic open and properties
        assert isinstance(nd2.version, tuple) and len(nd2.version) == 2
        assert nd2.store.sizeOnDisk > 0
        _ = nd2.store.lastModified

        gi = limnd2.generalImageInfo(nd2)
        assert isinstance(gi, dict)
        assert "dimension" in gi


def test_image_attributes_and_first_frame(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as nd2:
        a = nd2.imageAttributes
        assert a.width > 0 and a.height > 0
        assert a.componentCount >= 1
        assert 1 <= a.uiBpcSignificant <= 32

        if a.frameCount <= 0:
            pytest.skip("no frames")

        img0 = nd2.image(0)
        assert isinstance(img0, np.ndarray)
        assert img0.shape[0] == a.height and img0.shape[1] == a.width
        # components may be 3 for RGB, else use componentCount
        expected_comp = 3 if nd2.isRgb else a.componentCount
        assert img0.shape[-1] == expected_comp


# Experiment-focused tests moved to tests/test_reader_experiment.py


def test_text_info_and_app_info(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as nd2:
        # Image text info
        iti = nd2.imageTextInfo
        if iti is not None:
            d = iti.to_dict()
            assert isinstance(d, dict)

        # App info and software string
        assert isinstance(nd2.software, str)
        _ = nd2.appInfo


def test_metadata_channels_and_settings(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as nd2:
        md = nd2.pictureMetadata
        # Access channel list; count can be 1 for mono or 3 for RGB reported differently
        channels = getattr(md, "channels", [])
        assert channels is not None

        for ch in channels:
            # These fields may be zero if unspecified; just ensure they exist and are numeric
            assert hasattr(ch, "emissionWavelengthNm")
            assert hasattr(ch, "excitationWavelengthNm")
            assert isinstance(ch.emissionWavelengthNm, (int, float))
            assert isinstance(ch.excitationWavelengthNm, (int, float))

            # Sample/microscope settings for this channel
            settings = md.sampleSettings(ch)
            assert settings is not None
            assert hasattr(settings, "objectiveMagnification")


def test_optional_structures_and_ranges(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as nd2:
        # Optional wellplate descriptors should not raise
        _ = nd2.wellplateDesc
        _ = nd2.wellplateFrameInfo

        # Data range tuple
        lo, hi = nd2.imageDataRange
        assert isinstance(lo, (int, float)) and isinstance(hi, (int, float))
        assert lo <= hi

        # Recorded data access should not raise
        _ = nd2.recordedData
