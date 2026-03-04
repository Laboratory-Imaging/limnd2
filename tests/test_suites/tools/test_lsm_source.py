from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
from limnd2.tools.conversion.LimImageSource import LimImageSource
from limnd2.tools.conversion.LimImageSourceConvert import LIMND2Utils
from limnd2.tools.conversion.LimImageSourceLsm import LimImageSourceLsm
from limnd2.tools.conversion.LimImageSourceTiff_base import LimImageSourceTiffBase


def _write_lsm(path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(8 * 6, dtype=np.uint16).reshape(8, 6)
    tifffile.imwrite(path, arr)


def test_open_lsm_and_read(tmp_path: Path) -> None:
    path = tmp_path / "sample.lsm"
    _write_lsm(path)

    src = LimImageSource.open(path)
    assert isinstance(src, LimImageSourceLsm)

    frame = src.read()
    assert frame.shape == (8, 6)
    assert frame.dtype == np.uint16


def test_lsm_metadata_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "meta.lsm"
    _write_lsm(path)

    fake_meta = {
        "VoxelSizeX": 3.375e-7,
        "VoxelSizeZ": 2.0e-6,
        "TimeIntervall": 0.5,
        "DimensionChannels": 3,
        "ChannelColors": {
            "ColorNames": ["DAPI", "FITC", "TRITC"],
            "Colors": [
                [64, 128, 255, 0],
                [0, 176, 80, 0],
                [255, 48, 48, 0],
            ],
        },
        "ChannelWavelength": np.array(
            [
                [405.0, 460.0],
                [488.0, 520.0],
                [561.0, 605.0],
            ],
            dtype=np.float64,
        ),
    }

    monkeypatch.setattr(LimImageSourceLsm, "_lsm_metadata", lambda self: dict(fake_meta))

    src = LimImageSourceLsm(path)

    settings = src.metadata_as_pattern_settings()
    assert settings["pixel_calibration"] == "0.3375"
    assert settings["zstep"] == "2.000"
    assert settings["tstep"] == "500.000"
    assert [row[0] for row in settings["channels"]] == ["DAPI", "FITC", "TRITC"]
    assert settings["channels"][0][5] == "#4080FF"
    assert settings["channels"][1][5] == "#00B050"

    storage = ConversionSettings()
    storage.time_step = None
    storage.z_step = None
    src.parse_additional_metadata(storage)

    assert storage.metadata.pixel_calibration == pytest.approx(0.3375)
    assert storage.time_step == pytest.approx(500.0)
    assert storage.z_step == pytest.approx(2.0)
    assert storage.channels[0].name == "DAPI"
    assert storage.channels[1].emission_wavelength == 520


def test_tiff_nd2_attributes_channel_first_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Tag:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class _Tags:
        def values(self):
            return [
                _Tag("BitsPerSample", 16),
                _Tag("MaxSampleValue", 65535),
                _Tag("SamplesPerPixel", 3),
            ]

    class _Page:
        shape = (3, 4096, 4096)
        dtype = np.dtype(np.uint16)
        tags = _Tags()

    class _Reader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceTiff_base.tifffile.TiffReader",
        _Reader,
    )

    path = tmp_path / "dummy.lsm"
    path.write_bytes(b"")
    src = LimImageSourceTiffBase(path)
    attrs = src.nd2_attributes(sequence_count=1)

    assert attrs.uiHeight == 4096
    assert attrs.uiWidth == 4096
    assert attrs.uiComp == 3


def test_tiff_nd2_attributes_bits_tuple_with_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Tag:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class _Tags:
        def values(self):
            return [
                _Tag("BitsPerSample", (8, 8, 0)),
                _Tag("MaxSampleValue", (255, 255, 0)),
                _Tag("SamplesPerPixel", 3),
            ]

    class _Page:
        shape = (1024, 1024, 3)
        dtype = np.dtype(np.uint8)
        tags = _Tags()

    class _Reader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceTiff_base.tifffile.TiffReader",
        _Reader,
    )

    path = tmp_path / "dummy_bits_tuple.lsm"
    path.write_bytes(b"")
    src = LimImageSourceLsm(path)
    attrs = src.nd2_attributes(sequence_count=1)

    assert attrs.uiBpcSignificant == 8
    assert attrs.uiBpcInMemory == 8
    assert attrs.uiComp == 3


def test_tiff_unknown_dimension_defaults_to_multipoint(tmp_path: Path) -> None:
    path = tmp_path / "unknown_dim_default.lsm"
    path.write_bytes(b"")
    src = LimImageSourceLsm(path)

    src._additional_dimensions = {"axis_parsed": ["unknown"], "shape": [4]}

    expanded_sources, expanded_dims = src.parse_additional_dimensions({src: []}, {})

    assert "unknown" not in expanded_dims
    assert expanded_dims["multipoint"] == 4
    assert len(expanded_sources) == 4


def test_tiff_rgb_requires_at_least_three_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Photometric:
        name = "RGB"

    class _Page:
        photometric = _Photometric()
        samplesperpixel = 2

    class _Reader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [_Page()]
            self.series = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceTiff_base.tifffile.TiffReader",
        _Reader,
    )

    path = tmp_path / "two_sample_rgb_like.lsm"
    path.write_bytes(b"")
    src = LimImageSourceLsm(path)

    assert src.is_rgb is False


def test_coerce_components_pads_to_expected() -> None:
    class _Attrs:
        componentCount = 3

    arr = np.ones((5, 7, 2), dtype=np.uint16)
    out = LIMND2Utils._coerce_array_components(arr, _Attrs())

    assert out.shape == (5, 7, 3)
    assert out.dtype == np.uint16
    assert np.all(out[..., :2] == 1)
    assert np.all(out[..., 2] == 0)


def test_lsm_page_key_clamped_to_last_mapped_page(tmp_path: Path) -> None:
    path = tmp_path / "mapped_index_overflow.lsm"
    path.write_bytes(b"")
    src = LimImageSourceLsm(path)
    src._idf_page_keys = [10, 20, 30]

    assert src._page_key_for_idf(100) == 30


def test_lsm_read_channel_first_honors_channel_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _TiffFile:
        def __init__(self, *_args, **_kwargs):
            self._arr = np.arange(8 * 120 * 110, dtype=np.uint16).reshape(8, 120, 110)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def asarray(self, key=0):
            _ = key
            return self._arr

    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceLsm.tifffile.TiffFile",
        _TiffFile,
    )

    path = tmp_path / "channel_first_read.lsm"
    path.write_bytes(b"")
    src = LimImageSourceLsm(path, channel_index=3)
    src._idf_page_keys = [0]

    out = src.read()
    assert out.shape == (120, 110)
    assert np.array_equal(out, np.arange(8 * 120 * 110, dtype=np.uint16).reshape(8, 120, 110)[3])


def test_lsm_is_rgb_false_for_eight_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Photometric:
        name = "RGB"

    class _Page:
        photometric = _Photometric()
        samplesperpixel = 8

    class _Reader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [_Page()]
            self.series = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceLsm.tifffile.TiffReader",
        _Reader,
    )

    path = tmp_path / "eight_sample_rgb_like.lsm"
    path.write_bytes(b"")
    src = LimImageSourceLsm(path)

    assert src.is_rgb is False
