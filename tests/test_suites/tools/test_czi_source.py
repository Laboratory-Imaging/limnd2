from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("czifile")

import limnd2.tools.conversion.LimImageSourceCzi as czi_mod
from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
from limnd2.tools.conversion.LimImageSource import LimImageSource
from limnd2.tools.conversion.LimImageSourceCzi import LimImageSourceCzi


class _FakeCziFile:
    _datasets: dict[str, dict[str, object]] = {}

    def __init__(self, arg, multifile=True, filesize=None, detectmosaic=True):
        del multifile, filesize, detectmosaic
        key = str(Path(arg))
        if key not in self._datasets:
            raise FileNotFoundError(f"No fake CZI dataset for {key}")
        dataset = self._datasets[key]
        self.axes = dataset["axes"]
        self.shape = dataset["shape"]
        self.dtype = dataset["dtype"]
        self._array = dataset["array"]
        self._metadata = dataset.get("metadata", "")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def asarray(self, resize=True, order=0, out=None, max_workers=None):
        del resize, order, out, max_workers
        return np.asarray(self._array)

    def metadata(self, raw=True):
        del raw
        return self._metadata


def _register_fake_dataset(path: Path, *, axes: str, array: np.ndarray, metadata: str = ""):
    _FakeCziFile._datasets[str(path)] = {
        "axes": axes,
        "shape": tuple(int(v) for v in array.shape),
        "dtype": np.asarray(array).dtype,
        "array": np.asarray(array),
        "metadata": metadata,
    }


def _metadata_xml() -> str:
    return """<?xml version='1.0' encoding='utf-8'?>
<ImageDocument>
  <Metadata>
    <Scaling>
      <Items>
        <Distance Id='X'><Value>3.375e-07</Value></Distance>
        <Distance Id='Z'><Value>2.0e-06</Value></Distance>
      </Items>
    </Scaling>
    <Information>
      <Image>
        <Dimensions>
          <Channels>
            <Channel Id='Channel:0' Name='DAPI' ExcitationWavelength='405' EmissionWavelength='460' Color='#FF4080FF' />
            <Channel Id='Channel:1' Name='FITC' ExcitationWavelength='488' EmissionWavelength='520' Color='#FF00FF00' />
            <Channel Id='Channel:2' Name='TRITC' ExcitationWavelength='561' EmissionWavelength='605' Color='#FFFF3030' />
          </Channels>
        </Dimensions>
      </Image>
    </Information>
  </Metadata>
</ImageDocument>
"""


def test_czi_dimensions_and_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(czi_mod, "CziFile", _FakeCziFile)

    path = tmp_path / "sample.czi"
    path.write_bytes(b"fake-czi")

    arr = np.zeros((2, 3, 4, 8, 6, 1), dtype=np.uint16)  # T,C,Z,Y,X,0
    _register_fake_dataset(path, axes="TCZYX0", array=arr, metadata=_metadata_xml())

    src = LimImageSourceCzi(path)
    assert src.get_file_dimensions() == {"timeloop": 2, "channel": 3, "zstack": 4}

    settings = src.metadata_as_pattern_settings()
    assert settings["pixel_calibration"] == "0.3375"
    assert settings["zstep"] == "2.000"
    assert [row[0] for row in settings["channels"]] == ["DAPI", "FITC", "TRITC"]
    assert settings["channels"][0][5] == "#4080FF"

    storage = ConversionSettings()
    storage.z_step = None
    src.parse_additional_metadata(storage)
    assert storage.metadata.pixel_calibration == pytest.approx(0.3375)
    assert storage.z_step == pytest.approx(2.0)
    assert storage.channels[0].name == "DAPI"
    assert storage.channels[1].emission_wavelength == 520



def test_czi_parse_additional_dimensions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(czi_mod, "CziFile", _FakeCziFile)

    path = tmp_path / "dims.czi"
    path.write_bytes(b"fake-czi")

    arr = np.zeros((2, 3, 2, 5, 4, 1), dtype=np.uint16)  # T,C,Z,Y,X,0
    _register_fake_dataset(path, axes="TCZYX0", array=arr, metadata="")

    src = LimImageSourceCzi(path)
    new_sources, new_dims = src.parse_additional_dimensions({src: []}, {}, "multipoint")

    assert new_dims["timeloop"] == 2
    assert new_dims["channel"] == 3
    assert new_dims["zstack"] == 2
    assert len(new_sources) == 12

    channel_indexes = sorted({source.channel_index for source in new_sources})
    assert channel_indexes == [0, 1, 2]



def test_czi_read_and_rgb_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(czi_mod, "CziFile", _FakeCziFile)

    # Non-RGB multi-channel read, selecting specific T/C/Z plane.
    path_gray = tmp_path / "gray.czi"
    path_gray.write_bytes(b"fake-czi")

    gray = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 3, 2, 4, 5)
    gray = gray[..., None]  # T,C,Z,Y,X,0
    _register_fake_dataset(path_gray, axes="TCZYX0", array=gray, metadata="")

    src_gray = LimImageSourceCzi(path_gray, channel_index=2, axis_indices={"T": 1, "Z": 1})
    frame_gray = src_gray.read()
    expected_gray = gray[1, 2, 1, :, :, 0]
    assert np.array_equal(frame_gray, expected_gray)
    assert frame_gray.shape == (4, 5)

    # RGB sample axis path, output must be BGR like PNG/JPEG/TIFF readers.
    path_rgb = tmp_path / "rgb.czi"
    path_rgb.write_bytes(b"fake-czi")

    rgb = np.zeros((2, 4, 5, 3), dtype=np.uint8)  # Z,Y,X,0
    rgb[1, :, :, 0] = 10
    rgb[1, :, :, 1] = 20
    rgb[1, :, :, 2] = 30
    _register_fake_dataset(path_rgb, axes="ZYX0", array=rgb, metadata="")

    src_rgb = LimImageSourceCzi(path_rgb, axis_indices={"Z": 1})
    assert src_rgb.is_rgb is True

    frame_rgb = src_rgb.read()
    assert frame_rgb.shape == (4, 5, 3)
    assert np.array_equal(frame_rgb[:, :, 0], np.full((4, 5), 30, dtype=np.uint8))
    assert np.array_equal(frame_rgb[:, :, 1], np.full((4, 5), 20, dtype=np.uint8))
    assert np.array_equal(frame_rgb[:, :, 2], np.full((4, 5), 10, dtype=np.uint8))

    attrs = src_rgb.nd2_attributes(sequence_count=7)
    assert attrs.uiComp == 3
    assert attrs.uiWidth == 5
    assert attrs.uiHeight == 4
    assert attrs.uiSequenceCount == 7

    opened = LimImageSource.open(path_rgb)
    assert isinstance(opened, LimImageSourceCzi)
