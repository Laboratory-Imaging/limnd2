from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("oiffile")

import limnd2.tools.conversion.LimImageSourceOifOib as oif_mod
from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
from limnd2.tools.conversion.LimImageSource import LimImageSource
from limnd2.tools.conversion.LimImageSourceOifOib import LimImageSourceOifOib


class _FakeSeries:
    def __init__(self, axes: str):
        self.axes = axes


class _FakeOifFile:
    _datasets: dict[str, dict[str, object]] = {}

    def __init__(self, image: str, *args, **kwargs):
        del args, kwargs
        key = str(Path(image))
        if key not in self._datasets:
            raise FileNotFoundError(f"No fake OIF/OIB dataset for {key}")
        data = self._datasets[key]

        self._array = np.asarray(data["array"])
        self.axes = str(data["order"])
        self.shape = tuple(int(v) for v in self._array.shape)
        self.dtype = self._array.dtype
        self.mainfile = data.get("mainfile", {})
        self.series = (_FakeSeries(self.axes),)

    def asarray(self, series: int = 0, **kwargs):
        del series, kwargs
        return np.asarray(self._array)

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.close()


def _register(path: Path, *, order: str, array: np.ndarray, mainfile: dict | None = None):
    _FakeOifFile._datasets[str(path)] = {
        "order": order,
        "array": np.asarray(array),
        "mainfile": {} if mainfile is None else mainfile,
    }


def test_open_oib_and_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(oif_mod, "OifFile", _FakeOifFile)

    path = tmp_path / "sample.oib"
    path.write_bytes(b"fake-oib")

    arr = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)  # T,C,Z,Y,X
    _register(path, order="TCZYX", array=arr)

    src = LimImageSource.open(path)
    assert isinstance(src, LimImageSourceOifOib)

    frame = LimImageSourceOifOib(path, channel_index=2, axis_indices={"T": 1, "Z": 3}).read()
    expected = arr[1, 2, 3, :, :]
    assert np.array_equal(frame, expected)


def test_open_oif_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(oif_mod, "OifFile", _FakeOifFile)

    path = tmp_path / "sample.oif"
    path.write_bytes(b"fake-oif")
    arr = np.zeros((3, 4, 6), dtype=np.uint16)  # C,Y,X
    _register(path, order="CYX", array=arr)

    src = LimImageSource.open(path)
    assert isinstance(src, LimImageSourceOifOib)
    assert src.get_file_dimensions() == {"channel": 3}


def test_oib_dimensions_and_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(oif_mod, "OifFile", _FakeOifFile)

    path = tmp_path / "sample.oib"
    path.write_bytes(b"fake-oib")

    arr = np.zeros((2, 3, 4, 8, 6), dtype=np.uint16)  # T,C,Z,Y,X
    mainfile = {
        "Axis 0 Parameters Common": {"AxisCode": "X", "MaxSize": 6, "Interval": 0.3375, "UnitName": "um"},
        "Axis 1 Parameters Common": {"AxisCode": "Y", "MaxSize": 8, "Interval": 0.3375, "UnitName": "um"},
        "Axis 2 Parameters Common": {"AxisCode": "Z", "MaxSize": 4, "Interval": 2.0, "UnitName": "um"},
        "Axis 3 Parameters Common": {"AxisCode": "T", "MaxSize": 2, "Interval": 1.5, "UnitName": "s"},
        "Channel 1 Parameters": {"Name": "DAPI"},
        "Channel 2 Parameters": {"Name": "FITC"},
        "Channel 3 Parameters": {"Name": "TRITC"},
    }
    _register(path, order="TCZYX", array=arr, mainfile=mainfile)

    src = LimImageSourceOifOib(path)
    assert src.get_file_dimensions() == {"timeloop": 2, "channel": 3, "zstack": 4}

    settings = src.metadata_as_pattern_settings()
    assert settings["pixel_calibration"] == "0.3375"
    assert settings["zstep"] == "2.000"
    assert settings["tstep"] == "1500.000"
    assert [row[0] for row in settings["channels"]] == ["DAPI", "FITC", "TRITC"]
    assert settings["channels"][0][5] == "#4080FF"
    assert settings["channels"][1][5] == "#00B050"
    assert settings["channels"][2][5] == "#FF3030"

    storage = ConversionSettings()
    storage.z_step = None
    src.parse_additional_metadata(storage)
    assert storage.metadata.pixel_calibration == pytest.approx(0.3375)
    assert storage.z_step == pytest.approx(2.0)
    assert storage.time_step == pytest.approx(1500.0)
    assert storage.channels[1].name == "FITC"


def test_oib_parse_additional_dimensions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(oif_mod, "OifFile", _FakeOifFile)

    path = tmp_path / "dims.oib"
    path.write_bytes(b"fake-oib")

    arr = np.zeros((2, 3, 2, 5, 4), dtype=np.uint16)  # T,C,Z,Y,X
    _register(path, order="TCZYX", array=arr)

    src = LimImageSourceOifOib(path)
    new_sources, new_dims = src.parse_additional_dimensions({src: []}, {}, "multipoint")

    assert new_dims["timeloop"] == 2
    assert new_dims["channel"] == 3
    assert new_dims["zstack"] == 2
    assert len(new_sources) == 12

    channel_indexes = sorted({source.channel_index for source in new_sources})
    assert channel_indexes == [0, 1, 2]

