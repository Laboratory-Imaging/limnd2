"""End-to-end tests of wellplate options through the CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

import limnd2


def _run_cli(folder: Path, pattern: str, output: Path, *options: str):
    return subprocess.run(
        [sys.executable, "-c",
         "from limnd2.tools.conversion.LimConvertSequence import convert_sequence_to_nd2_cli; "
         "convert_sequence_to_nd2_cli()",
         str(folder), pattern, "--extension", "tif", *options,
         "-n", output.name, "-o", str(output.parent)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )


@pytest.fixture
def named_wells(tmp_path: Path) -> Path:
    folder = tmp_path / "inputs"
    folder.mkdir()
    for index, well in enumerate(("B2", "B3", "C2")):
        tifffile.imwrite(folder / f"image_{well}.tif", np.full((4, 4), index, dtype=np.uint8))
    return folder


def test_cli_auto_inference_and_descriptor_overrides(named_wells: Path, tmp_path: Path):
    output = tmp_path / "auto.nd2"
    result = _run_cli(named_wells, r"image_([A-Z]\d+)\.tif", output,
                      "-m", "1", "--wellplate-rows", "8", "--wellplate-columns", "12",
                      "--wellplate-name", "Test plate")
    assert result.returncode == 0, result.stdout + result.stderr
    with limnd2.Nd2Reader(output) as reader:
        plate = reader.wellplate
        assert (plate.description.name, plate.description.rows, plate.description.columns) == ("Test plate", 8, 12)
        assert (plate.description.rowNaming, plate.description.columnNaming) == ("letter", "number")
        assert [item.wellIndex for item in plate.frames] == [13, 14, 25]


def test_cli_wellplate_mode_off(named_wells: Path, tmp_path: Path):
    output = tmp_path / "off.nd2"
    result = _run_cli(named_wells, r"image_([A-Z]\d+)\.tif", output,
                      "-m", "1", "--wellplate-mode", "off")
    assert result.returncode == 0, result.stdout + result.stderr
    with limnd2.Nd2Reader(output) as reader:
        assert reader.wellplate is None


def test_cli_partial_range_across_time_and_z(tmp_path: Path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for time in range(2):
        for position in range(3):
            for z in range(2):
                tifffile.imwrite(folder / f"T{time}_P{position}_Z{z}.tif",
                                 np.full((4, 4), time * 100 + position * 10 + z, dtype=np.uint8))
    output = tmp_path / "range.nd2"
    result = _run_cli(folder, r"T(\d+)_P(\d+)_Z(\d+)\.tif", output,
                      "-t", "1", "-m", "2", "-z", "3", "--wellplate-start", "B2",
                      "--wellplate-end", "C3", "--wellplate-rows", "8", "--wellplate-columns", "12")
    assert result.returncode == 0, result.stdout + result.stderr
    with limnd2.Nd2Reader(output) as reader:
        plate = reader.wellplate
        assert [item.wellName for item in plate.frames] == ["B2", "B2", "B3", "B3", "C2", "C2"] * 2
        assert [item.wellIndex for item in plate.frames] == [13, 13, 14, 14, 25, 25] * 2
        assert [int(reader.image(i)[0, 0, 0]) for i in range(12)] == [0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121]


@pytest.mark.parametrize("options,error", [
    (["--wellplate-row-naming", "number"], "unrecognized arguments"),
    (["--wellplate-column-naming", "letter"], "unrecognized arguments"),
    (["--wellplate-start", "B2"], "must be provided together"),
    (["--wellplate-start", "B2", "--wellplate-end", "C3", "--wellplate-mode", "off"], "cannot be used"),
    (["--wellplate-start", "B2", "--wellplate-end", "B3"], "2 wells"),
    (["--wellplate-start", "B2", "--wellplate-end", "C3", "--wellplate-rows", "2"], "beyond"),
])
def test_cli_rejects_invalid_wellplate_options(named_wells: Path, tmp_path: Path, options, error):
    output = tmp_path / "invalid.nd2"
    result = _run_cli(named_wells, r"image_([A-Z]\d+)\.tif", output, "-m", "1", *options)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
    assert not output.exists()
