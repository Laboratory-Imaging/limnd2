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


def test_cli_sparse_well_list_with_repeated_wells_across_time_and_z(tmp_path: Path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for time in range(2):
        for position in range(3):
            for z in range(2):
                tifffile.imwrite(folder / f"T{time}_P{position}_Z{z}.tif",
                                 np.full((4, 4), time * 100 + position * 10 + z, dtype=np.uint8))
    output = tmp_path / "sparse.nd2"
    result = _run_cli(folder, r"T(\d+)_P(\d+)_Z(\d+)\.tif", output,
                      "-t", "1", "-m", "2", "-z", "3", "--wellplate-wells", " b2, B2, g11 ",
                      "--wellplate-rows", "8", "--wellplate-columns", "12")
    assert result.returncode == 0, result.stdout + result.stderr
    with limnd2.Nd2Reader(output) as reader:
        plate = reader.wellplate
        assert [item.wellName for item in plate.frames] == ["B2", "B2", "B2", "B2", "G11", "G11"] * 2
        assert [item.wellIndex for item in plate.frames] == [13, 13, 13, 13, 82, 82] * 2
        assert [int(reader.image(i)[0, 0, 0]) for i in range(12)] == [0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121]


@pytest.mark.parametrize("x_count,y_count,wells", [
    # A 10 x 2 suggested layout uses columns for X and rows for Y.
    (10, 2, [f"{row}{column}" for column in range(1, 11) for row in "AB"]),
    # A 2 x 10 suggested layout uses rows for X and columns for Y.
    (2, 10, [f"{row}{column}" for row in "AB" for column in range(1, 11)]),
    # An explicit sparse layout uses the supplied order, not plate order.
    (3, 2, ["D6", "B2", "G11", "A1", "C4", "F9"]),
])
def test_cli_explicit_wells_match_two_multipoint_axes_across_time_and_z(
    tmp_path: Path, x_count: int, y_count: int, wells: list[str],
):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for time in range(2):
        for x in range(x_count):
            for y in range(y_count):
                for z in range(2):
                    # Encode each coordinate independently so placement can be
                    # checked against the actual image in the resulting ND2.
                    value = time * 1000 + x * 100 + y * 10 + z
                    tifffile.imwrite(folder / f"T{time}_X{x:02}_Y{y:02}_Z{z}.tif",
                                     np.full((4, 4), value, dtype=np.uint16))

    output = tmp_path / "two_axes.nd2"
    result = _run_cli(
        folder, r"T(\d+)_X(\d+)_Y(\d+)_Z(\d+)\.tif", output,
        "-t", "1", "-mx", "2", "-my", "3", "-z", "4",
        "--wellplate-mode", "auto", "--wellplate-rows", "8", "--wellplate-columns", "12",
        "--wellplate-name", "Express test plate", "--wellplate-wells", ",".join(wells),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with limnd2.Nd2Reader(output) as reader:
        plate = reader.wellplate
        assert (plate.description.name, plate.description.rows, plate.description.columns) == (
            "Express test plate", 8, 12,
        )
        assert len(plate.frames) == 2 * x_count * y_count * 2
        seen = set()
        for item in plate.frames:
            value = int(reader.image(item.seqIndex)[0, 0, 0])
            time, remainder = divmod(value, 1000)
            x, remainder = divmod(remainder, 100)
            y, z = divmod(remainder, 10)
            seen.add((time, x, y, z))
            expected = wells[x * y_count + y]
            row, column = ord(expected[0]) - ord("A"), int(expected[1:]) - 1
            assert (item.wellName, item.wellRowIndex, item.wellColIndex, item.wellIndex) == (
                expected, row, column, row * 12 + column,
            )
        assert seen == {(t, x, y, z) for t in range(2) for x in range(x_count)
                        for y in range(y_count) for z in range(2)}


@pytest.mark.parametrize("options,error", [
    (["--wellplate-row-naming", "number"], "unrecognized arguments"),
    (["--wellplate-column-naming", "letter"], "unrecognized arguments"),
    (["--wellplate-start", "B2"], "must be provided together"),
    (["--wellplate-start", "B2", "--wellplate-end", "C3", "--wellplate-mode", "off"], "cannot be used"),
    (["--wellplate-start", "B2", "--wellplate-end", "B3"], "2 wells"),
    (["--wellplate-start", "B2", "--wellplate-end", "C3", "--wellplate-rows", "2"], "beyond"),
    (["--wellplate-wells", "B2,B3,C2", "--wellplate-start", "B2", "--wellplate-end", "C3"], "cannot be combined"),
    (["--wellplate-wells", "B2,B3,C2", "--wellplate-mode", "off"], "cannot be used"),
    (["--wellplate-wells", "B2,,C2"], "Invalid well"),
    (["--wellplate-wells", "B2,B0,C2"], "Invalid well"),
    (["--wellplate-wells", "B2,B3"], "counts must match"),
    (["--wellplate-wells", "B2,B3,C2,D3"], "counts must match"),
    (["--wellplate-wells", "B2,B3,C2", "--wellplate-rows", "2"], "beyond"),
    (["--wellplate-wells", "B2,B3,C2", "--wellplate-columns", "2"], "beyond"),
])
def test_cli_rejects_invalid_wellplate_options(named_wells: Path, tmp_path: Path, options, error):
    output = tmp_path / "invalid.nd2"
    result = _run_cli(named_wells, r"image_([A-Z]\d+)\.tif", output, "-m", "1", *options)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
    assert not output.exists()
