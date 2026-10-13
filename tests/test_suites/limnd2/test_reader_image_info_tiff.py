from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from limnd2.image_info import gatherImageInformation, imageInformationAsJSON


def _write_pyramid(path: Path, *, ome: bool = False) -> None:
    tifffile = pytest.importorskip("tifffile")
    base = np.zeros((16, 16), dtype=np.uint16)
    level_1 = np.zeros((8, 8), dtype=np.uint16)
    with tifffile.TiffWriter(path, ome=ome) as writer:
        writer.write(base, metadata={"axes": "YX"} if ome else None, subifds=1)
        writer.write(level_1, subfiletype=1)


@pytest.mark.parametrize(
    ("ome", "expected_created_by"),
    [(False, "TIFF, multi-res"), (True, "OME-TIFF, multi-res")],
)
def test_tiff_image_information_reports_pyramid_levels(
    tmp_path: Path, ome: bool, expected_created_by: str
) -> None:
    path = tmp_path / ("pyramid.ome.tif" if ome else "pyramid.tif")
    _write_pyramid(path, ome=ome)

    info = gatherImageInformation(path, format_hint="tiff")
    assert info["generalInfo"]["app_created"] == expected_created_by

    as_json = json.loads(imageInformationAsJSON(path, format_hint="tiff"))
    assert as_json["generalInfo"]["app_created"] == expected_created_by


def test_tiff_image_information_omits_multires_for_single_level(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "single-level.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint16))

    info = gatherImageInformation(path, format_hint="tiff")
    assert info["generalInfo"]["app_created"] == "TIFF"


def test_tiff_image_information_reports_partial_pyramid(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "partial-pyramid.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((16, 16), dtype=np.uint16), subifds=1)
        writer.write(np.zeros((8, 8), dtype=np.uint16), subfiletype=1)
        writer.write(np.zeros((16, 16), dtype=np.uint16))

    info = gatherImageInformation(path, format_hint="tiff")
    assert info["generalInfo"]["app_created"] == "TIFF, multi-res (partial)"
