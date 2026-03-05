from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from limnd2.tools.conversion.LimPlanSequence import plan_sequence


def _write_tiff(path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    arr = np.zeros((8, 8), dtype=np.uint16)
    tifffile.imwrite(path, arr)


def _canonical_well_tokens(values: list[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        text = str(value).strip().upper()
        match = re.fullmatch(r"([A-Z]+)0*([1-9]\d*)", text)
        if not match:
            continue
        out.add(f"{match.group(1)}{int(match.group(2))}")
    return out


def test_plan_sequence_reports_wellplate_settings_for_multipoint_xy(tmp_path: Path) -> None:
    for row in ("B", "G"):
        for col in ("02", "06", "11"):
            _write_tiff(tmp_path / f"06_translocation_v01{row}{col}.tif")

    result = plan_sequence(
        [
            str(tmp_path),
            r"06_translocation_v01(.+?)(.+?)\.tif",
            "--multipoint_x",
            "1",
            "--multipoint_y",
            "2",
            "--extra-dimension",
            "channel",
            "--extension",
            "tif",
            "--flatten_duplicates",
        ]
    )

    assert result["error"] is False
    assert result["has_wellplate_settings"] is True

    settings = result["wellplate_settings"]
    assert settings["rows"] == 7
    assert settings["columns"] == 11
    assert settings["frame_count"] == 6
    assert settings["unique_well_count"] == 6
    assert _canonical_well_tokens(settings["wells_preview"]) == {
        "B2",
        "B6",
        "B11",
        "G2",
        "G6",
        "G11",
    }
