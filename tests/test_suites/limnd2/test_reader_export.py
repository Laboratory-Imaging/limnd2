from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import limnd2
from limnd2.export import (
    map_dim_name,
    get_dim_sizes,
    delete_file,
    save_float_tiff,
    save_uint_tiff,
)
from limnd2 import export as export_mod


@pytest.fixture()
def sample_nd2_path(nd2_files: list[Path]) -> Path:
    return nd2_files[0]


def test_map_dim_name_basic():
    assert map_dim_name("time") == "t"
    assert map_dim_name("Timeloop") == "t"
    assert map_dim_name("zstack") == "z"
    assert map_dim_name("channel") == "c"
    assert map_dim_name("unknown-dim") is None


def test_delete_file(tmp_path: Path):
    p = tmp_path / "deleteme.txt"
    p.write_text("x")
    assert p.exists()
    delete_file(p)
    assert not p.exists()


def test_save_uint_and_float_tiff(tmp_path: Path):
    # uint scaling 8->16 RGB
    arr_rgb = np.random.randint(0, 256, (8, 6, 3), dtype=np.uint8)
    t1 = tmp_path / "rgb16.tiff"
    save_uint_tiff(arr_rgb, t1, source_bit_depth=8, target_bit_depth=16, is_rgb=True)
    from tifffile import imread
    img = imread(t1)
    assert img.dtype == np.uint16
    t1.unlink()

    # uint multi-channel non-RGB stays 8-bit
    arr_mc = np.random.randint(0, 256, (5, 4, 2), dtype=np.uint8)
    t2 = tmp_path / "mc8.tiff"
    save_uint_tiff(arr_mc, t2, source_bit_depth=8, target_bit_depth=8, is_rgb=False)
    img = imread(t2)
    assert img.dtype == np.uint8
    t2.unlink()

    # float save
    arr_f = np.random.rand(5, 7).astype(np.float32)
    t3 = tmp_path / "float.tiff"
    save_float_tiff(arr_f, t3)
    img = imread(t3)
    assert np.issubdtype(img.dtype, np.floating)
    t3.unlink()


def test_generate_frame_list_and_dims(sample_nd2_path: Path):
    with limnd2.Nd2Reader(sample_nd2_path) as r:
        dims = get_dim_sizes(r)
        # Build frames using the file's own dimension order
        frames = export_mod.generate_frame_list(r, None)

        # Expected number of frames equals product of dimension sizes (including 'c' if present)
        expected = 1
        for k, v in dims.items():
            expected *= max(1, int(v))
        assert len(frames) == expected

        # All unique coord combos should match expected
        seen = {tuple(sorted(coords.items())) for _, coords in frames}
        assert len(seen) == expected

        canon = list(r.experiment.dimnames()) if r.experiment else []
        if r.imageAttributes.componentCount > 1 and not r.isRgb and 'c' not in canon:
            canon.append('c')
        syn_map = dict(t="time", z="z", m="multipoint", c="channel")

        varying_dims = [d for d, v in dims.items() if v > 1]
        if varying_dims:
            drop = varying_dims[0]
            bad_order = [syn_map.get(d, d) for d in canon if d != drop]
            with pytest.raises(ValueError):
                export_mod.generate_frame_list(r, bad_order)

        singleton_dims = [d for d, v in dims.items() if v == 1]
        if singleton_dims:
            drop = singleton_dims[0]
            reduced_order = [syn_map.get(d, d) for d in canon if d != drop]
            frames_no_singleton = export_mod.generate_frame_list(r, reduced_order)
            assert len(frames_no_singleton) == expected


def test_frame_export_and_series_export(sample_nd2_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with limnd2.Nd2Reader(sample_nd2_path) as r:
        # frame export
        out_single = tmp_path / "single.tiff"
        limnd2.frameExport(r, frame_index=0, output_path=out_single, target_bit_depth=8, progress_to_json=True)
        captured = capsys.readouterr().out.strip()
        # Should be a JSON line
        data = json.loads(captured)
        assert data.get("progress") == 1 and Path(data.get("file")).exists()
        assert out_single.exists()
        out_single.unlink()

        # series export
        out_dir = tmp_path / "series_out"
        limnd2.seriesExport(
            r,
            folder=out_dir,
            prefix="exp",
            dimension_order=None,  # use file's own order
            bits=8,
            progress_to_json=True,
        )
        captured = capsys.readouterr().out
        # Ensure multiple JSON progress lines and files created
        lines = [json.loads(line) for line in captured.strip().splitlines() if line.strip()]
        dims = get_dim_sizes(r)
        expected = 1
        for v in dims.values():
            expected *= max(1, int(v))
        assert len(lines) == expected
        for item in lines:
            assert Path(item["file"]).exists()

    # Cleanup export dir
    if out_dir.exists():
        for p in out_dir.glob("*.tiff"):
            p.unlink()
        out_dir.rmdir()
