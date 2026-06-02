from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

import limnd2
from limnd2.export import (
    map_dim_name,
    get_dim_sizes,
    delete_file,
    frameToRGB,
    frame_to_rgb,
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


def test_frame_to_rgb_grayscale_and_alias():
    class _DummyMetadata:
        componentColors = [(1.0, 1.0, 1.0)]

    class _DummyImageAttributes:
        componentCount = 1

    class _DummyReader:
        isRgb = False
        compRange = np.array([[0.0, 255.0]], dtype=np.float32)
        pictureMetadata = _DummyMetadata()
        imageAttributes = _DummyImageAttributes()

        def image(self, frame_index: int) -> np.ndarray:
            return np.array([[0, 255]], dtype=np.uint8)

    reader = _DummyReader()
    rgb = frameToRGB(reader, 0)
    assert rgb.shape == (1, 2, 3)
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[0, 1].tolist() == [255, 255, 255]
    assert np.array_equal(rgb, frame_to_rgb(reader, 0))


def test_frame_to_rgb_reorders_native_rgb_bgr():
    class _DummyMetadata:
        componentColors = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]

    class _DummyImageAttributes:
        componentCount = 3

    class _DummyReader:
        isRgb = True
        compRange = np.array([[0.0, 255.0], [0.0, 255.0], [0.0, 255.0]], dtype=np.float32)
        pictureMetadata = _DummyMetadata()
        imageAttributes = _DummyImageAttributes()

        def image(self, frame_index: int) -> np.ndarray:
            return np.array([[[10, 20, 30], [0, 64, 255]]], dtype=np.uint8)

    rgb = frameToRGB(_DummyReader(), 0)
    assert rgb.tolist() == [[[30, 20, 10], [255, 64, 0]]]


def test_frame_to_rgb_composites_multichannel():
    class _DummyMetadata:
        componentColors = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]

    class _DummyImageAttributes:
        componentCount = 2

    class _DummyReader:
        isRgb = False
        compRange = np.array([[0.0, 255.0], [0.0, 255.0]], dtype=np.float32)
        pictureMetadata = _DummyMetadata()
        imageAttributes = _DummyImageAttributes()

        def image(self, frame_index: int) -> np.ndarray:
            return np.array([[[255, 0], [0, 255], [255, 255]]], dtype=np.uint8)

    rgb = frameToRGB(_DummyReader(), 0)
    assert rgb.shape == (1, 3, 3)
    assert rgb.dtype == np.uint8
    assert np.max(rgb[..., 2]) == 0
    assert rgb[0, 0].tolist() == [0, 255, 0]
    assert rgb[0, 1].tolist() == [255, 0, 0]
    assert rgb[0, 2].tolist() == [255, 255, 0]


def test_frame_export_and_series_export(sample_nd2_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with limnd2.Nd2Reader(sample_nd2_path) as r:
        # frame export
        out_single = tmp_path / "single.tiff"
        limnd2.frameExport(r, frame_index=0, output_path=out_single, target_bit_depth=8)
        captured = capsys.readouterr().out.strip()
        assert captured == ""
        assert out_single.exists()
        out_single.unlink()

        out_png = tmp_path / "single.png"
        limnd2.frameExport(r, frame_index=0, output_path=out_png)
        assert capsys.readouterr().out.strip() == ""
        from PIL import Image
        with Image.open(out_png) as img:
            assert img.mode == "RGB"
        out_png.unlink()

        out_jpeg = tmp_path / "single.jpeg"
        limnd2.frameExport(r, frame_index=0, output_path=out_jpeg)
        assert capsys.readouterr().out.strip() == ""
        with Image.open(out_jpeg) as img:
            assert img.mode == "RGB"
        out_jpeg.unlink()

        with pytest.raises(ValueError, match="Unsupported output file extension"):
            limnd2.frameExport(r, frame_index=0, output_path=tmp_path / "single.bmp")

        # series export
        out_dir = tmp_path / "series_out"
        limnd2.seriesExport(
            r,
            folder=out_dir,
            prefix="exp",
            dimension_order=None,  # use file's own order
            bits=8,
            extension=".png",
        )
        captured = capsys.readouterr().out
        assert captured == ""
        dims = get_dim_sizes(r)
        expected = 1
        for v in dims.values():
            expected *= max(1, int(v))
        assert len(list(out_dir.iterdir())) == expected
        for export_path in sorted(out_dir.iterdir()):
            assert export_path.exists()
            assert export_path.suffix.lower() == ".png"

        with pytest.raises(ValueError, match="Unsupported output file extension"):
            limnd2.seriesExport(r, folder=out_dir, prefix="bad", extension=".bmp")

    # Cleanup export dir
    if out_dir.exists():
        for p in out_dir.iterdir():
            p.unlink()
        out_dir.rmdir()


def test_frame_and_series_export_overwrite_requires_opt_in(sample_nd2_path: Path, tmp_path: Path) -> None:
    with limnd2.Nd2Reader(sample_nd2_path) as r:
        frame_path = tmp_path / "overwrite_frame.tiff"
        limnd2.frameExport(r, frame_index=0, output_path=frame_path)
        with pytest.raises(FileExistsError, match="Output file already exists"):
            limnd2.frameExport(r, frame_index=0, output_path=frame_path)
        limnd2.frameExport(r, frame_index=0, output_path=frame_path, overwrite=True)

        series_dir = tmp_path / "overwrite_series"
        limnd2.seriesExport(r, folder=series_dir, prefix="exp", bits=8, extension=".png")
        with pytest.raises(FileExistsError, match="Output file already exists"):
            limnd2.seriesExport(r, folder=series_dir, prefix="exp", bits=8, extension=".png")
        limnd2.seriesExport(r, folder=series_dir, prefix="exp", bits=8, extension=".png", overwrite=True)


def test_export_callbacks_report_final_completion(
    sample_nd2_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with limnd2.Nd2Reader(sample_nd2_path) as r:
        frame_events: list[tuple[int, int, str | Path | None, str]] = []
        frame_path = tmp_path / "callback_frame.png"
        limnd2.frameExport(
            r,
            frame_index=0,
            output_path=frame_path,
            progress_callback=lambda current, total, file, message: frame_events.append(
                (current, total, file, message)
            ),
        )
        assert capsys.readouterr().out == ""
        assert frame_events[0][0:3] == (1, 1, frame_path)
        assert frame_events[-1][0] == frame_events[-1][1] == 1
        assert frame_events[-1][2] == frame_path

        series_events: list[tuple[int, int, str | Path | None, str]] = []
        out_dir = tmp_path / "callback_series"
        limnd2.seriesExport(
            r,
            folder=out_dir,
            prefix="cb",
            extension=".png",
            bits=8,
            progress_callback=lambda current, total, file, message: series_events.append(
                (current, total, file, message)
            ),
        )
        assert capsys.readouterr().out == ""
        dims = get_dim_sizes(r)
        file_count = 1
        for value in dims.values():
            file_count *= max(1, int(value))
        assert len([event for event in series_events if event[2] is not None]) == file_count
        assert series_events[-1][0] == series_events[-1][1] == file_count
        assert series_events[-1][2] is None

        metadata_path = tmp_path / "metadata.json"
        limnd2.metadataAsJSON(
            r,
            output_path=metadata_path,
        )
        assert capsys.readouterr().out == ""
