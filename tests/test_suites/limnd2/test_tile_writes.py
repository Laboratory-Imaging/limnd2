from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pytest

import limnd2
import limnd2.attributes
import limnd2.file_legacy  # ensure legacy chunker is importable for isinstance checks
from limnd2.attributes import ImageAttributesCompression, ImageAttributesPixelType
from limnd2.base import (
    BaseChunker,
    ND2_CHUNK_NAME_ImageAttributes,
    ND2_CHUNK_NAME_ImageAttributesLV,
    ND2_CHUNK_NAME_ImageMetadata,
    ND2_CHUNK_NAME_ImageMetadataLV,
    ND2_CHUNK_FORMAT_ImageMetadataLV_1p,
    ND2_CHUNK_FORMAT_ImageMetadata_1p,
    ND2_CHUNKMAP_SIGNATURE,
    ND2_FILEMAP_SIGNATURE,
)

SKIP_CHUNKS = {
    ND2_CHUNK_NAME_ImageAttributesLV,
    ND2_CHUNK_NAME_ImageAttributes,
    ND2_CHUNK_NAME_ImageMetadataLV,
    ND2_CHUNK_NAME_ImageMetadata,
    ND2_CHUNK_FORMAT_ImageMetadataLV_1p % 0,
    ND2_CHUNK_FORMAT_ImageMetadata_1p % 0,
    ND2_CHUNKMAP_SIGNATURE,
    ND2_FILEMAP_SIGNATURE,
}
TILE_SIZE = 128


def _copy_metadata(reader: limnd2.Nd2Reader, writer: limnd2.Nd2Writer) -> None:
    legacy_chunker = getattr(limnd2.file_legacy, "LimJpeg2000Chunker", None)
    if legacy_chunker is not None and isinstance(reader.chunker, legacy_chunker):
        pytest.skip("Tile writer tests skip legacy JPEG2000 ND2 files.")
    writer.imageAttributes = reader.imageAttributes
    picture_md = reader.pictureMetadata
    if picture_md is not None:
        writer.pictureMetadata = picture_md
    experiment_data = reader.experiment
    if experiment_data is not None:
        writer.experiment = experiment_data


def _copy_fixed_chunks(reader: limnd2.Nd2Reader, writer: limnd2.Nd2Writer) -> None:
    _copy_metadata(reader, writer)
    for name in reader.chunker.chunk_names:
        if BaseChunker.isSkipChunk(name):
            continue
        if not BaseChunker._is_chunk_data(name):
            continue
        if BaseChunker.isImageChunk(name) is not None:
            continue
        if name in SKIP_CHUNKS:
            continue
        data = reader.chunk(name)
        if data is not None:
            writer.setChunk(name, data)


def _center_tile(attrs: limnd2.attributes.ImageAttributes) -> tuple[int, int, int, int]:
    tile_w = min(64, attrs.width)
    tile_h = min(64, attrs.height)
    x = max(0, (attrs.width - tile_w) // 2)
    y = max(0, (attrs.height - tile_h) // 2)
    return x, y, tile_w, tile_h


def _require_uncompressed(attrs: limnd2.attributes.ImageAttributes) -> None:
    if attrs.eCompression != ImageAttributesCompression.ictNone:
        pytest.skip("Tile writer tests require ImageAttributesCompression.ictNone")


def test_tile_write_rejects_out_of_range(nd2_path: Path, tmp_path: Path, prepare_conversion_output_dir: Path) -> None:
    copied = tmp_path / "tile_validation.nd2"
    shutil.copy2(nd2_path, copied)

    with limnd2.Nd2Reader(copied) as reader:
        attrs = reader.imageAttributes
        _require_uncompressed(attrs)
        x, y, tile_w, tile_h = _center_tile(attrs)

    with limnd2.Nd2Writer(copied) as writer:
        attrs = writer.imageAttributes
        bits = attrs.uiBpcSignificant
        bad_val = None
        if bits > 0:
            if attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
                bad_val = 1 << bits
            elif attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
                bad_val = 1 << (bits - 1)

        if bad_val is not None and attrs.uiBpcSignificant < attrs.uiBpcInMemory:
            bad_shape = (tile_h, tile_w, attrs.componentCount)
            bad_tile = np.full(bad_shape, bad_val, dtype=attrs.dtype)
            if attrs.componentCount == 1:
                bad_tile = bad_tile.reshape(tile_h, tile_w)
            with pytest.raises(ValueError):
                writer._chunker.setImageTile(0, x, y, bad_tile)

        if attrs.ePixelType == ImageAttributesPixelType.pxtReal:
            max_val = 1000.0
        elif bits > 0:
            if attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
                max_val = (1 << bits) - 1
            elif attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
                max_val = (1 << (bits - 1)) - 1
            else:
                max_val = 0
        else:
            max_val = 0

        if attrs.componentCount == 1:
            tile = np.full((tile_h, tile_w), max_val, dtype=attrs.dtype)
        else:
            tile = np.full((tile_h, tile_w, attrs.componentCount), max_val, dtype=attrs.dtype)

        for seqindex in range(attrs.frameCount):
            writer._chunker.setImageTile(seqindex, x, y, tile)

    with limnd2.Nd2Reader(copied) as reader:
        rect = (x, y, tile_w, tile_h)
        for seqindex in range(reader.imageAttributes.frameCount):
            after = reader.image(seqindex, rect=rect)
            assert np.all(after == max_val), f"Frame {seqindex} did not contain expected max range tile"
    shutil.copy2(copied, prepare_conversion_output_dir / f"{nd2_path.stem}_tile_validation.nd2")


def test_tile_random_mix_preserves_metadata(nd2_path: Path, tmp_path: Path, prepare_conversion_output_dir: Path) -> None:
    out_path = tmp_path / "tile_random_mix.nd2"
    rng = random.Random(1234)
    metadata_preserved = False
    any_modified = False

    with limnd2.Nd2Reader(nd2_path) as reader, limnd2.Nd2Writer(out_path) as writer:
        _copy_fixed_chunks(reader, writer)
        metadata_preserved = reader.experiment is None or writer.experiment is not None

        attrs = reader.imageAttributes
        _require_uncompressed(attrs)
        min_val, max_val = 0.0, 1.0
        bits = attrs.uiBpcSignificant
        if bits > 0:
            if attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
                max_val = float((1 << bits) - 1)
            elif attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
                min_val = float(-(1 << (bits - 1)))
                max_val = float((1 << (bits - 1)) - 1)

        for seqindex in range(attrs.frameCount):
            frame = np.array(reader.image(seqindex), copy=True)
            for y in range(0, attrs.height, TILE_SIZE):
                tile_h = min(TILE_SIZE, attrs.height - y)
                for x in range(0, attrs.width, TILE_SIZE):
                    tile_w = min(TILE_SIZE, attrs.width - x)
                    tile_view = frame[y : y + tile_h, x : x + tile_w]
                    if attrs.componentCount == 1:
                        tile_view = tile_view[:, :, 0]

                    roll = rng.random()
                    if roll < 0.3:
                        new_tile = np.array(tile_view, copy=True)
                        change = False
                    elif roll < 0.4:
                        new_tile = np.zeros_like(tile_view)
                        change = not np.array_equal(new_tile, tile_view)
                    elif roll < 0.5:
                        new_tile = np.full(tile_view.shape, max_val, dtype=attrs.dtype)
                        change = not np.array_equal(new_tile, tile_view)
                    else:
                        factor = rng.random()
                        if np.issubdtype(attrs.dtype, np.integer):
                            scaled = tile_view.astype(np.float32) * factor
                            scaled = np.clip(scaled, min_val, max_val)
                            new_tile = np.rint(scaled).astype(attrs.dtype)
                        else:
                            new_tile = (tile_view.astype(np.float32) * factor).astype(attrs.dtype)
                        change = not np.array_equal(new_tile, tile_view)

                    any_modified = any_modified or change
                    writer._chunker.setImageTile(seqindex, x, y, new_tile)

    assert metadata_preserved, "Experiment metadata missing after write"
    assert any_modified, "Random mix unexpectedly left all tiles unchanged"

    with limnd2.Nd2Reader(nd2_path) as source, limnd2.Nd2Reader(out_path) as mixed:
        assert mixed.pictureMetadata is not None or source.pictureMetadata is None
        src_frame = source.image(0)
        mixed_frame = mixed.image(0)
        assert src_frame.shape == mixed_frame.shape
        assert not np.array_equal(src_frame, mixed_frame)
    shutil.copy2(out_path, prepare_conversion_output_dir / f"{nd2_path.stem}_tile_random_mix.nd2")


def test_tile_full_copy_matches_source(nd2_path: Path, tmp_path: Path, prepare_conversion_output_dir: Path) -> None:
    out_path = tmp_path / "tile_full_copy.nd2"

    with limnd2.Nd2Reader(nd2_path) as reader, limnd2.Nd2Writer(out_path) as writer:
        _copy_fixed_chunks(reader, writer)
        attrs = reader.imageAttributes
        _require_uncompressed(attrs)

        for seqindex in range(attrs.frameCount):
            full_frame = np.array(reader.image(seqindex), copy=True)
            for y in range(0, attrs.height, TILE_SIZE):
                tile_h = min(TILE_SIZE, attrs.height - y)
                for x in range(0, attrs.width, TILE_SIZE):
                    tile_w = min(TILE_SIZE, attrs.width - x)
                    tile_arr = full_frame[y : y + tile_h, x : x + tile_w].copy()
                    if attrs.componentCount == 1 and tile_arr.ndim == 3:
                        tile_arr = tile_arr[:, :, 0]
                    writer._chunker.setImageTile(seqindex, x, y, tile_arr)

    with limnd2.Nd2Reader(nd2_path) as source, limnd2.Nd2Reader(out_path) as copy:
        attrs = source.imageAttributes
        for seqindex in (0, attrs.frameCount - 1):
            src_frame = source.image(seqindex)
            copy_frame = copy.image(seqindex)
            assert np.array_equal(src_frame, copy_frame), f"Frame {seqindex} differs after tile copy"
    shutil.copy2(out_path, prepare_conversion_output_dir / f"{nd2_path.stem}_tile_full_copy.nd2")
