from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import limnd2
from limnd2.attributes import ImageAttributesCompression, ImageAttributesPixelType
from limnd2.base import (
    BaseChunker,
    ND2_CHUNK_NAME_ImageAttributesLV,
    ND2_CHUNK_NAME_ImageAttributes,
    ND2_CHUNK_NAME_ImageMetadataLV,
    ND2_CHUNK_NAME_ImageMetadata,
    ND2_CHUNK_FORMAT_ImageMetadataLV_1p,
    ND2_CHUNK_FORMAT_ImageMetadata_1p,
)


SRC = Path("tst") / "test.nd2"
DST = Path("tst") / "test_tile_mix.nd2"
TILE_SIZE = 100
SEED = None  # set to an int for reproducible mixing


def pixel_limits(attrs) -> tuple[float, float]:
    bits = attrs.uiBpcSignificant
    if bits <= 0:
        return 0.0, 1.0
    if attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
        return 0.0, float((1 << bits) - 1)
    if attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
        return float(-(1 << (bits - 1))), float((1 << (bits - 1)) - 1)
    return 0.0, 1.0


def clamp_and_cast(arr: np.ndarray, min_val: float, max_val: float, dtype) -> np.ndarray:
    out = np.rint(arr).astype(np.float32, copy=False)
    np.clip(out, min_val, max_val, out=out)
    return out.astype(dtype, copy=False)


def main() -> int:
    if not SRC.exists():
        print(f"Missing source ND2: {SRC}")
        return 1

    if DST.exists():
        DST.unlink()

    rng = random.Random(SEED)

    with limnd2.Nd2Reader(SRC) as reader:
        attrs = reader.imageAttributes
        picture_md = reader.pictureMetadata
        experiment_data = reader.experiment
        print(f"Version: {reader.version}")
        print(f"Compression: {attrs.eCompression}")
        print(f"Frames: {attrs.frameCount}")
        print(f"Shape: {attrs.shape} dtype={attrs.dtype}")

        if attrs.eCompression != ImageAttributesCompression.ictNone:
            print("Tile write requires ictNone compression; aborting.")
            return 1

        with limnd2.Nd2Writer(DST) as writer:
            writer.imageAttributes = attrs
            if picture_md is not None:
                writer.pictureMetadata = picture_md
            if experiment_data is not None:
                writer.experiment = experiment_data

            skip_chunks = {
                ND2_CHUNK_NAME_ImageAttributesLV,
                ND2_CHUNK_NAME_ImageAttributes,
                ND2_CHUNK_NAME_ImageMetadataLV,
                ND2_CHUNK_NAME_ImageMetadata,
                ND2_CHUNK_FORMAT_ImageMetadataLV_1p % 0,
                ND2_CHUNK_FORMAT_ImageMetadata_1p % 0,
            }
            for name in reader.chunker.chunk_names:
                if BaseChunker.isSkipChunk(name):
                    continue
                if not BaseChunker._is_chunk_data(name):
                    continue
                if name in skip_chunks:
                    continue
                data = reader.chunk(name)
                if data is not None:
                    writer.setChunk(name, data)

        min_val, max_val = pixel_limits(attrs)
        max_val = float(max_val)

        with limnd2.Nd2Writer(DST, append=True) as writer:
            writer.imageAttributes = attrs
            if picture_md is not None:
                writer.pictureMetadata = picture_md
            if experiment_data is not None:
                writer.experiment = experiment_data

            for seqindex in range(attrs.frameCount):
                source_frame = np.array(reader.image(seqindex), copy=True)

                for y in range(0, attrs.height, TILE_SIZE):
                    tile_h = min(TILE_SIZE, attrs.height - y)
                    for x in range(0, attrs.width, TILE_SIZE):
                        tile_w = min(TILE_SIZE, attrs.width - x)
                        tile_view = source_frame[y : y + tile_h, x : x + tile_w]
                        if attrs.componentCount == 1:
                            tile_view = tile_view[:, :, 0]

                        roll = rng.random()
                        if roll < 0.3:
                            print(f"frame {seqindex} tile ({x},{y}) -> copying tile")
                            out_tile = np.array(tile_view, copy=True)
                        elif roll < 0.4:
                            print(f"frame {seqindex} tile ({x},{y}) -> skipping tile (black)")
                            if attrs.componentCount == 1:
                                out_tile = np.zeros((tile_h, tile_w), dtype=attrs.dtype)
                            else:
                                out_tile = np.zeros((tile_h, tile_w, attrs.componentCount), dtype=attrs.dtype)
                        elif roll < 0.5:
                            print(f"frame {seqindex} tile ({x},{y}) -> max brightness")
                            if attrs.componentCount == 1:
                                out_tile = np.full((tile_h, tile_w), max_val, dtype=attrs.dtype)
                            else:
                                out_tile = np.full((tile_h, tile_w, attrs.componentCount), max_val, dtype=attrs.dtype)
                        else:
                            factor = rng.random()
                            print(f"frame {seqindex} tile ({x},{y}) -> multiplying by {factor:.3f}")
                            if np.issubdtype(attrs.dtype, np.integer):
                                out_tile = clamp_and_cast(
                                    tile_view.astype(np.float32) * factor, min_val, max_val, attrs.dtype
                                )
                            else:
                                out_tile = (tile_view.astype(np.float32) * factor).astype(attrs.dtype)

                        writer._chunker.setImageTile(seqindex, x, y, out_tile)

    with limnd2.Nd2Reader(DST) as verify:
        print(f"Verification: compression={verify.imageAttributes.eCompression}")
        print(f"Verification: frames={verify.imageAttributes.frameCount}")
        experiment_present = verify.experiment is not None
        picture_metadata_present = verify.pictureMetadata is not None
        print(f"Experiment present after reopen: {experiment_present}")
        print(f"Picture metadata present after reopen: {picture_metadata_present}")

    print(f"Edited file: {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
