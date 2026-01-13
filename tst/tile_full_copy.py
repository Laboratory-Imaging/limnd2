from __future__ import annotations

from pathlib import Path

import numpy as np
import limnd2
from limnd2.attributes import ImageAttributesCompression
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
DST = Path("tst") / "test_tile_copy.nd2"
TILE_SIZE = 128


def main() -> int:
    if not SRC.exists():
        print(f"Missing source ND2: {SRC}")
        return 1

    if DST.exists():
        DST.unlink()

    with limnd2.Nd2Reader(SRC) as reader:
        attrs = reader.imageAttributes
        picture_md = reader.pictureMetadata
        experiment_data = reader.experiment

        print(f"Version: {reader.version}")
        print(f"Compression: {attrs.eCompression}")
        print(f"Frames: {attrs.frameCount}")
        print(f"Shape: {attrs.shape} dtype={attrs.dtype}")

        if attrs.eCompression != ImageAttributesCompression.ictNone:
            print("Tile copy requires ictNone compression; aborting.")
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

            for seqindex in range(attrs.frameCount):
                for y in range(0, attrs.height, TILE_SIZE):
                    tile_h = min(TILE_SIZE, attrs.height - y)
                    for x in range(0, attrs.width, TILE_SIZE):
                        tile_w = min(TILE_SIZE, attrs.width - x)
                        tile = reader.image(seqindex, rect=(x, y, tile_w, tile_h))
                        tile_arr = np.array(tile, copy=True)
                        if attrs.componentCount == 1 and tile_arr.ndim == 3:
                            tile_arr = tile_arr[:, :, 0]
                        writer._chunker.setImageTile(seqindex, x, y, tile_arr)

    print(f"Tiled copy complete: {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
