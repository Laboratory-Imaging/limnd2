from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import limnd2
from limnd2.attributes import ImageAttributesCompression, ImageAttributesPixelType


SRC = Path("tst") / "test.nd2"
DST = Path("tst") / "test_tile_edit.nd2"


def main() -> int:
    if not SRC.exists():
        print(f"Missing source ND2: {SRC}")
        return 1

    if DST.exists():
        DST.unlink()
    shutil.copy2(SRC, DST)

    with limnd2.Nd2Reader(DST) as reader:
        attrs = reader.imageAttributes
        print(f"Version: {reader.version}")
        print(f"Compression: {attrs.eCompression}")
        print(f"Frames: {attrs.frameCount}")
        print(f"Shape: {attrs.shape} dtype={attrs.dtype}")

        if attrs.eCompression != ImageAttributesCompression.ictNone:
            print("Tile write requires ictNone compression; aborting.")
            return 1

        frame_count = attrs.frameCount
        tile_w = min(64, attrs.width)
        tile_h = min(64, attrs.height)
        x = max(0, (attrs.width - tile_w) // 2)
        y = max(0, (attrs.height - tile_h) // 2)

    with limnd2.Nd2Writer(DST) as writer:
        attrs = writer.imageAttributes
        if attrs.eCompression != ImageAttributesCompression.ictNone:
            print("Writer reports non-ictNone compression; aborting.")
            return 1

        bits = attrs.uiBpcSignificant
        bad_val = None
        if bits > 0:
            if attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
                bad_val = (1 << bits)
            elif attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
                bad_val = (1 << (bits - 1))

        if bad_val is not None and np.issubdtype(attrs.dtype, np.integer):
            dtype_max = np.iinfo(attrs.dtype).max
            if bad_val <= dtype_max:
                if attrs.componentCount == 1:
                    bad_tile = np.full((tile_h, tile_w), bad_val, dtype=attrs.dtype)
                else:
                    bad_tile = np.full((tile_h, tile_w, attrs.componentCount), bad_val, dtype=attrs.dtype)
                try:
                    writer._chunker.setImageTile(0, x, y, bad_tile)
                    print("Unexpected: out-of-range write succeeded.")
                except ValueError as exc:
                    print(f"Expected out-of-range error: {exc}")
            else:
                print("Skipping out-of-range test: bad value exceeds dtype max.")
        else:
            print("Skipping out-of-range test: unsupported pixel type or non-integer dtype.")

        if bits > 0:
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

        for seqindex in range(frame_count):
            writer._chunker.setImageTile(seqindex, x, y, tile)

    with limnd2.Nd2Reader(DST) as reader, limnd2.Nd2Writer(DST, append=True) as writer:
        mismatches = []
        for seqindex in range(frame_count):
            after = reader.image(seqindex, rect=(x, y, tile_w, tile_h))
            if np.any(after != max_val):
                mismatches.append(seqindex)
            full_image = reader.image(seqindex).copy()
            writer._chunker.generateAndSetDownsampledImages(seqindex, full_image)

    if mismatches:
        print(f"Tile write failed for frames: {mismatches}")
        return 1

    print("Tile write succeeded for all frames; tile region set to max value.")

    print(f"Edited file: {DST}")
    print(f"Tile rect: x={x}, y={y}, w={tile_w}, h={tile_h}")
    print(f"Frames updated: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
