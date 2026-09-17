from __future__ import annotations

from collections import OrderedDict
import struct

from limnd2.base import (
    ND2_CHUNK_FORMAT_ImageDataSeq_1p,
    ND2_CHUNK_FORMAT_DownsampledColorData_2p,
    ND2_CHUNK_FORMAT_TiledRasterBinaryData_2p,
    ND2_CHUNK_FORMAT_TiledRasterBinaryData_4p,
    ND2_CHUNK_RE_ImageDataSeq_1p,
    ND2_CHUNK_RE_DownsampledColorData_2p,
    BaseChunker,
)
from limnd2.file_modern import LimBinaryIOChunker


def test_chunk_name_helpers_roundtrip():
    # Image chunk formatting and detection
    name_img0 = ND2_CHUNK_FORMAT_ImageDataSeq_1p % (0)
    assert ND2_CHUNK_RE_ImageDataSeq_1p.fullmatch(name_img0)
    assert BaseChunker.isImageChunk(name_img0) == 0

    # Downsampled image formatting and detection
    name_ds = ND2_CHUNK_FORMAT_DownsampledColorData_2p % (4, 0)
    assert ND2_CHUNK_RE_DownsampledColorData_2p.fullmatch(name_ds)
    idx, down = BaseChunker.isDownsampledImageChunk(name_ds) # type: ignore
    assert idx == 0 and down == 4

    # Binary raster names are recognized by helper (no real bin id needed here)
    name_bin2 = ND2_CHUNK_FORMAT_TiledRasterBinaryData_2p % (1, 0)
    assert BaseChunker.isBinaryRasterData(name_bin2) == (1, 0, 0, 0)

    name_bin4 = ND2_CHUNK_FORMAT_TiledRasterBinaryData_4p % (1, 0, 0, 0)
    assert BaseChunker.isBinaryRasterData(name_bin4) == (1, 0, 0, 0)


def test_chunkmap_writer_roundtrips_unknown_size_sentinel():
    name = b"CustomData|UnknownSize!"
    data = LimBinaryIOChunker._write_chunkmap(
        None, 4096, OrderedDict([(name, (128, -1))])
    )

    assert struct.Struct("QQ").unpack_from(data, len(name)) == (128, 128)

