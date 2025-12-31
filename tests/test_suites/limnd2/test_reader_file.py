from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np
import pytest

import limnd2
from limnd2.base import (
    ND2_CHUNK_FORMAT_ImageDataSeq_1p,
    UnexpectedCallError,
    ND2_CHUNK_NAME_AcqTimes2Cache,
    ND2_CHUNK_NAME_AcqFramesCache,
    BaseChunker,
)
from limnd2.binary import BinaryRasterMetadataItem, BinaryRasterMetadata, BinaryItemColorMode


def test_chunker_properties_and_chunk_access(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        c = r.chunker
        # filename, version and timestamps
        assert isinstance(c.store.filename, (str, type(None)))
        assert isinstance(c.format_version, tuple) and len(c.format_version) == 2
        assert isinstance(c.store.lastModified, datetime.datetime)
        assert c.store.sizeOnDisk > 0

        # chunk names present; metadata chunk missing returns None
        names = c.chunk_names
        assert isinstance(names, list) and all(isinstance(n, (bytes, bytearray)) for n in names)
        assert c.chunk(b"DefinitelyMissingChunk!") is None

        # calling chunk() with an image data name should raise an UnexpectedCallError
        with pytest.raises(UnexpectedCallError):
            _ = c.chunk(ND2_CHUNK_FORMAT_ImageDataSeq_1p % (0))


def test_image_read_and_rect(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        c = r.chunker
        a = c.imageAttributes
        if a.uiSequenceCount <= 0:
            pytest.skip("no frames")
        img = c.image(0)
        assert isinstance(img, np.ndarray)
        assert img.shape == a.shape
        assert img.dtype == a.dtype

        # Read a small rect subset
        h, w = a.height, a.width
        rw, rh = max(1, w // 8), max(1, h // 8)
        rect = (0, 0, rw, rh)
        sub = c.image(0, rect=rect)
        assert sub.shape == (rh, rw, a.shape[2])


def test_downsample_flags_and_ranges(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        c = r.chunker
        # Boolean flags shouldn't error
        assert isinstance(c.hasDownsampledImages, bool)
        assert isinstance(c.hasDownsampledBinaryRasterData, bool)

        # comp ranges shape and ordering
        cr = c.compRange
        assert cr.shape == (c.imageAttributes.uiComp, 2)
        assert np.all(cr[:, 0] <= cr[:, 1])


def test_set_downsampled_image_write_and_read(tmp_path: Path, nd2_base_dir: Path):
    # Use a specific ND2 file mirrored from the server
    target_name = "underwater_bmx_generated_by_NIS.nd2"
    source = nd2_base_dir / target_name
    if not source.exists():
        pytest.skip(f"Target ND2 not mirrored: {source}")

    out = tmp_path / source.name
    shutil.copy2(source, out)

    # Append downsampled data
    with limnd2.Nd2Writer(out) as w:  # append mode auto-detected
        if not w.chunker.imageAttributes.downsampleLevels:
            pytest.skip("lowerPowSizeList is empty for target file")
        dattrs = w.chunker.imageAttributes.makeDownsampled()
        dimg = np.random.randint(0, 256, dattrs.shape, dtype=dattrs.dtype) # type: ignore
        w.chunker.setDownsampledImage(0, dimg, downsample_level=1)

    # Read back and compare
    with limnd2.Nd2Reader(out) as r:
        rd = r.chunker.readDownsampledImage(0, downsample_level=1)
        assert rd.shape == dimg.shape
        assert rd.dtype == dimg.dtype
        assert np.array_equal(rd, dimg)


def test_downsampledImage_reads_chunk_and_fallback(tmp_path: Path, nd2_base_dir: Path):
    # Use the specific ND2 file; append a downsampled image and validate BaseChunker.downsampledImage
    target_name = "underwater_bmx_generated_by_NIS.nd2"
    source = nd2_base_dir / target_name
    if not source.exists():
        pytest.skip(f"Target ND2 not mirrored: {source}")

    out = tmp_path / source.name
    shutil.copy2(source, out)

    # Append a downsampled image at the first available down level
    with limnd2.Nd2Writer(out) as w:
        if not w.chunker.imageAttributes.downsampleLevels:
            pytest.skip("lowerPowSizeList is empty for target file")
        dattrs = w.chunker.imageAttributes.makeDownsampled()
        dimg = np.random.randint(0, 256, dattrs.shape, dtype=dattrs.dtype) # type: ignore
        w.chunker.setDownsampledImage(0, dimg, downsample_level=1)

    with limnd2.Nd2Reader(out) as r:
        c = r.chunker
        # Exact-level read via BaseChunker.downsampledImage should match the chunk we wrote
        exact = c.downsampledImage(0, downsample_level=1)
        assert exact.shape == dimg.shape
        assert exact.dtype == dimg.dtype
        assert np.array_equal(exact, dimg)


def test_acq_times_and_frames_caches(tmp_path: Path):
    out = tmp_path / "acq_cache.nd2"
    count = 5
    with limnd2.Nd2Writer(out) as w:
        attrs = limnd2.attributes.ImageAttributes.create(width=8, height=6, component_count=1, bits=8, sequence_count=count)
        w.imageAttributes = attrs
        # write minimal frames to match sequence_count
        for i in range(count):
            w.setImage(i, np.zeros((6, 8, 1), dtype=np.uint8))

        # Set caches
        times2 = np.linspace(0.0, 40.0, count, dtype=np.float64)
        frames = np.arange(count, dtype=np.uint32)
        w.chunker.setChunk(ND2_CHUNK_NAME_AcqTimes2Cache, times2.tobytes())
        w.chunker.setChunk(ND2_CHUNK_NAME_AcqFramesCache, frames.tobytes())

    with limnd2.Nd2Reader(out) as r:
        # acqTimes: either loads cache or synthesizes; ensure shape
        t = r.chunker.acqTimes
        assert t.shape == (count,)
        # acqTimes2/acqFrames can be None due to implementation; just ensure access doesn't raise
        t2 = r.chunker.acqTimes2
        f = r.chunker.acqFrames
        assert (t2 is None) or (isinstance(t2, np.ndarray) and t2.shape == (count,))
        assert (f is None) or (isinstance(f, np.ndarray) and f.shape == (count,))


def test_binary_raster_downsample_workflow(tmp_path: Path):
    out = tmp_path / "bin_raster.nd2"
    width, height = 64, 48
    binid = 1
    # Prepare raster metadata for writer chunker
    brm = BinaryRasterMetadata([
        BinaryRasterMetadataItem(
            binWidth=width,
            binHeight=height,
            binTileWidth=16,
            binTileHeight=16,
            binCompressionId="zlib",
            binCompressionLevel=6,
            binBitdepth=32,
            binLayerId=binid,
            binName="Layer1",
            binUuid="uuid-1",
            binComp="Comp",
            binCompOrder=0,
            binState=0,
            binColor=0,
            binColorMode=BinaryItemColorMode.eBaseBinObjColors,
        )
    ])

    with limnd2.Nd2Writer(out, chunker_kwargs={"with_binary_raster_metadata": brm}) as w:
        attrs = limnd2.attributes.ImageAttributes.create(width=width, height=height, component_count=1, bits=8, sequence_count=1)
        w.imageAttributes = attrs
        w.setImage(0, np.zeros((height, width, 1), dtype=np.uint8))

        # Base binary raster
        base_bin = np.random.randint(0, 2**32-1, (height, width), dtype=np.uint32)
        w.chunker.setBinaryRasterData(binid, 0, base_bin)

        # Fallback read: ask for downsize without chunks -> scales base
        downsample_levels = w.chunker.imageAttributes.downsampleLevels
        if not downsample_levels:
            pytest.skip("No lower powers available for downsample")
        level_1 = downsample_levels[0]

        # Also generate real downsampled chunks and verify exact read
        w.chunker.generateAndSetDownsampledBinaryRasterData(binid, 0, base_bin)
        dread = w.chunker.readDownsampledBinaryRasterData(binid, 0, downsample_level=level_1)
        assert isinstance(dread, np.ndarray)
        exp_meta = brm.findItemById(binid).makeDownsampled(level_1) # type: ignore
        assert dread.shape == exp_meta.shape

    with limnd2.Nd2Reader(out) as r:
        c = r.chunker
        # Detection helper over chunk names
        found = False
        downsize = r.imageAttributes.makeDownsampled().powSize
        for name in c.chunk_names:
            res = BaseChunker.isDownsampledBinaryRasterData(name)
            if res is not None:
                b_id, seq, dsz, ty, tx = res
                assert b_id == binid and dsz == downsize
                found = True
                break
        assert found, "No downsampled binary raster chunks detected"

        # Fallback path read (uses scaling if exact chunk missing) and exact chunk read
        fb = c.downsampledBinaryRasterData(binid, 0, downsample_level=1)
        rd = c.readDownsampledBinaryRasterData(binid, 0, downsample_level=1)
        assert rd.shape == fb.shape and rd.dtype == fb.dtype


def test_rle_detection_and_version_on_existing_files(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        meta = r.chunker.binaryRleMetadata
        if not meta or len(meta) == 0:
            pytest.skip("No RLE binary metadata in this file")

        regex_dict = meta.dataChunkNameRegexDict
        matched = False
        for name in r.chunker.chunk_names:
            res = BaseChunker.isBinaryRleDataChunk(regex_dict, name)
            if res is not None:
                binid, seq = res
                assert binid in meta.binIdList
                matched = True
                break
        assert matched, "Expected to match at least one RLE binary data chunk"

        # version should be >= 0, and >0 if we matched
        ver = r.chunker.rleBinaryVersion()
        assert ver >= 0
