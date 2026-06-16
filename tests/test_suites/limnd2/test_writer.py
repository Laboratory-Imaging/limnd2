import numpy as np
import pytest
from pathlib import Path

import limnd2
import limnd2.experiment_factory  # ensure submodule is loaded under limnd2
import limnd2.metadata_factory    # ensure submodule is loaded under limnd2
from limnd2.binary import BinaryRasterMetadata, BinaryRasterMetadataItem
from limnd2.base import NameNotInChunkmapError


def _create_random_noise(width: int, height: int, channels: int, bits: int) -> np.ndarray:
    if bits == 8:
        dtype = np.uint8
        max_value = 255
        return np.random.randint(0, max_value + 1, (height, width, channels), dtype=dtype)
    if bits == 16:
        dtype = np.uint16
        max_value = 65535
        return np.random.randint(0, max_value + 1, (height, width, channels), dtype=dtype)
    if bits == 32:
        dtype = np.float32
        return np.random.rand(height, width, channels).astype(dtype)
    raise ValueError("Unsupported bits. Use 8, 16, or 32.")


def test_write_basic_nd2(tmp_path: Path):
    # Keep sizes small for test speed
    width, height = 64, 32
    components = 2
    bits = 8
    seq_count = 6

    timeloop_count, timeloop_step = 3, 150
    zstack_count, zstack_step = 2, 100

    out_path = tmp_path / "writer_basic.nd2"

    with limnd2.Nd2Writer(out_path) as nd2:
        # attributes
        attrs = limnd2.attributes.ImageAttributes.create(
            width=width,
            height=height,
            component_count=components,
            bits=bits,
            sequence_count=seq_count,
        )
        nd2.imageAttributes = attrs

        # image data
        for i in range(seq_count):
            nd2.setImage(i, _create_random_noise(width, height, components, bits))

        # experiment
        expf = limnd2.experiment_factory.ExperimentFactory()
        expf.t.count = timeloop_count
        expf.t.step = timeloop_step
        expf.z.count = zstack_count
        expf.z.step = zstack_step
        nd2.experiment = expf.createExperiment() # type: ignore

        # metadata
        mdf = limnd2.metadata_factory.MetadataFactory(
            zoom_magnification=200.0,
            objective_magnification=1.0,
            pinhole_diameter=50,
            pixel_calibration=10.0,
        )
        mdf.addPlane(name="Blue channel", modality="Confocal, Fluo", color="blue")
        mdf.addPlane(name="Red channel", modality="Confocal, Fluo", color="red")
        nd2.pictureMetadata = mdf.createMetadata()

    assert out_path.exists(), "ND2 writer did not create the file"
    assert out_path.stat().st_size > 0, "ND2 file is empty"

    # Validate by reading back
    with limnd2.Nd2Reader(out_path) as r:
        a = r.imageAttributes
        assert a.width == width
        assert a.height == height
        assert a.componentCount == components
        assert a.frameCount == seq_count
        assert a.uiBpcSignificant == bits

        img0 = r.image(0)
        assert img0.shape == (height, width, components)

        exp = r.experiment
        assert exp is not None
        t_loop = exp.findLevel(limnd2.ExperimentLoopType.eEtTimeLoop)
        z_loop = exp.findLevel(limnd2.ExperimentLoopType.eEtZStackLoop)
        assert t_loop is not None and t_loop.count == timeloop_count and t_loop.uLoopPars.dPeriod == float(timeloop_step) # type: ignore
        assert z_loop is not None and z_loop.count == zstack_count and z_loop.uLoopPars.dZStep == float(zstack_step) # type: ignore

        md = r.pictureMetadata
        assert len(md.channels) == components
        names = [c.sDescription for c in md.channels]
        assert names == ["Blue channel", "Red channel"]


def test_writer_binary_raster_metadata_setter(tmp_path: Path):
    out_path = tmp_path / "writer_binary_raster_metadata.nd2"
    attrs = limnd2.attributes.ImageAttributes.create(
        width=32,
        height=24,
        component_count=1,
        bits=8,
        sequence_count=1,
    )
    binary_metadata = BinaryRasterMetadata(
        [
            BinaryRasterMetadataItem(
                binWidth=32,
                binHeight=24,
                binLayerId=1,
                binName="Layer 1",
                binUuid="uuid-1",
                binComp="",
                binColor=0xFF0000,
            )
        ]
    )

    with limnd2.Nd2Writer(out_path) as nd2:
        nd2.imageAttributes = attrs
        nd2.binaryRasterMetadata = binary_metadata

    with limnd2.Nd2Reader(out_path) as reader:
        persisted = reader.binaryRasterMetadata

    assert len(persisted) == 1
    assert persisted[0] == binary_metadata[0]


def test_writer_set_binary_raster_data_writes_downsampled_chunks(tmp_path: Path):
    out_path = tmp_path / "writer_binary_raster_data.nd2"
    width, height = 2048, 1024
    attrs = limnd2.attributes.ImageAttributes.create(
        width=width,
        height=height,
        component_count=1,
        bits=8,
        sequence_count=1,
    )
    binary_metadata = BinaryRasterMetadata(
        [
            BinaryRasterMetadataItem(
                binWidth=width,
                binHeight=height,
                binLayerId=1,
                binName="Layer 1",
                binUuid="uuid-1",
                binComp="",
                binColor=0xFF0000,
            )
        ]
    )
    data = np.zeros((height, width), dtype=np.uint32)
    data[::2, ::2] = np.arange((height // 2) * (width // 2), dtype=np.uint32).reshape(
        height // 2,
        width // 2,
    )

    with limnd2.Nd2Writer(out_path) as nd2:
        nd2.imageAttributes = attrs
        nd2.binaryRasterMetadata = binary_metadata
        nd2.setBinaryRasterData(1, 0, data)

    with limnd2.Nd2Reader(out_path) as reader:
        full_res = reader.binaryRasterData(1, 0)
        downsampled = reader.chunker.readDownsampledBinaryRasterData(1, 0, downsample_level=1)

    assert np.array_equal(full_res, data)
    assert np.array_equal(downsampled, data[::2, ::2])


def test_writer_set_binary_raster_data_tile_writes_single_tile(tmp_path: Path):
    out_path = tmp_path / "writer_binary_raster_data_tile.nd2"
    width, height = 48, 40
    attrs = limnd2.attributes.ImageAttributes.create(
        width=width,
        height=height,
        component_count=1,
        bits=8,
        sequence_count=1,
    )
    binary_metadata = BinaryRasterMetadata(
        [
            BinaryRasterMetadataItem(
                binWidth=width,
                binHeight=height,
                binTileWidth=16,
                binTileHeight=16,
                binLayerId=1,
                binName="Layer 1",
                binUuid="uuid-1",
                binComp="",
                binColor=0xFF0000,
            )
        ]
    )
    tile = np.arange(8 * 16, dtype=np.uint32).reshape(8, 16)

    with limnd2.Nd2Writer(out_path) as nd2:
        nd2.imageAttributes = attrs
        nd2.binaryRasterMetadata = binary_metadata
        nd2.setBinaryRasterDataTile(1, 0, 32, 32, tile)
        with pytest.raises(ValueError, match="align to tile boundaries"):
            nd2.setBinaryRasterDataTile(1, 0, 1, 0, tile)

    with limnd2.Nd2Reader(out_path) as reader:
        actual = reader.chunker.readBinaryRasterData(1, 0, rect=(32, 32, 16, 8))

    assert np.array_equal(actual, tile)


def test_writer_set_binary_raster_data_tile_downsampled_falls_back_to_full_tiles(tmp_path: Path):
    out_path = tmp_path / "writer_binary_raster_data_tile_downsampled.nd2"
    width, height = 2048, 1024
    attrs = limnd2.attributes.ImageAttributes.create(
        width=width,
        height=height,
        component_count=1,
        bits=8,
        sequence_count=1,
    )
    binary_metadata = BinaryRasterMetadata(
        [
            BinaryRasterMetadataItem(
                binWidth=width,
                binHeight=height,
                binTileWidth=1024,
                binTileHeight=1024,
                binLayerId=1,
                binName="Layer 1",
                binUuid="uuid-1",
                binComp="",
                binColor=0xFF0000,
            )
        ]
    )
    left_tile = np.full((1024, 1024), 5, dtype=np.uint32)
    right_tile = np.full((1024, 1024), 9, dtype=np.uint32)

    with limnd2.Nd2Writer(out_path) as nd2:
        nd2.imageAttributes = attrs
        nd2.binaryRasterMetadata = binary_metadata
        nd2.setBinaryRasterDataTile(1, 0, 0, 0, left_tile)
        nd2.setBinaryRasterDataTile(1, 0, 1024, 0, right_tile)

    with limnd2.Nd2Reader(out_path) as reader:
        with pytest.raises(NameNotInChunkmapError):
            reader.chunker.readDownsampledBinaryRasterData(1, 0, downsample_level=1)
        downsampled = reader.binaryRasterData(1, 0, downsample_level=1)

    expected = np.zeros((512, 1024), dtype=np.uint32)
    expected[:, :512] = left_tile[::2, ::2]
    expected[:, 512:] = right_tile[::2, ::2]
    assert np.array_equal(downsampled, expected)


def test_writer_create_downsampled_binary_raster_data_from_tiles(tmp_path: Path):
    out_path = tmp_path / "writer_create_downsampled_binary_raster_data.nd2"
    width, height = 2048, 1024
    attrs = limnd2.attributes.ImageAttributes.create(
        width=width,
        height=height,
        component_count=1,
        bits=8,
        sequence_count=1,
    )
    binary_metadata = BinaryRasterMetadata(
        [
            BinaryRasterMetadataItem(
                binWidth=width,
                binHeight=height,
                binTileWidth=1024,
                binTileHeight=1024,
                binLayerId=1,
                binName="Layer 1",
                binUuid="uuid-1",
                binComp="",
                binColor=0xFF0000,
            )
        ]
    )
    left_tile = np.arange(1024 * 1024, dtype=np.uint32).reshape(1024, 1024)
    right_tile = left_tile + np.uint32(100)

    with limnd2.Nd2Writer(out_path) as nd2:
        nd2.imageAttributes = attrs
        nd2.binaryRasterMetadata = binary_metadata
        nd2.setBinaryRasterDataTile(1, 0, 0, 0, left_tile)
        nd2.setBinaryRasterDataTile(1, 0, 1024, 0, right_tile)
        nd2.createDownsampledBinaryRasterData(1, 0)

    with limnd2.Nd2Reader(out_path) as reader:
        actual = reader.chunker.readDownsampledBinaryRasterData(1, 0, downsample_level=1)

    expected = np.zeros((512, 1024), dtype=np.uint32)
    expected[:, :512] = left_tile[::2, ::2]
    expected[:, 512:] = right_tile[::2, ::2]
    assert np.array_equal(actual, expected)


def test_writer_set_downsampled_binary_raster_data_tile_replaces_without_reading(tmp_path: Path):
    out_path = tmp_path / "writer_downsampled_binary_raster_data_tile_replace.nd2"
    width, height = 2048, 1024
    attrs = limnd2.attributes.ImageAttributes.create(
        width=width,
        height=height,
        component_count=1,
        bits=8,
        sequence_count=1,
    )
    binary_metadata = BinaryRasterMetadata(
        [
            BinaryRasterMetadataItem(
                binWidth=width,
                binHeight=height,
                binTileWidth=1024,
                binTileHeight=1024,
                binLayerId=1,
                binName="Layer 1",
                binUuid="uuid-1",
                binComp="",
                binColor=0xFF0000,
            )
        ]
    )
    first_tile = np.full((512, 1024), 5, dtype=np.uint32)
    replacement_tile = np.full((512, 1024), 9, dtype=np.uint32)

    with limnd2.Nd2Writer(out_path) as nd2:
        nd2.imageAttributes = attrs
        nd2.binaryRasterMetadata = binary_metadata
        nd2._chunker.setDownsampledBinaryRasterDataTile(1, 0, 0, 0, first_tile, downsample_level=1)

        def fail_read(*args, **kwargs):
            raise AssertionError("setDownsampledBinaryRasterDataTile must not read an existing tile")

        nd2._chunker._read_chunk = fail_read
        nd2._chunker.setDownsampledBinaryRasterDataTile(1, 0, 0, 0, replacement_tile, downsample_level=1)
        with pytest.raises(ValueError, match="align to tile boundaries"):
            nd2._chunker.setDownsampledBinaryRasterDataTile(1, 0, 512, 0, replacement_tile[:, :512], downsample_level=1)

    with limnd2.Nd2Reader(out_path) as reader:
        actual = reader.chunker.readDownsampledBinaryRasterData(1, 0, downsample_level=1)

    assert np.array_equal(actual, replacement_tile)


def test_write_wellplate_chunks_roundtrip(tmp_path: Path):
    out_path = tmp_path / "writer_wellplate_chunks.nd2"

    with limnd2.Nd2Writer(out_path) as nd2:
        attrs = limnd2.attributes.ImageAttributes.create(
            width=8,
            height=6,
            component_count=1,
            bits=8,
            sequence_count=2,
        )
        nd2.imageAttributes = attrs
        nd2.setImage(0, np.zeros((6, 8, 1), dtype=np.uint8))
        nd2.setImage(1, np.zeros((6, 8, 1), dtype=np.uint8))

        nd2.setWellplateDesc(
            limnd2.WellplateDesc(
                name="96 Well Plate",
                rows=8,
                columns=12,
                rowNaming="A-H",
                columnNaming="1-12",
            )
        )
        nd2.setWellplateFrameInfo(
            [
                {
                    "plateIndex": 0,
                    "plateUuid": "plate-uuid-1",
                    "seqIndex": 0,
                    "wellIndex": 0,
                    "wellName": "A1",
                    "wellColIndex": 0,
                    "wellRowIndex": 0,
                },
                {
                    "plateIndex": 0,
                    "plateUuid": "plate-uuid-1",
                    "seqIndex": 1,
                    "wellIndex": 1,
                    "wellCompactName": "A2",
                    "wellColIndex": 1,
                    "wellRowIndex": 0,
                },
            ]
        )

    with limnd2.Nd2Reader(out_path) as r:
        desc = r.wellplateDesc
        assert desc is not None
        assert desc.name == "96 Well Plate"
        assert desc.rows == 8
        assert desc.columns == 12
        assert desc.rowNaming == "A-H"
        assert desc.columnNaming == "1-12"

        frame_info = r.wellplateFrameInfo
        assert frame_info is not None
        assert len(frame_info) == 2
        assert frame_info[0].wellName == "A1"
        assert frame_info[1].wellName == "A2"
        assert frame_info.nwells == 2


def test_write_wellplate_combined_helper(tmp_path: Path):
    out_path = tmp_path / "writer_wellplate_helper.nd2"

    with limnd2.Nd2Writer(out_path) as nd2:
        attrs = limnd2.attributes.ImageAttributes.create(
            width=8,
            height=6,
            component_count=1,
            bits=8,
            sequence_count=1,
        )
        nd2.imageAttributes = attrs
        nd2.setImage(0, np.zeros((6, 8, 1), dtype=np.uint8))

        nd2.setWellplate(
            desc={
                "name": "Custom Plate",
                "rows": 3,
                "columns": 4,
                "rowNaming": "A-C",
                "columnNaming": "1-4",
            },
            frame_info=[
                {
                    "plateIndex": 0,
                    "plateUuid": "plate-uuid-2",
                    "seqIndex": 0,
                    "wellIndex": 5,
                    "wellName": "B2",
                    "wellColIndex": 1,
                    "wellRowIndex": 1,
                }
            ],
        )

    with limnd2.Nd2Reader(out_path) as r:
        desc = r.wellplateDesc
        info = r.wellplateFrameInfo
        assert desc is not None and info is not None
        assert desc.name == "Custom Plate"
        assert desc.rows == 3 and desc.columns == 4
        assert len(info) == 1
        assert info[0].wellName == "B2"


def test_wellplate_frame_info_payload_includes_compact_name_alias():
    payload = limnd2.Nd2Writer._wellplate_frame_info_payload(
        [
            {
                "plateIndex": 0,
                "plateUuid": "plate-uuid-3",
                "seqIndex": 0,
                "wellIndex": 0,
                "wellName": "B01",
                "wellColIndex": 0,
                "wellRowIndex": 1,
            }
        ]
    )
    assert payload[0]["wellName"] == "B01"
    assert payload[0]["wellCompactName"] == "B01"
