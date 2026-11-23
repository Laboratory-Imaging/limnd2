from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

import limnd2
from limnd2.tools.conversion.LimConvertSequence import convert_sequence_to_nd2_args
from limnd2.tools.conversion.LimImageSource import LimImageSource, get_file_dimensions_as_json
from limnd2.tools.conversion.LimPlanSequence import plan_sequence


CHANNEL_NAMES = [
    "DAPI", "NFH", "CC3", "NeuN", "MBP", "RECA1",
    "IBA1++", "TomatoLectin", "PCNA", "MAP2", "GAD67", "GFAP",
]


def _write_ome_without_tiffdata(path: Path, *, size_z: int = 1) -> None:
    channels = "".join(
        f'<Channel ID="Channel:0:{index}" Name="{name}" SamplesPerPixel="1"/>'
        for index, name in enumerate(CHANNEL_NAMES)
    )
    description = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
        f'Type="uint16" SizeX="8" SizeY="8" SizeZ="{size_z}" '
        f'SizeC="{len(CHANNEL_NAMES)}" SizeT="1">{channels}</Pixels></Image></OME>'
    )
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for index in range(len(CHANNEL_NAMES)):
            writer.write(
                np.full((8, 8), index + 1, dtype=np.uint16),
                photometric="minisblack", metadata=None,
                description=description if index == 0 else None,
            )


@pytest.mark.parametrize("with_tiffdata", [False, True])
@pytest.mark.parametrize("flatten_duplicates", [False, True])
def test_ome_channels_preserve_page_and_name_order(
    tmp_path: Path, capsys, with_tiffdata: bool, flatten_duplicates: bool,
) -> None:
    path = tmp_path / "sample.ome.tif"
    if with_tiffdata:
        image = np.stack([np.full((8, 8), i + 1, dtype=np.uint16)
                          for i in range(len(CHANNEL_NAMES))])
        tifffile.imwrite(path, image, ome=True, photometric="minisblack",
                         metadata={"axes": "CYX", "Channel": {"Name": CHANNEL_NAMES}})
    else:
        _write_ome_without_tiffdata(path)
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes in (("CYX",) if with_tiffdata else ("IYX", "QYX"))

    get_file_dimensions_as_json(path)
    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == len(CHANNEL_NAMES)
    assert "unknown" not in result
    assert [row[0] for row in result["qml_settings"]["channels"]] == CHANNEL_NAMES

    args = [str(tmp_path), r"sample\.ome\.tif", "--extension", "tif"]
    if flatten_duplicates:
        args.append("--flatten_duplicates")
    planned = plan_sequence(args)
    assert planned["channel"] == len(CHANNEL_NAMES)
    assert [row[1] for row in planned["qml_settings"]["channels"]] == CHANNEL_NAMES

    output = tmp_path / "output.nd2"
    convert_sequence_to_nd2_args(args + ["-n", output.name, "-o", str(tmp_path)])
    with limnd2.Nd2Reader(output) as reader:
        assert [channel.sDescription for channel in reader.pictureMetadata.channels] == CHANNEL_NAMES
        image = reader.image(0)
        assert [int(image[0, 0, index]) for index in range(len(CHANNEL_NAMES))] == list(
            range(1, len(CHANNEL_NAMES) + 1)
        )


def test_ome_mismatched_z_dimension_remains_unknown(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.ome.tif"
    _write_ome_without_tiffdata(path, size_z=2)
    assert LimImageSource.open(path).get_file_dimensions() == {"unknown": len(CHANNEL_NAMES)}


def test_nd2_metadata_keeps_numeric_channel_and_sample_order() -> None:
    factory = limnd2.MetadataFactory()
    for index, name in enumerate(CHANNEL_NAMES):
        factory.addPlane(name=name, camera_name=f"Camera {index}")
    metadata = limnd2.PictureMetadata.from_lv(factory.createMetadata().to_lv())
    assert metadata.channelNames == CHANNEL_NAMES
    assert [metadata.cameraName(i) for i in range(len(CHANNEL_NAMES))] == [
        f"Camera {i}" for i in range(len(CHANNEL_NAMES))
    ]
