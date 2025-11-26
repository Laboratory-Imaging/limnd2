import numpy as np
import pytest
from pathlib import Path

import limnd2
import limnd2.experiment_factory  # ensure submodule is loaded under limnd2
import limnd2.metadata_factory    # ensure submodule is loaded under limnd2


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
        nd2.experiment = expf.createExperiment()

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
        assert t_loop is not None and t_loop.count == timeloop_count and t_loop.uLoopPars.dPeriod == float(timeloop_step)
        assert z_loop is not None and z_loop.count == zstack_count and z_loop.uLoopPars.dZStep == float(zstack_step)

        md = r.pictureMetadata
        assert len(md.channels) == components
        names = [c.sDescription for c in md.channels]
        assert names == ["Blue channel", "Red channel"]
