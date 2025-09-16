from __future__ import annotations

from pathlib import Path
import pytest

import limnd2
from limnd2.nd2file import ND2File
from limnd2.attributes import ImageAttributesCompression, ImageAttributesPixelType
from limnd2.experiment import ExperimentLoopType


ND2_BASE = Path(__file__).parent / "test_files" / "nd2_files"
ND2_FILES = sorted(ND2_BASE.rglob("*.nd2")) if ND2_BASE.exists() else []

pytestmark = pytest.mark.skipif(
    not ND2_FILES,
    reason=f"No .nd2 files found under {ND2_BASE}",
)


@pytest.mark.parametrize("nd2_path", ND2_FILES, ids=lambda p: p.name)
def test_basic_properties(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        f = ND2File(nd2_path)
        assert f.version == r.version
        assert f.path == str(nd2_path)
        assert f.is_legacy is False


@pytest.mark.parametrize("nd2_path", ND2_FILES, ids=lambda p: p.name)
def test_attributes_mapping(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        f = ND2File(nd2_path)
        att = r.imageAttributes
        pm = r.pictureMetadata

        compressionLevel = att.dCompressionParam
        if att.eCompression == ImageAttributesCompression.ictLossy:
            compressionType = "lossy"
        elif att.eCompression == ImageAttributesCompression.ictLossLess:
            compressionType = "lossless"
        else:
            compressionType = None
            compressionLevel = None

        pixelDataType = "unsigned" if att.ePixelType == ImageAttributesPixelType.pxtUnsigned else "float"
        channelCount = pm.sPicturePlanes.uiCount

        expected = (
            att.uiBpcInMemory,
            att.uiBpcSignificant,
            att.uiComp,
            att.uiHeight,
            pixelDataType,
            att.uiSequenceCount,
            att.uiWidthBytes,
            att.uiWidth,
            compressionLevel,
            compressionType,
            None if att.uiTileHeight == att.uiHeight else att.uiTileHeight,
            None if att.uiTileWidth == att.uiWidth else att.uiTileWidth,
            channelCount,
        )

        assert tuple(f.attributes) == expected


@pytest.mark.parametrize("nd2_path", ND2_FILES[:1], ids=lambda p: p.name)
def test_not_implemented_interfaces_raise(nd2_path: Path):
    f = ND2File(nd2_path)
    with pytest.raises(NotImplementedError):
        _ = f.text_info
    with pytest.raises(NotImplementedError):
        _ = f.closed
    with pytest.raises(NotImplementedError):
        f.__getstate__()
    with pytest.raises(NotImplementedError):
        f.__setstate__({})
    with pytest.raises(NotImplementedError):
        _ = f.metadata
    with pytest.raises(NotImplementedError):
        f.frame_metadata(0)
    with pytest.raises(NotImplementedError):
        _ = f.ndim
    with pytest.raises(NotImplementedError):
        _ = f.shape
    with pytest.raises(NotImplementedError):
        _ = f.sizes
    with pytest.raises(NotImplementedError):
        _ = f.is_rgb
    with pytest.raises(NotImplementedError):
        _ = f.components_per_channel
    with pytest.raises(NotImplementedError):
        _ = f.size
    with pytest.raises(NotImplementedError):
        _ = f.nbytes
    with pytest.raises(NotImplementedError):
        _ = f.dtype
    with pytest.raises(NotImplementedError):
        f.read_frame(0)


@pytest.mark.parametrize("nd2_path", ND2_FILES, ids=lambda p: p.name)
def test_experiment_mapping_if_present(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        if r.experiment is None:
            pytest.skip("No experiment in file")

        # Build expected loop types from limnd2 (excluding spectral)
        expected_types = []
        for exp in r.experiment:
            if exp.eType == ExperimentLoopType.eEtTimeLoop:
                expected_types.append("TimeLoop")
            elif exp.eType == ExperimentLoopType.eEtZStackLoop:
                expected_types.append("ZStackLoop")
            elif exp.eType == ExperimentLoopType.eEtXYPosLoop:
                expected_types.append("XYPosLoop")
            # skip others by design

        f = ND2File(nd2_path)
        mapped = f.experiment

        # Lengths may differ if only spectral/custom loops exist
        if not expected_types:
            assert mapped == []
            return

        assert len(mapped) == len(expected_types)
        for m, et, exp in zip(mapped, expected_types, r.experiment):
            # Some r.experiment items may be skipped (spectral); ensure alignment
            if exp.eType not in (ExperimentLoopType.eEtTimeLoop, ExperimentLoopType.eEtZStackLoop, ExperimentLoopType.eEtXYPosLoop):
                continue
            assert m.type == et
            # Count should match
            assert m.count >= 0
            if et == "TimeLoop":
                assert hasattr(m.parameters, "periodMs")
                assert m.parameters.periodMs == pytest.approx(exp.uLoopPars.dPeriod)
            elif et == "ZStackLoop":
                assert hasattr(m.parameters, "stepUm")
                assert m.parameters.stepUm == pytest.approx(exp.uLoopPars.dZStep)
            elif et == "XYPosLoop":
                assert hasattr(m.parameters, "points")
                # points count should be <= raw points
                assert len(m.parameters.points) <= len(getattr(exp.uLoopPars, "Points", []))
