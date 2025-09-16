from __future__ import annotations

from pathlib import Path
import math
import pytest

import limnd2
from limnd2.experiment import ExperimentLoopType


ND2_BASE = Path(__file__).parent / "test_files" / "nd2_files"
ND2_FILES = sorted(ND2_BASE.rglob("*.nd2")) if ND2_BASE.exists() else []

pytestmark = pytest.mark.skipif(
    not ND2_FILES,
    reason=f"No .nd2 files found under {ND2_BASE}",
)


def _product(vals: list[int] | tuple[int, ...]) -> int:
    total = 1
    for v in vals:
        try:
            total *= int(v)
        except Exception:
            total *= 1
    return total


@pytest.mark.parametrize("nd2_path", ND2_FILES, ids=lambda p: p.name)
def test_dimnames_shape_and_indices(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        exp = r.experiment
        if exp is None:
            pytest.skip("No experiment in file")

        # Basic coherence between dimnames/ndim/shape
        names = exp.dimnames(skipSpectralLoop=True)
        shape = exp.shape(skipSpectralLoop=True)
        assert len(names) == exp.ndim(skipSpectralLoop=True)
        assert len(names) == len(shape)

        # dimensionSizes should map to names
        dims = r.dimensionSizes()
        assert set(dims.keys()) == set(names)

        # Index generation length must match product of sizes
        expected = _product(tuple(dims.values()))
        idx = r.generateLoopIndexes(named=True)
        assert len(idx) == expected


@pytest.mark.parametrize("nd2_path", ND2_FILES, ids=lambda p: p.name)
def test_find_levels_and_params(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        exp = r.experiment
        if exp is None:
            pytest.skip("No experiment in file")

        t_loop = exp.findLevel(ExperimentLoopType.eEtTimeLoop)
        if t_loop is not None:
            assert t_loop.count >= 0
            assert isinstance(t_loop.uLoopPars.dPeriod, (int, float))

        z_loop = exp.findLevel(ExperimentLoopType.eEtZStackLoop)
        if z_loop is not None:
            assert z_loop.count >= 0
            assert isinstance(z_loop.uLoopPars.dZStep, (int, float))

        m_loop = exp.findLevel(ExperimentLoopType.eEtXYPosLoop)
        if m_loop is not None:
            assert m_loop.count >= 0
            pts = getattr(m_loop.uLoopPars, "Points", [])
            assert isinstance(pts, list)


@pytest.mark.parametrize("nd2_path", ND2_FILES, ids=lambda p: p.name)
def test_reader_helpers_match_loops(nd2_path: Path):
    with limnd2.Nd2Reader(nd2_path) as r:
        exp = r.experiment
        if exp is None:
            # If helper exists, ensure it reflects no z-stack
            if hasattr(r, "hasZStack"):
                assert r.hasZStack is False
            return

        has_z = exp.findLevel(ExperimentLoopType.eEtZStackLoop) is not None
        if hasattr(r, "hasZStack"):
            assert r.hasZStack == has_z
