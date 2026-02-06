from pathlib import Path

import nd2
import pytest


def test_ome_meta(new_nd2: Path) -> None:
    ome = pytest.importorskip("ome_types")

    with nd2.ND2File(new_nd2) as f:
        meta = f.ome_metadata()
    assert isinstance(meta, ome.OME)

    # limnd2 compat does not guarantee position-based naming
