from __future__ import annotations

from limnd2.base import ND2_FILE_SIGNATURE
from limnd2.nd2file import ND2File

ND2_FILE_SIGNATURE = ND2_FILE_SIGNATURE


def get_version(path) -> tuple[int, int]:
    with ND2File(path) as f:
        ver = f.version
    if isinstance(ver, tuple) and len(ver) >= 2:
        return ver[0], ver[1]
    return (-1, -1)


def get_chunkmap(*_args, **_kwargs):
    raise NotImplementedError("nd2._parse._chunk_decode.get_chunkmap not implemented in limnd2 compat")
