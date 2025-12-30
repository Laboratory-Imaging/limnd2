from __future__ import annotations

from pathlib import Path
from datetime import datetime

from limnd2.base import FileStore, MemoryStore

def test_file_store(nd2_path: Path):
    fs = FileStore(nd2_path)
    assert fs.isFile == True
    assert fs.uri == nd2_path.as_uri()
    assert fs.sizeOnDisk == nd2_path.stat().st_size
    assert fs.lastModified == datetime.fromtimestamp(nd2_path.stat().st_mtime)
    assert fs.filename == nd2_path.as_posix()

    fs.open("rb")
    assert fs.fh is not None
    assert fs.mem is not None

    data = fs.mem[0:8]
    assert isinstance(data, bytes)

    fs.close()

def test_memory_store():
    SIZE = 1024
    URI = "memory://test_buffer"
    now = datetime.now()
    ms = MemoryStore(b"0"*SIZE, uri=URI, lastModified=now)
    assert ms.isFile == False
    assert ms.uri == URI
    assert ms.sizeOnDisk == SIZE
    assert ms.lastModified == now
    assert ms.filename == None

    ms.open("rb") # does nothing
    assert ms.fh is None
    assert ms.mem is not None

    data = ms.mem[0:8]
    assert isinstance(data, bytes)
    assert data == b"00000000"

    ms.close() # does nothing