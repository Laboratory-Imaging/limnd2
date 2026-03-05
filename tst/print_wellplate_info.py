#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import sys


DEFAULT_FILE = r"C:\Users\lukas.jirusek\Documents\NIS-Express\Images\Exports\96w-20Xpa-MCF-Flash-1h_.nd2"


def _repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _normalize_input_path(raw_path: str) -> Path:
    text = raw_path.strip().strip('"').strip("'")
    direct = Path(text)
    if direct.exists():
        return direct

    # Windows drive path (C:\... or C:/...)
    if len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        if os.name == "nt":
            return direct
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")

    # WSL-style path passed on Windows: /mnt/c/...
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")

    return direct


def main() -> int:
    parser = argparse.ArgumentParser(description="Print ND2 wellplate chunks.")
    parser.add_argument("file", nargs="?", default=DEFAULT_FILE, help="Path to ND2 file.")
    parser.add_argument("--preview", type=int, default=1000, help="Preview item count.")
    args = parser.parse_args()

    path = _normalize_input_path(args.file)
    if not path.exists():
        print(f"ERROR: File does not exist: {path}")
        return 1

    _repo_src_on_path()
    import limnd2  # pylint: disable=import-error

    with limnd2.Nd2Reader(path) as nd2:
        desc = nd2.wellplateDesc
        info = nd2.wellplateFrameInfo

    print(f"File: {path}")

    if desc is None:
        print("WellplateDesc: <missing>")
    else:
        print("WellplateDesc:")
        print(f"  name={desc.name!r}")
        print(f"  rows={desc.rows}, columns={desc.columns}")
        print(f"  rowNaming={desc.rowNaming!r}, columnNaming={desc.columnNaming!r}")

    if info is None:
        print("WellplateFrameInfo: <missing>")
        return 0

    items = list(info)
    well_names = [item.wellName for item in items]
    unique_wells = sorted(set(well_names))
    counts = Counter(well_names)

    print("WellplateFrameInfo:")
    print(f"  items={len(items)}, unique_wells={len(unique_wells)}")
    print(f"  wells={unique_wells}")
    print("  per-well frame counts:")
    for well_name in unique_wells:
        print(f"    {well_name}: {counts[well_name]}")

    preview_count = max(0, int(args.preview))
    if preview_count > 0:
        print(f"  preview first {min(preview_count, len(items))} item(s):")
        for item in items[:preview_count]:
            print(
                "    "
                f"seq={item.seqIndex}, "
                f"well={item.wellName}, "
                f"row={item.wellRowIndex}, col={item.wellColIndex}, "
                f"wellIndex={item.wellIndex}, plateIndex={item.plateIndex}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
