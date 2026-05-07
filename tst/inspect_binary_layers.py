from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import limnd2


def _iter_nd2_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.nd2"))


def _sample_seq_count(nd2: limnd2.Nd2Reader, limit: int) -> int:
    loop_indexes = nd2.generateLoopIndexes(named=True)
    if loop_indexes:
        return min(limit, len(loop_indexes))
    return 1


def inspect_file(path: Path, *, seq_limit: int = 3) -> bool:
    found_any = False
    with limnd2.Nd2Reader(path) as nd2:
        metadata = nd2.binaryRasterMetadata
        if not metadata:
            return False

        found_any = True
        print(f"FILE: {path}")
        print(f"  dims: {nd2.dimensionSizes()}")
        print(f"  binary layers: {len(metadata)}")

        seq_count = _sample_seq_count(nd2, seq_limit)
        for item in metadata:
            print(
                "  "
                f"layer id={item.id} name={item.name!r} "
                f"component={item.binComp!r} order={item.binCompOrder}"
            )

            if item.id <= 0:
                print("    cannot read through Nd2Reader.binaryRasterData(): bin_id <= 0")
                continue

            for seq_index in range(seq_count):
                try:
                    arr = np.asarray(nd2.binaryRasterData(item.id, seq_index))
                except Exception as exc:
                    print(
                        "    "
                        f"seq={seq_index} ERROR: {type(exc).__name__}: {exc}"
                    )
                    continue
                uniq = np.unique(arr)
                preview = uniq[:10].tolist()
                suffix = "" if len(uniq) <= 10 else " ..."
                print(
                    "    "
                    f"seq={seq_index} shape={arr.shape} dtype={arr.dtype} "
                    f"min={arr.min()} max={arr.max()} unique_count={len(uniq)} "
                    f"unique_head={preview}{suffix}"
                )
        print()

    return found_any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect binary layers embedded in ND2 test files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=(
            r"C:\Users\lukas.jirusek\Desktop\GitHubDesktop\limnd2\tests\test_files"
        ),
        help="ND2 file or directory to scan.",
    )
    parser.add_argument(
        "--seq-limit",
        type=int,
        default=3,
        help="Maximum number of sequence indices to sample per binary layer.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    files = _iter_nd2_files(root)
    found = False
    for path in files:
        try:
            found = inspect_file(path, seq_limit=args.seq_limit) or found
        except Exception as exc:
            print(f"FILE: {path}")
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print()

    if not found:
        print("No binary layers found.")


if __name__ == "__main__":
    main()
