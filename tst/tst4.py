from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import limnd2


# Minimal example input.
#INPUT_ND2 = Path(r"D:\stitch\Slide6_Region1_tiled.nd2")
INPUT_ND2 = Path(r"D:\stitch\60_0003_Region1_tiled.nd2")
#INPUT_ND2 = Path(r"D:\stitch\14_stitch_mp.nd2")


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> int:
    _log("Starting tst4 (stitch demo).")
    _log(f"Input:  {INPUT_ND2}")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_nd2 = INPUT_ND2.with_name(f"{INPUT_ND2.stem}-Stitch{stamp}.nd2")
    #output_nd2 = INPUT_ND2.with_name(f"{INPUT_ND2.stem}-Stitch{stamp}.ome.zarr")
    _log(f"Output: {output_nd2}")

    t0 = time.perf_counter()
    fused = limnd2.stitch(INPUT_ND2, output_filename=output_nd2, verbose=True)
    elapsed = time.perf_counter() - t0

    _log(f"Done in {elapsed:.2f}s.")
    if isinstance(fused, Path):
        _log(f"Wrote ND2: {fused}")
        return 0

    shape = getattr(fused, "shape", None)
    if shape is None and hasattr(fused, "data"):
        shape = getattr(fused.data, "shape", None)
    _log(f"Fused shape: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
