#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import statistics
import sys


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

    if len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        if os.name == "nt":
            return direct
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")

    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")

    return direct


def _iter_nd2_files(root: Path, recursive: bool):
    yield from root.rglob("*.nd2") if recursive else root.glob("*.nd2")


def _trimmed_median(values: list[float], *, trim_ratio: float = 0.1) -> float | None:
    if not values:
        return None
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) < 10:
        return statistics.median(vals)
    trim = int(len(vals) * trim_ratio)
    if trim * 2 >= len(vals):
        return statistics.median(vals)
    return statistics.median(vals[trim : len(vals) - trim])


def _estimate_tile_pitch(points: list[object], *, max_samples: int = 400) -> tuple[float | None, float | None, float | None]:
    if len(points) < 2:
        return None, None, None

    coords = [(float(p.dPosX), float(p.dPosY)) for p in points]
    n = len(coords)
    stride = max(1, n // max_samples)
    sample_indices = list(range(0, n, stride))[:max_samples]

    x_steps: list[float] = []
    y_steps: list[float] = []
    nn_dists: list[float] = []
    eps = 1e-9

    for i in sample_indices:
        xi, yi = coords[i]
        best_d2 = math.inf
        best_dx = 0.0
        best_dy = 0.0
        for j, (xj, yj) in enumerate(coords):
            if i == j:
                continue
            dx = abs(xj - xi)
            dy = abs(yj - yi)
            d2 = dx * dx + dy * dy
            if d2 <= eps:
                continue
            if d2 < best_d2:
                best_d2 = d2
                best_dx = dx
                best_dy = dy
        if not math.isfinite(best_d2):
            continue
        nn_dists.append(math.sqrt(best_d2))
        if best_dx >= best_dy and best_dx > eps:
            x_steps.append(best_dx)
        if best_dy >= best_dx and best_dy > eps:
            y_steps.append(best_dy)

    pitch_x = _trimmed_median(x_steps)
    pitch_y = _trimmed_median(y_steps)
    nearest = _trimmed_median(nn_dists)

    if pitch_x is None:
        pitch_x = nearest
    if pitch_y is None:
        pitch_y = nearest
    return pitch_x, pitch_y, nearest


def _overlap(step_um: float | None, fov_um: float | None) -> float | None:
    if step_um is None or fov_um is None or fov_um <= 0.0:
        return None
    return 1.0 - (step_um / fov_um)


def _print_file_report(nd2_path: Path, *, points_preview: int) -> tuple[int, bool]:
    import limnd2  # pylint: disable=import-error
    from limnd2.experiment import ExperimentLoopType, ExperimentXYPosLoop

    print("=" * 88)
    print(f"File: {nd2_path}")

    try:
        with limnd2.Nd2Reader(nd2_path) as nd2:
            exp = nd2.experiment
            if exp is None:
                print("  experiment: <missing>")
                return 0, False

            mp_level = exp.findLevel(ExperimentLoopType.eEtXYPosLoop)
            if mp_level is None:
                print("  multipoint: no")
                return 0, False

            if not isinstance(mp_level.uLoopPars, ExperimentXYPosLoop):
                print("  multipoint: present, but loop type is not ExperimentXYPosLoop")
                return 1, False

            mp = mp_level.uLoopPars
            dims = nd2.dimensionSizes(skipSpectralLoop=True)
            print("  multipoint: yes")
            print(f"  loop-count (uiCount): {mp.uiCount}")
            print(f"  parsed points: {len(mp.Points) if mp.Points is not None else 0}")
            print(f"  bUseZ={mp.bUseZ}, bZEnabled={mp.bZEnabled}, bRelativeXY={mp.bRelativeXY}")
            print(
                "  bSplitMultipoints="
                f"{mp.bSplitMultipoints}, bUseAFPlane={mp.bUseAFPlane}, bKeepPFSOn={mp.bKeepPFSOn}"
            )
            print(
                f"  reference XY: ({mp.dReferenceX:.3f}, {mp.dReferenceY:.3f}), "
                f"z-device={mp.sZDevice!r}"
            )
            print(f"  dimensions: {dims}")
            print(f"  canonical imageDataShape (T,M,Z,Y,X,C): {nd2.imageDataShape}")

            um_per_px: float | None = None
            if nd2.pictureMetadata is not None and nd2.pictureMetadata.bCalibrated:
                um_per_px = float(nd2.pictureMetadata.dCalibration)
                print(f"  xy calibration: {um_per_px:.6f} um/px")
            else:
                print("  xy calibration: <missing / uncalibrated>")

            attrs = nd2.imageAttributes
            fov_x_um = float(attrs.width) * um_per_px if um_per_px is not None else None
            fov_y_um = float(attrs.height) * um_per_px if um_per_px is not None else None
            if fov_x_um is not None and fov_y_um is not None:
                print(f"  fov: x={fov_x_um:.3f} um, y={fov_y_um:.3f} um")
            else:
                print("  fov: <unknown, needs calibration>")

            points = mp.Points if mp.Points is not None else []
            pitch_x, pitch_y, nearest = _estimate_tile_pitch(points)
            if nearest is not None:
                print(f"  estimated nearest tile-center distance: {nearest:.3f} um")
            else:
                print("  estimated nearest tile-center distance: <insufficient points>")
            if pitch_x is not None or pitch_y is not None:
                px = f"{pitch_x:.3f} um" if pitch_x is not None else "n/a"
                py = f"{pitch_y:.3f} um" if pitch_y is not None else "n/a"
                print(f"  estimated pitch: x={px}, y={py}")
            else:
                print("  estimated pitch: <unavailable>")

            ovx = _overlap(pitch_x, fov_x_um)
            ovy = _overlap(pitch_y, fov_y_um)
            if ovx is not None or ovy is not None:
                sx = f"{ovx * 100.0:.2f}%" if ovx is not None else "n/a"
                sy = f"{ovy * 100.0:.2f}%" if ovy is not None else "n/a"
                print(f"  estimated overlap: x={sx}, y={sy}")
            else:
                print("  estimated overlap: <needs calibration and pitch>")

            readiness_issues: list[str] = []
            if len(points) < 2:
                readiness_issues.append("not enough points")
            if um_per_px is None:
                readiness_issues.append("missing XY calibration")
            if pitch_x is None and pitch_y is None:
                readiness_issues.append("could not estimate tile pitch")
            if ovx is not None and not (0.0 <= ovx < 1.0):
                readiness_issues.append("x-overlap outside [0, 100)%")
            if ovy is not None and not (0.0 <= ovy < 1.0):
                readiness_issues.append("y-overlap outside [0, 100)%")
            if readiness_issues:
                print("  stitch-readiness: PARTIAL/UNCERTAIN")
                print(f"  stitch-issues: {', '.join(readiness_issues)}")
            else:
                print("  stitch-readiness: GOOD (geometry sufficient for initial stitching)")

            if mp.Points:
                n = max(0, points_preview)
                preview = min(n, len(mp.Points))
                print(f"  point preview ({preview}/{len(mp.Points)}):")
                for i, point in enumerate(mp.Points[:preview]):
                    name = point.dPosName if point.dPosName else f"#{i}"
                    print(
                        "    "
                        f"{i}: name={name!r}, "
                        f"x={point.dPosX:.3f}, y={point.dPosY:.3f}, "
                        f"z={point.dPosZ:.3f}, pfs={point.dPFSOffset:.3f}"
                    )

            wp_desc = nd2.wellplateDesc
            wp_info = nd2.wellplateFrameInfo
            if wp_desc is not None:
                print(
                    f"  wellplate desc: name={wp_desc.name!r}, "
                    f"rows={wp_desc.rows}, cols={wp_desc.columns}"
                )
            else:
                print("  wellplate desc: <missing>")
            if wp_info is not None:
                print(f"  wellplate frame entries: {len(wp_info)}, unique wells: {wp_info.nwells}")
            else:
                print("  wellplate frame info: <missing>")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        return 1, False

    return 0, True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a folder for ND2 files and print key multipoint settings."
    )
    parser.add_argument("folder", help="Folder containing .nd2 files.")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top-level folder (default scans recursively).",
    )
    parser.add_argument(
        "--points-preview",
        type=int,
        default=10,
        help="How many points to print per file.",
    )
    args = parser.parse_args()

    folder = _normalize_input_path(args.folder)
    if not folder.exists():
        print(f"ERROR: Folder does not exist: {folder}")
        return 1
    if not folder.is_dir():
        print(f"ERROR: Not a folder: {folder}")
        return 1

    _repo_src_on_path()
    files = sorted(_iter_nd2_files(folder, recursive=not args.non_recursive))
    if not files:
        print(f"No .nd2 files found in: {folder}")
        return 0

    print(f"Scanning {len(files)} file(s) in: {folder}")
    failures = 0
    mp_files = 0
    for nd2_path in files:
        rc, has_mp = _print_file_report(nd2_path, points_preview=args.points_preview)
        failures += 1 if rc else 0
        mp_files += 1 if has_mp else 0

    print("=" * 88)
    print(
        f"Done. files={len(files)}, multipoint-files={mp_files}, files-with-errors={failures}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
