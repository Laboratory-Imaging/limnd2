from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import limnd2
import limnd2.experiment_factory
import limnd2.metadata_factory


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w, _ = img.shape
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)


def _draw_grid(img: np.ndarray, step: int = 64) -> None:
    h, w, _ = img.shape
    grid_color = (45, 45, 45)
    for x in range(0, w, step):
        _draw_rect(img, x, 0, x + 1, h, grid_color)
    for y in range(0, h, step):
        _draw_rect(img, 0, y, w, y + 1, grid_color)


def _draw_digit(img: np.ndarray, digit: int, color: tuple[int, int, int]) -> None:
    # Seven-segment style drawing to avoid extra deps (Pillow/OpenCV).
    h, w, _ = img.shape
    seg = max(8, min(h, w) // 20)
    margin = max(24, min(h, w) // 8)
    cx0 = margin
    cx1 = w - margin
    cy0 = margin
    cy1 = h - margin
    mid = (cy0 + cy1) // 2

    # Segment definitions: A, B, C, D, E, F, G
    A = (cx0 + seg, cy0, cx1 - seg, cy0 + seg)
    B = (cx1 - seg, cy0 + seg, cx1, mid - seg // 2)
    C = (cx1 - seg, mid + seg // 2, cx1, cy1 - seg)
    D = (cx0 + seg, cy1 - seg, cx1 - seg, cy1)
    E = (cx0, mid + seg // 2, cx0 + seg, cy1 - seg)
    F = (cx0, cy0 + seg, cx0 + seg, mid - seg // 2)
    G = (cx0 + seg, mid - seg // 2, cx1 - seg, mid + seg // 2)

    mapping = {
        1: [B, C],
        2: [A, B, G, E, D],
        3: [A, B, G, C, D],
        4: [F, G, B, C],
    }
    for rect in mapping[digit]:
        _draw_rect(img, *rect, color)


def _draw_orientation_markers(img: np.ndarray, tile_id: int) -> None:
    h, w, _ = img.shape
    colors = {
        1: (220, 50, 50),
        2: (50, 170, 50),
        3: (50, 80, 220),
        4: (220, 160, 50),
    }
    c = colors[tile_id]
    # Top-left corner block
    _draw_rect(img, 10, 10, 80, 80, c)
    # L-shape marker
    _draw_rect(img, 100, 20, 220, 36, c)
    _draw_rect(img, 100, 20, 116, 160, c)
    # Bottom arrow-ish stripe
    _draw_rect(img, w // 2 - 120, h - 50, w // 2 + 120, h - 34, c)


def _make_tile(width: int, height: int, tile_id: int) -> np.ndarray:
    # Light canvas with slight tile-specific tint to spot seam mixups.
    base = np.full((height, width, 3), 232, dtype=np.uint8)
    tint = {
        1: np.array([20, 0, 0], dtype=np.uint8),
        2: np.array([0, 20, 0], dtype=np.uint8),
        3: np.array([0, 0, 20], dtype=np.uint8),
        4: np.array([12, 12, 0], dtype=np.uint8),
    }[tile_id]
    base = np.clip(base + tint, 0, 255).astype(np.uint8)
    _draw_grid(base, step=64)
    _draw_orientation_markers(base, tile_id)
    _draw_digit(base, tile_id, color=(15, 15, 15))
    return base


def build_debug_nd2(
    output: Path,
    *,
    width: int = 512,
    height: int = 512,
    step_px: int = 420,
    pixel_size_um: float = 1.0,
    rotation_deg: float = -89.834,
) -> Path:
    # Multipoint order:
    # 1 2
    # 3 4
    xcoords = [0.0, float(step_px), 0.0, float(step_px)]
    ycoords = [0.0, 0.0, float(step_px), float(step_px)]

    with limnd2.Nd2Writer(output) as nd2:
        attrs = limnd2.attributes.ImageAttributes.create(
            width=width,
            height=height,
            component_count=3,
            bits=8,
            sequence_count=4,
        )
        nd2.imageAttributes = attrs

        for i in range(4):
            nd2.setImage(i, _make_tile(width, height, i + 1))

        ef = limnd2.experiment_factory.ExperimentFactory(
            m={"count": 4, "xcoords": xcoords, "ycoords": ycoords}
        )
        nd2.experiment = ef.createExperiment()

        mf = limnd2.metadata_factory.MetadataFactory(
            pixel_calibration=pixel_size_um,
            objective_magnification=1.0,
            zoom_magnification=1.0,
        )
        mf.addPlane(name="RGB", modality="Brightfield", color="white")
        pm = mf.createMetadata(number_of_channels_fallback=1, is_rgb_fallback=True)

        # Inject stage->camera transform metadata used by stitch logic.
        th = math.radians(rotation_deg)
        c = math.cos(th)
        s = math.sin(th)
        object.__setattr__(pm, "dStgLgCT11", float(c))
        object.__setattr__(pm, "dStgLgCT12", float(-s))
        object.__setattr__(pm, "dStgLgCT21", float(s))
        object.__setattr__(pm, "dStgLgCT22", float(c))
        object.__setattr__(pm, "dAngle", float(th))
        object.__setattr__(pm, "bCalibrated", True)
        object.__setattr__(pm, "dCalibration", float(pixel_size_um))

        nd2.pictureMetadata = pm

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a synthetic 4-tile ND2 for stitching transform debugging."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\stitch\debug_4tiles_rot.nd2"),
        help="Output ND2 path.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--step-px",
        type=int,
        default=420,
        help="Multipoint stage step in pixels (with calibration=1.0, unit ~= um).",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=1.0,
        help="Pixel calibration stored in metadata.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=-89.834,
        help="dStgLgCT/dAngle rotation to store in metadata.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = build_debug_nd2(
        args.output,
        width=args.width,
        height=args.height,
        step_px=args.step_px,
        pixel_size_um=args.pixel_size_um,
        rotation_deg=args.rotation_deg,
    )
    print(f"Created debug ND2: {out}")
    print("Tile IDs are arranged as: [1 2; 3 4] with overlap and orientation markers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

