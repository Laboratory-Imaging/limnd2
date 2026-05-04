from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import limnd2
import numpy as np

# Optional explicit input (set to string path to force one file), otherwise
# first existing file from INPUT_ND2_CANDIDATES is used.
INPUT_ND2: str | None = None
INPUT_ND2_SWEEP = [
    r"D:\stitch\14_stitch_mp.nd2",
    r"D:\stitch\md2_crop.nd2",
    r"D:\stitch\debug_4tiles_rot.nd2",
    r"D:\stitch\convallaria_flim_crop.nd2",
]
INPUT_ND2_CANDIDATES = [
    r"D:\stitch\14_stitch_mp.nd2",
    r"D:\stitch\debug_4tiles_rot.nd2",
    r"D:\stitch\debug_4tiles_norot.nd2",
    r"D:\convallaria_flim_crop.nd2",
    r"D:\md2_crop.nd2",
    r"D:\60_0003_Region1_tiled.nd2",
    r"D:\Slide6_Region1_tiled.nd2",
    r"\\cork\images\big_stitch\Slide6_Region1_tiled.nd2",
    r"\\cork\images\big_stitch\Slide3_0014_Region1_tiled.nd2",
    r"\\cork\images\big_stitch\lung-Z MultiPoint.nd2",
]
DO_STITCH = True
SHOW_GUI = False
PLOT_DPI = 600
PLOT_SIZE_INCH = (14, 14)
STITCH_DOWNSAMPLE_LEVEL = 0
STITCH_OUTPUT_CHUNKSIZE = {"y": 4096, "x": 4096}
STITCH_OUTPUT_CHUNKSIZES = [
    {"y": 2048, "x": 2048},
    {"y": 8192, "x": 8192},
]
STITCH_BATCH_SIZE = 4
STITCH_BATCH_SIZES = [1, 8]
STAGE_TRANSFORM_MODES = ["inverse", "stitcher_inverse"]#["inverse", "stitcher_inverse"]
STITCH_DASK_SCHEDULER = "threads"
STITCH_DASK_NUM_WORKERS = 4
STITCH_DASK_SCHEDULERS = ["threads", "processes"]
STITCH_DASK_NUM_WORKERS_LIST = [4, 8]
STITCH_USE_RAY_BATCHES = False
STITCH_RAY_NUM_CPUS = 4
STITCH_ZARR_OPTIONS = {"ome_zarr": True}
CLEAN_OUTPUT_BEFORE_STITCH = True
OUTPUT_MODES = ["nd2", "zarr"]  # any of: "nd2", "zarr"
ND2_TILE_SIZE = (1024, 1024)
ND2_STREAM_ALL_TIMEPOINTS = True
STITCH_BLEND_PRESETS = ["balanced"]# ["hard_seams", "balanced", "wide_blend", "crisp_interp"]
STITCH_BATCH_EXEC_PRESETS = ["default", "joblib_threading", "joblib_loky"]  # default|joblib_threading|joblib_loky|ray
STITCH_TIMES_FILENAME = "stitch_times.txt"
PARALLELIZE_INPUT_SWEEP = False
INPUT_SWEEP_MAX_WORKERS = 4



def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def _blend_preset_kwargs(name: str) -> dict:
    key = name.strip().lower()
    presets = {
        # Use multiview-stitcher defaults.
        "default": {},
        # Fastest, minimal overlap smoothing (can show seams).
        "hard_seams": {
            "overlap_in_pixels": 0,
            "blending_widths": 0,
            "interpolation_order": 0,
        },
        # Good starting point for most tiled mosaics.
        "balanced": {
            "overlap_in_pixels": 64,
            "blending_widths": 24,
            "interpolation_order": 1,
        },
        # Stronger seam smoothing, a bit slower.
        "wide_blend": {
            "overlap_in_pixels": 128,
            "blending_widths": 48,
            "interpolation_order": 1,
        },
        # Keep edges sharper.
        "crisp_interp": {
            "overlap_in_pixels": 32,
            "blending_widths": 8,
            "interpolation_order": 0,
        },
    }
    if key not in presets:
        raise ValueError(
            f"Unknown STITCH_BLEND_PRESET={name!r}. "
            f"Use one of: {', '.join(sorted(presets.keys()))}."
        )
    return presets[key].copy()


def _sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())


def _chunksize_label(chunksize: dict[str, int]) -> str:
    y = int(chunksize.get("y", 0))
    x = int(chunksize.get("x", 0))
    z = chunksize.get("z", None)
    if z is None:
        return f"y{y}_x{x}"
    return f"z{int(z)}_y{y}_x{x}"


def _batch_exec_options(name: str, n_batch: int, workers: int) -> dict | None:
    key = name.strip().lower()
    if key == "default":
        return {"n_batch": int(n_batch)}
    if key in {"joblib_threading", "joblib_loky"}:
        from multiview_stitcher import misc_utils

        backend = "threading" if key == "joblib_threading" else "loky"
        return {
            "batch_func": misc_utils.process_batch_using_joblib,
            "n_batch": int(n_batch),
            "batch_func_kwargs": {"n_jobs": int(workers), "backend": backend},
        }
    if key == "ray":
        from multiview_stitcher import misc_utils

        return {
            "batch_func": misc_utils.process_batch_using_ray,
            "n_batch": int(n_batch),
            "batch_func_kwargs": {"num_cpus": int(workers)},
        }
    raise ValueError(
        f"Unknown batch execution preset: {name!r}. "
        "Use one of: default, joblib_threading, joblib_loky, ray."
    )


def _resolve_available_batch_exec_presets() -> list[str]:
    resolved: list[str] = []
    for preset in STITCH_BATCH_EXEC_PRESETS:
        key = preset.strip().lower()
        if key in {"joblib_threading", "joblib_loky"}:
            try:
                import joblib  # type: ignore  # noqa: F401
            except Exception:
                _log(f"Batch exec preset '{preset}' unavailable (joblib missing), skipping.")
                continue
        if key == "ray":
            try:
                import ray  # type: ignore  # noqa: F401
            except Exception:
                _log(f"Batch exec preset '{preset}' unavailable (ray missing), skipping.")
                continue
        resolved.append(preset)
    if not resolved:
        resolved = ["default"]
    return resolved


def _resolve_input_path() -> Path:
    if INPUT_ND2 is not None and INPUT_ND2.strip():
        return _normalize_input_path(INPUT_ND2)

    candidates: list[Path] = []
    for raw in INPUT_ND2_CANDIDATES:
        p = _normalize_input_path(raw)
        candidates.append(p)
        _log(f"Input candidate: {p} [{ 'exists' if p.exists() else 'missing' }]")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No ND2 input file found. Set INPUT_ND2 explicitly or update INPUT_ND2_CANDIDATES."
    )


def main(input_nd2_override: str | None = None) -> int:
    _log("Starting tst3.")
    import dask.array as da
    from dask.delayed import delayed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from multiview_stitcher import msi_utils, spatial_image_utils as si_utils, vis_utils
    from limnd2.experiment import ExperimentLoopType, ExperimentXYPosLoop, find_zstack

    nd2_path = (
        _normalize_input_path(input_nd2_override)
        if input_nd2_override is not None
        else _resolve_input_path()
    )
    times_log_path = nd2_path.with_name(STITCH_TIMES_FILENAME)
    positions_png = nd2_path.with_name(f"{nd2_path.stem}_positions_stage_metadata.png")
    available_batch_exec_presets = _resolve_available_batch_exec_presets()
    total_runs = (
        len(OUTPUT_MODES)
        * len(STITCH_OUTPUT_CHUNKSIZES)
        * len(STITCH_BATCH_SIZES)
        * len(STITCH_DASK_SCHEDULERS)
        * len(STITCH_DASK_NUM_WORKERS_LIST)
        * len(available_batch_exec_presets)
        * len(STAGE_TRANSFORM_MODES)
        * len(STITCH_BLEND_PRESETS)
    )
    _log(f"Input: {nd2_path}")
    _log(f"Total planned runs: {total_runs}")
    _log(f"Output modes: {OUTPUT_MODES}")
    _log(f"Transform modes: {STAGE_TRANSFORM_MODES}")
    _log(f"Blend presets: {STITCH_BLEND_PRESETS}")
    _log(f"Output chunksize sweep: {STITCH_OUTPUT_CHUNKSIZES}")
    _log(f"Batch size sweep: {STITCH_BATCH_SIZES}")
    _log(f"Dask scheduler sweep: {STITCH_DASK_SCHEDULERS}")
    _log(f"Dask worker sweep: {STITCH_DASK_NUM_WORKERS_LIST}")
    _log(f"Batch execution sweep (requested): {STITCH_BATCH_EXEC_PRESETS}")
    _log(f"Batch execution sweep (available): {available_batch_exec_presets}")
    _log(f"Positions preview image: {positions_png}")

    _log("Opening ND2 reader.")
    with limnd2.Nd2Reader(nd2_path) as nd2:
        _log("ND2 opened. Reading experiment metadata.")
        pm = nd2.pictureMetadata
        _log("Orientation metadata:")
        _log(
            "  dStgLgCT matrix = "
            f"[[{pm.dStgLgCT11:.6f}, {pm.dStgLgCT12:.6f}], "
            f"[{pm.dStgLgCT21:.6f}, {pm.dStgLgCT22:.6f}]]"
        )
        _log(f"  dAngle = {pm.dAngle}")
        _log(f"  ePictureXAxis = {pm.ePictureXAxis}")
        _log(f"  ePictureYAxis = {pm.ePictureYAxis}")
        exp = nd2.experiment
        if exp is None:
            raise RuntimeError("ND2 has no experiment metadata.")

        mp_level = exp.findLevel(ExperimentLoopType.eEtXYPosLoop)
        if mp_level is None or not isinstance(mp_level.uLoopPars, ExperimentXYPosLoop):
            raise RuntimeError("ND2 has no multipoint loop.")
        mp = mp_level.uLoopPars
        if not mp.Points:
            raise RuntimeError("Multipoint loop has no points.")

        _log("Reading dimensions.")
        dims = nd2.dimensionSizes(skipSpectralLoop=True)
        t_count = int(dims.get("t", 1))
        m_count = int(dims.get("m", mp.uiCount))
        z_count = int(dims.get("z", 1))
        _log(f"Dimensions resolved: t={t_count}, m={m_count}, z={z_count}.")
        time_index = 0
        if t_count <= time_index:
            raise RuntimeError(f"time_index {time_index} out of range.")

        _log("Building sequence lookup.")
        seq_lookup: dict[tuple[int, int, int], int] = {}
        for seq_index, idx in enumerate(exp.generateLoopIndexes(named=True)):
            seq_lookup[(idx.get("t", 0), idx.get("m", 0), idx.get("z", 0))] = seq_index
        _log(f"Sequence lookup built: {len(seq_lookup)} entries.")

        attrs = nd2.imageAttributes
        y_size, x_size, c_size = attrs.shape
        dtype = np.dtype(attrs.dtype)
        _log(f"Image shape per frame: (y={y_size}, x={x_size}, c={c_size}), dtype={dtype}.")

        xy_spacing = (
            float(nd2.pictureMetadata.dCalibration)
            if nd2.pictureMetadata is not None
            and nd2.pictureMetadata.bCalibrated
            and nd2.pictureMetadata.dCalibration > 0
            else 1.0
        )
        _log(f"XY spacing: {xy_spacing}.")
        z_spacing = 1.0
        z_loop = find_zstack(exp)
        if z_count > 1 and z_loop is not None and z_loop.step is not None and z_loop.step > 0:
            z_spacing = float(z_loop.step)
        _log(f"Z spacing: {z_spacing}.")

        dims_list = ["c", "y", "x"] if z_count == 1 else ["c", "z", "y", "x"]
        scale = {"y": xy_spacing, "x": xy_spacing}
        if z_count > 1:
            scale = {"z": z_spacing, **scale}
        _log(f"MSI dims={dims_list}, scale={scale}.")

        def _read_frame_cyx(seq_index: int) -> np.ndarray:
            arr = np.asarray(nd2.image(seq_index))
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            return np.moveaxis(arr, -1, 0)

        _log("Building msims.")
        msims = []
        for m_index in range(m_count):
            point = mp.Points[m_index]
            frames = []
            for z_index in range(z_count):
                key = (time_index, m_index, z_index)
                if key not in seq_lookup:
                    raise RuntimeError(f"Missing frame for indexes {key}.")
                seq_idx = seq_lookup[key]
                frm = da.from_delayed(
                    delayed(_read_frame_cyx)(seq_idx),
                    shape=(c_size, y_size, x_size),
                    dtype=dtype,
                )
                frames.append(frm)

            tile_array = frames[0] if z_count == 1 else da.stack(frames, axis=1)
            translation = {"y": float(point.dPosY), "x": float(point.dPosX)}
            if z_count > 1:
                translation["z"] = float(point.dPosZ)

            sim = si_utils.get_sim_from_array(
                tile_array,
                dims=dims_list,
                scale=scale,
                translation=translation,
                transform_key="stage_metadata",
            )
            msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
            if (m_index + 1) % 100 == 0 or m_index == m_count - 1:
                _log(f"Built msims: {m_index + 1}/{m_count}")

    _log("Calling vis_utils.plot_positions.")
    plt.figure(figsize=PLOT_SIZE_INCH)
    fig, _ = vis_utils.plot_positions(
        msims, transform_key="stage_metadata", use_positional_colors=False
    )
    _log("plot_positions finished. Saving figure.")
    fig.savefig(positions_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved positions plot: {positions_png}")

    if DO_STITCH:
        perf_rows = []
        combo_id = 0
        for output_mode in OUTPUT_MODES:
            if output_mode not in {"nd2", "zarr"}:
                raise ValueError(
                    f"Unsupported output mode {output_mode!r}. Use 'nd2' and/or 'zarr'."
                )
            for output_chunksize in STITCH_OUTPUT_CHUNKSIZES:
                chunk_label = _chunksize_label(output_chunksize)
                for batch_size in STITCH_BATCH_SIZES:
                    for dask_scheduler in STITCH_DASK_SCHEDULERS:
                        for dask_workers in STITCH_DASK_NUM_WORKERS_LIST:
                            for batch_exec in available_batch_exec_presets:
                                for transform_mode in STAGE_TRANSFORM_MODES:
                                    transform_slug = _sanitize_for_filename(transform_mode)
                                    for preset in STITCH_BLEND_PRESETS:
                                        combo_id += 1
                                        preset_slug = _sanitize_for_filename(preset)
                                        batch_exec_slug = _sanitize_for_filename(batch_exec)
                                        batch_options = None
                                        batch_opt_error = None
                                        try:
                                            batch_options = _batch_exec_options(
                                                batch_exec,
                                                n_batch=batch_size,
                                                workers=dask_workers,
                                            )
                                        except Exception as exc:
                                            batch_opt_error = f"{type(exc).__name__}: {exc}"

                                        stitch_kwargs = dict(
                                            time_index=0,
                                            register=False,  # set True for refinement if needed (slower on large tile counts)
                                            verbose=True,
                                            log_every_tiles=100,
                                            show_dask_progress=True,
                                            downsample_level=STITCH_DOWNSAMPLE_LEVEL,
                                            output_chunksize=dict(output_chunksize),
                                            fusion_batch_size=int(batch_size),
                                            stage_transform_mode=transform_mode,
                                            dask_scheduler=dask_scheduler,
                                            dask_num_workers=int(dask_workers),
                                            use_ray_batches=False,
                                            ray_num_cpus=STITCH_RAY_NUM_CPUS,
                                            zarr_options=STITCH_ZARR_OPTIONS,
                                            batch_options=batch_options,
                                        )
                                        stitch_kwargs.update(_blend_preset_kwargs(preset))

                                        output = nd2_path.with_name(
                                            f"{nd2_path.stem}_stitched_{combo_id:04d}_{transform_slug}_{preset_slug}_{chunk_label}.ome.zarr"
                                        )
                                        output_nd2 = nd2_path.with_name(
                                            f"{nd2_path.stem} - {combo_id:04d}_{transform_slug}_{preset_slug}_{chunk_label}.nd2"
                                        )
                                        tag = (
                                            f"{combo_id:04d}|{output_mode}|{transform_mode}|{preset}|{chunk_label}"
                                            f"|nb{batch_size}|{dask_scheduler}|w{dask_workers}|{batch_exec_slug}"
                                        )
                                        _log(f"[{tag}] Output (zarr): {output}")
                                        _log(f"[{tag}] Output (nd2 tiled): {output_nd2}")

                                        row = {
                                            "combo_id": combo_id,
                                            "output_mode": output_mode,
                                            "preset": preset,
                                            "transform_mode": transform_mode,
                                            "chunksize": chunk_label,
                                            "batch_size": int(batch_size),
                                            "dask_scheduler": dask_scheduler,
                                            "dask_workers": int(dask_workers),
                                            "batch_exec": batch_exec,
                                            "status": "ok",
                                            "error": "",
                                            "elapsed_s": 0.0,
                                            "elapsed_min": 0.0,
                                            "output": str(output_nd2 if output_mode == "nd2" else output),
                                        }
                                        if batch_opt_error is not None:
                                            row["status"] = "error"
                                            row["error"] = batch_opt_error
                                            perf_rows.append(row)
                                            _log(f"[{tag}] Skipped due to batch option error: {batch_opt_error}")
                                            continue

                                        t0 = time.perf_counter()
                                        _log(f"[{tag}] Started at {dt.datetime.now().strftime('%H:%M:%S')}.")
                                        try:
                                            if output_mode == "nd2":
                                                if CLEAN_OUTPUT_BEFORE_STITCH and output_nd2.exists():
                                                    _log(f"[{tag}] Removing existing output file: {output_nd2}")
                                                    output_nd2.unlink()
                                                _log(f"[{tag}] Starting stitch_to_nd2_tiled.")
                                                out_nd2 = limnd2.stitch_to_nd2_tiled(
                                                    nd2_path,
                                                    output_nd2,
                                                    tile_size=ND2_TILE_SIZE,
                                                    stream_all_timepoints=ND2_STREAM_ALL_TIMEPOINTS,
                                                    **stitch_kwargs,
                                                )
                                                _log(f"[{tag}] Done. Wrote tiled ND2: {out_nd2}")
                                            else:
                                                if CLEAN_OUTPUT_BEFORE_STITCH and output.exists():
                                                    _log(f"[{tag}] Removing existing output folder: {output}")
                                                    shutil.rmtree(output)
                                                _log(f"[{tag}] Starting stitch.")
                                                fused = limnd2.stitch(
                                                    nd2_path,
                                                    output_filename=output,
                                                    **stitch_kwargs,
                                                )
                                                shape = getattr(fused, "shape", None)
                                                if shape is None and hasattr(fused, "data"):
                                                    shape = getattr(fused.data, "shape", None)
                                                _log(f"[{tag}] Done. Fused shape: {shape}")
                                        except Exception as exc:
                                            row["status"] = "error"
                                            row["error"] = f"{type(exc).__name__}: {exc}"
                                            _log(f"[{tag}] ERROR: {row['error']}")

                                        elapsed_s = time.perf_counter() - t0
                                        row["elapsed_s"] = elapsed_s
                                        row["elapsed_min"] = elapsed_s / 60.0
                                        perf_rows.append(row)
                                        _log(
                                            f"[{tag}] Runtime: {elapsed_s:.2f} s ({elapsed_s / 60.0:.2f} min)."
                                        )

        if perf_rows:
            _log("Runtime summary:")
            for row in perf_rows:
                _log(
                    f"  id={row['combo_id']:04d}, status={row['status']}, "
                    f"mode={row['output_mode']}, preset={row['preset']}, transform={row['transform_mode']}, "
                    f"chunksize={row['chunksize']}, nbatch={row['batch_size']}, "
                    f"dask={row['dask_scheduler']}/{row['dask_workers']}, batch_exec={row['batch_exec']}, "
                    f"time={row['elapsed_s']:.2f}s ({row['elapsed_min']:.2f}m), output={row['output']}"
                )
                if row["error"]:
                    _log(f"    error={row['error']}")

            ok_rows = [r for r in perf_rows if r["status"] == "ok"]
            if ok_rows:
                fastest = sorted(ok_rows, key=lambda r: r["elapsed_s"])[:10]
                _log("Top fastest successful configs:")
                for row in fastest:
                    _log(
                        f"  id={row['combo_id']:04d}, time={row['elapsed_s']:.2f}s, "
                        f"mode={row['output_mode']}, transform={row['transform_mode']}, preset={row['preset']}, "
                        f"chunksize={row['chunksize']}, nbatch={row['batch_size']}, "
                        f"dask={row['dask_scheduler']}/{row['dask_workers']}, batch_exec={row['batch_exec']}"
                    )
            with open(times_log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"\n=== run {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                fh.write(f"input={nd2_path}\n")
                for row in perf_rows:
                    fh.write(
                        "id={combo_id}, status={status}, mode={mode}, preset={preset}, transform={transform}, "
                        "chunksize={chunksize}, batch_size={batch_size}, dask_scheduler={dask_scheduler}, "
                        "dask_workers={dask_workers}, batch_exec={batch_exec}, "
                        "time_s={time_s:.2f}, time_min={time_min:.2f}, output={output}, error={error}\n".format(
                            combo_id=row["combo_id"],
                            status=row["status"],
                            mode=row["output_mode"],
                            preset=row["preset"],
                            transform=row["transform_mode"],
                            chunksize=row["chunksize"],
                            batch_size=row["batch_size"],
                            dask_scheduler=row["dask_scheduler"],
                            dask_workers=row["dask_workers"],
                            batch_exec=row["batch_exec"],
                            time_s=row["elapsed_s"],
                            time_min=row["elapsed_min"],
                            output=row["output"],
                            error=row["error"],
                        )
                    )
            _log(f"Saved runtime summary to: {times_log_path}")
    _log("Finished tst3.")
    return 0


def _run_one_input(path: str) -> tuple[str, int, str]:
    try:
        rc = main(path)
        return (path, rc, "")
    except Exception as exc:
        return (path, 1, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    if INPUT_ND2_SWEEP:
        if PARALLELIZE_INPUT_SWEEP:
            max_workers = max(1, int(INPUT_SWEEP_MAX_WORKERS))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_run_one_input, p): p for p in INPUT_ND2_SWEEP}
                failed = False
                for fut in as_completed(futures):
                    path, rc, err = fut.result()
                    if rc != 0:
                        failed = True
                        print(f"[SWEEP ERROR] input={path} rc={rc} error={err}", flush=True)
                raise SystemExit(1 if failed else 0)
        for sweep_path in INPUT_ND2_SWEEP:
            rc = main(sweep_path)
            if rc != 0:
                raise SystemExit(rc)
        raise SystemExit(0)
    raise SystemExit(main())
