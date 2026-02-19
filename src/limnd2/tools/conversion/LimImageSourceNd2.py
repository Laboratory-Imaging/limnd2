from __future__ import annotations

from contextlib import contextmanager
import itertools
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import limnd2
from limnd2.attributes import ImageAttributes
from limnd2.metadata_factory import MetadataFactory, Plane

from .LimImageSource import LimImageSource, merge_four_fields
from .LimConvertUtils import ConvertSequenceArgs, ConversionSettings

_LOOP_NAME_MAP = {
    "t": "timeloop",
    "m": "multipoint",
    "z": "zstack",
    "s": "channel",
}


class LimImageSourceNd2(LimImageSource):
    """Image source reading from ND2 (supports ROI reads)."""

    def __init__(self, filename: str | Path, seq_index: int = 0, channel_index: int | None = None):
        super().__init__(filename)
        self.seq_index = int(seq_index)
        self.channel_index = channel_index

    def __repr__(self):
        return (
            f"LimImageSourceNd2({self.filename}, seq_index={self.seq_index}, "
            f"channel_index={self.channel_index})"
        )

    @property
    def supports_tile_read(self) -> bool:
        return True

    def _select_channel(self, frame: np.ndarray) -> np.ndarray:
        debug = os.getenv("LIMND2_DEBUG_CHANNEL_SELECT") == "1"
        if debug:
            print(
                f"[LimImageSourceNd2] _select_channel: frame.shape={getattr(frame, 'shape', None)} "
                f"ndim={getattr(frame, 'ndim', None)} channel_index={self.channel_index}",
                flush=True,
            )

        idx = self.channel_index
        if idx is None:
            return frame
        if idx < 0:
            raise ValueError(
                f"channel_index must be non-negative, got {idx}."
            )

        if frame.ndim == 2:
            # No channel axis available, return as-is.
            return frame
        if frame.ndim == 3:
            channels = frame.shape[-1]
            if channels <= 0:
                return frame
            if not (0 <= idx < channels):
                if debug:
                    print(
                        f"[LimImageSourceNd2] channel_index {idx} out of bounds for {frame.shape}; "
                        f"clamping to [0, {channels - 1}]",
                        flush=True,
                    )
                idx = min(max(idx, 0), channels - 1)
            return frame[..., idx]
        raise ValueError(f"Unsupported ND2 frame ndim={frame.ndim} for shape {frame.shape}.")

    def read(self) -> np.ndarray:
        with limnd2.Nd2Reader(self.filename) as nd2:
            frame = nd2.image(self.seq_index)

            # Convert while reader is alive
            frame = np.asarray(frame)

            # Select channel while reader is alive
            frame = self._select_channel(frame)

            # Ensure the returned array owns its memory (important if frame is a view)
            return np.array(frame, copy=True)

    def read_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        with self.open_tile_reader() as r:
            return np.array(r(x, y, w, h), copy=True)

    @contextmanager
    def open_tile_reader(self):
        nd2 = limnd2.Nd2Reader(self.filename)
        attrs = nd2.imageAttributes
        height = attrs.height
        width = attrs.width
        components = attrs.componentCount
        try:
            def _read_tile(x: int, y: int, w: int, h: int) -> np.ndarray:
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + int(w), y0 + int(h)

                xx0, xx1 = max(0, x0), min(width, x1)
                yy0, yy1 = max(0, y0), min(height, y1)

                if xx0 >= xx1 or yy0 >= yy1:
                    frame = np.empty((0, 0, components), dtype=attrs.dtype)
                else:
                    rect = (xx0, yy0, xx1 - xx0, yy1 - yy0)
                    frame = nd2.image(self.seq_index, rect=rect)
                return np.asarray(self._select_channel(frame))

            yield _read_tile
        finally:
            nd2.finalize()

    def _nd2_dimensions(self) -> list[tuple[str, int]]:
        with limnd2.Nd2Reader(self.filename) as nd2:
            dims: list[tuple[str, int]] = []
            channel_experiment_size = 1

            exp = nd2.experiment
            if exp is not None:
                names = exp.dimnames(skipSpectralLoop=False)
                shape = exp.shape(skipSpectralLoop=False)
                mapped_sizes: dict[str, int] = {}
                for name, size in zip(names, shape):
                    full = _LOOP_NAME_MAP.get(name, "unknown")
                    if full == "channel":
                        channel_experiment_size *= size
                    if full in mapped_sizes:
                        mapped_sizes[full] *= size
                    else:
                        mapped_sizes[full] = size
                dims.extend(
                    (full, size) for full, size in mapped_sizes.items() if size > 1
                )
            else:
                frame_count = nd2.imageAttributes.frameCount
                if frame_count > 1:
                    dims.append(("unknown", frame_count))

            if (
                not nd2.pictureMetadata.isRgb
                and nd2.imageAttributes.componentCount > 1
                and channel_experiment_size <= 1
            ):
                dims.append(("channel", nd2.imageAttributes.componentCount))

            return dims

    def get_file_dimensions(self) -> dict[str, int]:
        return dict(self._nd2_dimensions())

    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            with limnd2.Nd2Reader(self.filename) as nd2:
                self._is_rgb = nd2.pictureMetadata.isRgb
        return self._is_rgb

    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSource", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSource", list[int]], dict[str, int]]:
        debug_flatten = os.getenv("LIMND2_DEBUG_FLATTEN") == "1"

        dims_list = self._nd2_dimensions()
        if not dims_list:
            return sources, original_dimensions

        dims = dict(dims_list)
        kept_fulls = set(dims.keys())

        with limnd2.Nd2Reader(self.filename) as nd2:
            exp = nd2.experiment
            channel_in_experiment = False
            if exp is not None:
                raw_names = exp.dimnames(skipSpectralLoop=False)
                raw_shape = exp.shape(skipSpectralLoop=False)
                channel_experiment_size = 1
                for raw_name, size in zip(raw_names, raw_shape):
                    if _LOOP_NAME_MAP.get(raw_name, "unknown") == "channel":
                        channel_experiment_size *= size
                channel_in_experiment = channel_experiment_size > 1
                if raw_shape:
                    loop_indexes = list(
                        itertools.product(*[range(dim) for dim in raw_shape])
                    )
                else:
                    loop_indexes = [()]
                exp_index_map = {tuple(idx): i for i, idx in enumerate(loop_indexes)}

                combined_order: list[str] = []
                group_indices: dict[str, list[int]] = {}
                group_shapes: dict[str, list[int]] = {}
                for i, (raw_name, size) in enumerate(zip(raw_names, raw_shape)):
                    full = _LOOP_NAME_MAP.get(raw_name, "unknown")
                    if full == "channel" and not channel_in_experiment:
                        continue
                    if full not in kept_fulls:
                        continue
                    if full not in group_indices:
                        combined_order.append(full)
                        group_indices[full] = []
                        group_shapes[full] = []
                    group_indices[full].append(i)
                    group_shapes[full].append(size)

                exp_sizes = []
                for full in combined_order:
                    prod = 1
                    for s in group_shapes[full]:
                        prod *= s
                    exp_sizes.append(prod)

                exp_index_tuples = list(np.ndindex(*exp_sizes)) if exp_sizes else [()]

                def combined_to_raw(exp_idx: tuple[int, ...]) -> tuple[int, ...]:
                    raw_idx = [0] * len(raw_names)
                    for combined_pos, full in enumerate(combined_order):
                        combined_value = exp_idx[combined_pos] if exp_idx else 0
                        sizes = group_shapes[full]
                        indices = group_indices[full]
                        if len(sizes) == 1:
                            raw_idx[indices[0]] = combined_value
                            continue
                        remaining = combined_value
                        decoded = [0] * len(sizes)
                        for rev_i, size in enumerate(reversed(sizes)):
                            decoded[-(rev_i + 1)] = remaining % size
                            remaining //= size
                        for idx, val in zip(indices, decoded):
                            raw_idx[idx] = val
                    return tuple(raw_idx)
            else:
                combined_order = [name for name, _ in dims_list if name != "channel"]
                exp_sizes = [size for name, size in dims_list if name != "channel"]
                exp_index_tuples = list(np.ndindex(*exp_sizes)) if exp_sizes else [()]
                exp_index_map = {}

                def combined_to_raw(exp_idx: tuple[int, ...]) -> tuple[int, ...]:
                    return ()

        new_files: dict[LimImageSource, list[int]] = {}

        if respect_per_file_channel_count:
            target_order_all = list(dims.keys())
            target_order_without_channel = [dim for dim in target_order_all if dim != "channel"]
            max_sizes = {dim: int(size) for dim, size in dims.items()}
            if debug_flatten:
                print(
                    f"[LIMND2_DEBUG_FLATTEN] ND2 per-file expansion start: "
                    f"base_dims={dims}, target_order={target_order_all}, sources={len(sources)}",
                    flush=True,
                )

            for file, base_dims in sources.items():
                file_path = Path(file.filename)

                with limnd2.Nd2Reader(file_path) as src_nd2:
                    src_exp = src_nd2.experiment

                    local_channel_in_experiment = False
                    local_combined_order: list[str] = []
                    local_exp_index_map: dict[tuple[int, ...], int] = {}
                    local_dim_sizes: dict[str, int] = {}
                    use_linear_sequence_index = False

                    if src_exp is not None:
                        src_raw_names = src_exp.dimnames(skipSpectralLoop=False)
                        src_raw_shape = src_exp.shape(skipSpectralLoop=False)
                        suppress_channel_from_sequence = (
                            src_nd2.imageAttributes.componentCount > 1
                            and not src_nd2.pictureMetadata.isRgb
                        )
                        channel_experiment_size = 1
                        for raw_name, size in zip(src_raw_names, src_raw_shape):
                            if _LOOP_NAME_MAP.get(raw_name, "unknown") == "channel":
                                channel_experiment_size *= size
                        local_channel_in_experiment = (
                            channel_experiment_size > 1 and not suppress_channel_from_sequence
                        )
                        use_linear_sequence_index = suppress_channel_from_sequence

                        if src_raw_shape:
                            if suppress_channel_from_sequence:
                                seq_shape = [
                                    size
                                    for raw_name, size in zip(src_raw_names, src_raw_shape)
                                    if _LOOP_NAME_MAP.get(raw_name, "unknown") != "channel"
                                ]
                                src_loop_indexes = (
                                    list(itertools.product(*[range(dim) for dim in seq_shape]))
                                    if seq_shape
                                    else [()]
                                )
                            else:
                                src_loop_indexes = list(
                                    itertools.product(*[range(dim) for dim in src_raw_shape])
                                )
                        else:
                            src_loop_indexes = [()]
                        local_exp_index_map = {
                            tuple(idx): i for i, idx in enumerate(src_loop_indexes)
                        }

                        group_indices: dict[str, list[int]] = {}
                        group_shapes: dict[str, list[int]] = {}
                        for i, (raw_name, size) in enumerate(zip(src_raw_names, src_raw_shape)):
                            full = _LOOP_NAME_MAP.get(raw_name, "unknown")
                            if full == "channel" and not local_channel_in_experiment:
                                continue
                            if full not in kept_fulls:
                                continue
                            if full not in group_indices:
                                local_combined_order.append(full)
                                group_indices[full] = []
                                group_shapes[full] = []
                            group_indices[full].append(i)
                            group_shapes[full].append(size)

                        local_exp_sizes: list[int] = []
                        for full in local_combined_order:
                            prod = 1
                            for s in group_shapes[full]:
                                prod *= s
                            local_exp_sizes.append(prod)
                            local_dim_sizes[full] = prod

                        local_exp_index_tuples = (
                            list(np.ndindex(*local_exp_sizes)) if local_exp_sizes else [()]
                        )

                        def local_combined_to_raw(exp_idx: tuple[int, ...]) -> tuple[int, ...]:
                            raw_idx = [0] * len(src_raw_names)
                            for combined_pos, full in enumerate(local_combined_order):
                                combined_value = exp_idx[combined_pos] if exp_idx else 0
                                sizes = group_shapes[full]
                                indices = group_indices[full]
                                if len(sizes) == 1:
                                    raw_idx[indices[0]] = combined_value
                                    continue
                                remaining = combined_value
                                decoded = [0] * len(sizes)
                                for rev_i, size in enumerate(reversed(sizes)):
                                    decoded[-(rev_i + 1)] = remaining % size
                                    remaining //= size
                                for idx, val in zip(indices, decoded):
                                    raw_idx[idx] = val
                            return tuple(raw_idx)
                    else:
                        frame_count = int(src_nd2.imageAttributes.frameCount)
                        if "unknown" in kept_fulls and frame_count > 1:
                            local_combined_order = ["unknown"]
                            local_dim_sizes["unknown"] = frame_count
                        local_exp_sizes = [
                            local_dim_sizes[name] for name in local_combined_order
                        ]
                        local_exp_index_tuples = (
                            list(np.ndindex(*local_exp_sizes)) if local_exp_sizes else [()]
                        )

                        def local_combined_to_raw(exp_idx: tuple[int, ...]) -> tuple[int, ...]:
                            return ()

                    local_channel_size = int(dims.get("channel", 1))
                    if "channel" in dims and not local_channel_in_experiment:
                        local_channel_size = int(src_nd2.imageAttributes.componentCount)
                        if src_nd2.pictureMetadata.isRgb:
                            local_channel_size = 1
                        local_channel_size = max(local_channel_size, 1)
                        local_dim_sizes["channel"] = local_channel_size

                if debug_flatten:
                    print(
                        f"[LIMND2_DEBUG_FLATTEN] file={file_path.name} "
                        f"local_dim_sizes={local_dim_sizes} "
                        f"local_order={local_combined_order} "
                        f"local_exp_count={len(local_exp_index_tuples)} "
                        f"local_channel_in_exp={local_channel_in_experiment}",
                        flush=True,
                    )

                for dim, size in local_dim_sizes.items():
                    if dim in max_sizes and size > max_sizes[dim]:
                        max_sizes[dim] = int(size)

                for local_linear_idx, local_exp_idx in enumerate(local_exp_index_tuples):
                    if use_linear_sequence_index:
                        seq_index = local_linear_idx
                    else:
                        raw_idx = local_combined_to_raw(tuple(local_exp_idx))
                        seq_index = local_exp_index_map.get(raw_idx)
                        if seq_index is None:
                            fallback_seq = local_exp_idx[0] if local_exp_idx else 0
                            if debug_flatten:
                                print(
                                    f"[LIMND2_DEBUG_FLATTEN] raw index miss for {file_path.name}: "
                                    f"raw_idx={raw_idx}, local_exp_idx={local_exp_idx}, fallback_seq={fallback_seq}",
                                    flush=True,
                                )
                            seq_index = fallback_seq

                    local_idx_map = {
                        dim: int(local_exp_idx[pos])
                        for pos, dim in enumerate(local_combined_order)
                    }
                    ordered_all = [local_idx_map.get(dim, 0) for dim in target_order_all]
                    ordered_without_channel = [
                        local_idx_map.get(dim, 0) for dim in target_order_without_channel
                    ]

                    if "channel" in dims and not local_channel_in_experiment:
                        for c in range(local_channel_size):
                            file_copy = type(file)(
                                file.filename, seq_index=seq_index, channel_index=c
                            )
                            new_files[file_copy] = (
                                base_dims + ordered_without_channel + [c]
                            )
                    else:
                        file_copy = type(file)(file.filename, seq_index=seq_index)
                        new_files[file_copy] = base_dims + ordered_all

            for dim, size in max_sizes.items():
                dims[dim] = int(size)
            if debug_flatten:
                print(
                    f"[LIMND2_DEBUG_FLATTEN] ND2 per-file expansion done: "
                    f"new_files={len(new_files)}, final_dims={dims}",
                    flush=True,
                )

        elif "channel" in dims and not channel_in_experiment:
            channel_size = int(dims["channel"])
            per_file_channel_size: dict[LimImageSource, int] = {}
            channel_count_cache: dict[Path, int] = {}

            if preserve_duplicate_dimension_names:
                max_channel_size = channel_size
                for file in sources:
                    file_path = Path(file.filename)
                    if file_path not in channel_count_cache:
                        try:
                            with limnd2.Nd2Reader(file_path) as src_nd2:
                                comp_count = int(src_nd2.imageAttributes.componentCount)
                                if src_nd2.pictureMetadata.isRgb:
                                    comp_count = 1
                                channel_count_cache[file_path] = max(comp_count, 1)
                        except Exception:
                            channel_count_cache[file_path] = channel_size

                    file_channel_size = channel_count_cache[file_path]
                    per_file_channel_size[file] = file_channel_size
                    if file_channel_size > max_channel_size:
                        max_channel_size = file_channel_size

                dims["channel"] = max_channel_size

            for file, base_dims in sources.items():
                file_channel_size = per_file_channel_size.get(file, channel_size)
                for exp_idx in exp_index_tuples:
                    raw_idx = combined_to_raw(tuple(exp_idx))
                    seq_index = exp_index_map.get(raw_idx, exp_idx[0] if exp_idx else 0)
                    for c in range(file_channel_size):
                        file_copy = type(file)(
                            file.filename, seq_index=seq_index, channel_index=c
                        )
                        new_files[file_copy] = base_dims + list(exp_idx) + [c]
        else:
            for file, base_dims in sources.items():
                for exp_idx in exp_index_tuples:
                    raw_idx = combined_to_raw(tuple(exp_idx))
                    seq_index = exp_index_map.get(raw_idx, exp_idx[0] if exp_idx else 0)
                    file_copy = type(file)(file.filename, seq_index=seq_index)
                    new_files[file_copy] = base_dims + list(exp_idx)

        new_dims = original_dimensions.copy()
        for dim, size in dims.items():
            target_dim = dim
            if preserve_duplicate_dimension_names and target_dim in new_dims:
                suffix_index = 2
                while f"{dim}__dup{suffix_index}" in new_dims:
                    suffix_index += 1
                target_dim = f"{dim}__dup{suffix_index}"
            new_dims[target_dim] = size

        if new_dims.get("unknown", 0) > 1:
            if unknown_dimension_type is not None:
                target_dim = unknown_dimension_type
                if target_dim in new_dims and target_dim != "unknown":
                    suffix_index = 2
                    while f"{target_dim}__dup{suffix_index}" in new_dims:
                        suffix_index += 1
                    target_dim = f"{target_dim}__dup{suffix_index}"

                remapped_dims: dict[str, int] = {}
                for dim_name, size in new_dims.items():
                    if dim_name == "unknown":
                        remapped_dims[target_dim] = size
                    else:
                        remapped_dims[dim_name] = size
                new_dims = remapped_dims

        return new_files, new_dims

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        try:
            with limnd2.Nd2Reader(self.filename) as nd2:
                meta = nd2.pictureMetadata
                exp = nd2.experiment
                attrs = nd2.imageAttributes
        except Exception:
            return

        # Use ConvertSequenceArgs as a "sparse" partial B
        b = ConvertSequenceArgs()

        # --- time / z step from experiment ---
        try:
            from limnd2.experiment import canonical_experiment
            t, _, z = canonical_experiment(exp)
            if t and t.uLoopPars.step is not None and t.uLoopPars.step > 0:
                b.time_step = float(t.uLoopPars.step)
            if z and z.uLoopPars.step is not None and z.uLoopPars.step > 0:
                b.z_step = float(z.uLoopPars.step)
        except Exception:
            pass

        # --- metadata + channels from PictureMetadata ---
        pixel_cal = meta.dCalibration if meta.bCalibrated else -1.0

        def _pick_positive(*vals):
            for v in vals:
                try:
                    if v is not None and float(v) > 0:
                        return float(v)
                except Exception:
                    continue
            return None

        factory_kwargs = {}
        obj_mag = _pick_positive(meta.objectiveMagnification(), meta.dObjectiveMag)
        if obj_mag is not None:
            factory_kwargs["objective_magnification"] = obj_mag
        obj_na = _pick_positive(meta.objectiveNumericAperture(), meta.dObjectiveNA)
        if obj_na is not None:
            factory_kwargs["objective_numerical_aperture"] = obj_na
        refr = _pick_positive(meta.refractiveIndex(), meta.dRefractIndex1)
        if refr is not None:
            factory_kwargs["immersion_refractive_index"] = refr
        zoom = _pick_positive(meta.dZoom)
        if zoom is not None:
            factory_kwargs["zoom_magnification"] = zoom

        factory = MetadataFactory(pixel_calibration=pixel_cal, **factory_kwargs)

        planes = meta.channels
        channels: dict[int, Plane] = {}
        for idx, plane in enumerate(planes):
            name = "RGB" if plane.uiCompCount == 3 else (plane.sDescription or f"Channel {idx + 1}")

            ex = getattr(plane, "excitationWavelengthNm", 0.0) or 0.0
            em = getattr(plane, "emissionWavelengthNm", 0.0) or 0.0

            plane_kwargs = {
                "name": name,
                "modality": plane.uiModalityMask,
                "excitation_wavelength": int(round(ex)) if ex > 0 else 0,
                "emission_wavelength": int(round(em)) if em > 0 else 0,
                "color": plane.colorAsTuple,
            }

            if plane.dPinholeDiameter > 0:
                plane_kwargs["pinhole_diameter"] = plane.dPinholeDiameter

            # per-plane microscope settings if available
            per_obj_mag = meta.objectiveMagnification(idx)
            if per_obj_mag > 0:
                plane_kwargs["objective_magnification"] = per_obj_mag
            per_obj_na = meta.objectiveNumericAperture(idx)
            if per_obj_na > 0:
                plane_kwargs["objective_numerical_aperture"] = per_obj_na
            per_refr = meta.refractiveIndex(idx)
            if per_refr > 0:
                plane_kwargs["immersion_refractive_index"] = per_refr

            cam = meta.cameraName(idx)
            if cam:
                plane_kwargs["camera_name"] = cam
            scope = meta.microscopeName(idx)
            if scope:
                plane_kwargs["microscope_name"] = scope

            plane_obj = Plane(**plane_kwargs)
            #factory.addPlane(plane_obj)
            channels[idx] = plane_obj

        b.channels = channels
        b.metadata = factory
        merge_four_fields(metadata_storage, b)

    def metadata_as_pattern_settings(self) -> dict:
        """
        Return metadata in the exact shape your QML dialog expects for patternSettings.

        Keys:
        - tstep, zstep, pixel_calibration, pinhole_diameter, objective_magnification,
            objective_numerical_aperture, immersion_refractive_index, zoom_magnification : strings
        - channels: list of [nameFromFile, customName, modalityString, ex, em, colorString]

        Any missing value is returned as an empty string "".
        """

        # Helper maps to your QML lists
        MODALITIES = [
            "Undefined", "Wide-field", "Brightfield", "Phase", "DIC", "DarkField",
            "MC", "TIRF", "Confocal, Fluo", "Confocal, Trans", "Multi-Photon",
            "SFC pinhole", "SFC slit", "Spinning Disc", "DSD", "NSIM", "iSim",
            "RCM", "CSU W1-SoRa", "NSPARC"
        ]
        def _color_to_hex(rgb):
            if not rgb or len(rgb) < 3:
                return ""
            r, g, b = rgb[:3]

            def _norm(v):
                try:
                    v = float(v)
                except Exception:
                    return None
                if 0.0 <= v <= 1.0:
                    v = round(v * 255.0)
                else:
                    v = round(v)
                return int(min(255, max(0, v)))

            r = _norm(r)
            g = _norm(g)
            b = _norm(b)
            if r is None or g is None or b is None:
                return ""
            return f"#{r:02X}{g:02X}{b:02X}"

        def _fmt_number(val: float | None) -> str:
            if val is None or val <= 0:
                return ""
            return f"{float(val)}"

        def _fmt_step(val: float | None) -> str:
            if val is None or val <= 0:
                return ""
            return f"{float(val):.3f}"

        # Defaults for the final shape
        result = {
            "tstep": "",
            "zstep": "",
            "pixel_calibration": "",
            "pinhole_diameter": "",
            "objective_magnification": "",
            "objective_numerical_aperture": "",
            "immersion_refractive_index": "",
            "zoom_magnification": "",
            "channels": []
        }

        try:
            with limnd2.Nd2Reader(self.filename) as nd2:
                meta = nd2.pictureMetadata
                exp = nd2.experiment
        except Exception:
            return {}

        # --- tstep / zstep ---
        try:
            from limnd2.experiment import canonical_experiment
            t, _, z = canonical_experiment(exp)
            tstep_ms = t.uLoopPars.step if t and t.uLoopPars.step is not None else None
            zstep_um = z.uLoopPars.step if z and z.uLoopPars.step is not None else None
        except Exception:
            tstep_ms = None
            zstep_um = None

        result["tstep"] = _fmt_step(tstep_ms)
        result["zstep"] = _fmt_step(zstep_um)

        # --- pixel size / NA / RI / zoom ---
        pixel_cal = meta.dCalibration if meta.bCalibrated else -1.0
        result["pixel_calibration"] = _fmt_number(pixel_cal)

        obj_mag = meta.objectiveMagnification()
        if obj_mag <= 0:
            obj_mag = meta.dObjectiveMag
        result["objective_magnification"] = _fmt_number(obj_mag)

        obj_na = meta.objectiveNumericAperture()
        if obj_na <= 0:
            obj_na = meta.dObjectiveNA
        result["objective_numerical_aperture"] = _fmt_number(obj_na)

        refr_index = meta.refractiveIndex()
        if refr_index <= 0:
            refr_index = meta.dRefractIndex1
        result["immersion_refractive_index"] = _fmt_number(refr_index)

        result["zoom_magnification"] = _fmt_number(meta.dZoom)

        # --- pinhole diameter (first available plane) ---
        pinhole = None
        pinhole_vals = []
        for plane in meta.channels:
            val = getattr(plane, "dPinholeDiameter", -1.0)
            if val is None or val <= 0:
                pinhole_vals = []
                break
            pinhole_vals.append(float(val))
        if pinhole_vals and (max(pinhole_vals) - min(pinhole_vals) <= 1e-6):
            pinhole = pinhole_vals[0]
        result["pinhole_diameter"] = _fmt_number(pinhole)

        # --- Channels ---
        from limnd2.metadata import PicturePlaneModalityFlags
        modality_map = PicturePlaneModalityFlags.modality_string_map()
        if "Multi-Photon" not in modality_map and "Multi-photon" in modality_map:
            modality_map["Multi-Photon"] = modality_map["Multi-photon"]
        if "MC" not in modality_map:
            modality_map["MC"] = PicturePlaneModalityFlags.modNAMC

        def _modality_from_flags(flags: PicturePlaneModalityFlags) -> str:
            if flags == PicturePlaneModalityFlags.modUnknown:
                return "Undefined"
            for name in MODALITIES:
                req = modality_map.get(name)
                if req is None or req == PicturePlaneModalityFlags.modUnknown:
                    continue
                if (flags & req) == req:
                    return name
            return "Undefined"

        channels = []
        for idx, plane in enumerate(meta.channels):
            if plane.uiCompCount == 3:
                name_from_file = "RGB"
            else:
                name_from_file = plane.sDescription or f"Channel_{idx}"
            custom_name = name_from_file

            modality = _modality_from_flags(plane.uiModalityMask)
            if modality not in MODALITIES:
                modality = "Undefined"

            ex = getattr(plane, "excitationWavelengthNm", 0.0)
            em = getattr(plane, "emissionWavelengthNm", 0.0)
            ex_str = str(int(round(ex))) if ex and ex > 0 else "0"
            em_str = str(int(round(em))) if em and em > 0 else "0"

            rgb = plane.colorAsTuple if hasattr(plane, "colorAsTuple") else None
            color_name = _color_to_hex(rgb)

            channels.append([name_from_file, custom_name, modality, ex_str, em_str, color_name])

        result["channels"] = channels
        return result

    def nd2_attributes(self, *, sequence_count=1) -> ImageAttributes:
        with limnd2.Nd2Reader(self.filename) as nd2:
            attrs = nd2.imageAttributes
        return replace(attrs, uiSequenceCount=sequence_count)
