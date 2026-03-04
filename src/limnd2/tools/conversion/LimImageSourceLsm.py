from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np

from limnd2.metadata_factory import MetadataFactory, Plane

from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs, logprint
from .LimImageSource import LimImageSource, merge_four_fields
from .LimImageSourceTiff_base import LimImageSourceTiffBase, _normalize_to_yxs, tifffile


class LimImageSourceLsm(LimImageSourceTiffBase):
    """TIFF-based reader for Zeiss LSM files."""
    _IDF_RANGE_WARNING_PRINTED = False

    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            with tifffile.TiffReader(self.filename) as tiff:
                page = tiff.pages[0]
                pm_name = page.photometric.name
                if pm_name not in ("RGB", "YCBCR"):
                    self._is_rgb = False
                else:
                    samples_per_pixel = getattr(page, "samplesperpixel", None)
                    self._is_rgb = bool(samples_per_pixel is not None and int(samples_per_pixel) in (3, 4))
        return self._is_rgb

    def _calculate_additional_information(self) -> dict:
        grouped_axis_categories = {
            ("multipoint", "m"): "RPMJK",
            ("timeloop", "t"): "TVL",
            ("zstack", "z"): "ZA",
            ("channel", "c"): "CSEH",
            ("unknown", "unknown"): "IQ",
        }
        dimensional_axis = "XY"
        axis_category_by_letter = {
            letter: (full, short)
            for (full, short), letters in grouped_axis_categories.items()
            for letter in letters
        }

        result = copy.deepcopy(LimImageSource.DEFAULT_INFORMATION_TEMPLATE)

        with tifffile.TiffReader(self.filename) as tiff:
            _ = self.is_rgb

            if not tiff.series:
                return result

            primary = tiff.series[0]
            if len(tiff.series) > 1:
                candidates = [s for s in tiff.series if len(s.shape) >= 2 and len(s.pages) > 0]
                if candidates:
                    primary = max(
                        candidates,
                        key=lambda s: (int(s.shape[-2]) * int(s.shape[-1]), len(s.pages)),
                    )
                logprint(
                    "LSM file has multiple TIFF series. Using primary-resolution series for conversion.",
                    type="warning",
                )

            self._idf_page_keys = [
                int(getattr(page, "index", idx))
                for idx, page in enumerate(primary.pages)
            ] or None

            for axis, size in zip(primary.axes, primary.shape):
                if axis not in axis_category_by_letter and axis not in dimensional_axis:
                    raise ValueError(f"Error: {axis} is not a known axis type.")
                if axis in dimensional_axis:
                    continue

                full_category, _ = axis_category_by_letter[axis]
                if full_category == "channel" and self.is_rgb:
                    continue
                LimImageSource.update_axis_result(result, full_category, int(size))

        return result

    def read(self) -> np.ndarray:
        page_key = self._page_key_for_idf(self.idf)
        with tifffile.TiffFile(self.filename) as tif:
            arr = np.asarray(tif.asarray(key=page_key))

        if arr.ndim == 3:
            channel_first = arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64
            channel_last = arr.shape[-1] <= 64 and arr.shape[0] > 64 and arr.shape[1] > 64

            if self.channel_index is not None:
                if channel_first:
                    ch = max(0, min(int(self.channel_index), arr.shape[0] - 1))
                    return np.asarray(arr[ch, ...])
                if channel_last:
                    ch = max(0, min(int(self.channel_index), arr.shape[-1] - 1))
                    return np.asarray(arr[..., ch])

            if channel_first:
                arr = np.moveaxis(arr, 0, -1)

        arr = _normalize_to_yxs(np.asarray(arr))
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[..., ::-1]
        return arr

    @staticmethod
    def calculate_bpc_significant(
        numpy_bits_memory: int,
        tiff_bits_memory: int | tuple,
        max_sample_value: int | tuple,
    ) -> int:
        if isinstance(tiff_bits_memory, tuple):
            original_tiff_bits_memory = tiff_bits_memory
            valid_bits = [int(v) for v in tiff_bits_memory if int(v) > 0]
            if not valid_bits:
                tiff_bits_memory = numpy_bits_memory
            elif numpy_bits_memory in valid_bits:
                tiff_bits_memory = numpy_bits_memory
            else:
                tiff_bits_memory = max(set(valid_bits), key=valid_bits.count)
                if len(set(valid_bits)) != 1:
                    logprint(
                        f"LSM BitsPerSample contains mixed values {original_tiff_bits_memory!r}. "
                        f"Using {tiff_bits_memory} bits for conversion.",
                        type="warning",
                    )

        if isinstance(max_sample_value, tuple):
            valid_max_values = [int(v) for v in max_sample_value if int(v) > 0]
            max_sample_value = max(valid_max_values) if valid_max_values else -1

        if numpy_bits_memory != tiff_bits_memory:
            logprint(
                f"LSM dtype bits ({numpy_bits_memory}) differ from BitsPerSample ({tiff_bits_memory}). "
                "Using dtype bits.",
                type="warning",
            )
            tiff_bits_memory = numpy_bits_memory

        if max_sample_value <= 0:
            return tiff_bits_memory

        bpc_significant = max_sample_value.bit_length()
        if bpc_significant <= 8 and tiff_bits_memory > 8:
            return tiff_bits_memory
        return bpc_significant

    def _lsm_page_count(self) -> int:
        try:
            with tifffile.TiffReader(self.filename) as tiff:
                if self._idf_page_keys is not None:
                    return len(self._idf_page_keys)
                if len(tiff.series) == 1:
                    return len(tiff.series[0].pages)
                return len(tiff.pages)
        except Exception:
            return 0

    def _page_key_for_idf(self, idf: int) -> int:
        idx = int(idf)
        if self._idf_page_keys is not None:
            if idx < 0:
                idx = 0
            if idx >= len(self._idf_page_keys):
                if not self._IDF_RANGE_WARNING_PRINTED:
                    self.__class__._IDF_RANGE_WARNING_PRINTED = True
                    logprint(
                        f"LSM mapped page index {idx} is out of range ({len(self._idf_page_keys)}). "
                        "Using last mapped page.",
                        type="warning",
                    )
                idx = len(self._idf_page_keys) - 1
            return int(self._idf_page_keys[idx])

        page_count = self._lsm_page_count()
        if page_count > 0 and idx >= page_count:
            if not self._IDF_RANGE_WARNING_PRINTED:
                self.__class__._IDF_RANGE_WARNING_PRINTED = True
                logprint(
                    f"LSM page index {idx} is out of range ({page_count}). Using last available page.",
                    type="warning",
                )
            idx = page_count - 1
        return max(idx, 0)

    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSourceLsm", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSourceLsm", list[int]], dict[str, int]]:
        del respect_per_file_channel_count

        new_files: dict[LimImageSourceLsm, list[int]] = {}
        new_dimension = self.get_file_dimensions()
        if not new_dimension:
            return sources, original_dimensions

        dim_names = list(new_dimension.keys())
        ranges = [range(int(size)) for size in new_dimension.values()]
        index_tuples = list(itertools.product(*ranges))
        channel_axis_index = dim_names.index("channel") if "channel" in new_dimension else None
        page_count = self._lsm_page_count()

        idf_values: list[int] = []
        if channel_axis_index is not None and page_count > 0:
            key_to_page: dict[tuple[int, ...], int] = {}
            for idx_tuple in index_tuples:
                key = tuple(v for i, v in enumerate(idx_tuple) if i != channel_axis_index)
                if key not in key_to_page:
                    next_idx = len(key_to_page)
                    if self._idf_page_keys is not None:
                        if next_idx < len(self._idf_page_keys):
                            key_to_page[key] = int(self._idf_page_keys[next_idx])
                        else:
                            key_to_page[key] = int(self._idf_page_keys[-1])
                    else:
                        key_to_page[key] = min(next_idx, max(page_count - 1, 0))
                idf_values.append(key_to_page[key])
        else:
            if self._idf_page_keys is not None and len(self._idf_page_keys) >= len(index_tuples):
                idf_values = [int(v) for v in self._idf_page_keys[: len(index_tuples)]]
            else:
                idf_values = list(range(len(index_tuples)))

        index_to_idf = [(idx_tuple, idf_values[idx]) for idx, idx_tuple in enumerate(index_tuples)]

        for file, dims in sources.items():
            for axis_values, idf in index_to_idf:
                file_copy = copy.deepcopy(file)
                file_copy.idf = int(idf)
                if channel_axis_index is not None:
                    file_copy.channel_index = int(axis_values[channel_axis_index])
                else:
                    file_copy.channel_index = None
                new_files[file_copy] = dims + list(axis_values)

        new_dims = original_dimensions.copy()
        for dim, size in new_dimension.items():
            target_dim = dim
            if preserve_duplicate_dimension_names and target_dim in new_dims:
                suffix_index = 2
                while f"{dim}__dup{suffix_index}" in new_dims:
                    suffix_index += 1
                target_dim = f"{dim}__dup{suffix_index}"
            new_dims[target_dim] = size

        if new_dims.get("unknown", 0) > 1:
            if unknown_dimension_type is None:
                unknown_dimension_type = "multipoint"
                logprint(
                    "File contains unknown LSM dimension, but no unknown dimension type was provided. "
                    "Using 'multipoint' by default.",
                    type="warning",
                )

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

    @classmethod
    def has_lsm_metadata(cls, path: str | Path) -> bool:
        try:
            with tifffile.TiffFile(path) as tif:
                meta = tif.lsm_metadata
        except Exception:
            return False
        return isinstance(meta, dict) and len(meta) > 0

    def _lsm_metadata(self) -> dict:
        try:
            with tifffile.TiffFile(self.filename) as tif:
                meta = tif.lsm_metadata
        except Exception:
            return {}
        return dict(meta) if isinstance(meta, dict) else {}

    @staticmethod
    def _to_float(value) -> float | None:
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _color_to_hex(raw_color) -> str:
        if raw_color is None:
            return ""
        if not isinstance(raw_color, (list, tuple, np.ndarray)):
            return ""
        if len(raw_color) < 3:
            return ""
        try:
            r = int(raw_color[0])
            g = int(raw_color[1])
            b = int(raw_color[2])
        except Exception:
            return ""
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _channel_wavelength_pairs(lsm_meta: dict) -> list[tuple[int, int]]:
        raw = lsm_meta.get("ChannelWavelength")
        if raw is None:
            return []

        arr = np.asarray(raw)
        if arr.size == 0:
            return []

        if arr.ndim == 1:
            arr = arr.reshape((-1, 2)) if arr.size % 2 == 0 else arr.reshape((-1, 1))

        pairs: list[tuple[int, int]] = []
        if arr.ndim == 2 and arr.shape[1] >= 2:
            for row in arr:
                try:
                    lo = float(row[0])
                    hi = float(row[1])
                except Exception:
                    pairs.append((0, 0))
                    continue
                if lo <= 0 or hi <= 0:
                    pairs.append((0, 0))
                    continue
                ex = int(round(lo))
                em = int(round(hi))
                pairs.append((ex, em))
        return pairs

    def _channels_from_metadata(self, lsm_meta: dict) -> list[dict[str, object]]:
        declared_count = int(lsm_meta.get("DimensionChannels", 0) or 0)

        channel_colors = lsm_meta.get("ChannelColors")
        color_names: list[str] = []
        colors: list[str] = []

        if isinstance(channel_colors, dict):
            raw_names = channel_colors.get("ColorNames")
            raw_colors = channel_colors.get("Colors")

            if isinstance(raw_names, list):
                color_names = [str(name).strip() for name in raw_names]
            if isinstance(raw_colors, list):
                colors = [self._color_to_hex(color) for color in raw_colors]

        wavelengths = self._channel_wavelength_pairs(lsm_meta)

        channel_count = max(
            declared_count,
            len(color_names),
            len(colors),
            len(wavelengths),
        )

        channels: list[dict[str, object]] = []
        for idx in range(channel_count):
            name = color_names[idx] if idx < len(color_names) and color_names[idx] else f"Channel {idx + 1}"
            color = colors[idx] if idx < len(colors) else ""
            ex = 0
            em = 0
            if idx < len(wavelengths):
                ex, em = wavelengths[idx]

            channels.append(
                {
                    "name": name,
                    "color": color,
                    "excitation": ex,
                    "emission": em,
                }
            )

        return channels

    @staticmethod
    def _format_float(value: float | None, digits: int | None = None) -> str:
        if value is None:
            return ""
        if digits is None:
            return f"{value:g}"
        return f"{value:.{digits}f}"

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        lsm_meta = self._lsm_metadata()
        if not lsm_meta:
            logprint("LSM metadata is not available in file header. Skipping LSM metadata import.", type="warning")
            return

        voxel_x_m = self._to_float(lsm_meta.get("VoxelSizeX"))
        voxel_z_m = self._to_float(lsm_meta.get("VoxelSizeZ"))
        time_interval_s = self._to_float(lsm_meta.get("TimeIntervall"))

        pixel_cal_um = voxel_x_m * 1_000_000.0 if voxel_x_m and voxel_x_m > 0 else None
        z_step_um = voxel_z_m * 1_000_000.0 if voxel_z_m and voxel_z_m > 0 else None
        time_step_ms = time_interval_s * 1000.0 if time_interval_s and time_interval_s > 0 else None

        channels = self._channels_from_metadata(lsm_meta)

        if pixel_cal_um is None and z_step_um is None and time_step_ms is None and not channels:
            return

        b = ConvertSequenceArgs()
        b.metadata = MetadataFactory(pixel_calibration=pixel_cal_um if pixel_cal_um is not None else -1.0)
        b.channels = {
            idx: Plane(
                name=str(channel["name"]),
                modality=0,
                excitation_wavelength=int(channel["excitation"]),
                emission_wavelength=int(channel["emission"]),
                color=str(channel["color"]),
            )
            for idx, channel in enumerate(channels)
        }
        b.time_step = time_step_ms
        b.z_step = z_step_um

        merge_four_fields(metadata_storage, b)

    def metadata_as_pattern_settings(self) -> dict:
        lsm_meta = self._lsm_metadata()
        if not lsm_meta:
            return {}

        voxel_x_m = self._to_float(lsm_meta.get("VoxelSizeX"))
        voxel_z_m = self._to_float(lsm_meta.get("VoxelSizeZ"))
        time_interval_s = self._to_float(lsm_meta.get("TimeIntervall"))

        pixel_cal_um = voxel_x_m * 1_000_000.0 if voxel_x_m and voxel_x_m > 0 else None
        z_step_um = voxel_z_m * 1_000_000.0 if voxel_z_m and voxel_z_m > 0 else None
        time_step_ms = time_interval_s * 1000.0 if time_interval_s and time_interval_s > 0 else None

        channels = self._channels_from_metadata(lsm_meta)

        return {
            "tstep": self._format_float(time_step_ms, 3),
            "zstep": self._format_float(z_step_um, 3),
            "pixel_calibration": self._format_float(pixel_cal_um),
            "pinhole_diameter": "",
            "objective_magnification": "",
            "objective_numerical_aperture": "",
            "immersion_refractive_index": "",
            "zoom_magnification": "",
            "channels": [
                [
                    str(channel["name"]),
                    str(channel["name"]),
                    "Undefined",
                    str(int(channel["excitation"])),
                    str(int(channel["emission"])),
                    str(channel["color"]),
                ]
                for channel in channels
            ],
        }
