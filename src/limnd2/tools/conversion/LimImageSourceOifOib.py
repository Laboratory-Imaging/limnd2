from __future__ import annotations

import copy
import itertools
from pathlib import Path
import re

import numpy as np

from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.metadata_factory import MetadataFactory, Plane

from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs
from .LimImageSource import LimImageSource, merge_four_fields

_OLYMPUS_HINT = '[olympus] extra not installed. Install it with `pip install "limnd2[olympus]"`.'


def _missing_convert_dependency(package: str) -> ImportError:
    msg = (
        f'Missing optional dependency "{package}" required for Olympus OIF/OIB conversion. '
        f"{_OLYMPUS_HINT}"
    )
    return ImportError(msg)


def _require_oiffile():
    try:
        from oiffile import OifFile
    except ImportError as exc:
        raise _missing_convert_dependency("oiffile") from exc
    return OifFile


OifFile = _require_oiffile()


class LimImageSourceOifOib(LimImageSource):
    """Image source reading Olympus OIF/OIB files through oiffile."""

    axis_indices: dict[str, int]

    def __init__(
        self,
        filename: str | Path,
        seq_index: int = 0,
        channel_index: int | None = None,
        axis_indices: dict[str, int] | None = None,
    ):
        super().__init__(filename)
        self.seq_index = int(seq_index)
        self.channel_index = int(channel_index) if channel_index is not None else None
        self.axis_indices = dict(axis_indices or {})

    def __repr__(self):
        return (
            f"LimImageSourceOifOib({self.filename}, seq_index={self.seq_index}, "
            f"channel_index={self.channel_index}, axis_indices={self.axis_indices})"
        )

    def _open_image(self):
        return OifFile(str(self.filename))

    def _order_shape(self) -> tuple[str, tuple[int, ...]]:
        with self._open_image() as image:
            order = str(getattr(image, "axes", ""))
            shape_obj = getattr(image, "shape", ())
            shape = tuple(int(v) for v in shape_obj)
        if len(order) != len(shape):
            raise ValueError(
                f"OIF/OIB dims mismatch for {self.filename}: order={order!r}, shape={shape!r}."
            )
        return order, shape

    def _axis_sizes(self) -> dict[str, int]:
        order, shape = self._order_shape()
        return {axis: int(size) for axis, size in zip(order, shape)}

    def _is_rgb_from_dims(self, order: str, shape: tuple[int, ...]) -> bool:
        sizes = {axis: int(size) for axis, size in zip(order, shape)}
        sample_count = int(sizes.get("S", 1))
        channel_count = int(sizes.get("C", 1))
        return sample_count in (3, 4) and channel_count <= 1

    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            order, shape = self._order_shape()
            self._is_rgb = self._is_rgb_from_dims(order, shape)
        return self._is_rgb

    @staticmethod
    def _axis_to_dimension(axis: str, is_rgb: bool) -> str | None:
        if axis in {"X", "Y", "S"}:
            return None
        if axis == "T":
            return "timeloop"
        if axis == "Z":
            return "zstack"
        if axis == "C":
            return None if is_rgb else "channel"
        return "unknown"

    def _dimensions_with_axes(self) -> list[tuple[str, str, int]]:
        order, shape = self._order_shape()
        rgb = self._is_rgb_from_dims(order, shape)

        dims: list[tuple[str, str, int]] = []
        for axis, size in zip(order, shape):
            logical = self._axis_to_dimension(axis, rgb)
            if logical is None:
                continue
            if int(size) <= 1:
                continue
            dims.append((logical, axis, int(size)))
        return dims

    def get_file_dimensions(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for logical, _axis, size in self._dimensions_with_axes():
            if logical in dims:
                dims[logical] *= int(size)
            else:
                dims[logical] = int(size)
        return dims

    def _decode_seq_axes(self, order: str, shape: tuple[int, ...]) -> dict[str, int]:
        seq_axes: list[tuple[str, int]] = []
        for axis, size in zip(order, shape):
            if axis in {"X", "Y", "S", "C"}:
                continue
            if int(size) <= 1:
                continue
            seq_axes.append((axis, int(size)))

        if not seq_axes:
            return {}

        total = 1
        for _axis, size in seq_axes:
            total *= size

        linear = max(int(self.seq_index), 0)
        if total > 0:
            linear %= total

        out: dict[str, int] = {}
        for axis, size in reversed(seq_axes):
            out[axis] = linear % size
            linear //= size
        return out

    @staticmethod
    def _clamp(value: int, size: int) -> int:
        if size <= 0:
            return 0
        return min(max(int(value), 0), size - 1)

    @staticmethod
    def _normalize_to_yx_or_yxs(frame: np.ndarray, axes: list[str], *, rgb_samples: bool) -> np.ndarray:
        if "Y" not in axes or "X" not in axes:
            raise ValueError(f"OIF/OIB frame missing Y/X axes after indexing: axes={axes}")

        cur = np.asarray(frame)
        cur_axes = list(axes)

        for idx in range(cur.ndim - 1, -1, -1):
            axis = cur_axes[idx]
            if axis in {"X", "Y"}:
                continue
            if cur.shape[idx] == 1:
                cur = np.take(cur, 0, axis=idx)
                cur_axes.pop(idx)

        if "Y" not in cur_axes or "X" not in cur_axes:
            raise ValueError(f"OIF/OIB frame lost Y/X axes during normalization: axes={cur_axes}")

        sample_axis: str | None = None
        if "S" in cur_axes and cur.shape[cur_axes.index("S")] > 1:
            sample_axis = "S"
        elif "C" in cur_axes and cur.shape[cur_axes.index("C")] > 1:
            sample_axis = "C"

        if sample_axis is None:
            order = [cur_axes.index("Y"), cur_axes.index("X")]
            out = np.transpose(cur, axes=order)
            return np.asarray(out)

        order = [cur_axes.index("Y"), cur_axes.index("X"), cur_axes.index(sample_axis)]
        out = np.transpose(cur, axes=order)

        if sample_axis == "S" and rgb_samples:
            if out.shape[-1] >= 4:
                out = out[..., :3]
            if out.shape[-1] == 3:
                out = out[..., ::-1]
            elif out.shape[-1] == 1:
                out = out[..., 0]

        return np.asarray(out)

    def read(self) -> np.ndarray:
        with self._open_image() as image:
            arr = np.asarray(image.asarray(series=0))
            order = str(getattr(image, "axes", ""))
            shape = tuple(int(v) for v in arr.shape)

            if len(order) != arr.ndim:
                try:
                    series = getattr(image, "series", ())
                    if isinstance(series, (tuple, list)) and len(series) > 0:
                        series_order = str(getattr(series[0], "axes", ""))
                        if len(series_order) == arr.ndim:
                            order = series_order
                except Exception:
                    pass

        if len(order) != arr.ndim:
            raise ValueError(
                f"OIF/OIB axes/data mismatch for {self.filename}: order={order!r}, shape={arr.shape!r}."
            )

        kwargs: dict[str, int] = {}
        seq_axes = self._decode_seq_axes(order, shape)

        for axis, size in zip(order, shape):
            axis = str(axis)
            size = int(size)
            if axis in {"X", "Y", "S"}:
                continue

            if axis == "C":
                if self.is_rgb:
                    kwargs[axis] = 0
                    continue
                if self.channel_index is not None:
                    kwargs[axis] = self._clamp(self.channel_index, size)
                else:
                    kwargs[axis] = self._clamp(int(self.axis_indices.get(axis, 0)), size)
                continue

            val = self.axis_indices.get(axis)
            if val is None:
                val = seq_axes.get(axis, 0)
            kwargs[axis] = self._clamp(int(val), size)

        index: list[int | slice] = []
        for axis, _size in zip(order, shape):
            if axis in {"X", "Y", "S"}:
                index.append(slice(None))
                continue
            index.append(int(kwargs.get(axis, 0)))

        frame = np.asarray(arr[tuple(index)])
        remaining_axes = [axis for axis, idx in zip(order, index) if isinstance(idx, slice)]
        return self._normalize_to_yx_or_yxs(frame, remaining_axes, rgb_samples=self.is_rgb)

    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSourceOifOib", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSourceOifOib", list[int]], dict[str, int]]:
        del respect_per_file_channel_count

        dim_info = self._dimensions_with_axes()
        if not dim_info:
            return sources, original_dimensions

        axes_order = [axis for _logical, axis, _size in dim_info]
        ranges = [range(size) for _logical, _axis, size in dim_info]
        index_tuples = list(itertools.product(*ranges))
        channel_axis_idx = axes_order.index("C") if "C" in axes_order else None

        new_sources: dict[LimImageSourceOifOib, list[int]] = {}
        for file, dims in sources.items():
            for idx_tuple in index_tuples:
                file_copy = copy.deepcopy(file)

                next_axis_indices = dict(getattr(file_copy, "axis_indices", {}) or {})
                for axis, axis_index in zip(axes_order, idx_tuple):
                    if axis == "C":
                        continue
                    next_axis_indices[axis] = int(axis_index)
                file_copy.axis_indices = next_axis_indices

                if channel_axis_idx is not None:
                    file_copy.channel_index = int(idx_tuple[channel_axis_idx])
                else:
                    file_copy.channel_index = None

                new_sources[file_copy] = dims + list(idx_tuple)

        new_dims = original_dimensions.copy()
        for logical, _axis, size in dim_info:
            target_dim = logical
            if preserve_duplicate_dimension_names and target_dim in new_dims:
                suffix_index = 2
                while f"{logical}__dup{suffix_index}" in new_dims:
                    suffix_index += 1
                target_dim = f"{logical}__dup{suffix_index}"

            if target_dim in new_dims and not preserve_duplicate_dimension_names:
                new_dims[target_dim] = int(new_dims[target_dim]) * int(size)
            else:
                new_dims[target_dim] = int(size)

        if new_dims.get("unknown", 0) > 1 and unknown_dimension_type is not None:
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

        return new_sources, new_dims

    @staticmethod
    def _format_float(value: float | None, digits: int | None = None) -> str:
        if value is None:
            return ""
        if digits is None:
            return f"{value:g}"
        return f"{value:.{digits}f}"

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

    def _mainfile(self) -> dict:
        with self._open_image() as image:
            mainfile = getattr(image, "mainfile", None)
        if isinstance(mainfile, dict):
            return mainfile
        return {}

    @staticmethod
    def _infer_axis_step(props: dict, axis_size: int) -> tuple[float | None, str]:
        unit = str(
            props.get("UnitName")
            or props.get("AxisUnitName")
            or props.get("Unit")
            or props.get("CalibrateUnit")
            or ""
        ).strip()

        for key in ("Interval", "Increment", "Step", "Scale", "Calibration"):
            value = LimImageSourceOifOib._to_float(props.get(key))
            if value is not None and value > 0:
                return value, unit

        if axis_size > 1:
            start = LimImageSourceOifOib._to_float(props.get("StartPosition"))
            end = LimImageSourceOifOib._to_float(props.get("EndPosition"))
            if start is not None and end is not None and end != start:
                return (end - start) / float(axis_size - 1), unit

        return None, unit

    @staticmethod
    def _to_micrometers(value: float | None, unit: str) -> float | None:
        if value is None or value <= 0:
            return None
        unit = str(unit).strip().casefold()
        if unit in {"um", "µm", "μm", "micrometer", "micrometers"}:
            return value
        if unit in {"nm", "nanometer", "nanometers"}:
            return value / 1000.0
        if unit in {"mm", "millimeter", "millimeters"}:
            return value * 1000.0
        if unit in {"m", "meter", "meters"}:
            return value * 1_000_000.0
        return None

    @staticmethod
    def _to_milliseconds(value: float | None, unit: str) -> float | None:
        if value is None or value <= 0:
            return None
        unit = str(unit).strip().casefold()
        if unit in {"ms", "millisecond", "milliseconds"}:
            return value
        if unit in {"s", "sec", "second", "seconds"}:
            return value * 1000.0
        if unit in {"min", "minute", "minutes"}:
            return value * 60_000.0
        return None

    def _pixel_sizes(self) -> tuple[float | None, float | None, float | None]:
        mainfile = self._mainfile()
        if not mainfile:
            return None, None, None

        axis_data: dict[str, tuple[float | None, str]] = {}
        for section_name, props in mainfile.items():
            if not isinstance(props, dict):
                continue
            match = re.match(r"^Axis\s+\d+\s+Parameters Common$", str(section_name))
            if not match:
                continue
            axis_code = str(props.get("AxisCode", "")).strip().upper()
            if len(axis_code) != 1:
                continue

            axis_size = int(props.get("MaxSize", 0) or 0)
            axis_data[axis_code] = self._infer_axis_step(props, axis_size)

        x_step, x_unit = axis_data.get("X", (None, ""))
        y_step, y_unit = axis_data.get("Y", (None, ""))
        z_step, z_unit = axis_data.get("Z", (None, ""))
        t_step, t_unit = axis_data.get("T", (None, ""))

        pixel_x_um = self._to_micrometers(x_step, x_unit)
        pixel_y_um = self._to_micrometers(y_step, y_unit)
        pixel_cal_um = pixel_x_um if pixel_x_um is not None else pixel_y_um
        z_step_um = self._to_micrometers(z_step, z_unit)
        t_step_ms = self._to_milliseconds(t_step, t_unit)
        return pixel_cal_um, z_step_um, t_step_ms

    def _channel_count(self) -> int:
        sizes = self._axis_sizes()
        return int(sizes.get("C", 0))

    @staticmethod
    def _section_channel_index(section_name: str) -> int | None:
        match = re.search(r"(?:channel|ch|wave)\D*(\d+)", section_name.casefold())
        if not match:
            return None
        return int(match.group(1)) - 1

    @staticmethod
    def _pick_channel_name(props: dict) -> str:
        priority = (
            "DyeName",
            "ChannelName",
            "Name",
            "Label",
            "LUTName",
            "WaveName",
        )
        for key in priority:
            value = props.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _channel_names(self) -> list[str]:
        channel_count = self._channel_count()
        mainfile = self._mainfile()
        if not mainfile:
            return [f"Channel {idx + 1}" for idx in range(channel_count)]

        names: dict[int, str] = {}
        extras: list[str] = []

        for section_name, props in mainfile.items():
            if not isinstance(props, dict):
                continue

            section_text = str(section_name)
            section_lower = section_text.casefold()
            if not any(token in section_lower for token in ("channel", "wave", "lut")):
                continue

            name = self._pick_channel_name(props)
            if not name:
                continue

            idx = self._section_channel_index(section_text)
            if idx is None:
                extras.append(name)
                continue
            if idx < 0:
                continue
            names[idx] = name

        out: list[str] = []
        for idx in range(channel_count):
            if idx in names and str(names[idx]).strip():
                out.append(str(names[idx]).strip())
            elif idx < len(extras) and str(extras[idx]).strip():
                out.append(str(extras[idx]).strip())
            else:
                out.append(f"Channel {idx + 1}")
        return out

    @staticmethod
    def _infer_color_from_name(name: str) -> str:
        n = str(name).strip().casefold()
        if n == "":
            return ""
        if "blue" in n or "dapi" in n or "hoechst" in n:
            return "#4080FF"
        if "green" in n or "fitc" in n or "gfp" in n:
            return "#00B050"
        if "red" in n or "tritc" in n or "txr" in n:
            return "#FF3030"
        if "cyan" in n:
            return "#00FFFF"
        if "yellow" in n:
            return "#FFD400"
        if "magenta" in n:
            return "#FF00FF"
        return ""

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        pixel_cal_um, z_step_um, t_step_ms = self._pixel_sizes()
        channel_names = self._channel_names()

        if pixel_cal_um is None and z_step_um is None and t_step_ms is None and not channel_names:
            return

        b = ConvertSequenceArgs()
        b.metadata = MetadataFactory(pixel_calibration=pixel_cal_um if pixel_cal_um is not None else -1.0)
        b.channels = {
            idx: Plane(
                name=name,
                modality=0,
                excitation_wavelength=0,
                emission_wavelength=0,
                color=self._infer_color_from_name(name),
            )
            for idx, name in enumerate(channel_names)
        }
        b.time_step = t_step_ms
        b.z_step = z_step_um

        merge_four_fields(metadata_storage, b)

    def metadata_as_pattern_settings(self) -> dict:
        pixel_cal_um, z_step_um, t_step_ms = self._pixel_sizes()
        channel_names = self._channel_names()

        if not channel_names:
            dims = self.get_file_dimensions()
            channel_count = int(dims.get("channel", 0))
            channel_names = [f"Channel {idx + 1}" for idx in range(channel_count)]

        return {
            "tstep": self._format_float(t_step_ms, 3),
            "zstep": self._format_float(z_step_um, 3),
            "pixel_calibration": self._format_float(pixel_cal_um),
            "pinhole_diameter": "",
            "objective_magnification": "",
            "objective_numerical_aperture": "",
            "immersion_refractive_index": "",
            "zoom_magnification": "",
            "channels": [
                [name, name, "Undefined", "0", "0", self._infer_color_from_name(name)]
                for name in channel_names
            ],
        }

    def nd2_attributes(self, *, sequence_count=1) -> ImageAttributes:
        frame = np.asarray(self.read())

        if frame.ndim == 2:
            height, width = int(frame.shape[0]), int(frame.shape[1])
            components = 1
        elif frame.ndim == 3:
            height, width, comps = frame.shape
            height = int(height)
            width = int(width)
            components = int(comps)
            if components == 4:
                components = 3
        else:
            raise ValueError(f"Unsupported OIF/OIB frame shape {frame.shape}.")

        dtype = np.dtype(frame.dtype)
        bits = int(dtype.itemsize * 8)

        if np.issubdtype(dtype, np.signedinteger):
            pixel_type = ImageAttributesPixelType.pxtSigned
        elif np.issubdtype(dtype, np.unsignedinteger):
            pixel_type = ImageAttributesPixelType.pxtUnsigned
        elif np.issubdtype(dtype, np.floating):
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            raise ValueError(f"OIF/OIB file has unsupported pixel type: {dtype}.")

        width_bytes = ImageAttributes.calcWidthBytes(width, bits, components)

        return ImageAttributes(
            uiWidth=width,
            uiWidthBytes=width_bytes,
            uiHeight=height,
            uiComp=components,
            uiBpcInMemory=bits,
            uiBpcSignificant=bits,
            uiSequenceCount=sequence_count,
            uiTileWidth=width,
            uiTileHeight=height,
            uiVirtualComponents=components,
            ePixelType=pixel_type,
        )
