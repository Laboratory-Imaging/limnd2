from __future__ import annotations

import copy
import itertools
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.metadata_factory import MetadataFactory, Plane

from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs
from .LimImageSource import LimImageSource, merge_four_fields

_CZI_HINT = '[czi] extra not installed. Install it with `pip install "limnd2[czi]"`.'


def _missing_convert_dependency(package: str) -> ImportError:
    msg = (
        f'Missing optional dependency "{package}" required for CZI conversion. '
        f"{_CZI_HINT}"
    )
    return ImportError(msg)


def _require_czifile():
    try:
        from czifile import CziFile
    except ImportError as exc:
        raise _missing_convert_dependency("czifile") from exc
    return CziFile


CziFile = _require_czifile()


class LimImageSourceCzi(LimImageSource):
    """Image source reading from Carl Zeiss CZI files."""

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
            f"LimImageSourceCzi({self.filename}, seq_index={self.seq_index}, "
            f"channel_index={self.channel_index}, axis_indices={self.axis_indices})"
        )

    @staticmethod
    def _aggregate_shape_from_subblocks(czi, axes: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        entries = list(getattr(czi, "filtered_subblock_directory", ()) or ())
        if not entries:
            raise ValueError("CZI file does not contain readable subblocks.")

        axis_count = len(axes)
        min_start: list[int | None] = [None] * axis_count
        max_end: list[int | None] = [None] * axis_count

        for entry in entries:
            try:
                entry_axes = str(getattr(entry, "axes", ""))
                entry_start = tuple(int(v) for v in getattr(entry, "start", ()))
                entry_shape = tuple(int(v) for v in getattr(entry, "shape", ()))
            except Exception:
                continue

            if len(entry_axes) != len(entry_shape):
                continue

            axis_to_start = {
                axis: int(entry_start[idx]) if idx < len(entry_start) else 0
                for idx, axis in enumerate(entry_axes)
            }
            axis_to_size = {
                axis: max(int(entry_shape[idx]), 1)
                for idx, axis in enumerate(entry_axes)
            }

            for axis_idx, axis in enumerate(axes):
                start = int(axis_to_start.get(axis, 0))
                size = int(axis_to_size.get(axis, 1))
                end = start + max(size, 1)

                prev_min = min_start[axis_idx]
                prev_max = max_end[axis_idx]
                min_start[axis_idx] = start if prev_min is None else min(prev_min, start)
                max_end[axis_idx] = end if prev_max is None else max(prev_max, end)

        start_tuple = tuple(int(v) if v is not None else 0 for v in min_start)
        shape_tuple = tuple(
            max(1, int((max_end[idx] if max_end[idx] is not None else start_tuple[idx] + 1) - start_tuple[idx]))
            for idx in range(axis_count)
        )
        return shape_tuple, start_tuple

    def _asarray_from_subblocks(self, czi, axes: str) -> np.ndarray:
        shape, global_start = self._aggregate_shape_from_subblocks(czi, axes)
        out = np.zeros(shape, dtype=np.dtype(czi.dtype))
        entries = list(getattr(czi, "filtered_subblock_directory", ()) or ())

        for entry in entries:
            try:
                entry_axes = str(getattr(entry, "axes", ""))
                entry_start = tuple(int(v) for v in getattr(entry, "start", ()))
                tile = np.asarray(entry.data_segment().data(resize=True, order=0))
            except Exception:
                continue

            if tile.ndim != len(entry_axes):
                continue

            present_axes = [axis for axis in axes if axis in entry_axes]
            if len(present_axes) != tile.ndim:
                continue

            perm = [entry_axes.index(axis) for axis in present_axes]
            tile = np.transpose(tile, axes=perm)

            full_shape: list[int] = []
            tile_pos = 0
            for axis in axes:
                if axis in entry_axes:
                    full_shape.append(int(tile.shape[tile_pos]))
                    tile_pos += 1
                else:
                    full_shape.append(1)
            tile = tile.reshape(tuple(full_shape))

            slices: list[slice] = []
            for axis_idx, axis in enumerate(axes):
                if axis in entry_axes:
                    entry_pos = entry_axes.index(axis)
                    start = int(entry_start[entry_pos]) if entry_pos < len(entry_start) else 0
                    size = int(tile.shape[axis_idx])
                else:
                    start = 0
                    size = 1
                offset = start - int(global_start[axis_idx])
                slices.append(slice(offset, offset + size))

            try:
                out[tuple(slices)] = tile
            except Exception:
                continue

        return out

    def _axes_shape_dtype(self) -> tuple[str, tuple[int, ...], np.dtype]:
        with CziFile(self.filename) as czi:
            axes = str(czi.axes)
            try:
                shape = tuple(int(v) for v in czi.shape)
            except Exception:
                shape, _ = self._aggregate_shape_from_subblocks(czi, axes)
            dtype = np.dtype(czi.dtype)

        if len(axes) != len(shape):
            raise ValueError(
                f"CZI axes/shape mismatch for {self.filename}: axes={axes!r}, shape={shape!r}."
            )

        return axes, shape, dtype

    def _axis_size_map(self) -> dict[str, int]:
        axes, shape, _ = self._axes_shape_dtype()
        return {axis: int(size) for axis, size in zip(axes, shape)}

    def _is_rgb_from_axes(self, axes: str, shape: tuple[int, ...]) -> bool:
        axis_sizes = {axis: int(size) for axis, size in zip(axes, shape)}
        sample_count = int(axis_sizes.get("0", 1))
        channel_count = int(axis_sizes.get("C", 1))
        return sample_count in (3, 4) and channel_count <= 1

    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            axes, shape, _ = self._axes_shape_dtype()
            self._is_rgb = self._is_rgb_from_axes(axes, shape)
        return self._is_rgb

    @staticmethod
    def _axis_to_logical_dimension(axis: str, is_rgb: bool) -> str | None:
        if axis in {"X", "Y", "0", "M"}:
            return None
        if axis == "T":
            return "timeloop"
        if axis == "Z":
            return "zstack"
        if axis == "C":
            return None if is_rgb else "channel"
        return "unknown"

    def _logical_dimensions_with_axes(self) -> list[tuple[str, str, int]]:
        axes, shape, _ = self._axes_shape_dtype()
        rgb = self._is_rgb_from_axes(axes, shape)

        dims: list[tuple[str, str, int]] = []
        for axis, size in zip(axes, shape):
            logical = self._axis_to_logical_dimension(axis, rgb)
            if logical is None:
                continue
            if int(size) <= 1:
                continue
            dims.append((logical, axis, int(size)))
        return dims

    def get_file_dimensions(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for logical, _axis, size in self._logical_dimensions_with_axes():
            if logical in dims:
                dims[logical] *= int(size)
            else:
                dims[logical] = int(size)
        return dims

    def _decode_seq_index_axes(self, axes: str, shape: tuple[int, ...]) -> dict[str, int]:
        seq_axes: list[tuple[str, int]] = []
        for axis, size in zip(axes, shape):
            if axis in {"X", "Y", "0", "C", "M"}:
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
    def _clamp_index(value: int, size: int) -> int:
        if size <= 0:
            return 0
        return min(max(int(value), 0), size - 1)

    @staticmethod
    def _normalize_to_yx_or_yxs(frame: np.ndarray, axes: list[str], *, rgb_samples: bool) -> np.ndarray:
        if "Y" not in axes or "X" not in axes:
            raise ValueError(f"CZI frame missing Y/X axes after indexing: axes={axes}")

        # Drop singleton non-spatial dimensions.
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
            raise ValueError(f"CZI frame lost Y/X axes during normalization: axes={cur_axes}")

        sample_axis: str | None = None
        if "0" in cur_axes and cur.shape[cur_axes.index("0")] > 1:
            sample_axis = "0"
        elif "C" in cur_axes and cur.shape[cur_axes.index("C")] > 1:
            sample_axis = "C"

        if sample_axis is None:
            order = [cur_axes.index("Y"), cur_axes.index("X")]
            out = np.transpose(cur, axes=order)
            return np.asarray(out)

        order = [cur_axes.index("Y"), cur_axes.index("X"), cur_axes.index(sample_axis)]
        out = np.transpose(cur, axes=order)

        if sample_axis == "0" and rgb_samples:
            if out.shape[-1] >= 4:
                out = out[..., :3]
            if out.shape[-1] == 3:
                out = out[..., ::-1]
            elif out.shape[-1] == 1:
                out = out[..., 0]

        return np.asarray(out)

    def read(self) -> np.ndarray:
        with CziFile(self.filename) as czi:
            axes = str(czi.axes)
            try:
                arr = np.asarray(czi.asarray())
            except Exception:
                arr = self._asarray_from_subblocks(czi, axes)

        if len(axes) != arr.ndim:
            raise ValueError(
                f"CZI axes/data mismatch for {self.filename}: axes={axes!r}, shape={arr.shape!r}."
            )

        seq_axes = self._decode_seq_index_axes(axes, arr.shape)
        index: list[int | slice] = []

        for axis, size in zip(axes, arr.shape):
            if axis in {"X", "Y", "0", "M"}:
                index.append(slice(None))
                continue

            if axis == "C":
                if self.is_rgb:
                    index.append(0)
                    continue

                if self.channel_index is not None:
                    c_index = self._clamp_index(self.channel_index, int(size))
                    index.append(c_index)
                    continue

                axis_value = self.axis_indices.get("C", 0)
                index.append(self._clamp_index(axis_value, int(size)))
                continue

            axis_value = self.axis_indices.get(axis)
            if axis_value is None:
                axis_value = seq_axes.get(axis, 0)
            index.append(self._clamp_index(axis_value, int(size)))

        frame = np.asarray(arr[tuple(index)])
        remaining_axes = [ax for ax, idx in zip(axes, index) if isinstance(idx, slice)]
        return self._normalize_to_yx_or_yxs(frame, remaining_axes, rgb_samples=self.is_rgb)

    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSourceCzi", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSourceCzi", list[int]], dict[str, int]]:
        del respect_per_file_channel_count

        dim_info = self._logical_dimensions_with_axes()
        if not dim_info:
            return sources, original_dimensions

        ranges = [range(size) for _logical, _axis, size in dim_info]
        index_tuples = list(itertools.product(*ranges))

        # Build dimension plan. When duplicate logical dimensions are merged
        # (default behavior), merge their index tuples in the same way to keep
        # source index vectors aligned with the resulting `new_dims` order.
        dim_plan: list[dict[str, object]] = []
        dim_plan_by_name: dict[str, dict[str, object]] = {}

        for logical, axis, size in dim_info:
            target_dim = logical
            if preserve_duplicate_dimension_names:
                if target_dim in dim_plan_by_name:
                    suffix_index = 2
                    while f"{logical}__dup{suffix_index}" in dim_plan_by_name:
                        suffix_index += 1
                    target_dim = f"{logical}__dup{suffix_index}"
                plan_item = {"name": target_dim, "axes": [axis], "sizes": [int(size)]}
                dim_plan.append(plan_item)
                dim_plan_by_name[target_dim] = plan_item
                continue

            if target_dim not in dim_plan_by_name:
                plan_item = {"name": target_dim, "axes": [axis], "sizes": [int(size)]}
                dim_plan.append(plan_item)
                dim_plan_by_name[target_dim] = plan_item
            else:
                dim_plan_by_name[target_dim]["axes"].append(axis)
                dim_plan_by_name[target_dim]["sizes"].append(int(size))

        new_files: dict[LimImageSourceCzi, list[int]] = {}

        for file, dims in sources.items():
            for idx_tuple in index_tuples:
                file_copy = copy.deepcopy(file)
                axis_value = {
                    axis: int(axis_index)
                    for (_logical, axis, _size), axis_index in zip(dim_info, idx_tuple)
                }

                next_axis_indices = dict(getattr(file_copy, "axis_indices", {}) or {})
                for _logical, axis, _size in dim_info:
                    if axis == "C":
                        continue
                    next_axis_indices[axis] = axis_value[axis]
                file_copy.axis_indices = next_axis_indices

                merged_indices: list[int] = []
                channel_index: int | None = None
                for plan_item in dim_plan:
                    plan_name = str(plan_item["name"])
                    plan_axes = list(plan_item["axes"])
                    plan_sizes = [int(v) for v in plan_item["sizes"]]

                    values = [axis_value[a] for a in plan_axes]
                    if len(values) == 1:
                        merged_val = int(values[0])
                    else:
                        merged_val = 0
                        for val, radix in zip(values, plan_sizes):
                            merged_val = merged_val * int(radix) + int(val)

                    merged_indices.append(merged_val)
                    if plan_name == "channel":
                        channel_index = int(merged_val)

                file_copy.channel_index = channel_index

                new_files[file_copy] = dims + merged_indices

        new_dims = original_dimensions.copy()
        for plan_item in dim_plan:
            target_dim = str(plan_item["name"])
            size = 1
            for axis_size in plan_item["sizes"]:
                size *= int(axis_size)

            if preserve_duplicate_dimension_names and target_dim in new_dims:
                suffix_index = 2
                while f"{target_dim}__dup{suffix_index}" in new_dims:
                    suffix_index += 1
                target_dim = f"{target_dim}__dup{suffix_index}"

            if target_dim in new_dims and not preserve_duplicate_dimension_names:
                new_dims[target_dim] = int(new_dims[target_dim]) * int(size)
            else:
                new_dims[target_dim] = int(size)

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

    @staticmethod
    def _parse_float(value) -> float | None:
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
    def _parse_wavelength(value) -> int:
        parsed = LimImageSourceCzi._parse_float(value)
        if parsed is None or parsed <= 0:
            return 0
        return int(round(parsed))

    @staticmethod
    def _normalize_color(value: str | None) -> str:
        if value is None:
            return ""

        text = str(value).strip()
        if text == "":
            return ""

        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) >= 3:
                try:
                    rgb = [int(float(parts[i])) for i in range(3)]
                except Exception:
                    rgb = []
                if len(rgb) == 3:
                    rgb = [max(0, min(255, v)) for v in rgb]
                    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

        if text.startswith("#"):
            text = text[1:]

        if re.fullmatch(r"[0-9A-Fa-f]{8}", text):
            # ARGB -> RGB
            return f"#{text[2:].upper()}"

        if re.fullmatch(r"[0-9A-Fa-f]{6}", text):
            return f"#{text.upper()}"

        return ""

    def _metadata_xml(self) -> str | None:
        try:
            with CziFile(self.filename) as czi:
                xml_text = czi.metadata(raw=True)
        except Exception:
            return None

        if not xml_text:
            return None

        if isinstance(xml_text, bytes):
            return xml_text.decode("utf-8", errors="ignore")
        return str(xml_text)

    def _extract_metadata(self) -> tuple[float | None, float | None, list[dict[str, object]]]:
        xml_text = self._metadata_xml()
        if not xml_text:
            return None, None, []

        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None, None, []

        pixel_cal_um: float | None = None
        z_step_um: float | None = None

        for dist in root.findall(".//Distance"):
            dim_id = (dist.get("Id") or dist.get("ID") or "").strip().upper()
            value_text = dist.findtext("Value")
            if value_text is None:
                value_text = dist.findtext(".//Value")
            value_m = self._parse_float(value_text)
            if value_m is None or value_m <= 0:
                continue
            value_um = value_m * 1_000_000.0
            if dim_id in {"X", "Y"} and pixel_cal_um is None:
                pixel_cal_um = value_um
            elif dim_id == "Z":
                z_step_um = value_um

        channels: list[dict[str, object]] = []
        channel_nodes = root.findall(".//Channels/Channel")
        if not channel_nodes:
            channel_nodes = root.findall(".//Channel")

        for idx, channel in enumerate(channel_nodes):
            name = (
                channel.get("Name")
                or channel.get("ShortName")
                or channel.get("Id")
                or channel.get("ID")
                or f"Channel {idx + 1}"
            )
            name = str(name).strip() or f"Channel {idx + 1}"

            ex = self._parse_wavelength(
                channel.get("ExcitationWavelength")
                or channel.get("ExcitationWavelengthNm")
                or channel.findtext("ExcitationWavelength")
                or channel.findtext("ExcitationWavelengthNm")
                or channel.findtext(".//ExcitationWavelength")
            )
            em = self._parse_wavelength(
                channel.get("EmissionWavelength")
                or channel.get("EmissionWavelengthNm")
                or channel.findtext("EmissionWavelength")
                or channel.findtext("EmissionWavelengthNm")
                or channel.findtext(".//EmissionWavelength")
            )
            color = self._normalize_color(
                channel.get("Color")
                or channel.get("DisplayColor")
                or channel.findtext("Color")
                or channel.findtext("DisplayColor")
                or channel.findtext(".//Color")
            )

            channels.append(
                {
                    "name": name,
                    "excitation": ex,
                    "emission": em,
                    "color": color,
                }
            )

        # Deduplicate channel names while preserving order.
        deduped: list[dict[str, object]] = []
        seen_names: set[str] = set()
        for channel in channels:
            name = str(channel.get("name") or "")
            if name in seen_names:
                continue
            seen_names.add(name)
            deduped.append(channel)

        return pixel_cal_um, z_step_um, deduped

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        pixel_cal, z_step, channels = self._extract_metadata()

        has_channel_data = len(channels) > 0
        has_global_data = pixel_cal is not None or z_step is not None
        if not has_channel_data and not has_global_data:
            return

        b = ConvertSequenceArgs()
        b.metadata = MetadataFactory(pixel_calibration=pixel_cal if pixel_cal is not None else -1.0)
        b.channels = {
            idx: Plane(
                name=str(channel.get("name") or f"Channel {idx + 1}"),
                modality=0,
                excitation_wavelength=int(channel.get("excitation") or 0),
                emission_wavelength=int(channel.get("emission") or 0),
                color=str(channel.get("color") or ""),
            )
            for idx, channel in enumerate(channels)
        }
        b.z_step = z_step

        merge_four_fields(metadata_storage, b)

    @staticmethod
    def _format_float(value: float | None, digits: int | None = None) -> str:
        if value is None:
            return ""
        if digits is None:
            return f"{value:g}"
        return f"{value:.{digits}f}"

    def metadata_as_pattern_settings(self) -> dict:
        pixel_cal, z_step, channels = self._extract_metadata()

        if not channels:
            dims = self.get_file_dimensions()
            channel_count = int(dims.get("channel", 0))
            channels = [
                {
                    "name": f"Channel {idx + 1}",
                    "excitation": 0,
                    "emission": 0,
                    "color": "",
                }
                for idx in range(channel_count)
            ]

        return {
            "tstep": "",
            "zstep": self._format_float(z_step, 3),
            "pixel_calibration": self._format_float(pixel_cal),
            "pinhole_diameter": "",
            "objective_magnification": "",
            "objective_numerical_aperture": "",
            "immersion_refractive_index": "",
            "zoom_magnification": "",
            "channels": [
                [
                    str(channel.get("name") or f"Channel {idx + 1}"),
                    str(channel.get("name") or f"Channel {idx + 1}"),
                    "Undefined",
                    str(int(channel.get("excitation") or 0)),
                    str(int(channel.get("emission") or 0)),
                    str(channel.get("color") or ""),
                ]
                for idx, channel in enumerate(channels)
            ],
        }

    def nd2_attributes(self, *, sequence_count=1) -> ImageAttributes:
        axis_sizes = self._axis_size_map()
        _axes, _shape, dtype = self._axes_shape_dtype()

        height = int(axis_sizes.get("Y", 0))
        width = int(axis_sizes.get("X", 0))
        if height <= 0 or width <= 0:
            raise ValueError(f"CZI file {self.filename} does not contain valid X/Y dimensions.")

        if self.is_rgb:
            sample_count = int(axis_sizes.get("0", 3))
            components = 3 if sample_count >= 3 else sample_count
        elif self.channel_index is not None:
            components = 1
        else:
            components = int(axis_sizes.get("0", 1))
            if components <= 0:
                components = 1

        if components == 4:
            components = 3

        bits = int(dtype.itemsize * 8)
        width_bytes = ImageAttributes.calcWidthBytes(width, bits, components)

        if np.issubdtype(dtype, np.signedinteger):
            pixel_type = ImageAttributesPixelType.pxtSigned
        elif np.issubdtype(dtype, np.unsignedinteger):
            pixel_type = ImageAttributesPixelType.pxtUnsigned
        elif np.issubdtype(dtype, np.floating):
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            raise ValueError(f"CZI file has unsupported pixel type: {dtype}.")

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
