from __future__ import annotations

import json
import sys
from pathlib import Path
import re
from typing import Any

import numpy as np

from limnd2.metadata import PicturePlaneModality, PicturePlaneModalityFlags

from .crawler import FileCrawler
from .LimConvertSequence import (
    analyze_file_grid,
    build_dimensions_from_groups,
    convert_values,
    get_group_values,
)
from .LimConvertSequenceArgparser import convert_sequence_parse
from .LimConvertUtils import ConvertSequenceArgs, LogType
from . import LimConvertUtils as _convert_utils_mod
from .LimImageSource import LimImageSource
from .LimImageSourceConvert import (
    ConvertUtils,
    ZeroChannelSource,
    _derive_channel_labels_and_templates,
    _ensure_metadata_plane_count,
    flatten_duplicate_dimensions,
    group_by_channel_with_padding,
    reorder_grouped_files_by_auto_channel_labels,
)
from .LimImageSourceMapping import EXTENSION_MAP, READER_CLASS_MAP


_QML_MODALITIES = [
    "Undefined",
    "Wide-field",
    "Brightfield",
    "Phase",
    "DIC",
    "DarkField",
    "MC",
    "TIRF",
    "Confocal, Fluo",
    "Confocal, Trans",
    "Multi-Photon",
    "SFC pinhole",
    "SFC slit",
    "Spinning Disc",
    "DSD",
    "NSIM",
    "iSim",
    "RCM",
    "CSU W1-SoRa",
    "NSPARC",
]


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return ""
    if val <= 0:
        return ""
    return f"{val}"


def _format_step(value: Any) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return ""
    if val <= 0:
        return ""
    return f"{val:.3f}"


def _color_to_hex(color: Any) -> str:
    if isinstance(color, str):
        return color
    if isinstance(color, tuple) and len(color) >= 3:
        rgb: list[int] = []
        for comp in color[:3]:
            try:
                cval = float(comp)
            except Exception:
                return ""
            if 0.0 <= cval <= 1.0:
                cval *= 255.0
            rgb.append(int(min(255, max(0, round(cval)))))
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    return ""


def _modality_to_string(modality: Any) -> str:
    if isinstance(modality, str):
        return modality if modality else "Undefined"

    flags: PicturePlaneModalityFlags
    if isinstance(modality, PicturePlaneModalityFlags):
        flags = modality
    elif isinstance(modality, PicturePlaneModality):
        flags = PicturePlaneModalityFlags.from_modality(modality)
    else:
        return "Undefined"

    if flags == PicturePlaneModalityFlags.modUnknown:
        return "Undefined"

    modality_map = PicturePlaneModalityFlags.modality_string_map()
    if "Multi-Photon" not in modality_map and "Multi-photon" in modality_map:
        modality_map["Multi-Photon"] = modality_map["Multi-photon"]
    if "MC" not in modality_map:
        modality_map["MC"] = PicturePlaneModalityFlags.modNAMC

    for name in _QML_MODALITIES:
        required = modality_map.get(name)
        if required is None or required == PicturePlaneModalityFlags.modUnknown:
            continue
        if (flags & required) == required:
            return name
    return "Undefined"


def _common_plane_value(planes, key: str) -> Any:
    values = [getattr(plane, key, None) for plane in planes]
    values = [val for val in values if val is not None]
    if not values:
        return None
    first = values[0]
    if all(val == first for val in values):
        return first
    return None


def _build_qml_settings(parsed_args: ConvertSequenceArgs, channel_labels: list[str]) -> dict[str, Any]:
    metadata = parsed_args.metadata
    planes = list(getattr(metadata, "planes", []))
    other_settings = getattr(metadata, "_other_settings", {}) or {}

    def _setting_value(name: str) -> Any:
        if name in other_settings and other_settings[name] is not None:
            return other_settings[name]
        return _common_plane_value(planes, name)

    def _is_generic_channel_label(label: Any) -> bool:
        text = str(label).strip()
        if not text:
            return True
        if text.isdigit():
            return True
        return bool(re.match(r"(?i)^channel(?:[_ ]?\d+)?$", text))

    channels: list[list[str]] = []
    for idx, plane in enumerate(planes):
        label = plane.name if plane.name else f"Channel {idx + 1}"
        if idx < len(channel_labels) and channel_labels[idx]:
            inferred = channel_labels[idx]
            # Preserve richer metadata names (e.g. EGFP, TexasRed) when inferred
            # labels are only generic placeholders.
            if _is_generic_channel_label(label) or not _is_generic_channel_label(inferred):
                label = inferred
        ex = int(plane.excitation_wavelength) if plane.excitation_wavelength else 0
        em = int(plane.emission_wavelength) if plane.emission_wavelength else 0
        channels.append(
            [
                label,
                label,
                _modality_to_string(plane.modality),
                str(ex),
                str(em),
                _color_to_hex(plane.color),
            ]
        )

    return {
        "tstep": _format_step(parsed_args.time_step),
        "zstep": _format_step(parsed_args.z_step),
        "pixel_calibration": _format_number(metadata.pixel_calibration),
        "pinhole_diameter": _format_number(_setting_value("pinhole_diameter")),
        "objective_magnification": _format_number(_setting_value("objective_magnification")),
        "objective_numerical_aperture": _format_number(_setting_value("objective_numerical_aperture")),
        "immersion_refractive_index": _format_number(_setting_value("immersion_refractive_index")),
        "zoom_magnification": _format_number(_setting_value("zoom_magnification")),
        "channels": channels,
    }


def plan_sequence(args: list[str] | None = None) -> dict[str, Any]:
    """
    Plan sequence conversion without writing ND2.

    Resolves filename dimensions, in-file dimensions, duplicate-dimension
    flattening, missing-combination padding rules, and channel metadata shaping.
    Returns JSON-serializable data in the same high-level shape used by
    ``get_file_dimensions_as_json`` (dimensions + ``qml_settings`` + status).
    """
    parsed_args = convert_sequence_parse(args, require_output=False)
    _convert_utils_mod.LOG_TYPE = LogType.NONE

    crawler = FileCrawler(
        parsed_args.folder,
        file_extensions=EXTENSION_MAP[parsed_args.extension],
        regexp=parsed_args.regexp,
    )

    files: dict[Path, list[str]] = crawler.run(get_group_values, {"regexp": parsed_args.regexp}, True)
    if len(files) == 0:
        raise ValueError("No files matching given criteria were found.")

    file_sources: dict[LimImageSource, list[Any]] = {
        READER_CLASS_MAP[parsed_args.extension](path): dims for path, dims in files.items()
    }
    sample_file = next(iter(file_sources.keys()))

    convert_values(file_sources)

    missing_combinations: list[tuple[Any, ...]] = []
    if len(file_sources) != 1:
        found_values, missing_combinations = analyze_file_grid(list(file_sources.values()))
        if missing_combinations and not parsed_args.allow_missing_files:
            raise ValueError(f"Missing files with dimension values: {missing_combinations[0]}")
        dimensions = build_dimensions_from_groups(parsed_args.groups, found_values)
    else:
        dimensions = {}

    if sample_file.is_rgb and "channel" in dimensions:
        raise ValueError("Can not use channel dimension with RGB image.")

    channels_were_user_provided = bool(parsed_args.channels)

    sources, dimensions = sample_file.parse_additional_dimensions(
        file_sources,
        dimensions,
        parsed_args.unknown_dim,
        preserve_duplicate_dimension_names=parsed_args.flatten_duplicates,
        respect_per_file_channel_count=True,
    )
    ConvertUtils.convert_mx_my_to_m(sources, dimensions)
    if parsed_args.flatten_duplicates:
        sources, dimensions = flatten_duplicate_dimensions(sources, dimensions)
    sources, dimensions = ConvertUtils.reorder_experiments(sources, dimensions)

    sample_file.parse_additional_metadata(parsed_args)
    nd2_attributes_base = sample_file.nd2_attributes(sequence_count=1)

    if "channel" in dimensions:
        component_count = int(dimensions["channel"])
    elif sample_file.is_rgb:
        component_count = 3
    else:
        component_count = int(nd2_attributes_base.uiComp)

    height = int(nd2_attributes_base.height)
    width = int(nd2_attributes_base.width)
    out_dtype = np.dtype(getattr(nd2_attributes_base, "dtype", np.uint16))
    fallback_components = 1 if "channel" in dimensions else component_count

    def _source_wrapper_factory(source: Any, slot: int):
        return source

    def _zero_source_factory():
        return ZeroChannelSource(
            height=height,
            width=width,
            dtype=out_dtype,
            components=fallback_components,
        )

    grouped_files = group_by_channel_with_padding(
        sources,
        parsed_args,
        dimensions,
        component_count,
        _zero_source_factory,
        _source_wrapper_factory,
        allow_missing_files=parsed_args.allow_missing_files,
    )
    if not channels_were_user_provided and "channel" in dimensions and component_count > 1:
        grouped_files, _ = reorder_grouped_files_by_auto_channel_labels(grouped_files, component_count)
    channel_labels, channel_templates = _derive_channel_labels_and_templates(grouped_files, component_count)

    _ensure_metadata_plane_count(
        parsed_args,
        component_count,
        sample_file.is_rgb,
        channel_labels,
        rename_existing_with_labels=not channels_were_user_provided,
        channel_templates=channel_templates,
        apply_templates_to_existing=not channels_were_user_provided,
    )

    result: dict[str, Any] = dict(dimensions)
    result["is_rgb"] = bool(sample_file.is_rgb)
    result["has_file_info"] = bool(dimensions)
    result["qml_settings"] = _build_qml_settings(parsed_args, channel_labels)
    result["has_qml_settings"] = bool(result["qml_settings"])
    result["error"] = False
    result["error_message"] = ""
    result["sequence_info"] = {
        "input_files": len(file_sources),
        "missing_file_combinations": len(missing_combinations),
        "allow_missing_files": bool(parsed_args.allow_missing_files),
        "flatten_duplicates": bool(parsed_args.flatten_duplicates),
    }
    return result


def plan_sequence_as_json(args: list[str] | None = None) -> None:
    """CLI wrapper around :func:`plan_sequence` printing JSON to stdout."""
    if args is None:
        args = sys.argv[1:]

    _convert_utils_mod.LOG_TYPE = LogType.NONE
    try:
        result = plan_sequence(args)
    except SystemExit as exc:
        result = {
            "error": True,
            "error_message": f"Invalid arguments (exit code {exc.code}).",
        }
    except Exception as exc:
        result = {
            "error": True,
            "error_message": str(exc),
        }
    print(json.dumps(result, indent=2))
