"""TIFF and OME-TIFF implementation for Image Information JSON.

This module deliberately owns the optional :mod:`tifffile` dependency.  The
ND2-only :mod:`limnd2.image_info` entry point therefore remains usable without
TIFF support installed.
"""

from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .metadata import PictureMetadata
from .metadata_tiff import (
    picture_metadata_from_ome_tiff_file,
    picture_metadata_sequence_from_tiff_tag,
)
from .textinfo import ImageTextInfo
from .variant import decode_var
from .experiment_tiff import experiment_from_tiff_tag, recorded_time_step_ms_from_tiff_tag

_COMMONFF_HINT = (
    '[commonff] extra not installed. '
    'Install it with `pip install "limnd2[commonff]"`.'
)


def gather_image_info_from_tiff(path: str | Path, *, last_modified: str | None = None) -> dict[str, Any]:
    """Return Image Information JSON for a TIFF or OME-TIFF file.

    Native NIS private TIFF tags are authoritative.  OME-XML enriches or
    fills absent fields, and standard descriptive TIFF tags are the final
    fallback.
    """
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            'Missing optional dependency "tifffile" required for TIFF Image Information. '
            + _COMMONFF_HINT
        ) from exc

    file_path = Path(path)
    with tifffile.TiffFile(file_path) as tif:
        height, width, components, frame_count, dtype = _tiff_image_layout(tif, file_path)
        ome_xml, is_ome = tif.ome_metadata, bool(tif.ome_metadata)
        text_tags = {tag.name: str(tag.value) for tag in tif.pages[0].tags.values()
                     if tag.name in {"Artist", "Copyright", "DateTime", "DocumentName", "ImageDescription", "Software"}}
        native_text_info = _nis_image_text_info_from_tiff(tif)
        native_created_by = _nis_created_by_from_tiff(tif)
        native_experiment = _nis_experiment_from_tiff(tif)
        native_time_step_ms = _nis_recorded_time_step_from_tiff(tif)
        native_frame_metadata = _nis_frame_metadata_from_tiff(tif)

    picture_metadata = picture_metadata_from_ome_tiff_file(file_path)
    if picture_metadata is None:
        picture_metadata = PictureMetadata()
        picture_metadata.makeValid(components)

    ome_info = _ome_image_information(ome_xml, native_text_info) if ome_xml else {}
    from .image_info import _picture_planes_to_table, format_general_info_sizes
    acquisition_details = _ome_acquisition_details(
        _picture_planes_to_table(picture_metadata.sPicturePlanes), ome_info
    )
    _apply_legacy_objective_name(acquisition_details, picture_metadata)
    pixel_type = "float" if dtype.kind == "f" else "int" if dtype.kind == "i" else "uint"
    bit_depth = f"{dtype.itemsize * 8}bit {pixel_type}"
    dimension = f"{width} x {height} ({components} {'comps' if components > 1 else 'comp'} {bit_depth})" + (f" x {frame_count} frames" if frame_count > 1 else "")
    return {
        "generalInfo": {
            "filename": file_path.name, "path": str(file_path.parent) + os.sep,
            "bit_depth": bit_depth, "loops": "", "dimension": dimension,
            "calibration": f"{picture_metadata.dCalibration:.3f} µm/px" if picture_metadata.bCalibrated else "Uncalibrated",
            "mtime": last_modified or "", "app_created": native_created_by or ("OME-TIFF" if is_ome else "TIFF"),
            **format_general_info_sizes(file_path.stat().st_size, width * height * components * dtype.itemsize, 0),
        },
        "imageTextInfo": native_text_info or _tiff_image_text_info(ome_info, text_tags),
        "experimentData": _experiment_data(native_experiment) or ome_info.get("experimentData", []),
        "customMetadata": _tiff_custom_metadata(text_tags),
        "recordedData": _recorded_data(native_frame_metadata, native_experiment, native_time_step_ms) or ome_info.get("recordedData"),
        "acquisitionDetails": acquisition_details,
    }


def _tiff_image_layout(tif: Any, file_path: Path) -> tuple[int, int, int, int, Any]:
    """Return basic image layout, tolerating malformed OME series references.

    A file can have valid individual TIFF pages and valid NIS metadata while
    its OME ``TiffData`` keyframes disagree.  ``tifffile.series`` rejects that
    situation; Image Information only needs the first page's basic geometry.
    """
    try:
        series = tif.series[0]
    except (IndexError, RuntimeError):
        if not tif.pages:
            raise ValueError(f"TIFF contains no readable image pages: {file_path}")
        page = tif.pages[0]
        if not hasattr(page, "dtype") and hasattr(page, "aspage"):
            page = page.aspage()
        return page.imagelength, page.imagewidth, max(1, page.samplesperpixel), len(tif.pages), page.dtype
    shape = dict(zip(str(series.axes), series.shape))
    height, width = int(shape.get("Y", 1)), int(shape.get("X", 1))
    samples = int(shape.get("S", 0))
    components = samples if samples else int(shape.get("C", 1))
    return height, width, components, max(1, len(series.pages)), series.dtype


def _nis_image_text_info_from_tiff(tif: Any) -> dict[str, str] | None:
    """Decode modern and legacy NIS Image Fields private-tag records."""
    if (tag := _private_tiff_tag(tif, 65330)) is not None:
        payload = bytes(tag.value)
        offset = payload.find("SLxImageTextInfo".encode("utf-16le"))
        if offset >= 2:
            try:
                return ImageTextInfo.from_lv(payload[offset - 2:]).to_dict()
            except Exception:
                pass
    tag = _private_tiff_tag(tif, 65332)
    payload = _legacy_variant_data_from_tiff_tag(tag.value).get("TextInfoTiffV1_0") if tag is not None else None
    if payload is None:
        return None
    try:
        values = decode_var(payload)[0]
    except Exception:
        return None
    return {
        "imageId": values.get("sImageID", ""), "type": values.get("sType", ""),
        "group": values.get("sGroup", ""), "sampleId": values.get("sSampleID", ""),
        "author": values.get("sAuthor", ""), "description": values.get("sDescription", ""),
        "capturing": values.get("sCapturing", ""), "sampling": values.get("sSampling", ""),
        "location": values.get("sLocation", ""), "date": values.get("sDate", ""),
        "conclusion": values.get("sConclusion", ""), "info1": values.get("sInfo1", ""),
        "info2": values.get("sInfo2", ""), "optics": values.get("sOptics", ""),
    }


def _legacy_variant_data_from_tiff_tag(data: bytes | bytearray | memoryview) -> dict[str, bytes]:
    """Read name-to-XML entries from a legacy NIS custom-variant tag."""
    payload = memoryview(data).cast("B")
    if len(payload) < 4:
        return {}
    count, position = struct.unpack_from("<I", payload, 0)[0], 4
    result: dict[str, bytes] = {}
    for _ in range(count):
        if position + 4 > len(payload):
            return result
        name_length = struct.unpack_from("<I", payload, position)[0]
        position += 4
        name_end = position + name_length * 2
        if name_end + 8 > len(payload):
            return result
        try:
            name = bytes(payload[position:name_end]).decode("utf-16le")
        except UnicodeDecodeError:
            return result
        size, offset = struct.unpack_from("<II", payload, name_end)
        position = name_end + 8
        end = offset + size
        if offset <= len(payload) and end <= len(payload):
            result[name] = bytes(payload[offset:end])
    return result


def _nis_created_by_from_tiff(tif: Any) -> str | None:
    """Read NIS application name and version from private AppInfo tag 65332."""
    if (tag := _private_tiff_tag(tif, 65332)) is None:
        return None
    text = bytes(tag.value).decode("utf-16le", "ignore")
    name = re.search(r'm_SWNameString[^>]*value="([^"]+)"', text)
    version = re.search(r'm_VersionString[^>]*value="([^"]+)"', text)
    values = [match.group(1) for match in (name, version) if match is not None]
    return " ".join(values) or None


def _nis_experiment_from_tiff(tif: Any) -> Any | None:
    """Return native NIS experiment metadata from the final TIFF page."""
    if (tag := _private_tiff_tag(tif, 65330)) is None:
        return None
    return experiment_from_tiff_tag(tag.value)


def _nis_frame_metadata_from_tiff(tif: Any) -> list[PictureMetadata]:
    """Return the native per-frame ``MetadataTiffV1_0`` sequence, if present."""
    if (tag := _private_tiff_tag(tif, 65331)) is None:
        return []
    return picture_metadata_sequence_from_tiff_tag(tag.value)


def _nis_recorded_time_step_from_tiff(tif: Any) -> float | None:
    """Return the native Recorded Data interval stored with ExperimentTiff."""
    tag = _private_tiff_tag(tif, 65330)
    return None if tag is None else recorded_time_step_ms_from_tiff_tag(tag.value)


def _experiment_data(experiment: Any | None) -> list[dict[str, Any]]:
    """Convert a native ``SLxExperiment`` into Image Information loop tables."""
    if experiment is None:
        return []
    from .image_info import _experiment_to_table
    return [
        {"ClassName": level.name.lower(), "LoopName": f"{level.name} Loop",
         "data": _experiment_to_table(level)}
        for level in experiment
        if level.valid and level.uLoopPars.info
    ]


def _recorded_data(
    metadata: list[PictureMetadata], experiment: Any | None, time_step_ms: float | None,
) -> dict[str, Any] | None:
    """Return native TIFF frame values in the Image Information table schema.

    Every NIS metadata sequence supplies frame index and acquisition time.  A
    matching native Z-stack additionally supplies the frame-specific Z series,
    reconstructed from its authoritative bottom position and step size.
    """
    z_positions = _z_series_from_experiment(experiment, len(metadata))
    if z_positions is None and experiment is not None:
        for level in experiment:
            info = level.uLoopPars.info
            if level.shortName == "Z" and info and isinstance(info[0].get("Count"), int):
                z_positions = _z_series_from_experiment(experiment, info[0]["Count"])
                break
    if not metadata and z_positions is None:
        return None
    from .image_info import _get_format_fn
    number_style = {"text-align": "right"}
    coldefs = [{"id": "id", "hidden": True}]
    if z_positions is None:
        coldefs.extend([
            {"id": "INDEX", "title": "Index", "fmtfncode": "(coldef) => { coldef.fmtfn = String };", "style": number_style},
            {"id": "ACQTIME", "title": "Time [h:m:s.ms]", "fmtfncode": "(coldef) => { coldef.fmtfn = String };", "style": number_style},
        ])
    else:
        coldefs.extend([
            {"id": "TIME", "title": "Time [s]", "fmtfncode": _get_format_fn(4), "style": number_style},
            {"id": "DELTATIME", "title": "Delta Time [s]", "fmtfncode": _get_format_fn(4), "style": number_style},
            {"id": "INDEX", "title": "Index", "fmtfncode": "(coldef) => { coldef.fmtfn = String };", "style": number_style},
            {"id": "Z", "title": "Z series [µm]", "fmtfncode": _get_format_fn(3), "style": number_style},
        ])
    # Some NIS TIFFs save a single shared MetadataTiff record but preserve the
    # experiment's Recorded Data interval.  Expand that authoritative Z stack
    # into its logical frame rows in that case.
    synthetic_rows = z_positions is not None and len(metadata) != len(z_positions) and time_step_ms is not None
    frame_count = len(z_positions) if synthetic_rows and z_positions is not None else len(metadata)
    if not frame_count:
        return None
    rows = []
    previous_time: float | None = None
    for index in range(1, frame_count + 1):
        item = metadata[index - 1] if index <= len(metadata) else None
        time_seconds = ((index - 1) * time_step_ms / 1000.0) if synthetic_rows else (item.dTimeMSec / 1000.0 if item is not None else 0.0)
        row: dict[str, Any] = {"id": index, "INDEX": index}
        if z_positions is None:
            row["ACQTIME"] = _format_acquisition_time(item.dTimeMSec)
        else:
            row.update({
                "TIME": time_seconds,
                "DELTATIME": 0.0 if previous_time is None else time_seconds - previous_time,
                "Z": z_positions[index - 1],
            })
        previous_time = time_seconds
        rows.append(row)
    return {
        "coldefs": coldefs + [{"id": "tail"}],
        "rowdata": rows,
    }


def _z_series_from_experiment(experiment: Any | None, count: int) -> list[float] | None:
    """Return frame Z positions when a native Z-stack exactly covers frames."""
    if experiment is None:
        return None
    for level in experiment:
        info = level.uLoopPars.info
        if level.shortName != "Z" or not info:
            continue
        values = info[0]
        if values.get("Count") != count:
            continue
        bottom, step = values.get("Bottom"), values.get("Step")
        if isinstance(bottom, (int, float)) and isinstance(step, (int, float)):
            # The NIS UI expands the displayed three-decimal step.  Matching
            # that convention keeps the per-frame Z series consistent with
            # the Experiment Data table (e.g. 0.250 µm, not 0.249622...).
            display_step = round(step, 3)
            return [round(bottom + index * display_step, 3) for index in range(count)]
    return None


def _format_acquisition_time(milliseconds: float) -> str:
    """Format NIS millisecond timestamps exactly as the ND2 Recorded Data UI."""
    total = max(0, round(milliseconds))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _private_tiff_tag(tif: Any, code: int) -> Any | None:
    """Find a private tag, materializing lightweight OME frames if needed."""
    page_count = len(tif.pages)
    page_order = () if page_count == 0 else ((0,) if page_count == 1 else (0, page_count - 1, *range(1, page_count - 1)))
    for index in page_order:
        page = tif.pages[index]
        if not hasattr(page, "tags") and hasattr(page, "aspage"):
            page = page.aspage()
        tags = getattr(page, "tags", None)
        if tags is not None and (tag := tags.get(code)) is not None:
            return tag
    return None


def _local_name(element: ET.Element) -> str:
    """Return an XML tag name without its namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _first(element: ET.Element | None, name: str) -> ET.Element | None:
    """Return the first descendant with the requested namespace-independent name."""
    return next((item for item in element.iter() if _local_name(item) == name), None) if element is not None else None


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    """Return matching direct children without requiring a particular XML namespace."""
    return [item for item in element if _local_name(item) == name] if element is not None else []


def _number(value: str | None, unit: str | None = None, *, time: bool = False) -> float | None:
    """Parse an OME length as micrometres, or a time quantity as seconds."""
    try:
        number = float(value) if value is not None else None
    except ValueError:
        return None
    if number is None:
        return None
    units = {"ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0} if time else {"nm": 1e-3, "um": 1.0, "µm": 1.0, "mm": 1e3, "m": 1e6}
    return number * units.get((unit or ("s" if time else "um")).lower(), 1.0)


def _ome_image_information(ome_xml: str, native_text_info: dict[str, str] | None = None) -> dict[str, Any]:
    """Build UI loop and per-plane tables from standard OME-XML fields."""
    try:
        root = ET.fromstring(ome_xml)
    except (ET.ParseError, TypeError, ValueError):
        return {}
    image, pixels = _first(root, "Image"), _first(_first(root, "Image"), "Pixels")
    if pixels is None:
        return {}
    def dim(axis: str) -> int:
        try: return max(1, int(pixels.get(f"Size{axis}", "1")))
        except ValueError: return 1
    dimensions = {axis: dim(axis) for axis in "TCZYX"}
    steps = {"Z": _number(pixels.get("PhysicalSizeZ"), pixels.get("PhysicalSizeZUnit")), "T": _number(pixels.get("TimeIncrement"), pixels.get("TimeIncrementUnit"), time=True)}
    experiment_data = []
    # Channels are picture components, not an acquisition experiment loop.
    for axis, title in (("T", "Time Loop"), ("Z", "Z Stack Loop")):
        if dimensions[axis] <= 1: continue
        rows = [{"id": 1, "parameter": "Count", "value": dimensions[axis]}]
        if steps.get(axis) is not None: rows.append({"id": 2, "parameter": "Step size", "value": f"{steps[axis]:g} {'s' if axis == 'T' else 'µm'}"})
        experiment_data.append({"ClassName": axis.lower(), "LoopName": title, "data": {"coldefs": [{"id": "id", "hidden": True}, {"id": "parameter", "title": "Parameter"}, {"id": "value", "title": "Value"}], "rowdata": rows}})
    objective = _objective(root, image, native_text_info)
    columns = [("the_t", "T index"), ("the_c", "C index"), ("the_z", "Z index"), ("z", "Z position [µm]"), ("time", "Time [s]"), ("exposure", "Exposure [s]")]
    rows = []
    for index, plane in enumerate(_children(pixels, "Plane"), 1):
        row: dict[str, Any] = {"id": index}
        for attr, key in (("TheT", "the_t"), ("TheC", "the_c"), ("TheZ", "the_z")):
            if plane.get(attr) is not None: row[key] = int(plane.get(attr, "0"))
        for attr, key, time in (("PositionZ", "z", False), ("DeltaT", "time", True), ("ExposureTime", "exposure", True)):
            if (value := _number(plane.get(attr), plane.get(f"{attr}Unit"), time=time)) is not None: row[key] = value
        rows.append(row)
    # OME lists one Plane per (Z, C, T); Image Information is frame-oriented,
    # so represent one row per (Z, T) and retain the first channel's position.
    frames: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        frames.setdefault((row.get("the_t", 0), row.get("the_z", 0)), row)
    rows = list(frames.values())
    for index, row in enumerate(rows, 1):
        row["id"], row["index"] = index, index
    z_values = [row["z"] for row in rows if "z" in row]
    if z_values:
        z_center = (min(z_values) + max(z_values)) / 2
        for row in rows:
            if "z" in row:
                row["z"] -= z_center
    columns = [("time", "Time [s]"), ("index", "Index"), ("z", "Z series [µm]")]
    return {"root": root, "image": image, "objective": objective, "experimentData": experiment_data,
            "recordedData": {"coldefs": [{"id": "id", "hidden": True}] + [{"id": key, "title": title} for key, title in columns] + [{"id": "tail"}], "rowdata": rows} if rows else None}


def _objective(root: ET.Element, image: ET.Element | None, native_text_info: dict[str, str] | None) -> dict[str, Any]:
    """Return standard OME objective values for acquisition and optics fields."""
    instrument_ref, settings = _first(image, "InstrumentRef"), _first(image, "ObjectiveSettings")
    instrument_id, objective_id = (instrument_ref.get("ID") if instrument_ref is not None else None), (settings.get("ID") if settings is not None else None)
    for instrument in (item for item in root.iter() if _local_name(item) == "Instrument"):
        if instrument_id and instrument.get("ID") != instrument_id: continue
        objective = next((item for item in _children(instrument, "Objective") if item.get("ID") == objective_id), None)
        if objective is not None:
            magnification = _number(objective.get("NominalMagnification"))
            description = (native_text_info or {}).get("description", "")
            match = re.search(r"Refractive Index:\s*([0-9]+(?:[,.][0-9]+)?)", description)
            native_ri = float(match.group(1).replace(",", ".")) if match else None
            result = {"Objective name:": objective.get("Model"), "Objective numerical aperture:": _number(objective.get("LensNA")), "Objective magnification:": magnification, "Objective immersion:": objective.get("Immersion"), "Refractive index:": native_ri if native_ri is not None else (_number(settings.get("RefractiveIndex")) if settings is not None else None)}
            result["optics"] = f"{magnification:g}x" if magnification is not None else ""
            return {key: value for key, value in result.items() if value is not None}
    return {}


def _tiff_image_text_info(info: dict[str, Any], tags: dict[str, str]) -> dict[str, str]:
    """Map standard OME/TIFF descriptive fields to the Image Fields schema."""
    image = info.get("image")
    description = ((_first(image, "Description").text if _first(image, "Description") is not None else "") or (image.get("Description", "") if image is not None else tags.get("ImageDescription", "")))
    date = ((_first(image, "AcquisitionDate").text if _first(image, "AcquisitionDate") is not None else "") or tags.get("DateTime", ""))
    image_id = image.get("ID", tags.get("DocumentName", "")) if image is not None else tags.get("DocumentName", "")
    return {"imageId": image_id, "type": "", "group": "", "sampleId": "", "author": tags.get("Artist", ""), "description": description, "capturing": "", "sampling": "", "location": "", "date": date, "conclusion": "", "info1": "", "info2": "", "optics": info.get("objective", {}).get("optics", "")}


def _tiff_custom_metadata(tags: dict[str, str]) -> list[dict[str, Any]]:
    """Return compact descriptive TIFF tags without duplicating Image Fields."""
    return [{"name": name, "text": value, "type": 0} for name, value in tags.items() if name not in {"Artist", "DateTime", "DocumentName", "ImageDescription"}]


def _ome_acquisition_details(table: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Overlay standard and OME-only acquisition values and rebuild groups."""
    objective, rows = info.get("objective", {}), table.get("rowdata", [])
    camera_by_channel = {
        row["channel"]: row["camera"] for row in rows
        if row.get("channel") and row.get("camera")
    }
    for row in rows:
        if row.get("feature") in objective: row["value"] = objective[row["feature"]]
    for channel, details in _ome_channel_details(info).items():
        # OME represents a camera RGB composite as a single channel named
        # ``RGB``.  Native NIS metadata describes the same image as its sole
        # Brightfield plane; do not create a misleading second channel group.
        target_channel = channel
        if channel not in camera_by_channel and channel == "RGB" and len(camera_by_channel) == 1:
            target_channel = next(iter(camera_by_channel))
        for feature, value in details:
            rows.append({"id": str(len(rows) + 1), "camera": camera_by_channel.get(target_channel, "Unknown camera"), "channel": target_channel, "feature": feature, "value": value})
    for channel in camera_by_channel:
        if objective.get("Objective immersion:") is not None:
            rows.append({"id": str(len(rows) + 1), "camera": camera_by_channel[channel], "channel": channel, "feature": "Objective immersion:", "value": objective["Objective immersion:"]})
    if table.get("groupedby"):
        from .image_info import _create_treeview_grouping
        table["groups"] = _create_treeview_grouping(rows, table["groupedby"].copy())
    return table


def _ome_channel_details(info: dict[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    """Extract detector and filter metadata that has no ND2 plane equivalent."""
    root, image = info.get("root"), info.get("image")
    pixels = _first(image, "Pixels")
    if root is None or pixels is None:
        return {}
    instrument_ref = _first(image, "InstrumentRef")
    instrument_id = instrument_ref.get("ID") if instrument_ref is not None else None
    instrument = next((item for item in _children(root, "Instrument") if not instrument_id or item.get("ID") == instrument_id), None)
    if instrument is None:
        return {}
    detectors = {item.get("ID"): item for item in _children(instrument, "Detector")}
    filter_sets = {item.get("ID"): item for item in _children(instrument, "FilterSet")}
    filters = {item.get("ID"): item for item in _children(instrument, "Filter")}
    details: dict[str, list[tuple[str, Any]]] = {}
    for channel in _children(pixels, "Channel"):
        values: list[tuple[str, Any]] = []
        detector_settings = _first(channel, "DetectorSettings")
        if detector_settings is not None:
            detector = detectors.get(detector_settings.get("ID"))
            if detector is not None and detector.get("Model"):
                values.append(("Detector name:", detector.get("Model")))
            if detector_settings.get("Zoom"):
                values.append(("Detector zoom:", _number(detector_settings.get("Zoom"))))
        filter_set_ref = _first(channel, "FilterSetRef")
        filter_set = filter_sets.get(filter_set_ref.get("ID")) if filter_set_ref is not None else None
        if filter_set is not None:
            for ref_name, label in (("ExcitationFilterRef", "Excitation filter:"), ("EmissionFilterRef", "Emission filter:")):
                reference = _first(filter_set, ref_name)
                filter_ = filters.get(reference.get("ID")) if reference is not None else None
                if filter_ is not None:
                    values.append((label, _filter_description(filter_)))
        if values:
            details[channel.get("Name") or channel.get("ID") or "Unknown channel"] = values
    return details


def _filter_description(filter_: ET.Element) -> str:
    """Render an OME filter type and transmission band for the UI."""
    transmission = _first(filter_, "TransmittanceRange")
    filter_type = filter_.get("Type") or "Filter"
    if transmission is None:
        return filter_type
    low, high = transmission.get("CutIn"), transmission.get("CutOut")
    if low is None or high is None:
        return filter_type
    return f"{filter_type} {low}–{high} {transmission.get('CutInUnit') or 'nm'}"


def _apply_legacy_objective_name(table: dict[str, Any], metadata: PictureMetadata) -> None:
    """Fill blank objective rows from legacy global ``wsObjectiveName`` metadata."""
    if not metadata.wsObjectiveName:
        return
    for row in table.get("rowdata", []):
        if row.get("feature") == "Objective name:" and row.get("value") in ("", "N/A"):
            row["value"] = metadata.wsObjectiveName
