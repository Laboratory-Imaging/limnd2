"""Metadata interchange helpers for OME-TIFF and NIS Elements TIFF files.

The functions in this module intentionally operate on XML and tag *bytes*.
They do not require :mod:`tifffile`; :func:`picture_metadata_from_tiff_file`
is the optional convenience wrapper for callers that have it installed.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import struct
from typing import Any
from xml.etree import ElementTree as ET

from .metadata import PictureMetadata
from .metadata_factory import MetadataFactory, Plane

NIS_PICTURE_METADATA_TAG = 65331
"""NIS Elements TIFF private tag carrying ``SLxPictureMetadata`` records."""

NIS_PICTURE_METADATA_VARIANT_TAG = 65333
"""Legacy NIS TIFF tag containing XML-serialized ``RLxPictureMetadata``."""

NIS_METADATA_TIFF_KEY = "MetadataTiffV1_0"
"""Custom-data sequence key for the original per-frame picture metadata."""


def picture_metadata_from_tiff_tag(data: bytes | bytearray | memoryview) -> PictureMetadata | None:
    """Decode NIS ``SLxPictureMetadata`` from a TIFF tag payload.

    The LiteVariant header starts two bytes before the UTF-16LE record name.
    TIFF exports can contain other data before the record, so this helper
    searches for the marker rather than assuming a tag layout.  Invalid or
    unrelated private-tag data is deliberately non-fatal and returns ``None``.
    """

    payload = bytes(data)
    marker = "SLxPictureMetadata".encode("utf-16le")
    name_offset = payload.find(marker)
    if name_offset < 2:
        return None
    try:
        metadata = PictureMetadata.from_lv(payload[name_offset - 2:])
    except Exception:
        return None
    return metadata if metadata.channels else None


def picture_metadata_sequence_from_tiff_tag(
    data: bytes | bytearray | memoryview,
) -> list[PictureMetadata]:
    """Decode original per-frame metadata from a NIS tag-65331 payload.

    Tag 65331 is a NIS *custom-data sequence table*, not a single LiteVariant.
    Its ``MetadataTiffV1_0`` sequence contains one ``SLxPictureMetadata``
    record per acquired frame.  Invalid records are skipped individually.
    """
    records = _custom_data_sequence_from_tiff_tag(data).get(NIS_METADATA_TIFF_KEY, {})
    metadata: list[PictureMetadata] = []
    for _, payload in sorted(records.items()):
        item = picture_metadata_from_tiff_tag(payload)
        if item is not None:
            metadata.append(item)
    return metadata


def picture_metadata_from_tiff_variant_tag(
    data: bytes | bytearray | memoryview,
) -> PictureMetadata | None:
    """Decode legacy XML-variant ``MetadataTiffV1_0`` metadata from tag 65333."""
    payload = _custom_data_sequence_from_tiff_tag(data).get(NIS_METADATA_TIFF_KEY, {}).get(0)
    if payload is None:
        return None
    try:
        root = ET.fromstring(payload.decode("utf-16le"))
    except (UnicodeDecodeError, ET.ParseError):
        return None
    record = next((item for item in root if _local_name(item) == "no_name"), None)
    if record is None:
        return None
    metadata = PictureMetadata()
    float_fields = {"dTimeMSec", "dTimeAbsolute", "dXPos", "dYPos", "dZPos", "dCalibration", "dAspect", "dObjectiveMag", "dObjectiveNA", "dRefractIndex1", "dRefractIndex2", "dPinholeRadius", "dZoom"}
    for item in record:
        name, value = _local_name(item), item.get("value")
        if value is None:
            continue
        try:
            if name in float_fields:
                object.__setattr__(metadata, name, float(value))
            elif name in {"bZPosAbsolute", "bCalibrated"}:
                object.__setattr__(metadata, name, value.lower() == "true")
            elif name == "wsObjectiveName":
                object.__setattr__(metadata, name, value)
        except (TypeError, ValueError):
            continue
    components = _variant_component_count(record)
    metadata.makeValid(components)
    return metadata


def picture_metadata_from_tiff_file(path: str | Path) -> PictureMetadata | None:
    """Read tag 65331 from the final page of *path* and decode it if present.

    This is intentionally the only function in this module that imports the
    optional ``tifffile`` dependency.
    """

    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            'Missing optional dependency "tifffile"; use '
            "picture_metadata_from_tiff_tag() when tag bytes are available."
        ) from exc

    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return None
            tag = _private_tiff_tag(tif, NIS_PICTURE_METADATA_TAG)
            native = picture_metadata_from_tiff_tag(tag.value) if tag is not None else None
            if native is not None:
                return native
            variant = _private_tiff_tag(tif, NIS_PICTURE_METADATA_VARIANT_TAG)
            return picture_metadata_from_tiff_variant_tag(variant.value) if variant is not None else None
    except (OSError, ValueError):
        return None


def _custom_data_sequence_from_tiff_tag(
    data: bytes | bytearray | memoryview,
) -> dict[str, dict[int, bytes]]:
    """Read the NIS custom-data sequence-table layout used by tag 65331."""
    payload = memoryview(data).cast("B")
    if len(payload) < 4:
        return {}
    count = struct.unpack_from("<I", payload, 0)[0]
    position = 4
    result: dict[str, dict[int, bytes]] = {}
    for _ in range(count):
        if position + 4 > len(payload):
            return result
        name_length = struct.unpack_from("<I", payload, position)[0]
        position += 4
        name_end = position + name_length * 2
        if name_end + 4 > len(payload):
            return result
        try:
            name = bytes(payload[position:name_end]).decode("utf-16le")
        except UnicodeDecodeError:
            return result
        sequence_count = struct.unpack_from("<I", payload, name_end)[0]
        position = name_end + 4
        entries: dict[int, bytes] = {}
        for _ in range(sequence_count):
            if position + 12 > len(payload):
                return result
            index, length, offset = struct.unpack_from("<III", payload, position)
            position += 12
            end = offset + length
            if offset <= len(payload) and end <= len(payload):
                entries[index] = bytes(payload[offset:end])
        result[name] = entries
    return result


def _private_tiff_tag(tif: Any, code: int) -> Any | None:
    """Find a private tag, materializing lightweight OME frames if needed."""
    for index in _metadata_page_order(len(tif.pages)):
        page = tif.pages[index]
        if not hasattr(page, "tags") and hasattr(page, "aspage"):
            page = page.aspage()
        tags = getattr(page, "tags", None)
        if tags is not None and (tag := tags.get(code)) is not None:
            return tag
    return None


def _metadata_page_order(page_count: int) -> tuple[int, ...]:
    """Prefer primary-page metadata, then trailing metadata, then middle pages."""
    if page_count < 2:
        return (0,) if page_count else ()
    return (0, page_count - 1, *range(1, page_count - 1))


def _variant_component_count(record: ET.Element) -> int:
    """Get the component count from a legacy ``RLxPictureMetadata`` XML node."""
    planes = next((item for item in record if _local_name(item) == "sPicturePlanes"), None)
    if planes is None:
        return 1
    value = next((item.get("value") for item in planes if _local_name(item) == "uiCompCount"), None)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def picture_metadata_from_ome_xml(ome_xml: str | bytes) -> PictureMetadata | None:
    """Convert OME-XML directly into the closest representable ND2 metadata.

    A ``nd2.picture_metadata.full`` annotation emitted by :func:`to_ome_xml`
    takes precedence, providing a lossless limnd2 round trip.  Otherwise this
    maps standard OME pixels, channels, calibration, timing, stage position,
    and the common objective fields from the first OME image.
    """

    try:
        root = ET.fromstring(ome_xml)
    except (ET.ParseError, TypeError, ValueError):
        return None

    native_metadata = _full_picture_metadata_annotation(root)
    image = _first(root, "Image")
    pixels = _first(image, "Pixels") if image is not None else None
    if pixels is None:
        return None

    dimensions = {
        axis: _positive_int(pixels.get(f"Size{axis}"), 1)
        for axis in ("T", "C", "Z", "Y", "X")
    }
    pixel_calibration = _length_um(
        pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit")
    )
    z_step_um = _length_um(pixels.get("PhysicalSizeZ"), pixels.get("PhysicalSizeZUnit"))
    time_step_seconds = _time_seconds(
        pixels.get("TimeIncrement"), pixels.get("TimeIncrementUnit")
    )

    settings = _objective_settings(root, image)
    factory = MetadataFactory(
        pixel_calibration=pixel_calibration if pixel_calibration is not None else -1.0,
        **settings,
    )
    channels: list[Plane] = []
    for index, channel in enumerate(_children(pixels, "Channel")):
        plane = Plane(
            name=channel.get("Name") or f"Channel {index + 1}",
            color=_ome_color_to_rgb(channel.get("Color")),
            modality=_ome_modality(channel),
            excitation_wavelength=_float_or_none(channel.get("ExcitationWavelength")),
            emission_wavelength=_float_or_none(channel.get("EmissionWavelength")),
            pinhole_diameter=_length_um(channel.get("PinholeSize"), channel.get("PinholeSizeUnit")),
        )
        channels.append(plane)
        factory.addPlane(plane)

    if not channels:
        for index in range(dimensions["C"]):
            plane = Plane(name=f"Channel {index + 1}")
            channels.append(plane)
            factory.addPlane(plane)

    picture_metadata = native_metadata or factory.createMetadata(
        number_of_channels_fallback=dimensions["C"]
    )
    if native_metadata is None:
        _apply_ome_picture_values(
            picture_metadata, image, pixels, z_step_um, time_step_seconds
        )
    return picture_metadata


def picture_metadata_from_ome_tiff_file(path: str | Path) -> PictureMetadata | None:
    """Read an OME-TIFF and return native NIS metadata or an OME approximation.

    Native NIS tag 65331 wins when it is valid.  A regular OME-TIFF falls back
    to its OME-XML image description.  ``tifffile`` remains optional.
    """

    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            'Missing optional dependency "tifffile"; pass XML to '
            "picture_metadata_from_ome_xml() instead."
        ) from exc
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.pages:
                tag = _private_tiff_tag(tif, NIS_PICTURE_METADATA_TAG)
                if tag is not None:
                    native = picture_metadata_from_tiff_tag(tag.value)
                    if native is not None:
                        return native
                variant = _private_tiff_tag(tif, NIS_PICTURE_METADATA_VARIANT_TAG)
                if variant is not None:
                    native = picture_metadata_from_tiff_variant_tag(variant.value)
                    if native is not None:
                        return native
            return picture_metadata_from_ome_xml(tif.ome_metadata) if tif.ome_metadata else None
    except (OSError, ValueError):
        return None


def nd2_metadata_to_ome_xml(nd2_reader: Any, **kwargs: Any) -> str:
    """Create OME-XML from an ND2 reader without a module import cycle.

    This is a lazy facade for :func:`limnd2.export_ome_tiff.to_ome_xml`.
    ``metadata_tiff`` therefore remains safe for low-level metadata consumers.
    """

    from .export_ome_tiff import to_ome_xml

    return to_ome_xml(nd2_reader, **kwargs)


def _full_picture_metadata_annotation(root: ET.Element) -> PictureMetadata | None:
    """Restore the lossless PictureMetadata map annotation, when present."""
    for annotation in _iter(root, "MapAnnotation"):
        if _text(_first(annotation, "Description")) != "nd2.picture_metadata.full":
            continue
        for item in _iter(annotation, "M"):
            if item.get("K") != "json" or not item.text:
                continue
            try:
                return PictureMetadata(**json.loads(item.text))
            except (TypeError, ValueError):
                return None
    return None


def _apply_ome_picture_values(
    metadata: PictureMetadata,
    image: ET.Element | None,
    pixels: ET.Element,
    z_step_um: float | None,
    time_step_seconds: float | None,
) -> None:
    """Apply OME fields which have direct PictureMetadata counterparts."""
    if z_step_um is not None:
        object.__setattr__(metadata, "dZAxisCalibration", z_step_um)
    physical_y = _length_um(pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit"))
    physical_x = _length_um(pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit"))
    if physical_x and physical_y:
        object.__setattr__(metadata, "dAspect", physical_y / physical_x)
    if time_step_seconds is not None:
        object.__setattr__(metadata, "dTimeAxisCalibration", time_step_seconds * 1000.0)
    plane = _first(pixels, "Plane")
    if plane is not None:
        for ome_name, nd2_name in (("PositionX", "dXPos"), ("PositionY", "dYPos"), ("PositionZ", "dZPos")):
            value = _length_um(plane.get(ome_name), plane.get(f"{ome_name}Unit"))
            if value is not None:
                object.__setattr__(metadata, nd2_name, value)
    acquisition = _text(_first(image, "AcquisitionDate")) if image is not None else None
    if acquisition:
        try:
            dt = datetime.fromisoformat(acquisition.replace("Z", "+00:00"))
            object.__setattr__(metadata, "dTimeAbsolute", dt.timestamp() / 86400.0 + 2440587.5)
        except ValueError:
            pass
    if image is not None:
        description = _text(_first(image, "Description")) or image.get("Description")
        if description:
            object.__setattr__(metadata, "wsCustomData", description)


def _objective_settings(root: ET.Element, image: ET.Element | None) -> dict[str, float]:
    """Extract the global objective values supported by MetadataFactory."""
    if image is None:
        return {}
    objective_ref = _first(image, "ObjectiveSettings")
    instrument_ref = _first(image, "InstrumentRef")
    objective_id = objective_ref.get("ID") if objective_ref is not None else None
    instrument_id = instrument_ref.get("ID") if instrument_ref is not None else None
    objective = None
    for instrument in _iter(root, "Instrument"):
        if instrument_id and instrument.get("ID") != instrument_id:
            continue
        objective = next((item for item in _children(instrument, "Objective") if item.get("ID") == objective_id), None)
        if objective is not None:
            break
    values = {
        "objective_numerical_aperture": _float_or_none(objective.get("LensNA")) if objective is not None else None,
        "objective_magnification": _float_or_none(objective.get("NominalMagnification")) if objective is not None else None,
        "immersion_refractive_index": _float_or_none(objective_ref.get("RefractiveIndex")) if objective_ref is not None else None,
    }
    return {key: value for key, value in values.items() if value is not None}


def _local_name(element: ET.Element) -> str:
    """Return an XML element name without its optional namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _iter(element: ET.Element | None, name: str):
    """Iterate descendants whose namespace-independent name matches *name*."""
    return () if element is None else (item for item in element.iter() if _local_name(item) == name)


def _children(element: ET.Element | None, name: str):
    """Iterate direct children whose namespace-independent name matches *name*."""
    return () if element is None else (item for item in element if _local_name(item) == name)


def _first(element: ET.Element | None, name: str) -> ET.Element | None:
    """Return the first matching descendant, or ``None`` when absent."""
    return next(iter(_iter(element, name)), None)


def _text(element: ET.Element | None) -> str | None:
    """Return non-empty element text, normalized to ``None`` when missing."""
    return element.text if element is not None and element.text else None


def _positive_int(value: str | None, default: int) -> int:
    """Parse a positive integer, falling back safely for invalid OME values."""
    try:
        return max(1, int(value)) if value is not None else default
    except ValueError:
        return default


def _float_or_none(value: str | None) -> float | None:
    """Parse a floating-point XML attribute without propagating parse errors."""
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _length_um(value: str | None, unit: str | None) -> float | None:
    """Convert a supported OME length attribute to micrometres."""
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    scales = {None: 1.0, "µm": 1.0, "um": 1.0, "micrometer": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6}
    return numeric * scales.get((unit or "").lower() or None, 1.0)


def _time_seconds(value: str | None, unit: str | None) -> float | None:
    """Convert a supported OME time attribute to seconds."""
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    scales = {None: 1.0, "s": 1.0, "second": 1.0, "ms": 1e-3, "millisecond": 1e-3, "min": 60.0, "h": 3600.0}
    return numeric * scales.get((unit or "").lower() or None, 1.0)


def _ome_color_to_rgb(value: str | None) -> tuple[int, int, int] | None:
    """Convert OME's signed ``0xRRGGBBAA`` colour to normalized RGB."""
    if value is None:
        return None
    try:
        color = int(value) & 0xFFFFFFFF
    except ValueError:
        return None
    # ``MetadataFactory`` and ``calculateColor`` consume 8-bit RGB values,
    # not normalized floats.  OME stores RGBA as a signed 32-bit integer.
    return ((color >> 24) & 0xFF, (color >> 16) & 0xFF, (color >> 8) & 0xFF)


def _ome_modality(channel: ET.Element) -> str | None:
    """Select the OME modality field understood by ``metadata_factory.Plane``."""

    return channel.get("AcquisitionMode") or channel.get("ContrastMethod")
