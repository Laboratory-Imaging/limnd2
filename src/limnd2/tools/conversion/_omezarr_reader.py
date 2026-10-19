"""OME-Zarr image source supporting legacy v2 metadata and v3 metadata."""
from __future__ import annotations
import json
import copy
import itertools
import re
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import numpy as np
import limnd2
from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.binary import BinaryRasterMetadata, BinaryRasterMetadataFactory, BinaryRleMetadata
from limnd2.metadata import PictureMetadata
from limnd2.metadata_factory import Plane
from .LimImageSource import LimImageSource

def _metadata(path: Path) -> dict[str, Any] | None:
    for name in ("zarr.json", ".zattrs"):
        candidate = path / name
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return data.get("attributes", {}) if name == "zarr.json" else data
    return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None

class LimImageSourceOmeZarr(LimImageSource):
    supports_tile_read = True
    def __init__(self, filename: str | Path):
        super().__init__(filename)
        self.store_path = self.filename.parent
        self.group_path = Path(".")
        self.axis_indexes: dict[str, int] = {}
        self._groups: list[Path] | None = None
        self._native_metadata_models: dict[bytes, Any] | None = None
        self._rgb_by_group: dict[Path, bool] = {}
        self._selected_multipoint_groups: list[tuple[Path, dict[str, Any] | None]] | None = None
        self._multipoint_compatible = True
        # Set on expanded plate fields.  Keeping this on the source clone makes
        # the sequence-to-well mapping survive the generic conversion pipeline.
        self._ome_zarr_well: dict[str, Any] | None = None

    def native_nd2_metadata(self) -> dict[bytes, Any]:
        """Return supported limnd2 sidecar models, if this store provides them.

        ``nd2_chunks`` is deliberately optional OME-Zarr sidecar data.  Only
        models that can be written safely to a new ND2 are returned here; the
        generic converter passes them through to ``Nd2Writer``.
        """
        if self._native_metadata_models is not None:
            return self._native_metadata_models

        sidecar = self.store_path / "nd2_chunks"
        manifest_path = sidecar / "manifest.json"
        self._native_metadata_models = {}
        if manifest_path.is_file():
            try:
                # The shared importer validates schema version and reconstructs
                # all supported model kinds from the sidecar manifest.
                from limnd2.nd2_metadata_json import import_nd2_metadata_json
                self._native_metadata_models = import_nd2_metadata_json(sidecar)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        # Reuse the OME-TIFF XML parser for standard fields and limnd2's
        # lossless annotations. ND2 chunks remain authoritative when present.
        has_native_picture_metadata = any(
            isinstance(value, PictureMetadata)
            for value in self._native_metadata_models.values()
        )
        if not has_native_picture_metadata:
            xml_path = self._ome_xml_path()
            if xml_path is not None:
                try:
                    from limnd2.metadata_tiff import picture_metadata_from_ome_xml
                    picture = picture_metadata_from_ome_xml(
                        xml_path.read_bytes(), image_index=self._ome_xml_image_index()
                    )
                    if picture is not None:
                        self._native_metadata_models[b"ome-xml:pictureMetadata"] = picture
                except (OSError, ValueError):
                    pass
        xml_annotations = self._ome_xml_annotation_maps()
        if not any(isinstance(value, limnd2.ExperimentLevel) for value in self._native_metadata_models.values()):
            payload = xml_annotations.get("nd2.experiment.full")
            if isinstance(payload, dict):
                try:
                    experiment = limnd2.ExperimentLevel.fromDict(payload)
                    self._native_metadata_models[b"ome-xml:experiment"] = experiment
                except (TypeError, ValueError, KeyError):
                    pass
        if not any(isinstance(value, limnd2.ImageTextInfo) for value in self._native_metadata_models.values()):
            payload = xml_annotations.get("nd2.text_info.full")
            if isinstance(payload, dict):
                try:
                    text_info = limnd2.ImageTextInfo.fromDict(payload)
                    self._native_metadata_models[b"ImageTextInfoLV"] = text_info
                except (TypeError, ValueError, KeyError):
                    pass
        return self._native_metadata_models
    def _image_groups(self) -> list[Path]:
        if self._groups is None:
            groups: list[Path] = []
            for sentinel in (*self.store_path.rglob("zarr.json"), *self.store_path.rglob(".zattrs")):
                relative = sentinel.parent.relative_to(self.store_path)
                if "nd2_chunks" in relative.parts or "labels" in relative.parts:
                    continue
                metadata = _metadata(sentinel.parent) or {}
                if (metadata.get("ome", metadata)).get("multiscales"):
                    groups.append(relative)
            self._groups = sorted(set(groups), key=lambda item: (len(item.parts), str(item)))
            if not self._groups:
                raise ValueError(f"No OME-NGFF multiscale image was found in {self.store_path}.")
        return self._groups
    def _array(self):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("OME-Zarr import requires the ome-zarr extra.") from exc
        metadata = _metadata(self.store_path / self.group_path) or {}
        if not (metadata.get("ome", metadata)).get("multiscales"):
            # A root series/plate container has no pixel array of its own.
            self.group_path = self._image_groups()[0]
            metadata = _metadata(self.store_path / self.group_path) or {}
        multiscale = (metadata.get("ome", metadata))["multiscales"][0]
        axes = [axis if isinstance(axis, str) else axis["name"] for axis in multiscale["axes"]]
        level = multiscale["datasets"][0]["path"]
        return [axis.lower() for axis in axes], zarr.open_array(str(self.store_path / self.group_path / level), mode="r")

    def _plate(self) -> dict[str, Any] | None:
        """Return the root NGFF plate metadata, if this is a plate store."""
        root = _metadata(self.store_path) or {}
        plate = root.get("ome", root).get("plate")
        return plate if isinstance(plate, dict) else None

    def _plate_fields(self) -> list[tuple[Path, dict[str, Any]]]:
        """Read the exact row/well/field layout emitted by our OME-Zarr exporter."""
        plate = self._plate()
        if plate is None:
            return []

        row_names = [str(item.get("name", item)) for item in plate.get("rows", [])]
        col_names = [str(item.get("name", item)) for item in plate.get("columns", [])]
        fields: list[tuple[Path, dict[str, Any]]] = []
        for well in plate.get("wells", []):
            if not isinstance(well, dict) or not isinstance(well.get("path"), str):
                continue
            well_path = Path(well["path"])
            well_metadata = _metadata(self.store_path / well_path) or {}
            images = (well_metadata.get("ome", well_metadata).get("well", {}).get("images", []))
            row_name, col_name = (well_path.parts + ("", ""))[:2]
            row_index = row_names.index(row_name) if row_name in row_names else None
            col_index = col_names.index(col_name) if col_name in col_names else None
            for image in images:
                image_path = image.get("path") if isinstance(image, dict) else image
                if not isinstance(image_path, str):
                    continue
                group_path = well_path / image_path
                if group_path not in self._image_groups():
                    continue
                image_metadata = _metadata(self.store_path / group_path) or {}
                display_name = image_metadata.get("omero", {}).get("name")
                fields.append((group_path, {
                    "row": row_name,
                    "column": col_name,
                    "row_index": row_index,
                    "column_index": col_index,
                    "well_name": str(display_name or f"{row_name}{col_name}"),
                    "plate": plate,
                }))
        return fields

    def _restore_exported_plate_sequence(
        self, fields: list[tuple[Path, dict[str, Any]]]
    ) -> list[tuple[Path, dict[str, Any]]]:
        """Use limnd2's sidecar sequence indices to undo NGFF's path sorting."""
        native = self.native_nd2_metadata()
        wellplate = native.get(b"limnd2:wellplate")
        frame_info = wellplate.get("frameInfo") if isinstance(wellplate, dict) else None
        if not isinstance(frame_info, list) or not frame_info or not fields:
            return fields

        by_coordinate: dict[tuple[int, int], list[tuple[Path, dict[str, Any]]]] = {}
        for field in fields:
            well = field[1]
            row, column = well.get("row_index"), well.get("column_index")
            if row is None or column is None:
                return fields
            by_coordinate.setdefault((int(row), int(column)), []).append(field)

        ordered_records = sorted(
            (item for item in frame_info if isinstance(item, dict)),
            key=lambda item: int(item.get("seqIndex", 0)),
        )
        wanted: list[tuple[int, int]] = []
        for item in ordered_records:
            row, column = item.get("wellRowIndex"), item.get("wellColIndex")
            if row is None or column is None:
                return fields
            wanted.append((int(row), int(column)))

        # Only use the sidecar mapping when it describes this exact set of NGFF
        # fields.  A mismatched/partial sidecar should not drop or duplicate data.
        if len(wanted) != len(fields) or {
            coordinate: len(items) for coordinate, items in by_coordinate.items()
        } != {
            coordinate: wanted.count(coordinate) for coordinate in set(wanted)
        }:
            return fields

        return [by_coordinate[coordinate].pop(0) for coordinate in wanted]

    def _exported_position_groups(self) -> list[Path]:
        """Return multi-position groups only for limnd2's explicit series layout."""
        root = _metadata(self.store_path) or {}
        root_ome = root.get("ome", root)
        if root_ome.get("bioformats2raw.layout") != 3:
            return []
        ome_metadata = _metadata(self.store_path / "OME") or {}
        series = ome_metadata.get("ome", ome_metadata).get("series", [])
        image_groups = set(self._image_groups())
        return [Path(item) for item in series if isinstance(item, str) and Path(item) in image_groups]

    def _group_compatibility_signature(self, group_path: Path) -> dict[str, Any]:
        """Read metadata that must agree before image groups can be ND2 positions."""
        previous_group = self.group_path
        self.group_path = group_path
        try:
            axes, array = self._array()
            metadata = _metadata(self.store_path / group_path) or {}
            ome = metadata.get("ome", metadata)
            multiscales = ome.get("multiscales", [])
            multiscale = multiscales[0] if multiscales else {}
            axes_metadata = multiscale.get("axes", [])
            axes_signature = tuple(
                (
                    str(axis if isinstance(axis, str) else axis.get("name", "")).lower(),
                    None if isinstance(axis, str) else axis.get("type"),
                    None if isinstance(axis, str) else axis.get("unit"),
                )
                for axis in axes_metadata
            )
            dataset = (multiscale.get("datasets") or [{}])[0]
            transformations = dataset.get("coordinateTransformations", [])
            # Spatial translations identify where an otherwise-compatible
            # position lies; they are expected to differ between multipoints.
            # The physical scale must match, but translation is position data,
            # not an incompatibility.
            scale_signature = tuple(
                tuple(item.get("scale", []))
                for item in transformations
                if isinstance(item, dict) and item.get("type") == "scale"
            )
            omero = ome.get("omero") or metadata.get("omero") or {}
            channel_signature = tuple(
                (
                    str(channel.get("label", "")).strip(),
                    str(channel.get("color", "")).strip().lstrip("#").upper(),
                )
                for channel in (omero.get("channels", []) if isinstance(omero, dict) else [])
                if isinstance(channel, dict)
            )
            xml_channel_signature = tuple(
                tuple(
                    (key, channel.get(key))
                    for key in (
                        "Name", "Color", "SamplesPerPixel", "ExcitationWavelength",
                        "EmissionWavelength", "Fluor", "IlluminationType",
                        "AcquisitionMode", "ContrastMethod",
                    )
                    if key in channel
                )
                for channel in self._ome_xml_channels()
            )
            xml_pixels = self._ome_xml_pixels()
            xml_pixel_signature = tuple(
                (key, xml_pixels.get(key))
                for key in (
                    "Type", "SignificantBits", "PhysicalSizeX", "PhysicalSizeXUnit",
                    "PhysicalSizeY", "PhysicalSizeYUnit", "PhysicalSizeZ",
                    "PhysicalSizeZUnit", "TimeIncrement", "TimeIncrementUnit",
                )
                if xml_pixels is not None and key in xml_pixels.attrib
            )
            return {
                "axes": axes_signature or tuple(axes),
                "shape": tuple(array.shape),
                "pixel_type": (array.dtype.kind, array.dtype.itemsize),
                "scale": scale_signature,
                "channels": channel_signature,
                "xml_channels": xml_channel_signature,
                "xml_pixels": xml_pixel_signature,
                "rgb": self.is_rgb,
            }
        finally:
            self.group_path = previous_group

    def _multipoint_groups(self) -> list[tuple[Path, dict[str, Any] | None]]:
        """Only combine series/fields as ND2 positions when their layouts match."""
        if self._selected_multipoint_groups is not None:
            return self._selected_multipoint_groups

        plate_fields = self._restore_exported_plate_sequence(self._plate_fields())
        candidates = plate_fields or [
            (group, None) for group in self._exported_position_groups()
        ]
        self._multipoint_compatible = True
        if len(candidates) > 1:
            first_path = candidates[0][0]
            first_signature = self._group_compatibility_signature(first_path)
            mismatch: tuple[Path, str] | None = None
            for group_path, _well in candidates[1:]:
                signature = self._group_compatibility_signature(group_path)
                differences = [
                    key for key in first_signature
                    if first_signature[key] != signature.get(key)
                ]
                if differences:
                    mismatch = group_path, ", ".join(differences)
                    break
            if mismatch is not None:
                self._multipoint_compatible = False
                selected = [(first_path, None)]
                from .LimConvertUtils import logprint
                logprint(
                    "OME-Zarr images are not compatible for a single ND2 multipoint "
                    f"(groups '{first_path}' and '{mismatch[0]}' differ in "
                    f"{mismatch[1]}). Importing only the first image group '{first_path}'.",
                    type="warning",
                )
                self._selected_multipoint_groups = selected
                return selected

        self._selected_multipoint_groups = candidates
        return candidates
    def read(self) -> np.ndarray:
        axes, array = self._array()
        selection: list[Any] = []
        retained_axes: list[str] = []
        rgb = self.is_rgb
        for axis, size in zip(axes, array.shape):
            if axis in {"y", "x"} or (rgb and axis == "c"):
                selection.append(slice(None))
                retained_axes.append(axis)
            else:
                selection.append(self.axis_indexes.get(axis, 0))
        result = np.asarray(array[tuple(selection)])
        if rgb and "c" in retained_axes:
            result = np.moveaxis(result, retained_axes.index("c"), -1)
        return result

    def read_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        axes, array = self._array()
        selection: list[Any] = []
        retained_axes: list[str] = []
        rgb = self.is_rgb
        for axis in axes:
            if axis == "y":
                selection.append(slice(y, y + h))
                retained_axes.append(axis)
            elif axis == "x":
                selection.append(slice(x, x + w))
                retained_axes.append(axis)
            elif rgb and axis == "c":
                selection.append(slice(None))
                retained_axes.append(axis)
            else:
                selection.append(self.axis_indexes.get(axis, 0))
        result = np.asarray(array[tuple(selection)])
        if rgb and "c" in retained_axes:
            result = np.moveaxis(result, retained_axes.index("c"), -1)
        return result

    @contextmanager
    def open_tile_reader(self):
        axes, array = self._array()
        rgb = self.is_rgb

        def read_tile(x: int, y: int, w: int, h: int) -> np.ndarray:
            selection = []
            retained_axes = []
            for axis in axes:
                if axis == "y":
                    selection.append(slice(y, y + h))
                    retained_axes.append(axis)
                elif axis == "x":
                    selection.append(slice(x, x + w))
                    retained_axes.append(axis)
                elif rgb and axis == "c":
                    selection.append(slice(None))
                    retained_axes.append(axis)
                else:
                    selection.append(self.axis_indexes.get(axis, 0))
            result = np.asarray(array[tuple(selection)])
            if rgb and "c" in retained_axes:
                result = np.moveaxis(result, retained_axes.index("c"), -1)
            return result

        yield read_tile
    @property
    def is_rgb(self) -> bool:
        group = self.group_path
        if group not in self._rgb_by_group:
            metadata = _metadata(self.store_path / group) or {}
            ome = metadata.get("ome", metadata)
            omero = ome.get("omero") or metadata.get("omero") or {}
            multiscales = ome.get("multiscales", [])
            axes: list[str] = []
            if multiscales:
                axes = [
                    (axis if isinstance(axis, str) else axis.get("name", "")).lower()
                    for axis in multiscales[0].get("axes", [])
                ]
            channels = omero.get("channels", []) if isinstance(omero, dict) else []
            primary_colors = {"red": "FF0000", "green": "00FF00", "blue": "0000FF"}
            channel_signatures = {
                (
                    str(channel.get("label", "")).strip().casefold(),
                    str(channel.get("color", "")).strip().lstrip("#").upper(),
                )
                for channel in channels
                if isinstance(channel, dict)
            }
            # NGFF's color rendering model alone is not enough to distinguish
            # RGB samples from fluorescence channels. Require the three
            # canonical RGB component labels and their matching display colors.
            is_rgb_metadata = (
                "c" in axes
                and len(channels) == 3
                and channel_signatures == {
                    (name, color) for name, color in primary_colors.items()
                }
            )
            xml_channels = self._ome_xml_channels()
            is_rgb_xml = (
                "c" in axes
                and len(xml_channels) == 1
                and xml_channels[0].get("SamplesPerPixel") == "3"
            )
            self._rgb_by_group[group] = is_rgb_metadata or is_rgb_xml
        return self._rgb_by_group[group]

    def get_file_dimensions(self) -> dict[str, int]:
        axes, array = self._array()
        names = {"t": "timeloop", "z": "zstack", "c": "channel"}
        rgb = self.is_rgb
        dimensions = {
            names[axis]: size
            for axis, size in zip(axes, array.shape)
            if axis in names and size > 1 and not (rgb and axis == "c")
        }
        if groups := self._multipoint_groups():
            dimensions["multipoint"] = len(groups)
        return dimensions

    def _ome_xml_pixels(self) -> ET.Element | None:
        """Return the OME-XML Pixels element corresponding to this image group."""
        path = self._ome_xml_path()
        if path is None:
            return None
        try:
            root = ET.parse(path).getroot()
            namespace = root.tag.partition("}")[0].lstrip("{")
            tag = lambda name: f"{{{namespace}}}{name}" if namespace else name
            images = root.findall(tag("Image"))
            return images[self._ome_xml_image_index()].find(tag("Pixels"))
        except (ET.ParseError, IndexError, OSError):
            return None

    def _ome_xml_path(self) -> Path | None:
        """Find the optional companion XML, supporting the previous path too."""
        for relative in ("XML/METADATA.ome.xml", "OME/METADATA.ome.xml"):
            path = self.store_path / relative
            if path.is_file():
                return path
        return None

    def _ome_xml_annotation_maps(self) -> dict[str, Any]:
        """Read JSON map annotations emitted by limnd2's shared OME serializer."""
        path = self._ome_xml_path()
        if path is None:
            return {}
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            return {}
        result: dict[str, Any] = {}
        for annotation in root.iter():
            if annotation.tag.rsplit("}", 1)[-1] != "MapAnnotation":
                continue
            description = next(
                (item.text for item in annotation.iter()
                 if item.tag.rsplit("}", 1)[-1] == "Description"),
                None,
            )
            if not description:
                continue
            for item in annotation.iter():
                if item.tag.rsplit("}", 1)[-1] != "M" or item.attrib.get("K") != "json" or not item.text:
                    continue
                try:
                    result[description] = json.loads(item.text)
                except json.JSONDecodeError:
                    continue
        return result

    def _ome_xml_image_index(self) -> int:
        parts = self.group_path.parts
        return int(parts[0]) if parts and parts[0].isdigit() else 0

    def _ome_xml_channels(self) -> list[dict[str, str]]:
        """Return OME-XML channel attributes for this image group, if supplied."""
        pixels = self._ome_xml_pixels()
        if pixels is None:
            return []
        namespace = pixels.tag.partition("}")[0].lstrip("{")
        channel_tag = f"{{{namespace}}}Channel" if namespace else "Channel"
        return [dict(channel.attrib) for channel in pixels.findall(channel_tag)]

    def _ome_xml_significant_bits(self) -> int | None:
        """Read the meaningful pixel bit depth from the companion OME-XML."""
        pixels = self._ome_xml_pixels()
        if pixels is None:
            return None
        try:
            bits = int(pixels.attrib["SignificantBits"])
        except (KeyError, TypeError, ValueError):
            return None
        return bits if bits > 0 else None

    def metadata_as_pattern_settings(self) -> dict[str, Any]:
        """Return OME-Zarr metadata in the shape expected by the import dialog."""
        result: dict[str, Any] = {
            "tstep": "",
            "zstep": "",
            "pixel_calibration": "",
            "pinhole_diameter": "",
            "objective_magnification": "",
            "objective_numerical_aperture": "",
            "immersion_refractive_index": "",
            "zoom_magnification": "",
            "channels": [],
        }

        def format_number(value: Any, digits: int = 3) -> str:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return ""
            if not np.isfinite(number) or number <= 0:
                return ""
            return f"{number:.{digits}f}".rstrip("0").rstrip(".")

        def channel_color(value: Any) -> str:
            if isinstance(value, str):
                color = value.strip()
                if re.fullmatch(r"#?[0-9a-fA-F]{6}", color):
                    return color if color.startswith("#") else f"#{color}"
            return ""

        def modality_for(plane: Any, xml_channel: dict[str, str]) -> str:
            acquisition = (xml_channel.get("AcquisitionMode") or "").casefold()
            contrast = (xml_channel.get("ContrastMethod") or "").casefold()
            if "confocal" in acquisition or "laser" in acquisition:
                return "Confocal, Trans" if contrast and not any(
                    word in contrast for word in ("fluor", "emission")
                ) else "Confocal, Fluo"
            if "spinning" in acquisition or "spinning" in contrast or "disk" in acquisition:
                return "Spinning Disc"
            if "wide" in acquisition:
                return "Wide-field"
            if "bright" in contrast or bool(getattr(plane, "isBrightfield", False)):
                return "Brightfield"
            if "phase" in contrast:
                return "Phase"
            if "dic" in contrast:
                return "DIC"
            if "dark" in contrast or bool(getattr(plane, "isDarkfield", False)):
                return "DarkField"
            return "Undefined"

        metadata = _metadata(self.store_path / self.group_path) or {}
        ome = metadata.get("ome", metadata)
        multiscale = (ome.get("multiscales") or [{}])[0]
        axes_metadata = multiscale.get("axes", [])
        axes = [
            (axis if isinstance(axis, str) else axis.get("name", "")).lower()
            for axis in axes_metadata
        ]
        dataset = (multiscale.get("datasets") or [{}])[0]
        transforms = dataset.get("coordinateTransformations", [])
        scale = next(
            (item.get("scale") for item in transforms
             if isinstance(item, dict) and item.get("type") == "scale"),
            None,
        )
        axis_units = {
            name: (axis.get("unit") if isinstance(axis, dict) else None)
            for name, axis in zip(axes, axes_metadata)
        }
        scale_by_axis = dict(zip(axes, scale)) if scale and len(scale) == len(axes) else {}

        # NGFF scale units are converted to the units expected by the dialog:
        # micrometers for spatial calibration and Z, milliseconds for time.
        spatial_factors = {
            "m": 1e6, "meter": 1e6, "metre": 1e6,
            "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3,
            "µm": 1.0, "um": 1.0, "micrometer": 1.0, "micrometre": 1.0,
            "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3,
        }
        time_factors = {
            "s": 1e3, "second": 1e3, "seconds": 1e3,
            "ms": 1.0, "millisecond": 1.0, "milliseconds": 1.0,
            "µs": 1e-3, "us": 1e-3, "microsecond": 1e-3,
            "ns": 1e-6, "nanosecond": 1e-6,
        }

        def scaled_axis(axis: str, factors: dict[str, float]) -> str:
            value = scale_by_axis.get(axis)
            if value is None:
                return ""
            unit = str(axis_units.get(axis) or "").casefold()
            factor = factors.get(unit)
            return format_number(float(value) * factor) if factor is not None else ""

        xy_scale = scale_by_axis.get("x", scale_by_axis.get("y"))
        if xy_scale is not None and scale_by_axis.get("y", xy_scale) == xy_scale:
            result["pixel_calibration"] = scaled_axis(
                "x" if "x" in scale_by_axis else "y", spatial_factors
            )
        result["zstep"] = scaled_axis("z", spatial_factors)
        result["tstep"] = scaled_axis("t", time_factors)

        picture = next(
            (value for value in self.native_nd2_metadata().values()
             if isinstance(value, PictureMetadata)),
            None,
        )
        xml_path = self._ome_xml_path()
        if picture is None and xml_path is not None:
            try:
                from limnd2.metadata_tiff import picture_metadata_from_ome_xml
                picture = picture_metadata_from_ome_xml(
                    xml_path.read_bytes(), image_index=self._ome_xml_image_index()
                )
            except (OSError, ValueError):
                picture = None

        if picture is not None:
            for key, value in (
                ("pixel_calibration", getattr(picture, "dCalibration", None)),
                ("zstep", getattr(picture, "dZAxisCalibration", None)),
                ("tstep", getattr(picture, "dTimeAxisCalibration", None)),
                ("objective_magnification", getattr(picture, "dObjectiveMag", None)),
                ("objective_numerical_aperture", getattr(picture, "dObjectiveNA", None)),
                ("immersion_refractive_index", getattr(picture, "dRefractIndex1", None)),
                ("zoom_magnification", getattr(picture, "dZoom", None)),
            ):
                formatted = format_number(value)
                if formatted:
                    result[key] = formatted

        omero = ome.get("omero") or metadata.get("omero") or {}
        omero_channels = omero.get("channels", []) if isinstance(omero, dict) else []
        xml_channels = self._ome_xml_channels()
        if (
            picture is None
            and not omero_channels
            and not xml_channels
            and not any(
                result[key]
                for key in (
                    "tstep", "zstep", "pixel_calibration", "pinhole_diameter",
                    "objective_magnification", "objective_numerical_aperture",
                    "immersion_refractive_index", "zoom_magnification",
                )
            )
        ):
            return {}
        picture_channels = list(getattr(picture, "channels", []) or [])
        axes_array, array = self._array()
        channel_count = (
            1 if self.is_rgb else
            array.shape[axes_array.index("c")] if "c" in axes_array else 1
        )
        pinhole_values: list[float] = []

        for index in range(channel_count):
            plane = picture_channels[index] if index < len(picture_channels) else None
            xml_channel = xml_channels[index] if index < len(xml_channels) else {}
            omero_channel = omero_channels[index] if index < len(omero_channels) else {}
            label = (
                getattr(plane, "sDescription", None)
                or xml_channel.get("Name")
                or omero_channel.get("label")
                or ("RGB" if self.is_rgb else f"Channel_{index + 1}")
            )
            excitation = (
                _float_or_none(xml_channel.get("ExcitationWavelength"))
                or getattr(plane, "excitationWavelengthNm", None)
                or 0
            )
            emission = (
                _float_or_none(xml_channel.get("EmissionWavelength"))
                or getattr(plane, "emissionWavelengthNm", None)
                or 0
            )
            color = channel_color(omero_channel.get("color"))
            if not color and plane is not None:
                color = str(getattr(plane, "colorAsHtmlString", "") or "")
            result["channels"].append([
                str(label), str(label), modality_for(plane, xml_channel),
                format_number(excitation, 0) or "0",
                format_number(emission, 0) or "0",
                color,
            ])
            pinhole = getattr(plane, "dPinholeDiameter", None)
            if pinhole is not None and float(pinhole) > 0:
                pinhole_values.append(float(pinhole))

        if pinhole_values and len(pinhole_values) == channel_count and (
            max(pinhole_values) - min(pinhole_values) <= 1e-6
        ):
            result["pinhole_diameter"] = format_number(pinhole_values[0])

        return result

    def parse_additional_metadata(self, metadata_storage) -> None:
        """Translate OME-NGFF display channel labels and colors to ND2 planes."""
        metadata = _metadata(self.store_path / self.group_path) or {}
        ome = metadata.get("ome", metadata)
        multiscale = ome.get("multiscales", [{}])[0]
        axes = [axis if isinstance(axis, str) else axis.get("name", "") for axis in multiscale.get("axes", [])]
        dataset = multiscale.get("datasets", [{}])[0]
        transforms = dataset.get("coordinateTransformations", [])
        scale = next((item.get("scale") for item in transforms if item.get("type") == "scale"), None)
        if scale and len(scale) == len(axes):
            values = dict(zip((axis.lower() for axis in axes), scale))
            xy = values.get("x") or values.get("y")
            if xy and values.get("y", xy) == xy:
                metadata_storage.metadata.pixel_calibration = float(xy)
            if values.get("t"):
                metadata_storage.time_step = float(values["t"])
            if values.get("z"):
                metadata_storage.z_step = float(values["z"])
        # RGB components are samples of one pixel, not three fluorescence
        # planes. Leave the factory empty so is_rgb_fallback creates one RGB
        # picture plane with three bundled components.
        if self.is_rgb:
            return
        # OME-Zarr 0.5 metadata written by limnd2 nests OMERO display data
        # under ``ome.omero``.  Older writers commonly put it at the group
        # root, so accept both forms.
        omero = ome.get("omero") or metadata.get("omero") or {}
        channels = omero.get("channels", []) if isinstance(omero, dict) else []
        xml_channels = self._ome_xml_channels()
        xml_picture = None
        xml_path = self._ome_xml_path()
        if xml_path is not None:
            try:
                from limnd2.metadata_tiff import picture_metadata_from_ome_xml
                xml_picture = picture_metadata_from_ome_xml(
                    xml_path.read_bytes(), image_index=self._ome_xml_image_index()
                )
            except (OSError, ValueError):
                pass
        xml_planes = list(getattr(xml_picture, "channels", []) or [])
        if not channels and not xml_channels and not xml_planes:
            return
        if getattr(metadata_storage, "channels", None) is None:
            metadata_storage.channels = {}
        for index in range(max(len(channels), len(xml_channels), len(xml_planes))):
            channel = channels[index] if index < len(channels) else {}
            xml_channel = xml_channels[index] if index < len(xml_channels) else {}
            xml_plane = xml_planes[index] if index < len(xml_planes) else None
            name = (getattr(xml_plane, "name", None) or getattr(xml_plane, "sDescription", None)
                    or xml_channel.get("Name")
                    or channel.get("label") or f"Channel {index + 1}")
            color = getattr(xml_plane, "color", None)
            if not color and xml_picture is not None and xml_channel.get("Color"):
                component_colors = list(getattr(xml_picture, "componentColors", []) or [])
                if index < len(component_colors) and len(component_colors[index]) >= 3:
                    red, green, blue = component_colors[index][:3]
                    color = "#%02X%02X%02X" % tuple(
                        max(0, min(255, round(float(value) * 255)))
                        for value in (red, green, blue)
                    )
            color = color or channel.get("color")
            metadata_storage.channels[f"ome-zarr-{index}"] = Plane(
                name=name,
                modality=getattr(xml_plane, "modality", None) or "fluorescence",
                color=(color if str(color).startswith("#") else f"#{color}") if color else None,
                excitation_wavelength=(_float_or_none(xml_channel.get("ExcitationWavelength"))
                    or getattr(xml_plane, "excitation_wavelength", None)),
                emission_wavelength=(_float_or_none(xml_channel.get("EmissionWavelength"))
                    or getattr(xml_plane, "emission_wavelength", None)),
                filter_name=getattr(xml_plane, "filter_name", None) or xml_channel.get("Fluor"),
            )
        # OME-XML provides a more complete physical calibration than NGFF's
        # fallback scales; later native ND2 chunks still take precedence.
        if xml_picture is not None:
            calibration = getattr(xml_picture, "dCalibration", 0)
            if calibration and calibration > 0 and getattr(metadata_storage, "metadata", None) is not None:
                metadata_storage.metadata.pixel_calibration = float(calibration)
            z_step = getattr(xml_picture, "dZAxisCalibration", 0)
            if z_step and z_step > 0:
                metadata_storage.z_step = float(z_step)
            time_step_ms = getattr(xml_picture, "dTimeAxisCalibration", 0)
            if time_step_ms and time_step_ms > 0:
                metadata_storage.time_step = float(time_step_ms) / 1000.0
    def parse_additional_dimensions(self, sources, original_dimensions, unknown_dimension_type=None, **kwargs):
        # A plate has one OME image group per field.  Treat those fields as the
        # ND2 multipoint dimension; associated images in a non-plate store are
        # intentionally not guessed to be positions.
        groups = self._multipoint_groups() or [(self._image_groups()[0], None)]
        self.group_path = groups[0][0]
        axes, array = self._array()
        axis_names = {"t": "timeloop", "z": "zstack", "c": "channel"}
        rgb = self.is_rgb
        dimensions_in_store = [
            (axis, axis_names[axis], size)
            for axis, size in zip(axes, array.shape)
            if axis in axis_names and size > 1 and not (rgb and axis == "c")
        ]
        expanded = {}
        for source, values in sources.items():
            for group_index, (group_path, well) in enumerate(groups):
                for indexes in itertools.product(*(range(size) for _, _, size in dimensions_in_store)):
                    clone = copy.deepcopy(source)
                    clone.group_path = group_path
                    clone.axis_indexes = dict(zip((axis for axis, _, _ in dimensions_in_store), indexes))
                    clone._ome_zarr_well = well
                    expanded[clone] = [
                        *values,
                        *([group_index] if len(groups) > 1 else []),
                        *indexes,
                    ]
        dimensions = {**original_dimensions}
        if len(groups) > 1:
            dimensions["multipoint"] = len(groups)
        dimensions.update({name: size for _, name, size in dimensions_in_store})
        return expanded, dimensions

    def ome_zarr_wellplate(self, grouped_files) -> tuple[Any, Any] | None:
        """Build exact ND2 wellplate chunks from exporter-authored NGFF layout."""
        if not self._multipoint_compatible:
            return None
        plate = self._plate()
        if plate is None:
            return None
        rows = list(plate.get("rows", []))
        columns = list(plate.get("columns", []))
        if not rows or not columns:
            return None
        factory = limnd2.WellplateFactory(
            name=str(plate.get("name", "")), rows=len(rows), columns=len(columns),
        )
        for seq_index, sources in enumerate(grouped_files):
            source = next((item for item in sources if getattr(item, "_ome_zarr_well", None)), None)
            if source is None:
                return None
            well = source._ome_zarr_well
            row_index, col_index = well.get("row_index"), well.get("column_index")
            if row_index is None or col_index is None:
                return None
            factory.addItem(
                seqIndex=seq_index,
                wellRowIndex=int(row_index),
                wellColIndex=int(col_index),
                wellIndex=int(row_index) * len(columns) + int(col_index),
                wellName=str(well["well_name"]),
            )
        return factory.create()

    def binary_layer_writer(self, grouped_files, image_attributes):
        """Return a writer that restores OME-Zarr ``labels/<name>`` layers."""
        label_names: set[str] = set()
        groups = self._multipoint_groups() or [(self.group_path, None)]
        for group, _well in groups or [(self.group_path, None)]:
            labels_path = self.store_path / group / "labels"
            if labels_path.is_dir():
                label_names.update(item.name for item in labels_path.iterdir() if item.is_dir())
        if not label_names:
            return None

        def write_labels(nd2) -> None:
            native_models = self.native_nd2_metadata()
            native_raster_metadata = next(
                (model for model in native_models.values() if isinstance(model, BinaryRasterMetadata)),
                None,
            )
            if native_raster_metadata is None:
                native_rle_metadata = next(
                    (model for model in native_models.values() if isinstance(model, BinaryRleMetadata)),
                    None,
                )
                if native_rle_metadata is not None:
                    native_raster_metadata = native_rle_metadata.makeRasterMetadata(
                        image_attributes.width, image_attributes.height
                    )
            metadata = BinaryRasterMetadataFactory(
                native_raster_metadata.toDict() if native_raster_metadata is not None else [],
                imageAttributes=image_attributes,
            )
            used_names = {"labels"}
            original_ids: dict[str, int] = {}
            if native_raster_metadata is not None:
                for index, item in enumerate(native_raster_metadata):
                    base = item.name or f"Label_{index}"
                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base.strip()).strip("._") or "0"
                    candidate = safe
                    suffix = 1
                    while candidate in used_names:
                        candidate = f"{safe}_{suffix}"
                        suffix += 1
                    used_names.add(candidate)
                    original_ids[candidate] = int(item.id)

            ids: dict[str, int] = {}
            for name in sorted(label_names):
                if name in original_ids:
                    ids[name] = original_ids[name]
                else:
                    ids[name] = metadata.addNewItem(
                        name, self._label_color(name) or "#FFFFFF"
                    ).id
            nd2.binaryRasterMetadata = metadata
            for seq_index, sources in enumerate(grouped_files):
                source = next((item for item in sources if isinstance(item, LimImageSourceOmeZarr)), None)
                if source is None:
                    continue
                for name, layer_id in ids.items():
                    data = source._label_data(name)
                    if data is not None:
                        nd2.setBinaryRasterData(layer_id, seq_index, data.astype(np.uint32, copy=False))
        return write_labels

    def _label_color(self, name: str) -> str | None:
        """Return the first visible NGFF ``image-label`` color for a layer."""
        groups = self._multipoint_groups() or [(self.group_path, None)]
        for group, _well in groups or [(self.group_path, None)]:
            metadata = _metadata(self.store_path / group / "labels" / name) or {}
            image_label = (metadata.get("ome", metadata)).get("image-label", {})
            if not isinstance(image_label, dict):
                continue
            for item in image_label.get("colors", []):
                if not isinstance(item, dict):
                    continue
                value = item.get("label-value", item.get("labelValue", 0))
                rgba = item.get("rgba", item.get("color"))
                if value in (0, "0") or not isinstance(rgba, list) or len(rgba) < 3:
                    continue
                if len(rgba) > 3 and int(rgba[3]) == 0:
                    continue
                try:
                    return "#{:02X}{:02X}{:02X}".format(
                        int(rgba[0]), int(rgba[1]), int(rgba[2])
                    )
                except (TypeError, ValueError):
                    continue
        return None

    def _label_data(self, name: str) -> np.ndarray | None:
        """Load one T/Z plane of an exporter-written label pyramid at level zero."""
        try:
            import zarr
            metadata = _metadata(self.store_path / self.group_path / "labels" / name) or {}
            multiscale = (metadata.get("ome", metadata))["multiscales"][0]
            axes = [axis if isinstance(axis, str) else axis["name"] for axis in multiscale["axes"]]
            level = multiscale["datasets"][0]["path"]
            array = zarr.open_array(str(self.store_path / self.group_path / "labels" / name / level), mode="r")
        except (ImportError, KeyError, OSError, ValueError):
            return None
        selection: list[Any] = []
        for axis in axes:
            lower = axis.lower()
            selection.append(slice(None) if lower in {"y", "x"} else self.axis_indexes.get(lower, 0))
        return np.asarray(array[tuple(selection)])
    def nd2_attributes(self, *, sequence_count=1) -> ImageAttributes:
        axes, array = self._array()
        width, height = array.shape[axes.index("x")], array.shape[axes.index("y")]
        bits = array.dtype.itemsize * 8
        significant_bits = self._ome_xml_significant_bits()
        if significant_bits is None or significant_bits > bits:
            significant_bits = bits
        pixel = ImageAttributesPixelType.pxtReal if array.dtype.kind == "f" else (ImageAttributesPixelType.pxtSigned if array.dtype.kind == "i" else ImageAttributesPixelType.pxtUnsigned)
        components = 3 if self.is_rgb else 1
        return ImageAttributes(uiWidth=width, uiWidthBytes=ImageAttributes.calcWidthBytes(width, bits, components), uiHeight=height, uiComp=components, uiBpcInMemory=bits, uiBpcSignificant=significant_bits, uiSequenceCount=sequence_count, uiTileWidth=width, uiTileHeight=height, uiVirtualComponents=components, ePixelType=pixel)
