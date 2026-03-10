from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import html
import os
import re
from pathlib import Path

import limnd2

from limnd2.metadata_factory import MetadataFactory, Plane

from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs, logprint
from .LimImageSource import merge_four_fields
from .LimImageSourceTiff_base import LimImageSourceTiffBase

_PROP_RE = re.compile(r"<(?P<tag>prop|custom-prop)\s+(?P<attrs>[^>]*?)/>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z_][\w\-]*)="(?P<value>.*?)"')


@dataclass
class _MetaChannel:
    name: str
    modality_qml: str
    modality_flag: object
    excitation_wavelength: int
    emission_wavelength: int
    color_hex: str


class LimImageSourceTiffMeta(LimImageSourceTiffBase):
    """TIFF source for MetaSeries-style ImageDescription metadata payload."""

    @classmethod
    def has_meta_metadata(cls, path: str | Path) -> bool:
        return super().has_meta_metadata(path)

    @staticmethod
    def _to_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except Exception:
            return None

    @staticmethod
    def _to_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(round(float(str(value).strip())))
        except Exception:
            return None

    @staticmethod
    def _scale_to_um(units: str | None) -> float:
        unit = (units or "").strip().lower().replace("µ", "u")
        if unit in ("", "um"):
            return 1.0
        if unit == "nm":
            return 1e-3
        if unit == "mm":
            return 1e3
        if unit == "m":
            return 1e6
        return 1.0

    @classmethod
    def _parse_properties(cls, description: str | None) -> dict[str, str]:
        if not description:
            return {}

        props: dict[str, str] = {}
        for match in _PROP_RE.finditer(description):
            attrs_text = match.group("attrs")
            attrs = {m.group("key"): m.group("value") for m in _ATTR_RE.finditer(attrs_text)}
            prop_id = attrs.get("id")
            if not prop_id:
                continue
            props[prop_id] = html.unescape(attrs.get("value", ""))

        return props

    @classmethod
    def _props_from_file(cls, path: str | Path) -> dict[str, str]:
        return cls._parse_properties(cls._read_image_description(path))

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("LIMND2_DEBUG_TIFF_META") == "1"

    @classmethod
    def _debug(cls, message: str):
        if cls._debug_enabled():
            logprint(f"[TIFF-META] {message}")

    @staticmethod
    def _first(props: dict[str, str], *keys: str) -> str | None:
        for key in keys:
            value = props.get(key)
            if value is not None and str(value).strip() != "":
                return value
        return None

    @staticmethod
    def _extract_objective_magnification(props: dict[str, str]) -> float | None:
        direct = LimImageSourceTiffMeta._to_float(
            LimImageSourceTiffMeta._first(props, "objective-magnification", "objective magnification")
        )
        if direct is not None and direct > 0:
            return direct

        for key in ("_MagSetting_", "ImageXpress Micro Objective", "Description"):
            text = props.get(key, "")
            match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]", text)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
        return None

    @staticmethod
    def _extract_pinhole_um(props: dict[str, str]) -> float | None:
        direct = LimImageSourceTiffMeta._to_float(
            LimImageSourceTiffMeta._first(props, "pinhole-diameter", "pinhole diameter")
        )
        if direct is not None and direct > 0:
            return direct

        for key in ("IXConfocal Module Disk", "_IllumSetting_", "Description"):
            text = props.get(key, "")
            match = re.search(r"(\d+(?:\.\d+)?)\s*um\s*pinhole", text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
        return None

    @staticmethod
    def _parse_local_time(value: str | None) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        for fmt in ("%Y%m%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _description_kv(props: dict[str, str]) -> dict[str, str]:
        text = props.get("Description", "")
        values: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip()
        return values

    @classmethod
    def _extract_camera_name(cls, props: dict[str, str]) -> str | None:
        # Prefer explicit custom key if present
        camera = cls._first(props, "Camera Name", "camera-name")
        if camera:
            return camera

        desc_map = cls._description_kv(props)
        if "camera name" in desc_map:
            return desc_map["camera name"]

        desc = props.get("Description", "")
        m = re.search(r"Acquired from\s+(.+?Camera)", desc, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _color_hex(props: dict[str, str], emission_nm: int | None) -> str:
        raw = (props.get("threshold-color") or "").strip().lstrip("#")
        if re.fullmatch(r"[0-9A-Fa-f]{6}", raw):
            return f"#{raw.upper()}"

        if emission_nm is None:
            return ""
        if emission_nm < 490:
            return "#0080FF"
        if emission_nm < 570:
            return "#00B050"
        if emission_nm < 620:
            return "#FFC000"
        return "#FF3030"

    @staticmethod
    def _channel_info(props: dict[str, str]) -> _MetaChannel:
        name = LimImageSourceTiffMeta._first(props, "_IllumSetting_", "image-name", "stage-label") or "Channel_0"

        modality_text = " ".join(
            [
                props.get("IXConfocal Module Disk", ""),
                props.get("_IllumSetting_", ""),
                props.get("look-up-table-type", ""),
            ]
        ).lower()
        if "bright" in modality_text:
            modality_qml = "Brightfield"
            modality_flag = (
                limnd2.metadata.PicturePlaneModalityFlags.modBrightfield
                | limnd2.metadata.PicturePlaneModalityFlags.modCamera
            )
        elif "phase" in modality_text:
            modality_qml = "Phase"
            modality_flag = (
                limnd2.metadata.PicturePlaneModalityFlags.modBrightfield
                | limnd2.metadata.PicturePlaneModalityFlags.modCamera
                | limnd2.metadata.PicturePlaneModalityFlags.modPhaseContrast
            )
        elif "confocal" in modality_text or "pinhole" in modality_text or "laser" in modality_text:
            modality_qml = "Confocal, Fluo"
            modality_flag = (
                limnd2.metadata.PicturePlaneModalityFlags.modFluorescence
                | limnd2.metadata.PicturePlaneModalityFlags.modLaserScanConfocal
            )
        else:
            modality_qml = "Wide-field"
            modality_flag = (
                limnd2.metadata.PicturePlaneModalityFlags.modFluorescence
                | limnd2.metadata.PicturePlaneModalityFlags.modCamera
            )

        emission_nm = LimImageSourceTiffMeta._to_int(LimImageSourceTiffMeta._first(props, "wavelength", "emission-wavelength"))
        excitation_nm = LimImageSourceTiffMeta._to_int(props.get("excitation-wavelength"))
        color_hex = LimImageSourceTiffMeta._color_hex(props, emission_nm)
        # Prefer explicit color hints in channel name (e.g. "RED", "GREEN")
        # over stale/incorrect threshold-color metadata.
        inferred_from_name = LimImageSourceTiffMeta._infer_color_from_name(name)
        if inferred_from_name:
            color_hex = inferred_from_name

        return _MetaChannel(
            name=name,
            modality_qml=modality_qml,
            modality_flag=modality_flag,
            excitation_wavelength=excitation_nm or 0,
            emission_wavelength=emission_nm or 0,
            color_hex=color_hex,
        )

    @staticmethod
    def _format_float(val: float | None, precision: int | None = None) -> str:
        if val is None:
            return ""
        if precision is None:
            return str(float(val))
        return f"{float(val):.{precision}f}"

    @classmethod
    def _extract_metadata_values(cls, props: dict[str, str]):
        spatial_cal = cls._to_float(cls._first(props, "spatial-calibration-x", "spatial-calibration-y", "pixel-size-um"))
        if spatial_cal is not None:
            units = cls._first(props, "spatial-calibration-units")
            spatial_cal *= cls._scale_to_um(units)

        objective_na = cls._to_float(cls._first(props, "_MagNA_", "objective-numerical-aperture"))
        refractive_index = cls._to_float(cls._first(props, "_MagRI_", "immersion-refractive-index"))
        objective_mag = cls._extract_objective_magnification(props)
        pinhole_um = cls._extract_pinhole_um(props)
        zoom_mag = cls._to_float(cls._first(props, "zoom-magnification", "_MagZoom_", "zoom-percent"))

        zstep = cls._to_float(cls._first(props, "Z Projection Step Size", "z-step", "z-step-um"))
        if zstep is not None and zstep <= 0:
            zstep = None

        return spatial_cal, objective_na, refractive_index, objective_mag, pinhole_um, zoom_mag, zstep

    @classmethod
    def _collect_extra_qml_settings(cls, props: dict[str, str]) -> dict[str, str]:
        extras: dict[str, str] = {}

        key_map = {
            "ApplicationName": "application_name",
            "ApplicationVersion": "application_version",
            "MetaDataVersion": "metadata_version",
            "image-name": "image_name",
            "stage-label": "stage_label",
            "stage-position-x": "stage_position_x",
            "stage-position-y": "stage_position_y",
            "z-position": "z_position",
            "camera-binning-x": "camera_binning_x",
            "camera-binning-y": "camera_binning_y",
            "acquisition-time-local": "acquisition_time_local",
            "modification-time-local": "modification_time_local",
            "Z Projection Method": "z_projection_method",
            "Z Thickness": "z_thickness",
            "SiteX": "site_x",
            "SiteY": "site_y",
            "Exposure Time": "exposure_time",
            "Software Version": "software_version",
            "Instrument Serial Number": "instrument_serial_number",
        }

        for source_key, target_key in key_map.items():
            val = props.get(source_key)
            if val is not None and str(val).strip() != "":
                extras[target_key] = str(val)

        camera_name = cls._extract_camera_name(props)
        if camera_name:
            extras["camera_name"] = camera_name

        return extras

    @classmethod
    def _find_htd_file(cls, source_file: Path) -> Path | None:
        directory = source_file.parent
        if not directory.exists():
            return None
        candidates = sorted(
            [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".htd"],
            key=lambda p: p.name.casefold(),
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _read_text_best_effort(path: Path) -> str:
        data = path.read_bytes()
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")

    @classmethod
    def _parse_htd_keyvals(cls, source_file: Path) -> dict[str, list[str]]:
        htd = cls._find_htd_file(source_file)
        if not htd:
            return {}

        try:
            text = cls._read_text_best_effort(htd)
        except Exception:
            return {}

        keyvals: dict[str, list[str]] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                row = next(csv.reader([stripped], skipinitialspace=True))
            except Exception:
                continue
            if not row:
                continue
            key = row[0].strip().strip('"')
            values = [cell.strip().strip('"') for cell in row[1:]]
            keyvals[key] = values
        return keyvals

    @classmethod
    def _parse_htd_channel_names(cls, source_file: Path) -> list[str]:
        keyvals = cls._parse_htd_keyvals(source_file)
        if not keyvals:
            return []

        n_waves = cls._to_int((keyvals.get("NWavelengths") or [""])[0])
        wave_names: list[tuple[int, str]] = []

        name_re = re.compile(r"^WaveName(\d+)$", re.IGNORECASE)
        for key, values in keyvals.items():
            m = name_re.match(key)
            if not m or not values:
                continue
            idx = int(m.group(1))
            name = values[0].strip()
            if name == "":
                continue

            collect_key = f"WaveCollect{idx}"
            collect_values = keyvals.get(collect_key, [])
            if collect_values:
                collect_raw = collect_values[0].strip().upper()
                collect_on = collect_raw in {"1", "TRUE", "YES", "ON"}
                if not collect_on:
                    continue

            if n_waves is not None and n_waves > 0 and idx > n_waves:
                continue
            wave_names.append((idx, name))

        wave_names.sort(key=lambda item: item[0])
        return [name for _, name in wave_names]

    @classmethod
    def _parse_htd_plate_geometry(cls, source_file: Path) -> tuple[int, int] | None:
        keyvals = cls._parse_htd_keyvals(source_file)
        if not keyvals:
            return None

        x_wells = cls._to_int((keyvals.get("XWells") or [""])[0])
        y_wells = cls._to_int((keyvals.get("YWells") or [""])[0])
        if x_wells is None or y_wells is None:
            return None
        if x_wells <= 0 or y_wells <= 0:
            return None

        # HTD stores plate size as XWells (columns), YWells (rows).
        return (int(y_wells), int(x_wells))

    def get_htd_plate_geometry(self) -> tuple[int, int] | None:
        return self._parse_htd_plate_geometry(self.filename)

    @staticmethod
    def _infer_color_from_name(name: str) -> str:
        lowered = name.casefold()
        if "blue" in lowered:
            return "#4080FF"
        if "green" in lowered:
            return "#00B050"
        if "red" in lowered:
            return "#FF3030"
        if "cyan" in lowered:
            return "#00B8FF"
        if "yellow" in lowered:
            return "#FFC000"
        if "magenta" in lowered:
            return "#FF00FF"
        if "violet" in lowered:
            return "#7A3CFF"
        return ""

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        if not self.has_meta_metadata(self.filename):
            return

        props = self._props_from_file(self.filename)
        if not props:
            return

        self._debug(f"parsed {len(props)} metadata properties from {self.filename.name}")
        if self._debug_enabled():
            preview = sorted(props.keys())[:24]
            self._debug(f"property keys preview: {preview}")

        pixel_cal, objective_na, refractive_index, objective_mag, pinhole_um, zoom_mag, zstep = self._extract_metadata_values(props)
        channel = self._channel_info(props)
        camera_name = self._extract_camera_name(props)
        acquisition_time = self._parse_local_time(props.get("acquisition-time-local"))

        metadata_kwargs = {}
        if objective_na is not None:
            metadata_kwargs["objective_numerical_aperture"] = objective_na
        if refractive_index is not None:
            metadata_kwargs["immersion_refractive_index"] = refractive_index
        if objective_mag is not None:
            metadata_kwargs["objective_magnification"] = objective_mag
        if pinhole_um is not None:
            metadata_kwargs["pinhole_diameter"] = pinhole_um
        if zoom_mag is not None:
            metadata_kwargs["zoom_magnification"] = zoom_mag

        b = ConvertSequenceArgs()
        b.metadata = MetadataFactory(
            pixel_calibration=pixel_cal if pixel_cal is not None else -1.0,
            **metadata_kwargs,
        )
        b.channels = {
            0: Plane(
                name=channel.name,
                modality=channel.modality_flag,
                excitation_wavelength=channel.excitation_wavelength,
                emission_wavelength=channel.emission_wavelength,
                color=channel.color_hex,
                camera_name=camera_name,
                acquisition_time=acquisition_time,
            )
        }
        b.z_step = zstep

        self._debug(
            "mapped values: "
            f"pixel_cal={pixel_cal}, obj_na={objective_na}, obj_mag={objective_mag}, "
            f"ri={refractive_index}, pinhole={pinhole_um}, zoom={zoom_mag}, zstep={zstep}, "
            f"channel={channel.name}, emission={channel.emission_wavelength}, camera={camera_name}"
        )

        merge_four_fields(metadata_storage, b)

    def metadata_as_pattern_settings(self) -> dict:
        if not self.has_meta_metadata(self.filename):
            return {}

        props = self._props_from_file(self.filename)
        if not props:
            return {}

        pixel_cal, objective_na, refractive_index, objective_mag, pinhole_um, zoom_mag, zstep, = self._extract_metadata_values(props)
        channel = self._channel_info(props)
        extra_settings = self._collect_extra_qml_settings(props)

        result: dict[str, object] = {
            "tstep": "",
            "zstep": self._format_float(zstep, 3),
            "pixel_calibration": self._format_float(pixel_cal),
            "pinhole_diameter": self._format_float(pinhole_um),
            "objective_magnification": self._format_float(objective_mag),
            "objective_numerical_aperture": self._format_float(objective_na),
            "immersion_refractive_index": self._format_float(refractive_index),
            "zoom_magnification": self._format_float(zoom_mag),
            "channels": [],
        }

        # Keep single-file inspection strictly file-local:
        # do not expand channels from neighboring HTD files.
        result["channels"] = [
            [
                channel.name,
                channel.name,
                channel.modality_qml,
                str(channel.excitation_wavelength),
                str(channel.emission_wavelength),
                channel.color_hex,
            ]
        ]

        htd_plate_geometry = self.get_htd_plate_geometry()
        if htd_plate_geometry is not None:
            rows, cols = htd_plate_geometry
            result["plate_rows"] = str(rows)
            result["plate_columns"] = str(cols)
            result["plate_well_count"] = str(rows * cols)

        result.update(extra_settings)

        self._debug(
            "qml settings generated with keys: "
            f"{sorted(result.keys())}"
        )

        return result
