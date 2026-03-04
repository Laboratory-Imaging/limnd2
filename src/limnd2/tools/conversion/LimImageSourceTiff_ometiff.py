from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import limnd2
from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.metadata_factory import MetadataFactory, Plane

from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs
from .LimImageSource import merge_four_fields
from .LimImageSourceTiff_base import LimImageSourceTiffBase, _require_ome_types, tifffile

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ome_types


class LimImageSourceTiffOmeTiff(LimImageSourceTiffBase):
    """TIFF source with OME metadata parsing."""

    @classmethod
    def has_ome_metadata(cls, path: str | Path) -> bool:
        return super().has_ome_metadata(path)

    @staticmethod
    def get_significant_bits_from_ome(path: str | Path) -> int | None:
        if not LimImageSourceTiffOmeTiff.has_ome_metadata(path):
            return None
        try:
            ome_types = _require_ome_types()
        except ImportError:
            return None
        try:
            return ome_types.from_tiff(path).images[0].pixels.significant_bits
        except Exception:
            return None

    def nd2_attributes(self, *, sequence_count=1) -> ImageAttributes:
        with tifffile.TiffReader(self.filename) as tiff:
            page = tiff.pages[self.idf]

            shape = page.shape
            dtype = page.dtype
            tags = {tag.name: tag.value for tag in page.tags.values()}

        components = 1 if len(shape) <= 2 else shape[2]
        numpy_bits = dtype.itemsize * 8

        tiff_bits = tags.get("BitsPerSample", numpy_bits)
        max_value = tags.get("MaxSampleValue", -1)

        bpc_significant = self.get_significant_bits_from_ome(self.filename)
        if not bpc_significant:
            bpc_significant = self.calculate_bpc_significant(numpy_bits, tiff_bits, max_value)

        bpc_memory = bpc_significant if bpc_significant % 8 == 0 else math.ceil(bpc_significant / 8) * 8
        width_bytes = ImageAttributes.calcWidthBytes(shape[1], bpc_memory, components)

        if dtype in (np.int8, np.int16, np.int32):
            pixel_type = ImageAttributesPixelType.pxtSigned
        elif dtype in (np.uint8, np.uint16, np.uint32):
            pixel_type = ImageAttributesPixelType.pxtUnsigned
        elif dtype in (np.float16, np.float32, np.float64):
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            raise ValueError("TIFF file has unsupported pixel type.")

        return ImageAttributes(
            uiWidth=shape[1],
            uiWidthBytes=width_bytes,
            uiHeight=shape[0],
            uiComp=components,
            uiBpcInMemory=bpc_memory,
            uiBpcSignificant=bpc_significant,
            uiSequenceCount=sequence_count,
            uiTileWidth=shape[1],
            uiTileHeight=shape[0],
            uiVirtualComponents=components,
            ePixelType=pixel_type,
        )

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        if not self.has_ome_metadata(self.filename):
            return

        ome_types = _require_ome_types()

        try:
            ome = ome_types.from_tiff(self.filename)
        except Exception:
            return
        if not ome:
            return

        ome_dims = self.get_file_dimensions()

        b = ConvertSequenceArgs()

        if "timeloop" in ome_dims:
            b.time_step = OMEUtils.time_step_from_ome(ome)

        if "zstack" in ome_dims:
            b.z_step = OMEUtils.z_step_from_ome(ome)

        if "channel" in ome_dims:
            meta_from_ome, chans_from_ome = OMEUtils.channels_from_ome(ome, metadata_storage.metadata)
            b.metadata = meta_from_ome
            b.channels = chans_from_ome

        merge_four_fields(metadata_storage, b)

    def metadata_as_pattern_settings(self) -> dict:
        """
        Return metadata in the shape expected by QML pattern settings.
        """
        modalities = [
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

        def _modality_from(acq_mode, contrast_method):
            am = (acq_mode or "").lower()
            cm = (contrast_method or "").lower()

            if "confocal" in am or "laser" in am:
                if "fluor" in cm or cm == "" or "emission" in cm:
                    return "Confocal, Fluo"
                return "Confocal, Trans"

            if "spinning" in am or "disk" in am or "spinning" in cm:
                return "Spinning Disc"
            if "wide" in am or "widefield" in am or "wide field" in am:
                return "Wide-field"
            if "bright" in cm:
                return "Brightfield"
            if "phase" in cm:
                return "Phase"
            if "dic" in cm:
                return "DIC"
            if "dark" in cm:
                return "DarkField"

            return "Undefined"

        def _pinhole_um_from_channel(ch):
            val = getattr(ch, "pinhole_size", None)
            if val is None:
                return None
            unit = getattr(ch, "pinhole_size_unit", None)
            unit_value = "µm" if unit is None else getattr(unit, "value", str(unit))
            if unit_value in ("µm", "um"):
                scale = 1.0
            elif unit_value == "nm":
                scale = 1e-3
            elif unit_value == "mm":
                scale = 1e3
            elif unit_value == "m":
                scale = 1e6
            else:
                return None
            try:
                return float(val) * scale
            except Exception:
                return None

        result = {
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

        try:
            ome_types = _require_ome_types()
            ome = ome_types.from_tiff(self.filename)
        except Exception:
            return {}

        if not ome or not getattr(ome, "images", None):
            return {}

        image = ome.images[0]

        try:
            tstep_s = OMEUtils.time_step_from_ome(ome)
            zstep_um = OMEUtils.z_step_from_ome(ome)
        except Exception:
            tstep_s = None
            zstep_um = None

        if tstep_s is not None:
            result["tstep"] = f"{tstep_s * 1000.0:.3f}"

        if zstep_um is not None:
            result["zstep"] = f"{zstep_um:.3f}"

        objective_na = None
        objective_mag = None
        refractive_index = None
        pixel_size_x = None

        used_instrument = None
        used_objective = None

        if getattr(ome, "instruments", None) and getattr(image, "instrument_ref", None):
            for instrument in ome.instruments:
                if instrument.id == image.instrument_ref.id:
                    used_instrument = instrument
                    break

        if used_instrument and getattr(used_instrument, "objectives", None) and getattr(image, "objective_settings", None):
            for objective in used_instrument.objectives:
                if objective.id == image.objective_settings.id:
                    used_objective = objective
                    break

        if used_objective:
            objective_na = getattr(used_objective, "lens_na", None)
            objective_mag = getattr(used_objective, "nominal_magnification", None)
            if objective_mag is None:
                objective_mag = getattr(used_objective, "calibrated_magnification", None)

        if getattr(image, "objective_settings", None):
            refractive_index = getattr(image.objective_settings, "refractive_index", None)

        if getattr(image, "pixels", None):
            pixel_size_x = getattr(image.pixels, "physical_size_x", None)

        if pixel_size_x is not None:
            result["pixel_calibration"] = f"{float(pixel_size_x)}"
        if objective_na is not None:
            result["objective_numerical_aperture"] = f"{float(objective_na)}"
        if objective_mag is not None:
            result["objective_magnification"] = f"{float(objective_mag)}"
        if refractive_index is not None:
            result["immersion_refractive_index"] = f"{float(refractive_index)}"

        channels = []
        if getattr(image, "pixels", None) and getattr(image.pixels, "channels", None):
            chs = sorted(image.pixels.channels, key=lambda x: x.id)

            pinhole_vals = []
            for ch in chs:
                v = _pinhole_um_from_channel(ch)
                if v is None:
                    pinhole_vals = []
                    break
                pinhole_vals.append(v)
            if pinhole_vals and (max(pinhole_vals) - min(pinhole_vals) <= 1e-6):
                result["pinhole_diameter"] = f"{float(pinhole_vals[0])}"

            for idx, ch in enumerate(chs):
                name_from_file = ch.name or f"Channel_{idx}"
                custom_name = name_from_file

                acq_mode = getattr(ch, "acquisition_mode", None)
                contrast = getattr(ch, "contrast_method", None)
                modality = _modality_from(
                    acq_mode.value if acq_mode else None,
                    contrast.value if contrast else None,
                )
                if modality not in modalities:
                    modality = "Undefined"

                ex = getattr(ch, "excitation_wavelength", None)
                em = getattr(ch, "emission_wavelength", None)
                ex_str = str(int(round(ex))) if ex is not None else "0"
                em_str = str(int(round(em))) if em is not None else "0"

                rgb = ch.color.as_rgb_tuple(alpha=False) if getattr(ch, "color", None) else None
                color_name = _color_to_hex(rgb)

                channels.append([name_from_file, custom_name, modality, ex_str, em_str, color_name])

        result["channels"] = channels
        return result


class OMEUtils:
    @staticmethod
    def time_step_from_ome(ome: "ome_types.model.OME") -> float | None:
        if not ome.images or not ome.images[0].pixels or not ome.images[0].pixels.planes:
            return None
        planes = ome.images[0].pixels.planes
        times = [plane.delta_t for plane in planes if plane.delta_t is not None]
        if len(times) < 2:
            return None
        times = sorted(set(times))
        return (times[-1] - times[0]) / (len(times) - 1)

    @staticmethod
    def z_step_from_ome(ome: "ome_types.model.OME") -> float | None:
        if not ome.images or not ome.images[0].pixels or not ome.images[0].pixels.planes:
            return None

        planes = ome.images[0].pixels.planes
        zpositions = [plane.position_z for plane in planes if plane.position_z is not None]
        if len(zpositions) < 2:
            return None

        zpositions = sorted(set(zpositions))
        return (zpositions[-1] - zpositions[0]) / (len(zpositions) - 1)

    @staticmethod
    def channel_from_ome(channel: "ome_types.model.Channel"):
        try:
            acquisition = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(
                channel.acquisition_mode.value if channel.acquisition_mode else "unknown"
            )
        except ValueError:
            acquisition = limnd2.metadata.PicturePlaneModalityFlags.modUnknown
        try:
            contrast = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(
                channel.contrast_method.value if channel.contrast_method else "unknown"
            )
        except ValueError:
            contrast = limnd2.metadata.PicturePlaneModalityFlags.modUnknown

        color = channel.color.as_rgb_tuple(alpha=False) if getattr(channel, "color", None) else None

        plane = Plane(
            name=channel.name,
            modality=acquisition | contrast,
            color=color,
            excitation_wavelength=channel.excitation_wavelength,
            emission_wavelength=channel.emission_wavelength,
        )
        return plane

    @staticmethod
    def channels_from_ome(
        ome: "ome_types.model.OME",
        metadata_factory: MetadataFactory | None = None,
    ):
        if metadata_factory is not None:
            provided_settings = metadata_factory._other_settings
            base_pixel_cal = metadata_factory.pixel_calibration
        else:
            provided_settings = {}
            base_pixel_cal = None

        def _is_missing(val):
            if val is None:
                return True
            if isinstance(val, (int, float)):
                return val <= 0
            if isinstance(val, str):
                return val.strip() == ""
            return False

        def _pick(provided, detected):
            return provided if not _is_missing(provided) else detected

        image = ome.images[0]

        objective_numerical_aperture = None
        objective_magnification = None
        immersion_refractive_index = None
        pixel_calibration = None

        used_instrument = None
        used_objective = None

        if getattr(ome, "instruments", None) and getattr(image, "instrument_ref", None):
            for instrument in ome.instruments:
                if instrument.id == image.instrument_ref.id:
                    used_instrument = instrument
                    break

        if used_instrument and getattr(used_instrument, "objectives", None) and getattr(image, "objective_settings", None):
            for objective in used_instrument.objectives:
                if objective.id == image.objective_settings.id:
                    used_objective = objective
                    break

        if used_objective:
            objective_numerical_aperture = used_objective.lens_na
            objective_magnification = (
                used_objective.nominal_magnification
                if getattr(used_objective, "nominal_magnification", None) is not None
                else getattr(used_objective, "calibrated_magnification", None)
            )

        immersion_refractive_index = image.objective_settings.refractive_index if getattr(image, "objective_settings", None) else None
        pixel_calibration = image.pixels.physical_size_x if getattr(image, "pixels", None) else None

        def _pinhole_um_from_channel(ch):
            val = getattr(ch, "pinhole_size", None)
            if val is None:
                return None
            unit = getattr(ch, "pinhole_size_unit", None)
            unit_value = "µm" if unit is None else getattr(unit, "value", str(unit))
            if unit_value in ("µm", "um"):
                scale = 1.0
            elif unit_value == "nm":
                scale = 1e-3
            elif unit_value == "mm":
                scale = 1e3
            elif unit_value == "m":
                scale = 1e6
            else:
                return None
            try:
                return float(val) * scale
            except Exception:
                return None

        pinhole_diameter = None
        if getattr(image, "pixels", None) and getattr(image.pixels, "channels", None):
            vals = []
            for ch in image.pixels.channels:
                v = _pinhole_um_from_channel(ch)
                if v is None:
                    vals = []
                    break
                vals.append(v)
            if vals and (max(vals) - min(vals) <= 1e-6):
                pinhole_diameter = vals[0]

        pixel_calibration = _pick(base_pixel_cal, pixel_calibration)
        if _is_missing(pixel_calibration):
            pixel_calibration = -1.0

        new_factory_kwargs = {}
        imm_val = _pick(provided_settings.get("immersion_refractive_index"), immersion_refractive_index)
        if not _is_missing(imm_val):
            new_factory_kwargs["immersion_refractive_index"] = imm_val
        na_val = _pick(provided_settings.get("objective_numerical_aperture"), objective_numerical_aperture)
        if not _is_missing(na_val):
            new_factory_kwargs["objective_numerical_aperture"] = na_val
        mag_val = _pick(provided_settings.get("objective_magnification"), objective_magnification)
        if not _is_missing(mag_val):
            new_factory_kwargs["objective_magnification"] = mag_val
        pin_val = _pick(provided_settings.get("pinhole_diameter"), pinhole_diameter)
        if not _is_missing(pin_val):
            new_factory_kwargs["pinhole_diameter"] = pin_val

        new_factory = MetadataFactory(
            pixel_calibration=pixel_calibration,
            **new_factory_kwargs,
        )

        for key, value in provided_settings.items():
            if key not in new_factory._other_settings and not _is_missing(value):
                new_factory._other_settings[key] = value

        channels: dict[int, Plane] = {}
        if getattr(image, "pixels", None) and getattr(image.pixels, "channels", None):
            for index, channel in enumerate(sorted(image.pixels.channels, key=lambda x: x.id)):
                channels[index] = OMEUtils.channel_from_ome(channel)

        return new_factory, channels
