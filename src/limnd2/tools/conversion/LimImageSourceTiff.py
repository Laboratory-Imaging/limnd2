from __future__ import annotations
import copy
import itertools
import math
from pathlib import Path

import numpy as np

_COMMON_FF_HINT = (
    '[commonff] extra not installed. Install it with `pip install "limnd2[commonff]"`.'
)


def _missing_convert_dependency(package: str) -> ImportError:
    msg = (
        f'Missing optional dependency "{package}" required for TIFF conversion. '
        f"{_COMMON_FF_HINT}"
    )
    return ImportError(msg)


def _require_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise _missing_convert_dependency("tifffile") from exc
    return tifffile


def _require_ome_types():
    try:
        import ome_types
    except ImportError as exc:
        raise _missing_convert_dependency("ome-types") from exc
    return ome_types


tifffile = _require_tifffile()

import limnd2
from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.metadata_factory import MetadataFactory, Plane
from .LimImageSource import LimImageSource, merge_four_fields
from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs, logprint


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ome_types


class LimImageSourceTiff(LimImageSource):
    """Class for reading images from TIFF files."""
    idf: int

    def __init__(self, filename: str | Path, idf: int = 0):
        super().__init__(filename)
        self.idf = idf

    def __repr__(self):
        return f"LimImageSourceTiff({self.filename}, {self.idf})"

    def read(self):
        with tifffile.TiffReader(self.filename) as tiff:
            arr = tiff.asarray(self.idf)
        if arr.ndim == 3:
            arr = arr[:, :, ::-1]
        return arr

    @property
    def additional_information(self) -> dict:
        if self._additional_dimensions is None:
            self._additional_dimensions = self._calculate_additional_information()
        return self._additional_dimensions


    def get_file_dimensions(self) -> dict[str, int]:
        file_info = self.additional_information
        new_dimensions = {}
        for dimension_type, count in zip(file_info["axis_parsed"], file_info["shape"]):
            new_dimensions[dimension_type] = count
        return new_dimensions


    def _calculate_additional_information(self) -> dict:
        """
        Map axes character codes to dimension names (from tifffile source code):
        - X : width          (image width)
        - Y : height         (image length)
        - Z : depth          (image depth)
        - S : sample         (color space and extra samples)
        - I : sequence       (generic sequence of images, frames, planes, pages)
        - T : time           (time series)
        - C : channel        (acquisition path or emission wavelength)
        - A : angle          (OME)
        - P : phase          (OME. In LSM, **P** maps to **position**)
        - R : tile           (OME. Region, position, or mosaic)
        - H : lifetime       (OME. Histogram)
        - E : lambda         (OME. Excitation wavelength)
        - Q : other          (OME)
        - L : exposure       (FluoView)
        - V : event          (FluoView)
        - M : mosaic         (LSM 6)
        - J : column         (NDTiff)
        - K : row            (NDTiff)
        """

        # Following dictionary will try to correctly map tifffile dimension names to limnd2 dimensions:
        GROUPED_AXIS_CATEGORIES = {
            ("multipoint", "m"): "RPMJK",
            ("timeloop", "t"): "TVL",
            ("zstack", "z"): "ZA",
            ("channel", "c"): "CSEH",
            ("unknown", "unknown"): "IQ",
        }

        DIMENSIONAL_AXIS = "XY"     # tiffile reports those as dimensions, but they are not added in experiments or metadata

        # turns dictionary above to mapping of letters to dimension names
        AXIS_CATEGORY_BY_LETTER = {
            letter: (full, short)
            for (full, short), letters in GROUPED_AXIS_CATEGORIES.items()
            for letter in letters
        }

        result = copy.deepcopy(LimImageSource.DEFAULT_INFORMATION_TEMPLATE)

        with tifffile.TiffReader(self.filename) as tiff:
            if tiff.pages[0].photometric.name == "RGB":
                result["is_rgb"] = True
                self._is_rgb = True
            else:
                self._is_rgb = False

            if len(tiff.series) == 1:
                prod = 1
                for axis, size in zip(tiff.series[0].axes, tiff.series[0].shape):
                    if axis not in AXIS_CATEGORY_BY_LETTER and axis not in DIMENSIONAL_AXIS:
                        raise ValueError(f"Error: {axis} is not a known axis type.")
                    if axis in DIMENSIONAL_AXIS:
                        continue

                    full_category, _ = AXIS_CATEGORY_BY_LETTER[axis]
                    if full_category == "channel" and self.is_rgb:
                        # RGB images are not channels, so we skip this axis
                        continue
                    LimImageSource.update_axis_result(result, full_category, size)
                    prod *= size

                """
                # TODO: attempt to detect OME file spanning over several TIFF files, which is not supported yet
                # wanted to at least throw an error in this case, but it doesnt work with RGB files

                if len(tiff.pages) * tiff.pages[0].samplesperpixel != prod:
                    raise ValueError(f"Incorrect number of images (could be OME TIFF file spanning over several TIFF files).")
                """
            else:
                # Allow multi-series tiff file ONLY IF all series have the same resolution
                # in this case, we will treat it as a single series with multiple images

                resolutions = [s.shape[-2:] for s in tiff.series if len(s.shape) >= 2]
                same_res = all(r == resolutions[0] for r in resolutions)
                if same_res:
                    total_images = sum(len(s.pages) for s in tiff.series)
                    LimImageSource.update_axis_result(result, "unknown", total_images)
                else:
                    raise ValueError(f"Multi-page TIFF file with different resolutions not supported")
        return result


    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            with tifffile.TiffReader(self.filename) as tiff:
                self._is_rgb = tiff.pages[0].photometric.name == "RGB"
        return self._is_rgb



    def parse_additional_dimensions(self, sources: dict[list["LimImageSource"], tuple], original_dimensions: dict[str, int], unknown_dimension_type: str = None) \
        -> tuple[list["LimImageSource"], dict[str, int]]:
        """
        Parse additional dimensions from the tiff source.
        Adds IDF to tiff file based on dimensions found in the TIFF file.
        Returns a list of new sources and a dictionary of new dimensions.
        """
        new_files = {}
        new_dimension = self.get_file_dimensions()

        # TODO: add check if dimensions from file do not include any of the dimensions in the original_dimensions

        if not new_dimension:
            return sources, original_dimensions

        ranges = [range(r) for r in new_dimension.values()]
        index_tuples = list(itertools.product(*ranges))
        index_to_idf = [(idx_tuple, idf) for idf, idx_tuple in enumerate(index_tuples)]

        for file, dims in sources.items():
            for ome_dims, idf in index_to_idf:
                file_copy = copy.deepcopy(file)
                file_copy.idf = idf
                new_files[file_copy] = dims + list(ome_dims)

        new_dims = original_dimensions.copy()
        for dim, size in new_dimension.items():
            new_dims[dim] = size

        if new_dims.get("unknown", 0) > 1:
            if unknown_dimension_type is None:
                logprint("WARNING: File contains unknown dimension, but no unknown dimension type was provided. This may lead to data loss.", type="warning")
            else:
                new_dims[unknown_dimension_type] = new_dims.pop("unknown")

        return new_files, new_dims

    @staticmethod
    def calculate_bpc_significant(numpy_bits_memory: int, tiff_bits_memory: int | tuple, max_sample_value: int | tuple) -> int:
        """
        Numpy bits memory is derived from the numpy dtype, which is the dtype used for actual array when reading data from the TIFF file.
        TIFF bits memory is derived from the TIFF file itself using BitsPerSample tag.

        Those 2 values should be the same.

        Significant bits are calculated from the max sample value, which is derived from the MaxSampleValue tag.
        It can be lower than the TIFF bits memory, if the image is not full range (for example 16 bit image in memory - maximum theoritical value is 65535, but the max sample value is 4095).
        """

        # turns out that the BitsPerSample tag can be a tuple, for example (8, 8, 8) for RGB image.
        # in that case, we need to check if all values are the same, and if not, raise an error.
        # this is not a common case, but it can happen.
        if isinstance(tiff_bits_memory, tuple):
            if len(set(tiff_bits_memory)) != 1:
                raise ValueError(f"TIFF bits memory is a tuple with different values: {tiff_bits_memory}.")
            tiff_bits_memory = tiff_bits_memory[0]

        if isinstance(max_sample_value, tuple):
            if len(set(max_sample_value)) != 1:
                raise ValueError(f"Max sample value is a tuple with different values: {max_sample_value}.")
            max_sample_value = max_sample_value[0]

        if numpy_bits_memory != tiff_bits_memory:
            raise ValueError(f"Mismatch between numpy dtype bits ({numpy_bits_memory}) and TIFF bits ({tiff_bits_memory}).")

        if max_sample_value <= 0:
            return tiff_bits_memory

        bpc_significant = max_sample_value.bit_length()
        if bpc_significant <= 8 and tiff_bits_memory > 8:
            return tiff_bits_memory
        return bpc_significant

    @staticmethod
    def get_significant_bits_from_ome(path) -> int:
        ome_types = _require_ome_types()
        try:
            return ome_types.from_tiff(path).images[0].pixels.significant_bits             # try to get significant bits from OME
        except:
            return None

    def nd2_attributes(self, *, sequence_count = 1) -> ImageAttributes:
        with tifffile.TiffReader(self.filename) as tiff:
            page = tiff.pages[self.idf]

            shape = page.shape
            dtype = page.dtype
            tags = {tag.name: tag.value for tag in page.tags.values()}

        components = (1 if len(shape) <= 2 else shape[2])
        numpy_bits = dtype.itemsize * 8

        tiff_bits = tags.get('BitsPerSample', numpy_bits)
        max_value = tags.get('MaxSampleValue', -1)

        if not (bpc_significant:= LimImageSourceTiff.get_significant_bits_from_ome(self.filename)):
            bpc_significant = LimImageSourceTiff.calculate_bpc_significant(numpy_bits, tiff_bits, max_value)

        bpc_memory = bpc_significant if bpc_significant % 8 == 0 else math.ceil(bpc_significant / 8) * 8

        if dtype in (np.int8, np.int16, np.int32):
            pixel_type = ImageAttributesPixelType.pxtSigned
        elif dtype in (np.uint8, np.uint16, np.uint32):
            pixel_type = ImageAttributesPixelType.pxtUnsigned
        elif dtype in (np.float16, np.float32, np.float64):
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            raise ValueError("TIFF file has unsupported pixel type.")

        return ImageAttributes(
            uiWidth = shape[1],
            uiWidthBytes = shape[1] * components * bpc_memory,
            uiHeight = shape[0],
            uiComp = components,
            uiBpcInMemory = bpc_memory,
            uiBpcSignificant = bpc_significant,
            uiSequenceCount = sequence_count,
            uiTileWidth = shape[1],
            uiTileHeight = shape[0],
            uiVirtualComponents = components,
            ePixelType = pixel_type
        )

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        ome_types = _require_ome_types()

        try:
            ome = ome_types.from_tiff(self.filename)
        except Exception:
            return
        if not ome:
            return

        ome_dims = self.get_file_dimensions()

        # Use ConvertSequenceArgs as a "sparse" partial B
        b = ConvertSequenceArgs()

        if "timeloop" in ome_dims:
            b.time_step = OMEUtils.time_step_from_ome(ome)

        if "zstack" in ome_dims:
            b.z_step = OMEUtils.z_step_from_ome(ome)

        if "channel" in ome_dims:
            # Pass through existing metadata to preserve your current extraction behavior
            meta_from_ome, chans_from_ome = OMEUtils.channels_from_ome(ome, metadata_storage.metadata)
            b.metadata = meta_from_ome
            b.channels = chans_from_ome

        # Merge into A (only fills None/missing fields in A)
        merge_four_fields(metadata_storage, b)


    def metadata_as_pattern_settings(self) -> dict:
        """
        Return metadata in the exact shape your QML dialog expects for patternSettings.

        Keys:
        - tstep, zstep, pixel_calibration, pinhole_diameter, objective_magnification,
            objective_numerical_aperture, immersion_refractive_index, zoom_magnification : strings
        - channels: list of [nameFromFile, customName, modalityString, ex, em, colorName]

        Any missing value is returned as an empty string "".
        """

        # Helper maps to your QML lists
        MODALITIES = [
            "Undefined", "Wide-field", "Brightfield", "Phase", "DIC", "DarkField",
            "MC", "TIRF", "Confocal, Fluo", "Confocal, Trans", "Multi-Photon",
            "SFC pinhole", "SFC slit", "Spinning Disc", "DSD", "NSIM", "iSim",
            "RCM", "CSU W1-SoRa", "NSPARC"
        ]
        COLOR_NAMES = [
            "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta",
            "Orange", "Pink", "Purple", "Brown", "Gray",
            "Black", "White"
        ]

        def _color_name_from_rgb(rgb):
            if not rgb or len(rgb) < 3:
                return ""
            r, g, b = rgb[:3]

            # --- Strict checks for vivid colors ---
            if r > 200 and g < 80 and b < 80:
                return "Red"
            if g > 200 and r < 80 and b < 80:
                return "Green"
            if b > 200 and r < 80 and g < 80:
                return "Blue"

            # Orange: strong red + moderate green, low blue
            if r > 200 and 100 < g < 180 and b < 80:
                return "Orange"

            # Pink: strong red + moderate blue, low green
            if r > 200 and g < 150 and b > 150:
                return "Pink"

            # Purple/Violet: red + blue both strong, green low
            if r > 120 and b > 120 and g < 100:
                return "Purple"

            # Yellow: high red + green, low blue
            if r > 200 and g > 200 and b < 100:
                return "Yellow"

            # Cyan: high green + blue, low red
            if g > 200 and b > 200 and r < 100:
                return "Cyan"

            # Magenta: high red + blue, low green
            if r > 200 and b > 200 and g < 100:
                return "Magenta"

            # Brown: moderate red + green, very low blue
            if r > 120 and g > 80 and b < 60:
                return "Brown"

            # Gray: all channels balanced, mid intensity
            if abs(r - g) < 20 and abs(g - b) < 20 and 50 < r < 200:
                return "Gray"

            # Black/White extremes
            if r < 40 and g < 40 and b < 40:
                return "Black"
            if r > 220 and g > 220 and b > 220:
                return "White"

            # --- Fallback heuristic: max component ---
            mx = max((r, "Red"), (g, "Green"), (b, "Blue"), key=lambda x: x[0])[1]
            return mx

        def _modality_from(acq_mode, contrast_method):
            # Normalize to strings
            am = (acq_mode or "").lower()
            cm = (contrast_method or "").lower()

            # Fluorescence vs transmitted confocal guess (very rough)
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

            # As a safe default
            return "Undefined"

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

        # --- Read OME ---
        try:
            ome_types = _require_ome_types()
            ome = ome_types.from_tiff(self.filename)
        except Exception:
            return {}

        if not ome or not getattr(ome, "images", None):
            return {}

        image = ome.images[0]

        # --- tstep / zstep ---
        # If you already have OMEUtils, you can keep using it; otherwise compute roughly
        try:
            tstep_s = OMEUtils.time_step_from_ome(ome)  # seconds or None
            zstep_um = OMEUtils.z_step_from_ome(ome)    # micrometers or None
        except Exception:
            tstep_s = None
            zstep_um = None

        if tstep_s is not None:
            t_ms = tstep_s * 1000.0
            result["tstep"] = f"{t_ms:.3f}"

        if zstep_um is not None:
            result["zstep"] = f"{zstep_um:.3f}"

        # --- pixel size / NA / RI ---
        objective_na = None
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

        if getattr(image, "objective_settings", None):
            refractive_index = getattr(image.objective_settings, "refractive_index", None)

        if getattr(image, "pixels", None):
            pixel_size_x = getattr(image.pixels, "physical_size_x", None)

        if pixel_size_x is not None:
            # Keep as float string; QML uses DoubleValidator here
            result["pixel_calibration"] = f"{float(pixel_size_x)}"
        if objective_na is not None:
            result["objective_numerical_aperture"] = f"{float(objective_na)}"
        if refractive_index is not None:
            result["immersion_refractive_index"] = f"{float(refractive_index)}"

        # --- Channels ---
        channels = []
        if getattr(image, "pixels", None) and getattr(image.pixels, "channels", None):
            # Sort by channel.id (same as your earlier logic)
            chs = sorted(image.pixels.channels, key=lambda x: x.id)
            for idx, ch in enumerate(chs):
                name_from_file = ch.name or f"Channel_{idx}"
                custom_name = name_from_file  # default custom = same as file name

                # modality
                acq_mode = getattr(ch, "acquisition_mode", None)
                contrast = getattr(ch, "contrast_method", None)
                modality = _modality_from(
                    acq_mode.value if acq_mode else None,
                    contrast.value if contrast else None
                )
                if modality not in MODALITIES:
                    modality = "Undefined"

                # wavelengths -> strings; default "0" to match your UI behavior
                ex = getattr(ch, "excitation_wavelength", None)
                em = getattr(ch, "emission_wavelength", None)
                ex_str = str(int(round(ex))) if ex is not None else "0"
                em_str = str(int(round(em))) if em is not None else "0"

                # color -> closest UI color name
                rgb = ch.color.as_rgb_tuple(alpha=False) if getattr(ch, "color", None) else None
                color_name = _color_name_from_rgb(rgb)
                if color_name not in COLOR_NAMES:
                    color_name = "Red"  # stable fallback

                channels.append([name_from_file, custom_name, modality, ex_str, em_str, color_name])

        result["channels"] = channels
        return result


        """ example output:
        {
            "tstep": "422",
            "zstep": "0",
            "pixel_calibration": "0.103174604",
            "pinhole_diameter": "",
            "objective_magnification": "",
            "objective_numerical_aperture": "1.4",
            "immersion_refractive_index": "1.518",
            "zoom_magnification": "",
            "channels": [
                ["DAPI",  "DAPI",  "Wide-field", "353", "465", "Cyan"],
                ["AF488", "AF488", "Wide-field", "493", "517", "Green"],
                ["AF555", "AF555", "Wide-field", "553", "568", "Red"],
                ["AF647", "AF647", "Wide-field", "653", "668", "Magenta"]
            ]
        }
        """

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
        # returns estimated z step in OME model
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
        # returns limnd2 Plane object from OME channel object
        try:
            acquisition = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(channel.acquisition_mode.value if channel.acquisition_mode else "unknown")
        except ValueError:
            acquisition = limnd2.metadata.PicturePlaneModalityFlags.modUnknown
        try:
            contrast = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(channel.contrast_method.value if channel.contrast_method else "unknown")
        except ValueError:
            contrast = limnd2.metadata.PicturePlaneModalityFlags.modUnknown

        plane = Plane(name = channel.name,
                    modality = acquisition | contrast,
                    color = channel.color.as_rgb_tuple(alpha=False),
                    excitation_wavelength = channel.excitation_wavelength,
                    emission_wavelength = channel.emission_wavelength
        )
        return plane

    @staticmethod
    def channels_from_ome(ome: "ome_types.model.OME", metadata_factory: MetadataFactory = None):
        # parses metadata from OME-TIFF file and returns a factory for such metadata and a dictionary of channels
        if metadata_factory is not None:
            provided_settings = metadata_factory._other_settings
        else:
            provided_settings = {}
        image = ome.images[0]

        objective_numerical_aperture = None
        immersion_refractive_index = None
        pixel_calibration = None

        used_instrument = None
        used_objective = None
        if ome.instruments:
            for instrument in ome.instruments:
                if instrument.id == image.instrument_ref.id:
                    used_instrument = instrument
        if used_instrument:
            if used_instrument.objectives and image.objective_settings:
                for objective in used_instrument.objectives:
                    if objective.id == image.objective_settings.id:
                        used_objective = objective

        if used_objective:
            objective_numerical_aperture = used_objective.lens_na

        immersion_refractive_index = image.objective_settings.refractive_index if image.objective_settings else None
        pixel_calibration = image.pixels.physical_size_x

        new_factory = MetadataFactory(pixel_calibration = metadata_factory.pixel_calibration if metadata_factory.pixel_calibration else pixel_calibration,
                                    immersion_refractive_index = provided_settings.get("immersion_refractive_index", immersion_refractive_index),
                                    objective_numerical_aperture = provided_settings.get("objective_numerical_aperture", objective_numerical_aperture))

        for key, value in provided_settings.items():
            if key not in new_factory._other_settings:
                new_factory._other_settings[key] = value

        channels = {}
        for index, channel in enumerate(sorted(image.pixels.channels, key=lambda x: x.id)):
            channels[index] = OMEUtils.channel_from_ome(channel)


        return new_factory, channels


