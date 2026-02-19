from __future__ import annotations
from contextlib import contextmanager
import copy
import itertools
import math
from pathlib import Path
import numpy as np
import zarr

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

_SAMPLE_DIMS = (1, 2, 3, 4)

def _height_width_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        if shape[-1] in _SAMPLE_DIMS:
            return shape[0], shape[1]
        if shape[0] in _SAMPLE_DIMS:
            return shape[1], shape[2]
        return shape[-2], shape[-1]
    if shape[-1] in _SAMPLE_DIMS:
        return shape[-3], shape[-2]
    return shape[-2], shape[-1]


def _normalize_to_yxs(tile: np.ndarray) -> np.ndarray:
    """Normalize to (Y,X) or (Y,X,S)."""
    if tile.ndim == 2:
        return tile
    if tile.ndim == 3:
        # planar (S,Y,X) -> (Y,X,S)
        if tile.shape[0] in _SAMPLE_DIMS and tile.shape[-1] not in _SAMPLE_DIMS:
            tile = np.moveaxis(tile, 0, -1)
        return tile
    raise ValueError(f"Unsupported ndim {tile.ndim} for shape {tile.shape}")


def _tile_index(arr, xx0: int, xx1: int, yy0: int, yy1: int):
    """
    Index ROI without exploding leading dims:
      - (Y,X)
      - (Y,X,S)
      - (S,Y,X)
      - (...,Y,X) and (...,Y,X,S) -> pins leading dims to 0
    """
    shape = arr.shape
    ndim = len(shape)

    if ndim == 2:
        return (slice(yy0, yy1), slice(xx0, xx1))

    if ndim == 3:
        # (Y,X,S)
        if shape[-1] in _SAMPLE_DIMS:
            return (slice(yy0, yy1), slice(xx0, xx1), slice(None))
        # (S,Y,X)
        if shape[0] in _SAMPLE_DIMS:
            return (slice(None), slice(yy0, yy1), slice(xx0, xx1))
        # fallback: treat as (Z,Y,X) and pick Z=0
        return (0, slice(yy0, yy1), slice(xx0, xx1))

    # ndim >= 4
    if shape[-1] in _SAMPLE_DIMS:
        # (..., Y, X, S)
        lead = (0,) * (ndim - 4)
        return lead + (slice(yy0, yy1), slice(xx0, xx1), slice(None))

    # (..., Y, X)
    lead = (0,) * (ndim - 3)
    return lead + (slice(yy0, yy1), slice(xx0, xx1))


def _slice_yx(arr, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    idx = _tile_index(arr, x0, x1, y0, y1)
    tile = np.asarray(arr[idx])          # materialize only ROI
    tile = _normalize_to_yxs(tile)       # planar -> interleaved
    if tile.ndim == 3 and tile.shape[-1] == 3:
        tile = tile[..., ::-1]           # RGB -> BGR
    return tile


class LimImageSourceTiff(LimImageSource):
    """Image source reading from TIFF (supports ROI reads)."""
    idf: int
    channel_index: int | None

    def __init__(self, filename: str | Path, idf: int = 0, channel_index: int | None = None):
        super().__init__(filename)
        self.idf = idf
        self.channel_index = int(channel_index) if channel_index is not None else None

    def __repr__(self):
        return f"LimImageSourceTiff({self.filename}, {self.idf}, channel_index={self.channel_index})"

    @property
    def additional_information(self) -> dict:
        if self._additional_dimensions is None:
            self._additional_dimensions = self._calculate_additional_information()
        return copy.deepcopy(self._additional_dimensions)

    @property
    def supports_tile_read(self) -> bool:
        return True

    def read(self) -> np.ndarray:
        # Full read (will load whole frame); only used when you decide frame is small enough.
        with tifffile.TiffFile(self.filename) as tif:
            arr = tif.asarray(key=self.idf)

        arr = _normalize_to_yxs(np.asarray(arr))
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[..., ::-1]  # RGB -> BGR
        return arr

    def read_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        # One-off tile read convenience. Your writer should prefer open_tile_reader().
        with self.open_tile_reader() as r:
            return np.array(r(x, y, w, h), copy=True)

    @contextmanager
    def open_tile_reader(self):
        """
        Open TIFF and return a callable read_tile(x,y,w,h)->ndarray
        without reopening TIFF / rebuilding Zarr per tile.
        """
        tif = tifffile.TiffFile(self.filename)
        store = None
        try:
            page = tif.pages[self.idf]

            # Fast path: memmap when possible
            try:
                arr = tifffile.memmap(self.filename, page=self.idf)
            except Exception:
                store = page.aszarr(chunkmode="strile")
                root = zarr.open(store, mode="r")
                # unwrap groups
                arr = root
                while not hasattr(arr, "shape"):
                    key = "0" if "0" in arr else sorted(arr.keys())[0]
                    arr = arr[key]

            H, W = _height_width_from_shape(arr.shape)

            def _read_tile(x: int, y: int, w: int, h: int) -> np.ndarray:
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + int(w), y0 + int(h)

                xx0, xx1 = max(0, x0), min(W, x1)
                yy0, yy1 = max(0, y0), min(H, y1)

                if xx0 >= xx1 or yy0 >= yy1:
                    # empty ROI
                    if len(arr.shape) >= 3 and (arr.shape[-1] in _SAMPLE_DIMS or arr.shape[0] in _SAMPLE_DIMS):
                        s = arr.shape[-1] if arr.shape[-1] in _SAMPLE_DIMS else arr.shape[0]
                        return np.empty((0, 0, s), dtype=arr.dtype)
                    return np.empty((0, 0), dtype=arr.dtype)

                return _slice_yx(arr, xx0, xx1, yy0, yy1)

            yield _read_tile

        finally:
            try:
                tif.close()
            finally:
                if store is not None:
                    try:
                        store.close()
                    except Exception:
                        pass

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
            pm_name = tiff.pages[0].photometric.name

            if pm_name in ("RGB", "YCBCR"):
                # Treat YCbCr as RGB-like (color image, not "channels")
                result["is_rgb"] = True
                self._is_rgb = True
            else:
                result["is_rgb"] = False
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

    @staticmethod
    def _has_ome_metadata(path: str | Path) -> bool:
        """Cheap check whether the TIFF contains OME-XML in ImageDescription."""
        try:
            # you already have a module-level `tifffile = _require_tifffile()`
            with tifffile.TiffFile(path) as tif:
                desc_tag = tif.pages[0].tags.get("ImageDescription")
                if desc_tag is None:
                    return False

                value = desc_tag.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
        except Exception:
            return False

        text = value.lstrip()
        return (
            (text.startswith("<?xml") and "<OME" in text)
            or "http://www.openmicroscopy.org/Schemas/OME" in text
        )

    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            with tifffile.TiffReader(self.filename) as tiff:
                pm_name = tiff.pages[0].photometric.name
                self._is_rgb = pm_name in ("RGB", "YCBCR")
        return self._is_rgb



    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSource", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSource", list[int]], dict[str, int]]:
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
        channel_axis_index = list(new_dimension.keys()).index("channel") if "channel" in new_dimension else None

        for file, dims in sources.items():
            for ome_dims, idf in index_to_idf:
                file_copy = copy.deepcopy(file)
                file_copy.idf = idf
                if channel_axis_index is not None:
                    file_copy.channel_index = int(ome_dims[channel_axis_index])
                else:
                    file_copy.channel_index = None
                new_files[file_copy] = dims + list(ome_dims)

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
                logprint("WARNING: File contains unknown dimension, but no unknown dimension type was provided. This may lead to data loss.", type="warning")
            else:
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
    def get_significant_bits_from_ome(path) -> int | None:
        has_ome = LimImageSourceTiff._has_ome_metadata(path)
        if not has_ome:
            return None
        try:
            ome_types = _require_ome_types()
        except ImportError:
            return None
        try:
            return ome_types.from_tiff(path).images[0].pixels.significant_bits             # try to get significant bits from OME
        except Exception:
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
            uiWidth = shape[1],
            uiWidthBytes = width_bytes,
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
        has_ome = LimImageSourceTiff._has_ome_metadata(self.filename)
        if not has_ome:
            return

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
        - channels: list of [nameFromFile, customName, modalityString, ex, em, colorString]

        Any missing value is returned as an empty string "".
        """

        # Helper maps to your QML lists
        MODALITIES = [
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

        def _pinhole_um_from_channel(ch):
            val = getattr(ch, "pinhole_size", None)
            if val is None:
                return None
            unit = getattr(ch, "pinhole_size_unit", None)
            # OME default is micrometers when unit is omitted
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
            # Keep as float string; QML uses DoubleValidator here
            result["pixel_calibration"] = f"{float(pixel_size_x)}"
        if objective_na is not None:
            result["objective_numerical_aperture"] = f"{float(objective_na)}"
        if objective_mag is not None:
            result["objective_magnification"] = f"{float(objective_mag)}"
        if refractive_index is not None:
            result["immersion_refractive_index"] = f"{float(refractive_index)}"

        # --- Channels ---
        channels = []
        if getattr(image, "pixels", None) and getattr(image.pixels, "channels", None):
            # Sort by channel.id (same as your earlier logic)
            chs = sorted(image.pixels.channels, key=lambda x: x.id)
            # --- pinhole diameter (only if all channels share the same value) ---
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

                # color -> hex string (preferred by QML)
                rgb = ch.color.as_rgb_tuple(alpha=False) if getattr(ch, "color", None) else None
                color_name = _color_to_hex(rgb)

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

        color = channel.color.as_rgb_tuple(alpha=False) if getattr(channel, "color", None) else None

        plane = Plane(name = channel.name,
                    modality = acquisition | contrast,
                    color = color,
                    excitation_wavelength = channel.excitation_wavelength,
                    emission_wavelength = channel.emission_wavelength
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
            pixel_calibration = pixel_calibration,
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


