from __future__ import annotations
import copy
import itertools
import math
from pathlib import Path

import numpy as np
import tifffile

import limnd2
from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.metadata_factory import MetadataFactory, Plane
from .LimImageSource import LimImageSource
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
        indices = list(itertools.product(*ranges))
        index_to_idf = [(indices, idf) for idf, indices in enumerate(indices)]

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
        import ome_types
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
        import ome_types
        try:
            ome = ome_types.from_tiff(self.filename)
        except Exception as e:
            return

        if ome:
            ome_dims = self.get_file_dimensions()
            if "timeloop" in ome_dims:
                metadata_storage.time_step = OMEUtils.time_step_from_ome(ome)

            if "zstep" in ome_dims:
                metadata_storage.z_step = OMEUtils.z_step_from_ome(ome)

            if "channel" in ome_dims:
                metadata_storage.metadata, metadata_storage.channels = OMEUtils.channels_from_ome(ome, metadata_storage.metadata)

class OMEUtils:
    @staticmethod
    def time_step_from_ome(ome: "ome_types.model.OME"):
        # returns estimated time step in OME model
        planes = ome.images[0].pixels.planes
        times = list(set([plane.delta_t for plane in planes]))
        times.sort()
        return ((times[-1] - times[0]) / (len(times) - 1))

    @staticmethod
    def z_step_from_ome(ome: "ome_types.model.OME"):
        # returns estimated z step in OME model
        planes = ome.images[0].pixels.planes
        zpositions = list(set([plane.position_z for plane in planes]))
        zpositions.sort()
        return ((zpositions[-1] - zpositions[0]) / (len(zpositions) - 1))

    @staticmethod
    def channel_from_ome(channel: "ome_types.model.Channel"):
        # returns limnd2 Plane object from OME channel object
        acquisition = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(channel.acquisition_mode.value)
        contrast = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(channel.contrast_method.value)

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
        if len(ome.instruments):
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

        immersion_refractive_index = image.objective_settings.refractive_index
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


