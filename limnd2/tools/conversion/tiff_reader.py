import tifffile
import logging
from pathlib import Path
from typing import Type, Any, Union
import ome_types

import numpy as np
import warnings
import math

from limnd2.attributes import ImageAttributes, ImageAttributesPixelType

warnings.filterwarnings("ignore", category=RuntimeWarning, module="tifffile")
logging.getLogger('tifffile').setLevel(logging.ERROR)

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

def get_significant_bits_from_ome(path) -> int:
    try:
        return ome_types.from_tiff(path).images[0].pixels.significant_bits             # try to get significant bits from OME
    except:
        return None


class TiffPageMetadata:
    page_index: int
    shape: tuple[int, ...]
    dtype: Type[np.dtype]
    number_of_dimensions: int
    tags: dict[str, Any]
    color_space: str            # color space (e.g., RGB, GRAY, CMYK).
    compression: str            # compression used for the image (e.g., LZW, JPEG, NONE).

    def __init__(self, page: tifffile.TiffPage, page_index: int) -> None:
        self.page_index = page_index
        self.shape = page.shape
        self.dtype = page.dtype
        self.number_of_dimensions = len(page.shape)
        self.tags = {tag.name: tag.value for tag in page.tags.values()}
        self.color_space = page.photometric.name if page.photometric else 'Unknown'
        self.compression = page.compression.name if page.compression else 'None'

    def __str__(self):
        return f"\n\t\tPage number: {self.page_index}, Shape: {self.shape}, Type: {self.dtype}, Color space: {self.color_space}" #+ "".join(f"\n\t\t\t{k}: {v}" for k, v in self.tags.items())

class TiffPagesMetadata:
    pages: list[TiffPageMetadata]
    same_dimensions: bool

    def __init__(self, pages: tifffile.TiffPages):
        self.pages = []
        self.same_dimensions = True

        first_shape = None

        for page_index, page in enumerate(pages):
            page_obj = TiffPageMetadata(page, page_index)
            self.pages.append(page_obj)
            if not first_shape:
                first_shape = page_obj.shape

            if self.same_dimensions:
                if first_shape != page_obj.shape:
                    self.same_dimensions = False

    def __str__(self):
        return f"\n\tNumber of pages: {len(self.pages)}, same shape: {self.same_dimensions}" + "".join(str(page) for page in self.pages)

class TiffReader:
    path: Path
    pages_metadata: TiffPagesMetadata
    number_of_pages: int

    def __init__(self, path: Union[Path, str]) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        with tifffile.TiffFile(path) as tif:
            self.pages_metadata = TiffPagesMetadata(tif.pages)
        self.number_of_pages = len(self.pages_metadata.pages)

    def __str__(self) -> str:
        return str(self.path) + str(self.pages_metadata)

    def asarray(self, page_index: int = 0) -> np.array:
        """
        Usage of this function is not recommended, as this class loads a lot of additional data,
        if you just want to load the image data, use the `asarray` method of `tifffile.TiffFile`.

        This class is designed to provide metadata, tags and information about the TIFF file, reading might be unnecesisary slow.
        """
        if page_index < 0 or page_index >= self.number_of_pages:
            raise ValueError(f"Invalid page index, requested index: {page_index}, valid indices: 0 - {self.number_of_pages - 1}")

        with tifffile.TiffFile(self.path) as tif:
            return tif.pages[page_index].asarray()

    def get_nd2_attributes(self, page_index: int = 0, *, sequence_count = 1) -> ImageAttributes:
        if page_index < 0 or page_index >= self.number_of_pages:
            raise ValueError(f"Invalid page index, requested index: {page_index}, valid indices: 0 - {self.number_of_pages - 1}")
        page = self.pages_metadata.pages[page_index]
        shape = page.shape


        numpy_bits = page.dtype.itemsize * 8
        tiff_bits = page.tags.get('BitsPerSample', numpy_bits)
        max_value = page.tags.get('MaxSampleValue', -1)


        # get significant bits from OME, if not available, calculate it using bits per sample and max sample value
        if not (bits:= get_significant_bits_from_ome(self.path)):
            bits = calculate_bpc_significant(numpy_bits, tiff_bits, max_value)

        bpc_memory = bits if bits % 8 == 0 else math.ceil(bits / 8) * 8

        if page.dtype in (np.int8, np.int16, np.int32):
            pixel_type = ImageAttributesPixelType.pxtSigned
        elif page.dtype in (np.uint8, np.uint16, np.uint32):
            pixel_type = ImageAttributesPixelType.pxtUnsigned
        elif page.dtype in (np.float16, np.float32, np.float64):
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            raise ValueError("Tiff file has unsupported pixel type.")

        components = (1 if len(shape) <= 2 else shape[2])

        return ImageAttributes(
            uiWidth = shape[1],
            uiWidthBytes = shape[1] * components * bpc_memory,
            uiHeight = shape[0],
            uiComp = components,
            uiBpcInMemory = bpc_memory,
            uiBpcSignificant = bits,
            uiSequenceCount = sequence_count,
            uiTileWidth = shape[1],
            uiTileHeight = shape[0],
            uiVirtualComponents = components,
            ePixelType = pixel_type
        )

if __name__ == "__main__":
    a = TiffReader("C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\export\\file.tif")
    print(a.get_nd2_attributes())