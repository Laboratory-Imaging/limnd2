from __future__ import annotations

import math
import enum, typing
import numpy as np
from dataclasses import dataclass

from .lite_variant import decode_lv, encode_lv, LVSerializable, ELxLiteVariantType as LVType, LV_field
from .variant import decode_var

NumpyDTypeLike: typing.TypeAlias = np._typing.DTypeLike
NumpyArrayLike: typing.TypeAlias = np.ndarray

ND2_MIN_DOWNSAMPLED_SIZE = 512

def _full_res_base_pow2(*shape) -> int:
    target = max(*shape)
    if target <= 1:
        return 0
    for i in range(1, int(target)):
        if (2 ** i >= target):
            return i

def full_res_size(*shape) -> int:
    return 2 ** _full_res_base_pow2(shape)

class ImageAttributesCompression(enum.IntEnum):
    """
    Enum for image compression.
    """
    ictLossLess = 0
    ictLossy = 1
    ictNone = 2

class ImageAttributesPixelType(enum.IntEnum):
    """
    Enum for pixel type.
    """
    pxtSigned = 0
    pxtUnsigned = 1
    pxtReal = 2

    @staticmethod
    def short_name(type : ImageAttributesPixelType) -> str:
        match type:
            case ImageAttributesPixelType.pxtSigned:
                return "int"
            case ImageAttributesPixelType.pxtUnsigned:
                return "uint"
            case ImageAttributesPixelType.pxtReal:
                return "float"
            case _:
                return "unknown"

@dataclass(frozen=True, kw_only=True)
class ImageAttributes(LVSerializable):
    """
    Dataclass for ND2 Image attributes chunk, stores mostly information about image width, height,
    number of components and bit depth of each pixel, as well as information about compression and total number of images.
    """
    uiWidth: int                                = LV_field(0,                                       LVType.UINT32)
    uiWidthBytes: int                           = LV_field(0,                                       LVType.UINT32)
    uiHeight: int                               = LV_field(0,                                       LVType.UINT32)
    uiComp: int                                 = LV_field(0,                                       LVType.UINT32)
    uiBpcInMemory: int                          = LV_field(0,                                       LVType.INT32)
    uiBpcSignificant: int                       = LV_field(0,                                       LVType.INT32)
    uiSequenceCount: int                        = LV_field(0,                                       LVType.UINT32)
    uiTileWidth: int                            = LV_field(0,                                       LVType.UINT32)
    uiTileHeight: int                           = LV_field(0,                                       LVType.UINT32)
    eCompression: ImageAttributesCompression    = LV_field(ImageAttributesCompression.ictNone,      LVType.INT32)
    dCompressionParam: float                    = LV_field(0.0,                                     LVType.DOUBLE)
    ePixelType: ImageAttributesPixelType        = LV_field(ImageAttributesPixelType.pxtUnsigned,    LVType.INT32)
    uiVirtualComponents: int                    = LV_field(0,                                       LVType.UINT32)

    MinDownsampledSize: typing.ClassVar[int] = ND2_MIN_DOWNSAMPLED_SIZE

    def __post_init__(self):
        object.__setattr__(self, 'eCompression', ImageAttributesCompression(self.eCompression))
        object.__setattr__(self, 'ePixelType', ImageAttributesPixelType(self.ePixelType))

    @staticmethod
    def calcWidthBytes(width: int, bits: int, comps: int) -> int:
        """
        Calculates number of bytes per single image row.
        """
        return (width * comps * (bits + 7) // 8 + 3) // 4 * 4

    @staticmethod
    def create(shape: tuple[int], bits: int, sequence_count: int) -> ImageAttributes:
        """
        Create ImageAttributes instance from simplified parameters:

        shape: tuple[int]   - tuple in following format: (width_pixels, height_pixels, [component_count]), component count is optional, 1 by default
        bits: int           - number of bits per pixel component
        sequence_count: int - total number of frames in ND2 file (product of size of each dimension)
        """
        if bits == 32:
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            pixel_type = ImageAttributesPixelType.pxtUnsigned

        components = (1 if len(shape) <= 2 else shape[2])

        return ImageAttributes(
            uiWidth = shape[1],
            uiWidthBytes = ImageAttributes.calcWidthBytes(shape[0], bits, components),
            uiHeight = shape[0],
            uiComp = components,
            uiBpcInMemory = bits if bits % 8 == 0 else math.ceil(bits / 8) * 8,
            uiBpcSignificant = bits,
            uiSequenceCount = sequence_count,
            uiTileWidth = shape[1],
            uiTileHeight = shape[0],
            uiVirtualComponents = components,
            ePixelType = pixel_type
        )

    @property
    def width(self) -> int:
        """
        Returns width of the image in pixels.
        """
        return self.uiWidth

    @property
    def height(self) -> int:
        """
        Returns height of the image in pixels.
        """
        return self.uiHeight

    @property
    def componentCount(self) -> int:
        """
        Returns number of components in the image.
        """
        return self.uiComp

    @property
    def imageBytes(self) -> int:
        """
        Total size of the image in bytes.
        """
        return self.uiWidthBytes * self.uiHeight

    @property
    def widthBytes(self) -> int:
        """
        Size of image row in bytes.
        """
        return self.uiWidthBytes

    @property
    def componentBytes(self) -> int:
        """
        Size of component in bytes.
        """
        return self.uiBpcInMemory // 8

    @property
    def pixelBytes(self) -> int:
        """
        Size of pixel in bytes.
        """
        return (self.uiBpcInMemory // 8) * self.uiComp

    @property
    def dtype(self) -> NumpyDTypeLike:
        """
        Returns numpy datatype used for storing image data in Python.
        """
        SINT_DTYPES = { 1 : np.int8, 2 : np.int16, 4 : np.int32 }
        UINT_DTYPES = { 1 : np.uint8, 2 : np.uint16, 4 : np.uint32 }
        REAL_DTYPES = { 2 : np.float16, 4 : np.float32, 8: np.float64 }
        if self.ePixelType == ImageAttributesPixelType.pxtSigned:
            return SINT_DTYPES[self.componentBytes]
        elif self.ePixelType == ImageAttributesPixelType.pxtUnsigned:
            return UINT_DTYPES[self.componentBytes]
        elif self.ePixelType == ImageAttributesPixelType.pxtReal:
            return REAL_DTYPES[self.componentBytes]
        else:
            raise RuntimeError("Unsupported data type")

    @property
    def safe_dtype(self) -> NumpyDTypeLike:
        """
        Returns numpy datatype that will always be big enough to fit pixel component data.
        """
        if self.ePixelType == ImageAttributesPixelType.pxtSigned:
            return np.int32
        elif self.ePixelType == ImageAttributesPixelType.pxtUnsigned:
            return np.int32
        elif self.ePixelType == ImageAttributesPixelType.pxtReal:
            return np.float32
        else:
            raise RuntimeError("Unsupported data type")

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns shape of the image which can be used with in array (height_pixels, width_pixels, component_count)
        """
        return (self.uiHeight, self.uiWidth, self.uiComp)

    @property
    def strides(self) -> tuple[int, int, int]:
        """
        Returns the strides of the image which can be used in numpy array
        """
        return (self.widthBytes, self.pixelBytes, self.componentBytes)

    @property
    def frameCount(self) -> int:
        """
        Returns number of frames in the ND2 file.
        """
        return self.uiSequenceCount

    @property
    def powSize(self) -> int:
        """
        Returns next power of 2 for bigger dimension.
        """
        return full_res_size(self.uiWidth, self.uiHeight)

    @property
    def powSizeBase(self) -> int:
        """
        Returns exponent used in powSize() function.
        """
        return _full_res_base_pow2(self.uiWidth, self.uiHeight)

    @property
    def lowerPowSizeList(self) -> list[int]:
        """
        Returns list of powers of 2 between ND2_MIN_DOWNSAMPLED_SIZE and image resolution.
        """
        ret, size = [], full_res_size(self.uiWidth, self.uiHeight)
        while ND2_MIN_DOWNSAMPLED_SIZE < size:
            size //= 2
            ret.append(size)
        return ret

    def makeDownsampledFromPowBase(self, powBase: int) -> ImageAttributes:
        """
        Returns ImageAttributes for downsampled image using power of 2 exponent.
        """
        return self.makeDownsampled(2 ** powBase)

    def makeDownsampled(self, downsize : int|None = None) -> ImageAttributes:
        """
        Returns ImageAttributes for downsampled image using power of 2 image size.
        """
        full_size = full_res_size(self.uiWidth, self.uiHeight)
        if downsize is None:
            downsize = full_size // 2
        if downsize <= 2:
            raise ValueError(f"Unexpected downsize: {downsize}")
        w = self.uiWidth * downsize // full_size
        h = self.uiHeight * downsize // full_size
        wb = ((self.uiBpcInMemory // 8) * self.uiComp * w + 3) // 4 * 4
        return ImageAttributes(uiWidth=w, uiWidthBytes=wb, uiHeight=h, uiComp=self.uiComp,
                               uiBpcInMemory=self.uiBpcInMemory, uiBpcSignificant=self.uiBpcSignificant,
                               uiSequenceCount=self.uiSequenceCount, ePixelType=self.ePixelType)


    def to_lv(self) -> bytes:
        """
        Encodes ImageAttributes to ND2 lite variant chunk.
        """
        return encode_lv({"SLxImageAttributes" : self.to_serializable_dict()})

    @staticmethod
    def from_lv(data: bytes|memoryview) -> ImageAttributes:
        """
        Decodes ImageAttributes from ND2 lite variant chunk.
        """
        return ImageAttributes(**(decode_lv(data).get('SLxImageAttributes', {})))

    @staticmethod
    def from_var(data: bytes|memoryview) -> ImageAttributes:
        """
        Decodes ImageAttributes from ND2 XML chunk.
        """
        decoded = decode_var(data)
        return ImageAttributes(**decoded[0])
