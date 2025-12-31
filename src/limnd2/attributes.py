from __future__ import annotations

import enum, functools, math, typing
import numpy as np
from dataclasses import dataclass

from .lite_variant import decode_lv, encode_lv, LVSerializable, ELxLiteVariantType as LVType, LV_field
from .variant import decode_var

try:
    from numpy.typing import DTypeLike as NumpyDTypeLike
except Exception:
    NumpyDTypeLike = typing.Any
NumpyArrayLike: typing.TypeAlias = np.ndarray

ND2_MIN_DOWNSAMPLED_SIZE = 512

def _full_res_base_pow2(*shape) -> int:
    target = max(*shape)
    if target <= 1:
        return 0
    for i in range(1, int(target)):
        if (2 ** i >= target):
            return i
    return 0

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
        return (width * comps * ((bits + 7) // 8) + 3) // 4 * 4

    @staticmethod
    def create(height: int, width: int, component_count: int, bits: int, sequence_count: int) -> ImageAttributes:
        """
        !!! warning
            This function is used for creating new ImageAttributes instance, usually for creating new .nd2 files with [Nd2Writer](nd2.md#limnd2.nd2.Nd2Writer) class.
            Do not use this function if you only read `.nd2` file.

        Create ImageAttributes instance from following arguments (all must be passed as named arguments)

        Parameters
        ----------
        height : int
            height in pixels
        width : int
            width in pixels
        component_count : int
            number of components
        bits : int
            number of bits per pixel component
        sequence_count : int
            total number of frames in ND2 file (product of size of each dimension)
        """

        if bits == 32:
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            pixel_type = ImageAttributesPixelType.pxtUnsigned

        return ImageAttributes(
            uiWidth = width,
            uiWidthBytes = ImageAttributes.calcWidthBytes(width, bits, component_count),
            uiHeight = height,
            uiComp = component_count,
            uiBpcInMemory = bits if bits % 8 == 0 else math.ceil(bits / 8) * 8,
            uiBpcSignificant = bits,
            uiSequenceCount = sequence_count,
            uiTileWidth = width,
            uiTileHeight = height,
            uiVirtualComponents = component_count,
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

    @functools.cached_property
    def powSize(self) -> int:
        """
        Returns next power of 2 for bigger dimension.
        """
        return full_res_size(self.uiWidth, self.uiHeight)

    @functools.cached_property
    def powSizeBase(self) -> int:
        """
        Returns exponent used in powSize() function.
        """
        return _full_res_base_pow2(self.uiWidth, self.uiHeight)

    @functools.cached_property
    def downsampleLevels(self) -> list[int]:
        """
        Returns list containing levels of downsampled images up to ND2_MIN_DOWNSAMPLED_SIZE.
        """
        i = 1
        p = self.powSize // 2
        ret = []
        while ND2_MIN_DOWNSAMPLED_SIZE < p:
            ret.append(i)
            p = p // 2
            i += 1
        return ret

    def findDownsampledLevelFor(self, size: int) -> int:
        """
        Returns downsample level for given size.
        """
        l = 0
        s = max(self.uiWidth, self.uiHeight)
        while size < s:
            s = s // 2
            l += 1
        return max(0, l - 1)

    def makeDownsampled(self, downsample_level: int = 1) -> ImageAttributes:
        """
        Returns ImageAttributes for downsampled image.

        downsample_level: int
            Determines downsampling d = 2^downsample_level that produces
            lower level frames of size (w // d, h // d).
        """
        assert 0 <= downsample_level, f"Downsample level must be positive non-negative but got {downsample_level}"
        if 0 == downsample_level:
            return self
        w = self.uiWidth // (1 << downsample_level)
        h = self.uiHeight // (1 << downsample_level)
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
        return ImageAttributes(**decoded[0]) # type: ignore
