from __future__ import annotations

import enum, typing
import numpy as np
from dataclasses import dataclass
from .lite_variant import decode_lv
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
      ictLossLess = 0
      ictLossy = 1
      ictNone = 2

class ImageAttributesPixelType(enum.IntEnum):
      pxtSigned = 0
      pxtUnsigned = 1
      pxtReal = 2

@dataclass(frozen=True, kw_only=True)
class ImageAttributes:
    uiWidth: int = 0
    uiWidthBytes: int = 0
    uiHeight: int = 0
    uiComp: int = 0
    uiBpcInMemory: int = 0
    uiBpcSignificant: int = 0
    uiSequenceCount: int = 0
    uiTileWidth: int = 0
    uiTileHeight: int = 0
    eCompression: ImageAttributesCompression = ImageAttributesCompression.ictNone
    dCompressionParam: float = 0.0
    ePixelType: ImageAttributesPixelType = ImageAttributesPixelType.pxtUnsigned
    uiVirtualComponents: int = 0
    
    MinDownsampledSie: typing.ClassVar[int] = ND2_MIN_DOWNSAMPLED_SIZE

    def __post_init__(self):
        if self.uiBpcInMemory <= 0:
            self.uiBpcInMemory = (self.uiBpcSignificant + 7) // 8 * 8
        if self.uiWidthBytes <= 0:
            self.uiWidthBytes = (self.uiWidth * self.uiComp * self.uiBpcInMemory // 8 + 3) // 4 * 4
            
    @property
    def width(self) -> int:
        return self.uiWidth
    
    @property
    def height(self) -> int:
        return self.uiHeight
    
    @property
    def componentCount(self) -> int:
        return self.uiComp    
    
    @property
    def imageBytes(self) -> int:
        return self.uiWidthBytes * self.uiHeight
    
    @property
    def widthBytes(self) -> int:
        return self.uiWidthBytes
    
    @property
    def componentBytes(self) -> int:
        return self.uiBpcInMemory // 8
    
    @property
    def pixelBytes(self) -> int:
        return (self.uiBpcInMemory // 8) * self.uiComp

    @property
    def dtype(self) -> NumpyDTypeLike:
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
        return (self.uiHeight, self.uiWidth, self.uiComp)
    
    @property
    def strides(self) -> tuple[int, int, int]:
        return (self.widthBytes, self.pixelBytes, self.componentBytes)
    
    @property
    def frameCount(self) -> int:
        return self.uiSequenceCount
    
    @property
    def powSizeBase(self) -> int:
        return _full_res_base_pow2(self.uiWidth, self.uiHeight)
    
    @property
    def powSize(self) -> int:
        return full_res_size(self.uiWidth, self.uiHeight)
    
    @property
    def lowerPowSizeList(self) -> list[int]:
        ret, size = [], full_res_size(self.uiWidth, self.uiHeight)
        while ND2_MIN_DOWNSAMPLED_SIZE < size:
            size //= 2
            ret.append(size)
        return ret
    
    def makeDownsampledFromPowBase(self, powBase: int) -> ImageAttributes:
        return self.makeDownsampled(2 ** powBase)
            
    def makeDownsampled(self, downsize : int|None = None) -> ImageAttributes:
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
        raise NotImplementedError()

    @staticmethod
    def from_lv(data: bytes|memoryview) -> ImageAttributes:
        return ImageAttributes(**(decode_lv(data).get('SLxImageAttributes', {})))
    
    @staticmethod
    def from_var(data: bytes|memoryview) -> ImageAttributes:
        decoded = decode_var(data)
        return ImageAttributes(**decoded[0])    
    