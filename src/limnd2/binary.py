from __future__ import annotations

import collections, enum, json, re
from typing_extensions import Literal
import numpy as np
from dataclasses import dataclass, asdict
from .attributes import full_res_size
from .variant import decode_var


class BinaryItemStateFlags(enum.IntFlag):
      eBinLayerWorking     = 0x00000001
      eBinLayerTracked     = 0x00000010
      eBinLayerAssayTracked= 0x00000020
      eBinLayerSpots       = 0x00000040
      eBinLayerLines       = 0x00000080
      eBinLayer3DConnected = 0x00000100
      eBinLayerClassified  = 0x00001000
      eBinLayerSel         = 0x00010000
      eBinLayerComponent   = 0x00080000
      eBinLayerReserved    = 0xF0000000

class BinaryItemColorMode(enum.IntEnum):
      eBaseBinObjColors = 0
      eCustomBinObjColors = 1
      eClassifierBinObjColors = 2
      eObjectsBinObjColors = 4
      e3DObjectsBinObjColors = 8
      eTrackedBinObjColors = 16
      eObjectsBinObjColorsById = 32
      eObjectsBinObjColorsByValue = 64
      e3DObjectsBinObjColorsByValue = 128

@dataclass(frozen=True, kw_only=True)
class BinaryRleMetadataItem:
    uiBinLayerID: int = 0
    strUuid: str = ""
    uiState: BinaryItemStateFlags | Literal[0] = 0
    uiColor: int = 0
    strName: str = ""
    strCompName: str = ""
    uiCompOrder: int = 0
    strFileTag: str = ""
    uiColorMode: BinaryItemColorMode = BinaryItemColorMode.eBaseBinObjColors

    def dataChunkName(self, seq_index: int) -> bytes:
        return f"CustomDataSeq|{self.strFileTag}|{seq_index}!".encode("ascii")

    @property
    def id(self) -> int:
        return self.uiBinLayerID

    @property
    def dataChunkNameRegex(self) -> re.Pattern:
        return re.compile(f'^CustomDataSeq\\|{self.strFileTag}\\|(\\d+)!$'.encode("ascii"))

    def makeRasterMetadata(self, width: int, height: int, tilesize: int = 1024, bitdepth: int = 32) -> BinaryRasterMetadataItem:
        return BinaryRasterMetadataItem(binWidth=width, binHeight=height,
            binTileWidth=tilesize, binTileHeight=tilesize,
            binCompressionId="zlib", binCompressionLevel=6,
            binBitdepth=bitdepth, binLayerId=self.uiBinLayerID,
            binName=self.strName, binUuid=self.strUuid,
            binComp=self.strCompName, binCompOrder=self.uiCompOrder,
            binState=self.uiState, binColor=self.uiColor,
            binColorMode=self.uiColorMode,
            emulatedOverRle=True)

class BinaryRleMetadata(collections.UserList):
    def __init__(self, iterable):
        super().__init__(BinaryRleMetadataItem(**item) for item in iterable)

    def findItemById(self, id: int) -> BinaryRleMetadataItem|None:
        for item in self.data:
            if item.uiBinLayerID == id:
                return item
        return None

    @property
    def dataChunkNameRegexDict(self) -> dict[int, re.Pattern]:
        return { item.uiBinLayerID: item.dataChunkNameRegex for item in self.data}

    @property
    def binIdList(self) -> list[int]:
        return [item.uiBinLayerID for item in self.data]

    def makeRasterMetadata(self, width: int, height: int, tilesize: int = 1024, bitdepth: int = 32) -> BinaryRasterMetadata:
        return BinaryRasterMetadata([item.makeRasterMetadata(width, height, tilesize, bitdepth) for item in self.data])

    @staticmethod
    def from_var(data: bytes|memoryview) -> BinaryRleMetadata:
        decoded = decode_var(data)
        return BinaryRleMetadata(decoded.get('BinaryMetadata_v1', []))

@dataclass(frozen=True, kw_only=True)
class BinaryRasterMetadataItem:
    binWidth: int = 0
    binHeight: int = 0
    binTileWidth: int = 0
    binTileHeight: int = 0
    binCompressionId: str = "zlib"
    binCompressionLevel: int = 6
    binBitdepth: int = 32
    binLayerId: int = 0
    binName: str
    binUuid: str
    binComp: str
    binCompOrder: int = 0
    binState: BinaryItemStateFlags | Literal[0] = 0
    binColor: int = 0
    binColorMode: BinaryItemColorMode = BinaryItemColorMode.eBaseBinObjColors
    emulatedOverRle: bool = False

    @property
    def id(self) -> int:
        return self.binLayerId

    @property
    def name(self) -> str:
        return self.binName

    @property
    def color(self) -> tuple[float, float, float]:
        color = self.binColor
        b = ((color >> 16) & 0xFF) / 255.0
        g = ((color >> 8) & 0xFF) / 255.0
        r = (color & 0xFF) / 255.0
        return (r, g, b)

    @property
    def dtype(self):
        return np.uint32

    @property
    def shape(self):
        return (self.binHeight, self.binWidth)

    @property
    def tileShape(self):
        return (self.binTileHeight, self.binTileWidth)

    @property
    def strides(self):
        return (self.binWidth * 4, 4)

    @property
    def tileStrides(self):
        return (self.binTileWidth * 4, 4)

    @property
    def imageBytes(self):
        return self.binWidth * 4 * self.binHeight

    @property
    def tileBytes(self):
        return self.binTileWidth * 4 * self.binTileHeight

    def makeDownsampled(self, downsize : int) -> BinaryRasterMetadataItem:
        fullsize = full_res_size(self.binWidth, self.binHeight)
        w = self.binWidth * downsize // fullsize
        h = self.binHeight * downsize // fullsize
        return BinaryRasterMetadataItem(
            binWidth=w, binHeight=h, binTileWidth=self.binTileWidth, binTileHeight=self.binTileHeight,
            binCompressionId=self.binCompressionId, binCompressionLevel=self.binCompressionLevel,
            binBitdepth=self.binBitdepth, binLayerId=self.binLayerId, binName=self.binName,
            binUuid=self.binUuid, binComp=self.binComp, binCompOrder=self.binCompOrder,
            binState=self.binState, binColor=self.binColor, binColorMode=self.binColorMode)


class BinaryRasterMetadata(collections.UserList):
    def __init__(self, iterable):
        super().__init__(BinaryRasterMetadataItem(**item) if type(item) == dict else item  for item in iterable)

    def findItemById(self, id: int) -> BinaryRasterMetadataItem|None:
        for item in self.data:
            if item.binLayerId == id:
                return item
        return None

    @property
    def binIdList(self) -> list[int]:
        return [item.binLayerId for item in self.data]

    @property
    def binNameList(self) -> list[str]:
        return [item.binName for item in self.data]

    @property
    def binColorList(self) -> list[str]:
        return [item.binColor for item in self.data]

    def to_json(self) -> bytes:
        return json.dumps([asdict(item) for item in self.data]).encode('utf-8')

    @staticmethod
    def from_json(data: bytes|memoryview) -> BinaryRasterMetadata:
        if type(data) == memoryview:
            data = data.tobytes()
        decoded = json.loads(data.decode('utf-8'))
        return BinaryRasterMetadata(decoded)
