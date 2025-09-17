from __future__ import annotations

from pathlib import Path
import abc, datetime, io, itertools, re, struct, typing, zlib
import numpy as np
from .attributes import ImageAttributes, NumpyArrayLike
from .binary import BinaryRleMetadata, BinaryRasterMetadata
from .experiment import ExperimentLevel, ExperimentLoopType
from .metadata import PictureMetadata
from .textinfo import ImageTextInfo

FileLikeObject: typing.TypeAlias = str | Path | typing.BinaryIO | memoryview
ChunkMap: typing.TypeAlias = typing.Mapping[bytes, tuple]

Nd2LoggerEnabled = False
if Nd2LoggerEnabled:
    import logging
    logger = logging.getLogger("limnd2")

ND2_FILE_SIGNATURE:     typing.Final                    = b"ND2 FILE SIGNATURE CHUNK NAME01!" # len 32, at the start of file
ND2_CHUNKMAP_SIGNATURE: typing.Final                    = b"ND2 CHUNK MAP SIGNATURE 0000001!" # len 32, at end of file and last chunk in chunkmap

ND2_FILEMAP_SIGNATURE:  typing.Final                    = b"ND2 FILEMAP SIGNATURE NAME 0001!" # len 32, name of chunkmap
ND2_CHUNK_MAGIC:        typing.Final                    = 0x0ABECEDA
JP2_MAGIC:              typing.Final                    = 0x0C000000

ND2_CHUNK_NAME_AppInfo                                  = b'CustomDataVar|AppInfo_V1_0!'

ND2_CHUNK_NAME_ImageAttributes                          = b'ImageAttributes!'
ND2_CHUNK_NAME_ImageAttributesLV                        = b'ImageAttributesLV!'                         #basic attributes

ND2_CHUNK_NAME_ImageMetadata                            = b'ImageMetadata!'
ND2_CHUNK_NAME_ImageMetadataLV                          = b'ImageMetadataLV!'                           #experiment

ND2_CHUNK_NAME_ImageTextInfo                            = b'ImageTextInfo!'
ND2_CHUNK_NAME_ImageTextInfoLV                          = b'ImageTextInfoLV!'

ND2_CHUNK_FORMAT_ImageMetadata_1p                       = b'ImageMetadataSeq|%u!' # |seq_index!
ND2_CHUNK_RE_ImageMetadata_1p                           = re.compile(b'^ImageMetadataSeq\\|(\\d+)!$')

ND2_CHUNK_NAME_AcqTimesCache                            = b'CustomData|AcqTimesCache!'
ND2_CHUNK_NAME_AcqTimes2Cache                           = b'CustomData|AcqTimes2Cache!'
ND2_CHUNK_NAME_AcqFramesCache                           = b'CustomData|AcqFramesCache!'
ND2_CHUNK_NAME_TimeSourceCache                          = b'CustomData|TimeSourceCache!'
ND2_CHUNK_NAME_FloatRangeCache                          = b'CustomData|FloatRangeCache!'
ND2_CHUNK_FORMAT_FloatCompRangeCache_1p                 = b'CustomData|FloatCompRangeCache%u!'
ND2_CHUNK_RE_FloatCompRangeCache_1p                     = re.compile(b'^CustomData\\|FloatCompRangeCache(\\d+)!$')

ND2_CHUNK_NAME_CustomDescription                        = b'CustomData|CustomDescriptionV1_0!'
ND2_CHUNK_NAME_CustomDataVar                            = b'CustomDataVar|CustomDataV2_0!'
ND2_CHUNK_NAME_CustomDataVarLI                          = b'CustomDataVar|CustomDataV2_0LI!'

ND2_CHUNK_NAME_WellPlateDesc                            = b'CustomData|WellPlateDesc_0!'
ND2_CHUNK_NAME_WellPlateFrameInfo                       = b'CustomData|WellPlateFrameInfoZJSON!'

ND2_CHUNK_FORMAT_ImageMetadataLV_1p                     = b'ImageMetadataSeqLV|%u!' # |seq_index!           #metadata
ND2_CHUNK_RE_ImageMetadataLV_1p                         = re.compile(b'^ImageMetadataSeqLV\\|(\\d+)!$')

ND2_CHUNK_NAME_BinaryMetadata_v1                        = b'CustomDataVar|BinaryMetadata_v1!' # xml variant
ND2_CHUNK_NAME_BinaryMetadata_v2                        = b'CustomData|BinaryMetadata_v2!'    # zipped json

ND2_CHUNK_FORMAT_ImageDataSeq_1p                        = b'ImageDataSeq|%u!' # |seq_index!
ND2_CHUNK_RE_ImageDataSeq_1p                            = re.compile(b'^ImageDataSeq\\|(\\d+)!$')

ND2_CHUNK_FORMAT_DownsampledColorData_2p                = b'CustomDataSeq|DownsampledColorData_%u|%u!' # _size|seq_index!
ND2_CHUNK_RE_DownsampledColorData_2p                    = re.compile(b'^CustomDataSeq\\|DownsampledColorData_(\\d+)\\|(\\d+)!$')

ND2_CHUNK_FORMAT_TiledRasterBinaryData_4p               = b'CustomDataSeq|TiledRasterBinaryData_%u_%u_%u|%u!' # _binid_ytile_xtile|seq_index!
ND2_CHUNK_FORMAT_TiledRasterBinaryData_2p               = b'CustomDataSeq|TiledRasterBinaryData_%u|%u!' # _binid|seq_index! (only on S3 ZARR)
ND2_CHUNK_RE_TiledRasterBinaryData_4p                   = re.compile(b'^CustomDataSeq\\|TiledRasterBinaryData_(\\d+)_(\\d+)_(\\d+)\\|(\\d+)!$')
ND2_CHUNK_RE_TiledRasterBinaryData_2p                   = re.compile(b'^CustomDataSeq\\|TiledRasterBinaryData_(\\d+)\\|(\\d+)!$')

ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_5p    = b'CustomDataSeq|DownsampledTiledRasterBinaryData_%u_%u_%u_%u|%u!' # _binid_size_ytile_xtile|seq_index!
ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_3p    = b'CustomDataSeq|DownsampledTiledRasterBinaryData_%u_%u|%u!' # _binid_size|seq_index! (only on S3 ZARR)
ND2_CHUNK_RE_DownsampledTiledRasterBinaryData_5p        = re.compile(b'^CustomDataSeq\\|DownsampledTiledRasterBinaryData_(\\d+)_(\\d+)_(\\d+)_(\\d+)\\|(\\d+)!$')
ND2_CHUNK_RE_DownsampledTiledRasterBinaryData_3p        = re.compile(b'^CustomDataSeq\\|DownsampledTiledRasterBinaryData_(\\d+)_(\\d+)\\|(\\d+)!$')

ND2_CHUNK_FORMAT_DeepSIMRawChannel                      = b'CustomDataSeq|AUXIMAGE_21_%u|%u!'       # component index | sequence index
ND2_CHUNK_RE_DeepSIMRawChannel                          = re.compile(b'^CustomDataSeq\\|AUXIMAGE_21_(\\d+)\\|(\\d+)!$')

class NotNd2Format(Exception):
    def __init__(self):
        self.message = "Not an ND2 file format: header does not match"
        super().__init__(self.message)

class NameNotInChunkmapError(Exception):
    def __init__(self, name: bytes|str):
        self.chunk_name = name if type(name) == str else name.decode('ascii')
        self.message = f"Name not in Chunk Map: {self.chunk_name}"
        super().__init__(self.message)

class UnsupportedChunkmapError(Exception):
    def __init__(self, version : tuple, name: bytes|str):
        self.file_version = version
        self.chunk_name = name if type(name) == str else name.decode('ascii')
        self.message = f"File version {self.file_version} with unsupported Chunk Map signature: {self.chunk_name}"
        super().__init__(self.message)

class BinaryIdNotFountError(Exception):
    def __init__(self, id: int):
        self.binid = id
        self.message = f"Binary Id not found: {self.binid}"
        super().__init__(self.message)

class UnexpectedCallError(Exception):
    def __init__(self, function_name: str, name: bytes|str):
        self.function_name = function_name
        self.chunk_name = name if type(name) == str else name.decode('ascii')
        self.message = f"Unexpected call ({self.function_name}): {self.chunk_name}"
        super().__init__(self.message)

class BaseChunker(abc.ABC):
    def __init__(self,
                 *,
                 with_image_attributes: ImageAttributes|None = None,
                 with_experiment: ExperimentLevel|None = None,
                 with_picture_metadata: PictureMetadata|None = None,
                 with_binary_rle_metadata: BinaryRleMetadata|None = None,
                 with_binary_raster_metadata: BinaryRasterMetadata|None = None,
                 with_image_text_info: ImageTextInfo|None = None) -> None:
        super().__init__()
        self._image_attributes: ImageAttributes|None = with_image_attributes
        self._experiment: ExperimentLevel|None = with_experiment
        self._picture_metadata: PictureMetadata|None = with_picture_metadata
        self._binary_rle_metadata: BinaryRleMetadata|None = with_binary_rle_metadata
        self._binary_tiled_raster_metadata: BinaryRasterMetadata|None = with_binary_raster_metadata
        self._image_text_info: ImageTextInfo|None = with_image_text_info
        self._acq_times: np.ndarray|None = None
        self._acq_times2: np.ndarray|None = None
        self._acq_frames: np.ndarray|None = None
        self._comp_range: np.ndarray|None = None

    @property
    @abc.abstractmethod
    def filename(self) -> str|None:
        pass

    @property
    @abc.abstractmethod
    def size_on_disk(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def last_modified(self) -> datetime.datetime:
        pass

    @property
    @abc.abstractmethod
    def format_version(self) -> tuple[int, int]:
        pass

    @property
    @abc.abstractmethod
    def chunker_name(self) -> str:
        return ""

    @property
    @abc.abstractmethod
    def is_readonly(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def chunk_names(self) -> list[bytes]:
        pass

    @abc.abstractmethod
    def chunk(self, name: bytes|str, asbytes: bool|None = None) -> bytes|memoryview|None:
        pass

    @abc.abstractmethod
    def setChunk(self, name: bytes|str, data: bytes|memoryview) -> None:
        pass

    @abc.abstractmethod
    def image(self, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        pass

    @abc.abstractmethod
    def setImage(self, seqindex: int, image: NumpyArrayLike, acqtime: float = -1.0) -> None:
        pass

    @abc.abstractmethod
    def readDownsampledImage(self, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        pass

    @abc.abstractmethod
    def setDownsampledImage(self, seqindex: int, downsize: int, image: NumpyArrayLike) -> None:
        pass

    @abc.abstractmethod
    def binaryRasterData(self, binid: int, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        pass

    @abc.abstractmethod
    def setBinaryRasterData(self, binid: int, seqindex: int, binimage: NumpyArrayLike) -> None:
        pass

    @abc.abstractmethod
    def readDownsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        pass

    @abc.abstractmethod
    def setDownsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int, binimage: NumpyArrayLike) -> None:
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        pass

    @abc.abstractmethod
    def rollback(self) -> None:
        pass


    @property
    def imageAttributes(self) -> ImageAttributes:
        if self._image_attributes is None:
            if (data := self.chunk(ND2_CHUNK_NAME_ImageAttributesLV)) is not None:
                self._image_attributes = ImageAttributes.from_lv(data)
            elif (data := self.chunk(ND2_CHUNK_NAME_ImageAttributes)) is not None:
                self._image_attributes = ImageAttributes.from_var(data)
            else:
                raise RuntimeError("Missing ImageAttributes")
        return self._image_attributes

    @imageAttributes.setter
    def imageAttributes(self, val: ImageAttributes) -> None:
        if self.is_readonly:
            raise PermissionError("Cannot set ImageAttributes to readonly chunker")
        data = val.to_lv()
        self.setChunk(ND2_CHUNK_NAME_ImageAttributesLV, data)
        self._image_attributes = val


    @property
    def pictureMetadata(self) -> PictureMetadata:
        if self._picture_metadata is None:
            if (data := self.chunk(ND2_CHUNK_FORMAT_ImageMetadataLV_1p % (0))) is not None:
                self._picture_metadata = PictureMetadata.from_lv(data)
            elif (data := self.chunk(ND2_CHUNK_FORMAT_ImageMetadata_1p % (0))) is not None:
                self._picture_metadata = PictureMetadata.from_var(data)
            else:
                raise RuntimeError("Missing PictureMetadata")
            if not self._picture_metadata.valid:
                self._picture_metadata.makeValid(self.imageAttributes.componentCount)
        return self._picture_metadata

    @pictureMetadata.setter
    def pictureMetadata(self, val: PictureMetadata) -> None:
        if self.is_readonly:
            raise PermissionError("Cannot set PictureMetadata to readonly chunker")
        data = val.to_lv()
        self.setChunk(ND2_CHUNK_FORMAT_ImageMetadataLV_1p % (0), data)
        self._picture_metadata = val

    @property
    def experiment(self) -> ExperimentLevel|None:
        if self._experiment is None:
            if (data := self.chunk(ND2_CHUNK_NAME_ImageMetadataLV)) is not None:
                self._experiment = ExperimentLevel.from_lv(data)
            elif (data := self.chunk(ND2_CHUNK_NAME_ImageMetadata)) is not None:
                self._experiment = ExperimentLevel.from_var(data)
            else:
                return None
            spectralLoop = self._experiment.findLevel(ExperimentLoopType.eEtSpectLoop)
            if spectralLoop is not None and self.pictureMetadata is not None:
                spectralLoop.uLoopPars.replacePlanes(self.pictureMetadata.sPicturePlanes)
        return self._experiment

    @experiment.setter
    def experiment(self, val: ExperimentLevel) -> None:
        if self.is_readonly:
            raise PermissionError("Cannot set ExperimentLevel to readonly chunker")
        data = val.to_lv()
        self.setChunk(ND2_CHUNK_NAME_ImageMetadataLV, data)
        self._experiment = val


    @property
    def imageTextInfo(self) -> ImageTextInfo:
        if self._image_text_info is None:
            if (data := self.chunk(ND2_CHUNK_NAME_ImageTextInfoLV)) is not None:
                self._image_text_info = ImageTextInfo.from_lv(data)
            elif (data := self.chunk(ND2_CHUNK_NAME_ImageTextInfo)) is not None:
                self._image_text_info = ImageTextInfo.from_var(data)
            else:
                return None
        return self._image_text_info

    @property
    def binaryRleMetadata(self) -> BinaryRleMetadata:
        if self._binary_rle_metadata is None:
            if (data := self.chunk(ND2_CHUNK_NAME_BinaryMetadata_v1)) is not None:
                self._binary_rle_metadata = BinaryRleMetadata.from_var(data)
            else:
                self._binary_rle_metadata = BinaryRleMetadata([])
        return self._binary_rle_metadata

    @property
    def binaryRasterMetadata(self) -> BinaryRasterMetadata:
        if self._binary_tiled_raster_metadata is None:
            if (data := self.chunk(ND2_CHUNK_NAME_BinaryMetadata_v2)) is not None:
                self._binary_tiled_raster_metadata = BinaryRasterMetadata.from_json(data)
            else:
                self._binary_tiled_raster_metadata = BinaryRasterMetadata([])
        return self._binary_tiled_raster_metadata

    @property
    def hasDownsampledImages(self) -> bool:
        attrs = self.imageAttributes
        chnames = self.chunk_names
        downsizes = attrs.lowerPowSizeList
        if len(downsizes) == 0:
            return False
        for seqindex in range(attrs.frameCount):
            for downsize in downsizes:
                chname = ND2_CHUNK_FORMAT_DownsampledColorData_2p % (downsize, seqindex)
                if chname not in chnames:
                    return False
        return True

    @property
    def hasDownsampledBinaryRasterData(self) -> bool:
        attrs, binmeta = self.imageAttributes, self.binaryRasterMetadata
        chnames = self.chunk_names
        if not binmeta:
            return False
        for binmetaitem in binmeta:
            h, w = binmetaitem.shape
            th, tw = binmetaitem.tileShape
            alltiles = itertools.product(list(range(0, h//th, 1)), list(range(0, w//tw, 1)))
            for seqindex in range(attrs.frameCount):
                for downsize in attrs.lowerPowSizeList:
                    chname = ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_3p % (binmetaitem.id, downsize, seqindex)
                    if chname not in chnames:
                        for tile in alltiles:
                            chname = ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_5p % (binmetaitem.id, downsize, tile[0], tile[1], seqindex)
                            if chname not in chnames:
                                return False
        return True

    def downsampledImage(self, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        img = None
        denomPowBase = 0
        while downsize < self.imageAttributes.powSize:
            try:
                img = self.readDownsampledImage(seqindex, downsize, rect)
                break
            except NameNotInChunkmapError:
                if Nd2LoggerEnabled:
                    logger.debug(f"Downsampled (downsize={downsize}) image (index={seqindex}) not found!")
                if rect is not None:
                    rect = (2*rect[0], 2*rect[1], 2*rect[2], 2*rect[3])
                denomPowBase += 1
                downsize *= 2
        img = self.image(seqindex, rect) if img is None else img
        return self.scale_2xN_down_linear(img, denomPowBase) if img is not None else img

    def downsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        img = None
        denomPowBase = 0
        while downsize < self.imageAttributes.powSize:
            try:
                img = self.readDownsampledBinaryRasterData(binid, seqindex, downsize, rect)
                break
            except BinaryIdNotFountError or NameNotInChunkmapError:
                if Nd2LoggerEnabled:
                    logger.debug(f"Downsampled (downsize={downsize}) binary (binid={binid}) at (index={seqindex}) not found!")
                if rect is not None:
                    rect = (2*rect[0], 2*rect[1], 2*rect[2], 2*rect[3])
                denomPowBase += 1
                downsize *= 2
        img = self.binaryRasterData(binid, seqindex, rect) if img is None else img
        return self.scale_2xN_down_00(img, denomPowBase) if img is not None else img

    def generateAndSetDownsampledImages(self, seqindex: int, image: NumpyArrayLike) -> None:
        src_attrs, src_image = self.imageAttributes, image
        while ImageAttributes.MinDownsampledSize < src_attrs.powSize:
            downsampled_attrs = src_attrs.makeDownsampled()
            downsampled_image = np.zeros(shape=downsampled_attrs.shape, dtype=downsampled_attrs.safe_dtype)
            _downsample_2x_linear(downsampled_image, src_image)
            self.setDownsampledImage(seqindex=seqindex, downsize=downsampled_attrs.powSize, image=downsampled_image.astype(dtype=downsampled_attrs.dtype))
            src_attrs, src_image = downsampled_attrs, downsampled_image

    def generateAndSetDownsampledBinaryRasterData(self, binid: int, seqindex: int, binimage: NumpyArrayLike) -> None:
        src_attrs, src_binimage = self.imageAttributes, binimage
        while ImageAttributes.MinDownsampledSize < src_attrs.powSize:
            downsampled_attrs = src_attrs.makeDownsampled()
            downsampled_image = np.zeros(shape=downsampled_attrs.shape[:2], dtype=np.uint32)
            _downsample_2x_00(downsampled_image, src_binimage)
            self.setDownsampledBinaryRasterData(binid=binid, seqindex=seqindex, downsize=downsampled_attrs.powSize, binimage=downsampled_image)
            src_attrs, src_binimage = downsampled_attrs, downsampled_image

    def scale_2xN_down_linear(self, img : NumpyArrayLike, n : int) -> NumpyArrayLike:
        src_safe_dtype = self.imageAttributes.safe_dtype
        src_shape = img.shape
        while n:
            downsampled_image = np.zeros(shape=(src_shape[0]//2, src_shape[1]//2, src_shape[2]), dtype=src_safe_dtype)
            _downsample_2x_linear(downsampled_image, img)
            src_shape, img = downsampled_image.shape, downsampled_image
            n -= 1
        return img.astype(self.imageAttributes.dtype)

    def scale_2xN_down_00(self, img : NumpyArrayLike, n : int) -> NumpyArrayLike:
        src_shape = img.shape
        while n:
            downsampled_image = np.zeros(shape=(src_shape[0]//2, src_shape[1]//2), dtype=np.uint32)
            _downsample_2x_00(downsampled_image, img)
            src_shape, img = downsampled_image.shape, downsampled_image
            n -= 1
        return img


    def _set_metadata(self, name: bytes, data: bytes) -> None:
        if ND2_CHUNK_NAME_ImageAttributesLV == name:
            self._image_attributes = ImageAttributes.from_lv(data)
        elif ND2_CHUNK_NAME_BinaryMetadata_v1 == name:
            self._binary_rle_metadata = BinaryRleMetadata.from_var(data)
        elif ND2_CHUNK_NAME_BinaryMetadata_v2 == name:
            self._binary_tiled_raster_metadata = BinaryRasterMetadata.from_json(data)

    @property
    def acqTimes(self) -> np.ndarray:
        if self._acq_times is None:
            if (data := self.chunk(ND2_CHUNK_NAME_AcqTimesCache)) is not None:
                acq_times = np.ndarray(
                    buffer=data, dtype=np.float64,
                    shape=(self.imageAttributes.uiSequenceCount, ),
                    strides=(8, ))
                if np.all(np.diff(acq_times) > 0):
                    self._acq_times = acq_times
            if self._acq_times is None:
                self._acq_times = np.array([i*10.0 for i in range(self.imageAttributes.uiSequenceCount) ])
        return self._acq_times

    @property
    def acqTimes2(self) -> np.ndarray:
        if self._acq_times is None:
            if (data := self.chunk(ND2_CHUNK_NAME_AcqTimes2Cache)) is not None:
                self._acq_times = np.ndarray(
                    buffer=data, dtype=np.float64,
                    shape=(self.imageAttributes.uiSequenceCount, ),
                    strides=(8, ))
        return self._acq_times2



    @property
    def acqFrames(self) -> np.ndarray:
        if self._acq_times is None:
            if (data := self.chunk(ND2_CHUNK_NAME_AcqFramesCache)) is not None:
                self._acq_times = np.ndarray(
                    buffer=data, dtype=np.uint32,
                    shape=(self.imageAttributes.uiSequenceCount, ),
                    strides=(4, ))
        return self._acq_frames

    @property
    def compFrameRange(self) -> np.ndarray:
        if self._comp_range is None:
            self._comp_range = np.zeros((self.imageAttributes.uiComp, 2, self.imageAttributes.uiSequenceCount))
            for comp in range(self.imageAttributes.uiComp):
                if (data := self.chunk(ND2_CHUNK_FORMAT_FloatCompRangeCache_1p % (comp))) is not None:
                    pairs = np.ndarray(
                        buffer=data, dtype=np.float32,
                        shape=(self.imageAttributes.uiSequenceCount, 2),
                        strides=(2*4, 4))
                    self._comp_range[comp, :, :] = np.moveaxis(pairs, 1, 0)
                else:
                    self._comp_range[comp, 0, :] = 0
                    self._comp_range[comp, 1, :] = (1 << self.imageAttributes.uiBpcInMemory) - 1
        return self._comp_range

    @property
    def compRange(self) -> np.ndarray:
        ret = np.zeros((self.imageAttributes.uiComp, 2))
        for comp in range(self.imageAttributes.uiComp):
            ret[comp, 0] = np.min(self.compFrameRange[comp, 0, :])
            ret[comp, 1] = np.max(self.compFrameRange[comp, 1, :])
        return ret



    @staticmethod
    def _is_chunk_data(chunkname: bytes) -> bool:
        return not (
            ND2_CHUNK_RE_ImageDataSeq_1p.fullmatch(chunkname)
            or ND2_CHUNK_RE_DownsampledColorData_2p.fullmatch(chunkname)
            or ND2_CHUNK_RE_TiledRasterBinaryData_2p.fullmatch(chunkname)
            or ND2_CHUNK_RE_TiledRasterBinaryData_4p.fullmatch(chunkname)
            or ND2_CHUNK_RE_DownsampledTiledRasterBinaryData_3p.fullmatch(chunkname)
            or ND2_CHUNK_RE_DownsampledTiledRasterBinaryData_5p.fullmatch(chunkname))

    @staticmethod
    def isSkipChunk(chunkname: bytes) -> bool:
        return chunkname in (ND2_FILE_SIGNATURE, ND2_CHUNKMAP_SIGNATURE, ND2_FILEMAP_SIGNATURE)

    def isBinaryRleMetadata(chunkname: bytes) -> bool:
        return ND2_CHUNK_NAME_BinaryMetadata_v1 == chunkname

    @staticmethod
    def isImageChunk(chunkname: bytes) -> int|None:
        if match := ND2_CHUNK_RE_ImageDataSeq_1p.fullmatch(chunkname):
            return int(match.group(1))
        return None

    @staticmethod
    def isDownsampledImageChunk(chunkname: bytes) -> tuple[int, int]|None:
        if match := ND2_CHUNK_RE_DownsampledColorData_2p.fullmatch(chunkname):
            return (int(match.group(2)), int(match.group(1)))
        return None

    @staticmethod
    def isBinaryRleDataChunk(regex_dict: dict[int, re.Pattern], chunkname: bytes) -> tuple[int, int]|None:
        for binid, regex in regex_dict.items():
            if match := regex.fullmatch(chunkname):
                return (int(binid), int(match.group(1)))
        return None

    @staticmethod
    def isBinaryRasterData(chunkname: bytes) -> tuple[int, int, int, int]|None:
        if match := ND2_CHUNK_RE_TiledRasterBinaryData_2p.fullmatch(chunkname):
            return (int(match.group(1)), int(match.group(2)), 0, 0)
        elif match := ND2_CHUNK_RE_TiledRasterBinaryData_4p.fullmatch(chunkname):
            return (int(match.group(1)), int(match.group(4)), int(match.group(2)), int(match.group(3)))
        return None

    @staticmethod
    def isDownsampledBinaryRasterData(chunkname: bytes) -> tuple[int, int, int, int, int]|None:
        if match := ND2_CHUNK_RE_DownsampledTiledRasterBinaryData_3p.fullmatch(chunkname):
            return (int(match.group(1)), int(match.group(3)), int(match.group(2)), 0, 0)
        elif match := ND2_CHUNK_RE_DownsampledTiledRasterBinaryData_5p.fullmatch(chunkname):
            return (int(match.group(1)), int(match.group(5)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
        return None

    def rleBinaryVersion(self) -> int:
        attrs = self.imageAttributes
        for meta in self.binaryRleMetadata:
            data = self.chunk(meta.dataChunkName(0))
            if 4 <= len(data):
                version_struct = struct.Struct("<I")
                version_struct_size = version_struct.size
                decompress = zlib.decompressobj()
                decompressed_header = decompress.decompress(data[4:], version_struct_size)
                (version, ) =  version_struct.unpack(decompressed_header)
                return version
        return 0

    def rleChunkToArray(self, data: bytes|memoryview, rect : tuple[int, int, int, int]|None = None, *, no_obj_info: bool = False) -> tuple[NumpyArrayLike, dict[int, dict|None]]:
        if len(data) < 4:
            height, width = self.imageAttributes.shape[0:2]
            y0, y1 = (rect[1], min(rect[1] + rect[3], height)) if rect is not None else (0, height)
            x0, x1 = (rect[0], min(rect[0] + rect[2], width)) if rect is not None else (0, width)
            return np.zeros(shape=(y1-y0, x1-x0), dtype=np.uint32), {}
        (_uncompressed_size, ) = struct.unpack_from("<I", data)
        stream = io.BytesIO(zlib.decompress(data[4:]))

        def _unpack(stream: io.BufferedIOBase, strct: struct.Struct) -> tuple:
            return strct.unpack(stream.read(strct.size))

        rle_header = struct.Struct("<IIIIIII")
        (version, width, height, obj_count, _nbytes, _last_object_offset, _custom_data_size) = _unpack(stream, rle_header)

        ver_rle_obj_struct: dict[int, dict[str, any]] = { 2: struct.Struct("<IIIIIIIIIII"), 3: struct.Struct("<IIIIIIIII") }

        if version == 1:
            raise NotImplementedError()

        elif version in (2, 3):
            rle_object = ver_rle_obj_struct[version]
            rle_seg = rle_row = struct.Struct("<II")

            ret_obj_info_dict = {}
            y0, y1 = (rect[1], min(rect[1] + rect[3], height)) if rect is not None else (0, height)
            x0, x1 = (rect[0], min(rect[0] + rect[2], width)) if rect is not None else (0, width)
            ret_binimage = np.zeros(shape=(y1-y0, x1-x0), dtype=np.uint32)
            for _i in range(obj_count):
                obj_id, left, top, right, bottom, nbytes, nrows, _last_row_offset, obj_status, *_ = _unpack(stream, rle_object)
                if rect is not None:
                    if x1 <= left or y1 <= top or right <= x0 or bottom <= y0:
                        stream.seek(nbytes-rle_object.size, 1)
                        continue
                if not no_obj_info:
                    obj_info = dict(bb=(left, top, right, bottom), status=obj_status)
                    ret_obj_info_dict[obj_id] = obj_info
                    for j in range(nrows):
                        (y, nsegments) = _unpack(stream, rle_row)
                        for k in range(nsegments):
                            (x, n) = _unpack(stream, rle_seg)
                            if 0 == j and 0 == k:
                                obj_info["seed"] = (x, y)
                            x_slice = slice(max(x, x0) - x0, min(x + n, x1) - x0)
                            if y0 <= y and y < y1:
                                ret_binimage[y-y0, x_slice] = obj_id
                            pxls += n
                            xx += sum(range(x, x+n))
                            yy += n*y
                    obj_info["pixels"] = pxls
                    obj_info["center"] = (xx // pxls, yy // pxls)
                else:
                    ret_obj_info_dict[obj_id] = None
                    for _j in range(nrows):
                        (y, nsegments) = _unpack(stream, rle_row)
                        if y0 <= y and y < y1:
                            for _k in range(nsegments):
                                (x, n) = _unpack(stream, rle_seg)
                                if x + n <= x0 or x1 <= x:
                                    continue
                                x_slice = slice(max(x-x0, 0), min(x + n, x1) - x0)
                                ret_binimage[y-y0, x_slice] = obj_id
                        else:
                            stream.seek(nsegments*rle_seg.size, 1)
            return (ret_binimage, ret_obj_info_dict)
        else:
            raise NotImplementedError()

    def crestDeepSimRawData(self, seqindex: int, componentindex: int) -> \
        tuple[NumpyArrayLike, str, str, tuple[float, float], tuple[int, int], tuple[int, int]]:

        data = self.chunk(ND2_CHUNK_FORMAT_DeepSIMRawChannel % (componentindex, seqindex))
        if not data:
            raise NameNotInChunkmapError(f"DeepSIM chunk for sequence index {seqindex} and component index {componentindex} not found.")

        # deepSim chunk data structure

        # Size                                      size_t        Q
        # Width                                     int16         h
        # Height                                    int16         h
        # Raw channel count                         int16         h
        # Raw channel size                          size_t        Q
        # PSF correction                            double        d
        # PSF correction - default value            double        d
        # Number of iterations                      uint16        H
        # Number of iterations - default value      uint16        H
        # ROI offset x                              int16         h
        # ROI offset y                              int16         h

        # ADDED PADDING TO OFFSET 166

        # Calibration data size                     size_t        Q
        # Calibration key size                      uint32        I

        # OFFSET 178 - DATA_START_OFFSET

        # Calibration key (optional)
        # Calibration data as XML (optional)
        # Raw channel data                          starts at CHANNEL_DATA_START_OFFSET


        header_struct = struct.Struct('<QhhhQddHHhh')

        # Unpack the data using struct.unpack
        size, width, height, channel_count, channel_size, psf, psf_def, iter, iter_def, roi_offset_x, roi_offset_y = header_struct.unpack(data[:header_struct.size])

        DATA_START_OFFSET = 178

        calibration_sizes_struct = struct.Struct("<QI")
        calibration_data_size, calibration_key_size = calibration_sizes_struct.unpack(data[DATA_START_OFFSET - calibration_sizes_struct.size:DATA_START_OFFSET])

        CHANNEL_DATA_START_OFFSET = DATA_START_OFFSET + calibration_data_size + calibration_key_size

        calibration_key = data[DATA_START_OFFSET : DATA_START_OFFSET + calibration_key_size].decode('utf-16-le')
        calibration_data = data[DATA_START_OFFSET + calibration_key_size: DATA_START_OFFSET + calibration_key_size + calibration_data_size].decode('utf-8')

        channels = []
        for i in range(channel_count):
            subarray = data[CHANNEL_DATA_START_OFFSET + i * channel_size : CHANNEL_DATA_START_OFFSET + (i + 1) * channel_size]
            if not len(subarray):
                raise ValueError("Missing DeepSIM data to parse")       # should not happen
            np_array = np.frombuffer(subarray, dtype=np.uint16).reshape((height, width))
            channels.append(np_array)

        channels = np.stack(channels, axis=0)
        return channels, calibration_key, calibration_data, (psf, psf_def), (iter, iter_def), (roi_offset_x, roi_offset_y)

    def crestDeepSimRawDataIndices(self) -> list[tuple[int, int]]:
        result = []
        for chunk_name in self.chunk_names:
            match = ND2_CHUNK_RE_DeepSIMRawChannel.match(chunk_name)
            if match:
                result.append((int(match.group(2)), int(match.group(1))))
        return result



def _downsample_2x_linear(dst: NumpyArrayLike, src: NumpyArrayLike) -> None:
    s0, s1 = dst.shape[0:2]
    dst += src[0:2*s0:2, 0:2*s1:2]
    dst += src[1:2*s0:2, 0:2*s1:2]
    dst += src[0:2*s0:2, 1:2*s1:2]
    dst += src[1:2*s0:2, 1:2*s1:2]
    dst //= 4

def _downsample_2x_00(dst: NumpyArrayLike, src: NumpyArrayLike) -> None:
    s0, s1 = dst.shape[0:2]
    np.copyto(dst, src[0:2*s0:2, 0:2*s1:2])


