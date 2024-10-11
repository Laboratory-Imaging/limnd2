import datetime, functools, numpy as np, os

from .attributes import ImageAttributesPixelType
from .base import FileLikeObject, NumpyArrayLike, Nd2LoggerEnabled, BinaryRleMetadata, BinaryRasterMetadata, ImageAttributes, NumpyArrayLike
from .custom_data import CustomDescription, RecordedData, RecordedDataItem, RecordedDataType
from .experiment import ExperimentLevel, WellplateDesc, WellplateFrameInfo
from .file import LimBinaryIOChunker
from .metadata import PictureMetadata
from .textinfo import ImageTextInfo, AppInfo
from .variant import decode_var


if Nd2LoggerEnabled:
    import logging
    logger = logging.getLogger("limnd2")

class Nd2Reader:
    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        kwargs["readonly"] = True
        return _create_chunker(*args, **kwargs)

    def __init__(self, file : FileLikeObject, *, chunker_kwargs:dict = {}) -> None:
        self._chunker = self.create_chunker(file, chunker_kwargs=chunker_kwargs)

    @property
    def filename(self) -> str|None:
        return self._chunker.filename

    @property
    def version(self) -> tuple[int, int]:
        return self._chunker.fileVersion

    @functools.cached_property
    def is3d(self) -> bool:
        exp = self.experiment
        if exp is None:
            return False
        dims = exp.dimnames()
        return dims and 'z' in dims

    @functools.cached_property
    def is8bitRgb(self) -> bool:
        return 8 == self.imageAttributes.uiBpcSignificant and self.pictureMetadata.isRgb

    @property
    def imageAttributes(self) -> ImageAttributes:
        return self._chunker.imageAttributes

    @property
    def pictureMetadata(self) -> PictureMetadata:
        return self._chunker.pictureMetadata

    @property
    def experiment(self) -> ExperimentLevel|None:
        return self._chunker.experiment

    @property
    def imageTextInfo(self) -> ImageTextInfo:
        return self._chunker.imageTextInfo

    @functools.cached_property
    def wellplateDesc(self) -> WellplateDesc|None:
        data = self.chunk(b'CustomData|WellPlateDesc_0!')
        return WellplateDesc.from_lv(data) if data is not None else None

    @functools.cached_property
    def wellplateFrameInfo(self) -> WellplateFrameInfo|None:
        data = self.chunk(b'CustomData|WellPlateFrameInfoZJSON!')
        return WellplateFrameInfo.from_json(data) if data is not None else None

    @functools.cached_property
    def appInfo(self) -> AppInfo:
        data = self.chunk(b'CustomDataVar|AppInfo_V1_0!')
        return AppInfo.from_var(data)

    @property
    def software(self) -> str:
        return self.appInfo.software

    @property
    def acqFrames(self) -> NumpyArrayLike:
        return self._chunker.acqFrames

    @property
    def acqTimes(self) -> NumpyArrayLike|None:
        return self._chunker.acqTimes

    @property
    def acqTimes2(self) -> NumpyArrayLike|None:
        return self._chunker.acqTimes2

    @property
    def compFrameRange(self) -> NumpyArrayLike:
        return self._chunker.compFrameRange

    @property
    def compRange(self) -> NumpyArrayLike:
        return self._chunker.compRange

    @property
    def recordedData(self) -> RecordedData:
        recData = RecordedData()
        if self.acqTimes is not None:
            strings = []
            for ms in self.acqTimes:
                ss = int((ms/1_000)%60)
                mm = int((ms/(60_000))%60)
                hh = int(ms/(3_600_000))
                ms -= (3_600_000)*hh + (60_000)*mm + 60*ss
                strings.append("%d:%02d:%02d.%03d" % (hh, mm, ss, ms % 1000))
            recData.append(RecordedDataItem(ID='ACQTIME', Desc='Time', Unit='h:m:s.ms', Type=RecordedDataType.eString, Group=0, Size=len(strings), Data=np.array(strings)))
        data = self.chunk(b'CustomDataVar|CustomDataV2_0!')
        if data is not None:
            decoded = decode_var(data)
            desc = decoded.get('CustomTagDescription_v1.0', {})
            for i in range(len(desc)):
                itemDesc = desc.get(f"Tag{i}", None)
                if itemDesc is not None:
                    colData = self.chunk(b'CustomData|%s!' % (itemDesc.get('ID').encode('utf-8')))
                    recData.append(RecordedDataItem.from_desc_and_data(itemDesc, colData))
        if 0 < len(recData):
            recData.insert(0, RecordedDataItem(ID='INDEX', Desc='Index', Unit='', Type=RecordedDataType.eInt, Group=0, Size=recData.rowCount, Data=np.arange(1, recData.rowCount+1)))
            recData.sort()
        return recData

    @functools.cached_property
    def generalImageInfo(self) -> dict[str, any]:
        ia = self.imageAttributes
        stat = os.stat(self.filename)
        loops = ", ".join([ f"{exp_level.shortName}({exp_level.count})" for exp_level in self.experiment if 0 < exp_level.count ]) if self.experiment else ""
        filename = os.path.basename(self.filename)
        path = os.path.dirname(self.filename)
        bit_depth = f"{ia.uiBpcSignificant}bit {ImageAttributesPixelType.short_name(ia.ePixelType)}"
        frame_res = f"{ia.width} x {ia.height}"
        dimension = f"{frame_res} ({ia.componentCount} {"comps" if 1 < ia.componentCount else "comp"} {bit_depth})" + (f" x {ia.uiSequenceCount} frames" if 1 < ia.uiSequenceCount else "") +(f": {loops}" if loops else "")
        file_size = _size_fmt(stat.st_size)
        frame_size = _size_fmt(ia.height*ia.widthBytes)
        z_count = self.experiment.dims.get('z', 0) if self.experiment is not None else 0
        volume_size = _size_fmt(ia.height*ia.widthBytes*z_count)
        sizes = f"{_size_fmt(stat.st_size)} on disk, {frame_size} frame" + (f", {volume_size} volume" if z_count else "")
        calibration = f"{self.pictureMetadata.dCalibration:.3f} µm/px" if self.pictureMetadata.bCalibrated else "Uncalibrated"
        mtime = f"{datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%x %X')}"
        app_created = self.appInfo.software
        return dict(filename=filename, path=path, bit_depth=bit_depth, loops=loops, dimension=dimension, file_size=file_size, frame_res=frame_res, volume_size=volume_size, sizes=sizes, calibration=calibration, mtime=mtime, app_created=app_created)

    @functools.cached_property
    def customDescription(self) -> CustomDescription|None:
        data = self.chunk(b'CustomData|CustomDescriptionV1_0!')
        if data is None:
            return None
        return CustomDescription.from_lv(data)


    def generateLoopIndexes(self, named: bool = False) -> list:
        exp = self.experiment
        if exp is None:
            return []
        wp_desc = self.wellplateDesc
        wp_frameinfo = self.wellplateFrameInfo
        names, shape = self.experiment.dimnames(skipSpectralLoop=True), self.experiment.shape(skipSpectralLoop=True)
        if isinstance(wp_desc, WellplateDesc) and isinstance(wp_frameinfo, WellplateFrameInfo) and 'm' in names and len(wp_frameinfo):
            ret = []
            i = names.index('m')
            names = ('w', ) + names
            mp_size, wp_size = shape[i], wp_frameinfo.nwells
            true_mp_size = mp_size // wp_size
            for idexes in self.experiment.generateLoopIndexes(named=False):
                lst = list(idexes)
                windex, lst[i] = lst[i] // true_mp_size, lst[i] % true_mp_size
                lst = [windex] + lst
                ret.append(dict(zip(names, lst)) if named else lst)
            return ret

        else:
            return self.experiment.generateLoopIndexes(named=named)

    @property
    def binaryRleMetadata(self) -> BinaryRleMetadata:
        return self._chunker.binaryRleMetadata

    @property
    def binaryRasterMetadata(self) -> BinaryRasterMetadata:
        if 0 == len(self._chunker.binaryRasterMetadata) and 0 < len(self._chunker.binaryRleMetadata):
            return self._chunker.binaryRleMetadata.makeRasterMetadata(self.imageAttributes.width, self.imageAttributes.height)
        else:
            return self._chunker.binaryRasterMetadata

    @property
    def chunker(self):
        return self._chunker

    def chunk(self, name : bytes|str, asbytes : bool|None = None) -> bytes|memoryview|None:
        return self._chunker.chunk(name)

    def image(self, seqindex: int) -> NumpyArrayLike:
        return self._chunker.image(seqindex)

    def downsampledImage(self, seqindex: int, downsize: int) -> NumpyArrayLike:
        return self._chunker.downsampledImage(seqindex, downsize)

    def binaryRasterData(self, binid: int, seqindex: int, xtile:int|None = None, ytile:int|None = None) -> NumpyArrayLike:
        return self._chunker.binaryRasterData(binid, seqindex)

    def downsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int, xtile:int|None = None, ytile:int|None = None) -> NumpyArrayLike:
        return self._chunker.downsampledBinaryRasterData(binid, seqindex, downsize)

    def finalize(self) -> None:
        return self._chunker.finalize()

    def load_chunks(self) -> dict[str, bytes]:
        data = {}
        for chunk_name in self.chunker.chunk_names:
            pos = self.chunker._chunk_pos(chunk_name)
            data[chunk_name] = self.chunker._read_chunk(pos)
        return data

    def write(self, filename: str) -> None:
        data = self.load_chunks()
        for x, y in data.items():
            print(x, ":", y[:20])


        for chunk_name, (offset, size) in self.chunker._chunkmap.items():
            print(chunk_name, offset)

class Nd2Writer:
    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        kwargs["readonly"] = False
        return _create_chunker(*args, **kwargs)

    def __init__(self, file : FileLikeObject, *, append : bool|None = None, chunker_kwargs:dict = {}) -> None:
        self._chunker = self.create_chunker(file, append=append, chunker_kwargs=chunker_kwargs)

    @property
    def version(self) -> tuple[int, int]:
        return self._chunker.fileVersion

    @property
    def imageAttributes(self) -> ImageAttributes:
        return self._chunker.imageAttributes

    @imageAttributes.setter
    def imageAttributes(self, val: ImageAttributes) -> None:
        self._chunker.imageAttributes = val

    @property
    def chunker(self):
        return self._chunker

    def setChunk(self, name : bytes|str, data : bytes|memoryview) -> None:
        return self._chunker.setChunk(name, data)

    def finalize(self) -> None:
        return self._chunker.finalize()

    def rollback(self) -> None:
        return self._chunker.rollback()

def _create_chunker(file : FileLikeObject, *, readonly: bool = True, append: bool|None = None, chunker_kwargs: dict = {}):
    import os
    if type(file) == str:
        if readonly:
            mode = "rb"
        else:
            if append is None:
                append = os.path.isfile(file)
            mode = "rb+" if append else "wb"

        fh = open(file, mode)
        chunker_kwargs.update(dict(filename=file))
        return LimBinaryIOChunker(fh, **chunker_kwargs)

    elif (hasattr(file, "read") or hasattr(file, "write")) and hasattr(file, "seek") and hasattr(file, "tell") and hasattr(file, "mode"):
        if readonly and "rb" != file.mode:
            raise ValueError("File handle passed to LimNd2Reader must have \"rb\" mode")
        elif not readonly and file.mode not in ("rb+", "wb"):
            raise ValueError("File handle passed to LimNd2Wrtier must have \"rb+\" or \"wb\" mode")
        return LimBinaryIOChunker(file, **chunker_kwargs)

def _size_fmt(size):
    kB = 1024
    MB = kB*1024
    GB = MB*1024
    TB = GB*1024
    if TB <= size:
        return f"{size/TB:.0f}TB"
    if GB <= size:
        return f"{size/GB:.0f}GB"
    if MB <= size:
        return f"{size/MB:.0f}MB"
    if kB <= size:
        return f"{size/kB:.0f}kB"
    return f"{size} B"
