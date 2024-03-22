from .base import FileLikeObject, NumpyArrayLike, Nd2LoggerEnabld, BinaryRleMetadata, BinaryRasterMetadata, ImageAttributes, NumpyArrayLike
from .experiment import ExperimentLevel, WellplateDesc, WellplateFrameInfo
from .file import LimBinaryIOChunker
from .metadata import PictureMetadata

if Nd2LoggerEnabld:
    import logging
    logger = logging.getLogger("limnd2")

class Nd2Reader:    
    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        kwargs["readonly"] = True
        return _create_chunker(*args, **kwargs)

    def __init__(self, file : FileLikeObject, *, chunker_kwargs:dict = {}) -> None:
        self._chunker = self.create_chunker(file, chunker_kwargs=chunker_kwargs)

    @property
    def version(self) -> tuple[int, int]:
        return self._chunker.fileVersion

    @property
    def is3d(self) -> bool:
        exp = self.experiment
        if exp is None:
            return False
        dims = exp.dimnames()
        return dims and 'z' in dims

    @property
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
    def wellplateDesc(self) -> WellplateDesc|None:
        data = self.chunk(b'CustomData|WellPlateDesc_0!')
        return WellplateDesc.from_lv(data) if data is not None else None
    
    @property
    def wellplateFrameInfo(self) -> WellplateFrameInfo|None:
        data = self.chunk(b'CustomData|WellPlateFrameInfoZJSON!')
        return WellplateFrameInfo.from_json(data) if data is not None else None
    
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
    
def _create_chunker(file : FileLikeObject, *, readonly: bool = True, append: bool|None = None, chunker_kwargs:dict = {}):
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

