import datetime, functools, numpy as np, os
import warnings
import limnd2
from pathlib import Path

from .attributes import ImageAttributesPixelType
from .base import FileLikeObject, Nd2LoggerEnabled, BinaryRleMetadata, BinaryRasterMetadata, ImageAttributes, NumpyArrayLike
from .custom_data import CustomDescription, CustomDescriptionItemType, RecordedData, RecordedDataItem, RecordedDataType
from .experiment import ExperimentLevel, ExperimentLoopType, WellplateDesc, WellplateFrameInfo
from .file import LimBinaryIOChunker
from .metadata import PictureMetadata
from .results import create_table_data_from_h5, read_results_from_h5, TableData, ResultItem, ResultPane
from .textinfo import ImageTextInfo, AppInfo
from .variant import decode_var


if Nd2LoggerEnabled:
    import logging
    logger = logging.getLogger("limnd2")

class StorageInfo:
    def __init__(self, filename: str | None, url: str | None, size_on_disk: int, last_modified: datetime.datetime):
        self._filename = filename
        self._url = url
        self._size_on_disk = size_on_disk
        self._last_modified = last_modified

    @property
    def filename(self) -> str | None:
        return self._filename

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def size_on_disk(self) -> int:
        return self._size_on_disk

    @property
    def last_modified(self) -> datetime.datetime:
        return self._last_modified

class Nd2Reader():
    """
    Specific implementation of `Nd2ReaderProtocol` specific to `.nd2` files, implementing additional methods.

    See [`Nd2ReaderProtocol`](protocols.md#limnd2.protocols.Nd2ReaderProtocol) for more information.
    """

    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        kwargs["readonly"] = True
        return _create_chunker(*args, **kwargs)

    @property
    def chunker(self):
        return self._chunker

    def __init__(self, file : FileLikeObject, *, chunker_kwargs: dict = {}) -> None:
        """
        Parameters
        -----------
        file : str | Path | int | typing.BinaryIO
            Filename of the ND2 file.
        chunker_kwargs
            Additional parameters for chunker.
        """
        super().__init__()
        self._chunker = self.create_chunker(file, chunker_kwargs=chunker_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    def finalize(self) -> None:
        return self._chunker.finalize()

    # Static methods

    @staticmethod
    def file_size_on_disk(filename: str|Path|None) -> int:
        if filename is None:
            raise ValueError()

        if isinstance(filename, str):
            filename = Path(filename)
        size = filename.stat().st_size
        filename = filename.with_suffix('.h5')
        try:
            size += filename.stat().st_size
        except:
            pass

        return size

    @staticmethod
    def file_last_modified(filename: str|Path|None) -> datetime.datetime:
        if filename is None:
            raise ValueError()

        if isinstance(filename, str):
            filename = Path(filename)
        mtime = filename.stat().st_mtime

        filename = filename.with_suffix('.h5')
        try:
            h5_mtime = filename.stat().st_mtime
            if mtime < h5_mtime:
                mtime = h5_mtime
        except:
            pass

        return datetime.datetime.fromtimestamp(mtime)

    # DEPRECATED properties and methods, will be removed in future versions

    @property
    def filename(self) -> str|None:
        warnings.warn(
            "Nd2Reader.filename is deprecated; use Nd2Reader.storage_info.filename instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.chunker.filename

    @property
    def url(self) -> str|None:
        warnings.warn(
            "Nd2Reader.url is deprecated; use Nd2Reader.storage_info.url instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return Path(self.chunker.filename).absolute().as_uri()

    @property
    def size_on_disk(self) -> int:
        warnings.warn(
            "Nd2Reader.size_on_disk is deprecated; use Nd2Reader.storage_info.size_on_disk instead.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            return Nd2Reader.file_size_on_disk(self.filename)
        except ValueError or FileNotFoundError or PermissionError:
            return self.chunker.size_on_disk

    @property
    def last_modified(self) -> datetime.datetime:
        warnings.warn(
            "Nd2Reader.last_modified is deprecated; use Nd2Reader.storage_info.last_modified instead.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            return Nd2Reader.file_last_modified(self.filename)
        except ValueError or FileNotFoundError or PermissionError:
            return self.chunker.last_modified


    def series_export(
        self,
        folder: str | Path | None = None,
        prefix: str | None = None,
        dimension_order: list[str] | None = None,
        bits: int | None = None,
        *,
        progress_to_json: bool = False
    ) -> None:
        warnings.warn(
            "Nd2Reader.series_export is deprecated and will be removed in future versions; use limnd2.series_export() function instead.",
            DeprecationWarning,
        )
        limnd2.series_export(self, folder=folder, prefix=prefix, dimension_order=dimension_order, bits=bits, progress_to_json=progress_to_json)

    def frame_export(
        self,
        frame_index: int = 0,
        output_path: str | Path | None = None,
        target_bit_depth: int | None = None,
        *,
        progress_to_json: bool = False
    ):
        warnings.warn(
            "Nd2Reader.frame_export is deprecated and will be removed in future versions; use limnd2.frame_export() function instead.",
            DeprecationWarning,
        )
        limnd2.frame_export(self, frame_index=frame_index, output_path=output_path, target_bit_depth=target_bit_depth, progress_to_json=progress_to_json)

    @functools.cached_property
    def generalImageInfo(self) -> dict[str, any]:
        warnings.warn(
            "Nd2Reader.generalImageInfo is deprecated and will be removed in future versions. Use limnd2.generalImageInfo() function instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return limnd2.generalImageInfo(self)

    # METHODS AND PROPERTIES IMPLEMENTING Nd2ReaderProtocol -> those should be documented in protocols.py

    @property
    def version(self) -> tuple[int, int]:
        return self.chunker.format_version

    @property
    def storage_info(self) -> StorageInfo:
        try:
            filename = self.chunker.filename
            url = Path(filename).absolute().as_uri() if filename else None
            size_on_disk = Nd2Reader.file_size_on_disk(filename)
            last_modified = Nd2Reader.file_last_modified(filename)
        except (ValueError, FileNotFoundError, PermissionError):
            filename = getattr(self.chunker, "filename", None)
            url = Path(filename).absolute().as_uri() if filename else None
            size_on_disk = getattr(self.chunker, "size_on_disk", 0)
            last_modified = getattr(self.chunker, "last_modified", datetime.datetime.fromtimestamp(0))
        return StorageInfo(filename, url, size_on_disk, last_modified)

    @functools.cached_property
    def is3d(self) -> bool:
        exp = self.experiment
        if exp is None:
            return False
        z_loop = exp.findLevel(ExperimentLoopType.eEtZStackLoop)
        if z_loop is None:
            return False
        return 1 < z_loop.count

    @functools.cached_property
    def isMono(self) -> bool:
        return 1 == self.imageAttributes.componentCount

    @functools.cached_property
    def isRgb(self) -> bool:
        return self.pictureMetadata.isRgb

    @functools.cached_property
    def is8bitRgb(self) -> bool:
        return 8 == self.imageAttributes.uiBpcSignificant and self.isRgb

    @functools.cached_property
    def isFloat(self) -> bool:
        return 32 == self.imageAttributes.uiBpcSignificant

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
        from .base import ND2_CHUNK_NAME_WellPlateDesc
        data = self.chunk(ND2_CHUNK_NAME_WellPlateDesc)
        return WellplateDesc.from_lv(data) if data is not None else None

    @functools.cached_property
    def wellplateFrameInfo(self) -> WellplateFrameInfo|None:
        from .base import ND2_CHUNK_NAME_WellPlateFrameInfo
        data = self.chunk(ND2_CHUNK_NAME_WellPlateFrameInfo)
        return WellplateFrameInfo.from_json(data) if data is not None else None

    @functools.cached_property
    def appInfo(self) -> AppInfo:
        from .base import ND2_CHUNK_NAME_AppInfo
        data = self.chunk(ND2_CHUNK_NAME_AppInfo)
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

    @functools.cached_property
    def imageDataRange(self) -> tuple[int, int]:
        return (np.min(self.compRange[:, 0]), np.max(self.compRange[:, 1])) if self.isFloat else (0, 2 ** self.imageAttributes.uiBpcSignificant - 1)

    @property
    def recordedData(self) -> RecordedData:
        from .base import ND2_CHUNK_NAME_CustomDataVar
        recData = RecordedData()
        if self.acqTimes is not None:
            strings = []
            for ms in self.acqTimes:
                sign = '-' if ms < 0 else ''
                total_ms = abs(ms)
                hh = int(total_ms // 3_600_000)
                total_ms -= hh * 3_600_000

                mm = int(total_ms // 60_000)
                total_ms -= mm * 60_000

                ss = int(total_ms // 1_000)
                total_ms -= ss * 1_000
                rem_ms = int(total_ms)
                strings.append(f"{sign}{hh}:{mm:02d}:{ss:02d}.{rem_ms:03d}")
            recData.append(RecordedDataItem(ID='ACQTIME', Desc='Time', Unit='h:m:s.ms', Type=RecordedDataType.eString, Group=0, Size=len(strings), Data=np.array(strings)))
        data = self.chunk(ND2_CHUNK_NAME_CustomDataVar)
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
    def shape(self) -> tuple[int, ...]:
        return (self.experiment.shape(skipSpectralLoop=True) if self.experiment else tuple()) + self.imageAttributes.shape

    def dimensionSizes(self, skipSpectralLoop=True) -> dict[str, int]:
        if self.experiment is None:
            return {}

        shape = self.experiment.shape(skipSpectralLoop=skipSpectralLoop)
        names = self.experiment.dimnames(skipSpectralLoop=skipSpectralLoop)
        return dict(zip(names, shape))

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
            for indexes in self.experiment.generateLoopIndexes(named=False):
                lst = list(indexes)
                windex, lst[i] = lst[i] // true_mp_size, lst[i] % true_mp_size
                lst = [windex] + lst
                ret.append(dict(zip(names, lst)) if named else lst)
            return ret

        else:
            return self.experiment.generateLoopIndexes(named=named)



        return self._chunker

    def chunk(self, name : bytes|str) -> bytes|memoryview|None:
        return self._chunker.chunk(name)

    def image(self, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.image(seqindex, rect)

    def downsampledImage(self, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.downsampledImage(seqindex, downsize, rect)

    def binaryRasterData(self, bin_id: int, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.binaryRasterData(bin_id, seqindex, rect)

    def downsampledBinaryRasterData(self, bin_id: int, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.downsampledBinaryRasterData(bin_id, seqindex, downsize, rect)

    @functools.cached_property
    def results(self) -> dict[str, ResultItem]:
        """
        Returns a dictionary of all results in the accompanying `.h5` file.

        Each result potentially contains tabular results (tables, graphs, ...) and binary layers.
        """
        return read_results_from_h5(self.filename.replace(".nd2", ".h5"))

    # ADDITIONAL PROPERTIES AND METHODS NOT IN THE PROTOCOL SPECIFIC TO ND2Reader, THOSE SHOULD BE DOCUMENTED HERE

    @functools.cached_property
    def customDescription(self) -> CustomDescription|None:
        from .base import ND2_CHUNK_NAME_CustomDescription
        data = self.chunk(ND2_CHUNK_NAME_CustomDescription)
        if data is None:
            return None
        return CustomDescription.from_lv(data)

    @functools.cached_property
    def smart_experiment_description(self) -> dict[str, any]|None:
        if self.customDescription is None or self.customDescription.name != "onepush":
            return None
        se_custom_data = {}
        for item in self.customDescription:
            if item.name in [ 'Assay', 'Date', 'Name', 'Plate', 'User', 'Notes' ]:
                if item.type == CustomDescriptionItemType.Date:
                    se_custom_data[item.name.lower()] = item.date.isoformat()
                else:
                    se_custom_data[item.name.lower()] = item.valueAsText
        return se_custom_data

    @property
    def chunk_size(self) -> tuple[int,int]|None:
        return None

    def crestDeepSimRawData(self, seqindex: int, component_index: int) -> NumpyArrayLike:
        """
        This method retrieves deepSIM data for a specific sequence index and component.

        The data is returned as a tuple containing deepSIM data (see `Returns` section below).

        !!! warning
            Even if file contains deepSIM data, not all channels or sequence indices must have deepSim data,
            use [`crestDeepSimRawDataIndices()`](nd2.md#limnd2.nd2.Nd2Reader.crestDeepSimRawDataIndices) method to
            see all valid indices for deepSIM data.

        Reading all deepSIM data can be done using following code:
        ```py
        file = "deepSIM.nd2"

        with limnd2.Nd2Reader(file) as nd2:
            results = {}
            for indices in nd2.crestDeepSimRawDataIndices():
                results[indices] = nd2.crestDeepSimRawData(*indices)

            for i, r in results.items():
                print(f"Input indices: sequence index: {i[0]}, component index: {i[1]}")
                print("    Final image: (shape)", r[0].shape)
                print("    Calibration key:", r[1])
                print("    Calibration data: (length of XML)", len(r[2]))
                print("    PSF: (set, default)", r[3])
                print("    Iter: (set, default)", r[4])
                print("    ROI offsets: (x, y)", r[5])
        ```

        ??? example "See example output"
            ```
            Input indices: sequence index: 0, component index: 0
                Final image: (shape) (65, 1024, 1024)
                Calibration key: zoom60000_na1400_ex561_pitch15000_size1500_im65_tm0
                Calibration data: (length of XML) 880341
                PSF: (set, default) (2.0, 2.0)
                Iter: (set, default) (25, 25)
                ROI offsets: (x, y) (0, 0)
            Input indices: sequence index: 0, component index: 1
                Final image: (shape) (65, 1024, 1024)
                Calibration key: zoom60000_na1400_ex488_pitch15000_size1500_im65_tm0
                Calibration data: (length of XML) 884538
                PSF: (set, default) (2.0, 2.0)
                Iter: (set, default) (25, 25)
                ROI offsets: (x, y) (0, 0)
            Input indices: sequence index: 1, component index: 0
                Final image: (shape) (65, 1024, 1024)
            ...
            ```

        Parameters
        ----------
        seqindex : int
            The index of the sequence for which the deepSIM data is being retrieved.

        component_index : str
            The index of the component for which the deepSIM data is requested.

        Returns
        -------
        numpy.ndarray
            A 3D array of deepSIM image data with shape (channel_count, height, width).
        str
            A string that contains the significant calibration data in compressed form,
            for example `zoom60000_na1400_ex561_pitch15000_size1500_im65_tm0`.
        str
            DeepSIM data detailed outcome in XML Format as string.
        tuple[float, float]
            PSF values (used, default). Depends on the objective used.
        tuple[int, int]
            Number of iterations for reconstruction (used, default).
        tuple[int, int]
            ROI width and height listed in ndarray (x, y).

        Raises
        ------
        NameNotInChunkmapError
            If chunk with given sequence and component index is missing.
        """
        return self._chunker.crestDeepSimRawData(seqindex, component_index)

    def crestDeepSimRawDataIndices(self) -> list[tuple[int, int]]:
        """
        Returns all valid indices for deepSIM data in the `.nd2` file.

        Indices are returned as tuples of (sequence_index, component_index) in a list.

        See [`crestDeepSimRawData()`](nd2.md#limnd2.nd2.Nd2Reader.crestDeepSimRawData) method that uses those indices to retrieve deepSIM data.
        """
        return self._chunker.crestDeepSimRawDataIndices()

    def result_size_on_disk(self, result_name: str) -> int|None:
        """
        Returns size of the result.
        """
        return None

    def result_binary_data(self, bin_id: int, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        pass

    def result_private_table(self, result_name: str, pane: str, table_name: str) -> TableData:
        the_pane: ResultPane = None
        try:
            the_pane = self.results[result_name].result_panes[pane]
        except KeyError:
            raise KeyError(f"Result {result_name} or pane {pane} not found in H5.")

        try:
            return the_pane.private_tables[table_name]
        except KeyError:
            pass

        try:
            loc = the_pane.private_table_locations[table_name]
        except KeyError:
            raise KeyError(f"Table name {table_name} not found in H5 {result_name}/{pane} .")

        table_data: TableData = create_table_data_from_h5(self.filename.replace(".nd2", ".h5"), loc)
        the_pane.private_tables[table_name] = table_data

        return the_pane.private_tables[table_name]


class Nd2Writer():
    """
    Writer class for writing `.nd2` files.

    See [`Nd2WriterProtocol`](protocols.md#limnd2.protocols.Nd2WriterProtocol) for more information.

    """
    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        kwargs["readonly"] = False
        return _create_chunker(*args, **kwargs)

    def __init__(self, file : FileLikeObject, *, append : bool|None = None, chunker_kwargs:dict = {}) -> None:
        """
        Parameters
        -----------
        file : str | Path | int | typing.BinaryIO
            Filename of the ND2 file.
        chunker_kwargs
            Additional parameters for chunker.
        """
        super().__init__()
        self._chunker = self.create_chunker(file, append=append, chunker_kwargs=chunker_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    @property
    def filename(self) -> str|None:
        return self.chunker.filename

    @property
    def imageAttributes(self) -> ImageAttributes:
        return self._chunker.imageAttributes

    @imageAttributes.setter
    def imageAttributes(self, val: ImageAttributes) -> None:
        if val != None:
            self._chunker.imageAttributes = val

    @property
    def experiment(self) -> ExperimentLevel:
        return self._chunker.experiment

    @experiment.setter
    def experiment(self, val: ExperimentLevel) -> None:
        if val != None:
            self._chunker.experiment = val

    @property
    def pictureMetadata(self) -> PictureMetadata:
        return self._chunker.pictureMetadata

    @pictureMetadata.setter
    def pictureMetadata(self, val: PictureMetadata) -> None:
        if val != None:
            self._chunker.pictureMetadata = val

    @property
    def chunker(self):
        return self._chunker

    def setChunk(self, name : bytes|str, data : bytes|memoryview) -> None:
        return self._chunker.setChunk(name, data)

    def setImage(self, seq_index: int, data: NumpyArrayLike) -> None:
        return self._chunker.setImage(seq_index, data)

    def finalize(self) -> None:
        return self._chunker.finalize()

    def rollback(self) -> None:
        return self._chunker.rollback()

def _create_chunker(file : FileLikeObject, *, readonly: bool = True, append: bool|None = None, chunker_kwargs: dict = {}) -> LimBinaryIOChunker:
    if isinstance(file, (str, Path)):
        if readonly:
            mode = "rb"
        else:
            if append is None:
                append = os.path.isfile(file)
            mode = "rb+" if append else "wb"

        #if mode == "rb+":
        #    raise FileExistsError("This file already exists, can not open for writing.")

        fh = open(file, mode)
        chunker_kwargs.update(dict(filename=file))
        return LimBinaryIOChunker(fh, **chunker_kwargs)

    elif isinstance(file, memoryview):
        return LimBinaryIOChunker(file, **chunker_kwargs)

    elif (hasattr(file, "read") or hasattr(file, "write")) and hasattr(file, "seek") and hasattr(file, "tell") and hasattr(file, "mode"):
        if readonly and "rb" != file.mode:
            raise ValueError("File handle passed to LimNd2Reader must have \"rb\" mode")
        elif not readonly and file.mode not in ("rb+", "wb"):
            raise ValueError("File handle passed to LimNd2Writer must have \"rb+\" or \"wb\" mode")
        return LimBinaryIOChunker(file, **chunker_kwargs)

    raise ValueError("Invalid chunker source, must be filename, file handle or memoryview.")
