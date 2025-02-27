import datetime, functools, numpy as np, os
from pathlib import Path

from .attributes import ImageAttributesPixelType
from .base import FileLikeObject, NumpyArrayLike, Nd2LoggerEnabled, BinaryRleMetadata, BinaryRasterMetadata, ImageAttributes, NumpyArrayLike
from .custom_data import CustomDescription, RecordedData, RecordedDataItem, RecordedDataType
from .experiment import ExperimentLevel, ExperimentLoopType, WellplateDesc, WellplateFrameInfo
from .file import LimBinaryIOChunker
from .metadata import PictureMetadata
from .textinfo import ImageTextInfo, AppInfo
from .variant import decode_var

if Nd2LoggerEnabled:
    import logging
    logger = logging.getLogger("limnd2")

class Nd2Reader:
    """
    Creates Nd2Read instance for reading `.nd2` files and its attributes, metadata, properties, image data and so on.

    Also see [Quickstart](index.md#reading-nd2-files) for an
    example of how to use this class and how to read individual chunks, attributes, metadata and so on.


    """
    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        kwargs["readonly"] = True
        return _create_chunker(*args, **kwargs)

    def __init__(self, file : FileLikeObject, *, chunker_kwargs: dict = {}) -> None:
        """
        Parameters
        -----------
        file : str | Path | int | typing.BinaryIO
            Filename of the ND2 file.
        chunker_kwargs
            Additional parameters for chunker.
        """
        self._chunker = self.create_chunker(file, chunker_kwargs=chunker_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    @property
    def filename(self) -> str|None:
        return self._chunker.filename

    @property
    def version(self) -> tuple[int, int]:
        """
        Returns the version of the `.nd2` file as a tuple of two integers.
        """
        return self._chunker.fileVersion

    @functools.cached_property
    def is3d(self) -> bool:
        """
        Returns `True` if the file contains valid z-stack, otherwise `False`.
        """
        exp = self.experiment
        if exp is None:
            return False
        z_loop = exp.findLevel(ExperimentLoopType.eEtZStackLoop)
        if z_loop is None:
            return False
        return 1 < z_loop.count

    @functools.cached_property
    def isMono(self) -> bool:
        """
        Returns `True` if the file contains only one component, otherwise `False`.
        """
        return 1 == self.imageAttributes.componentCount

    @functools.cached_property
    def isRgb(self) -> bool:
        """
        Returns `True` if the file contains RGB data, otherwise `False`.
        """
        return self.pictureMetadata.isRgb

    @functools.cached_property
    def is8bitRgb(self) -> bool:
        """
        Returns `True` if the file contains 8-bit RGB data, otherwise `False`.
        """
        return 8 == self.imageAttributes.uiBpcSignificant and self.isRgb

    @functools.cached_property
    def isFloat(self) -> bool:
        """
        Returns `True` if the file data is 32-bit float, otherwise `False`.
        """
        return 32 == self.imageAttributes.uiBpcSignificant

    @property
    def imageAttributes(self) -> ImageAttributes:
        """
        Attribute to get attributes of an `.nd2` file.

        See [`ImageAttributes`](attributes.md#limnd2.attributes.ImageAttributes) class for more information.

        In order to create an instance of `ImageAttributes` class from simple parameters, use [`ImageAttributes.create`](attributes.md#limnd2.attributes.ImageAttributes.create) method.
        """
        return self._chunker.imageAttributes

    @property
    def pictureMetadata(self) -> PictureMetadata:
        """
        Attribute to get metadata of an `.nd2` file.

        See [`PictureMetadata`](metadata.md#limnd2.metadata.PictureMetadata) class for more information.

        In order to create an instance of `PictureMetadata` class, use [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) class.
        """
        return self._chunker.pictureMetadata

    @property
    def experiment(self) -> ExperimentLevel|None:
        """
        Attribute to get experiments in an `.nd2` file.

        See [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel) class for more information.

        In order to create an instance of `ExperimentLevel` class, use [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) class.
        """
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

    @functools.cached_property
    def imageDataRange(self) -> tuple[int, int]:
        return (np.min(self.compRange[:, 0]), np.max(self.compRange[:, 1])) if self.isFloat else (0, 2 ** self.imageAttributes.uiBpcSignificant - 1)
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
        """
        Returns general information about the image as a dictionary.
        """
        ia = self.imageAttributes
        loops = ", ".join([ f"{exp_level.shortName}({exp_level.count})" for exp_level in self.experiment if 0 < exp_level.count ]) if self.experiment else ""
        filename = os.path.basename(self.filename)
        path = os.path.dirname(self.filename)
        bit_depth = f"{ia.uiBpcSignificant}bit {ImageAttributesPixelType.short_name(ia.ePixelType)}"
        frame_res = f"{ia.width} x {ia.height}"
        dimension = f"{frame_res} ({ia.componentCount} {"comps" if 1 < ia.componentCount else "comp"} {bit_depth})" + (f" x {ia.uiSequenceCount} frames" if 1 < ia.uiSequenceCount else "") +(f": {loops}" if loops else "")
        file_size = self.chunker.filesize
        frame_size = format_file_size(ia.height*ia.widthBytes)
        z_count = self.experiment.dims.get('z', 0) if self.experiment is not None else 0
        volume_size = format_file_size(ia.height*ia.widthBytes*z_count)
        sizes = f"{format_file_size(self.chunker.filesize)} on disk, {frame_size} frame" + (f", {volume_size} volume" if z_count else "")
        calibration = f"{self.pictureMetadata.dCalibration:.3f} µm/px" if self.pictureMetadata.bCalibrated else "Uncalibrated"
        mtime = f"{self.chunker.filelastmodified.strftime('%x %X')}"
        app_created = self.appInfo.software
        return dict(filename=filename, path=path, bit_depth=bit_depth, loops=loops, dimension=dimension, file_size=file_size, frame_res=frame_res, volume_size=volume_size, sizes=sizes, calibration=calibration, mtime=mtime, app_created=app_created)

    @functools.cached_property
    def customDescription(self) -> CustomDescription|None:
        data = self.chunk(b'CustomData|CustomDescriptionV1_0!')
        if data is None:
            return None
        return CustomDescription.from_lv(data)


    def generateLoopIndexes(self, named: bool = False) -> list:
        """
        Generates indexes for all loops in the experiment.
        """
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
        """
        Returns data for specific chunk name

        Parameters
        ----------
        name : bytes|str
            Name of the chunk to retrieve.
        """
        return self._chunker.chunk(name)

    def image(self, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        """
        Get image data from specified frame as NumPy array.

        Parameters
        ----------
        seqindex: int
            Image sequence index you want to get.
        rect: tuple[int, int, int, int]|None
            Rectangle (x, y, w, h) of the image to get image to get.
        """
        return self._chunker.image(seqindex, rect)

    def downsampledImage(self, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.downsampledImage(seqindex, downsize, rect)

    def binaryRasterData(self, binid: int, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.binaryRasterData(binid, seqindex, rect)

    def downsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        return self._chunker.downsampledBinaryRasterData(binid, seqindex, downsize, rect)

    def crestDeepSimRawData(self, seqindex: int, componentindex: int) -> NumpyArrayLike:
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

        componentindex : str
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
        return self._chunker.crestDeepSimRawData(seqindex, componentindex)

    def crestDeepSimRawDataIndices(self) -> list[tuple[int, int]]:
        """
        Returns all valid indices for deepSIM data in the `.nd2` file.

        Indices are returned as tuples of (sequence_index, component_index) in a list.

        See [`crestDeepSimRawData()`](nd2.md#limnd2.nd2.Nd2Reader.crestDeepSimRawData) method that uses those indices to retrieve deepSIM data.
        """
        return self._chunker.crestDeepSimRawDataIndices()

    def finalize(self) -> None:
        return self._chunker.finalize()


class Nd2Writer:
    """
    Experimental ND2 file writer.

    Supports encoding od all image attributes, most commonly used experiments and most of image metadata.
    Currently does not support encoding of Wellplates, binary layers, ROIs and any custom data and text into chunk.

    !!! info
        Data is written in chunks, so you can write data in any order you want, imade data however can
        only be written after image attributes are set, if you write same chunk multiple times, all chunks will be saved,
        however only the last one will be used.

    !!! tip
        As explained in [Quickstart](index.md#writing-to-nd2-file), image data can only be written **after**
        image attributes are set, but if you want to write image data into `.nd2` file without knowing how many frames there is
        (for example with continuous writing),
        you can pass `ImageAttributes` instance when creating `.nd2` using custom chunker argument as shown below.

        Setting `ImageAttributes` this way **will not store them in `.nd2` file** and you still **have to store them at some point**,
        however you can do so after you know how many frames there is.

        ```py title="Example os using chunker arguments to set image attributes"
        attributes = limnd2.attributes.ImageAttributes.create(
            width = WIDTH,
            height = HEIGHT,
            component_count = COMPONENT_COUNT,
            bits = BITS,
            sequence_count = ...  # will be set later
        )

        with limnd2.Nd2Writer("outfile.nd2", chunker_kwargs={"with_image_attributes": attributes}) as nd2:
            # you can now set image data without setting attributes
        ```

    See [Quickstart](index.md#writing-to-nd2-file) for an example of how to use this class and how to write individual chunks.
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
        self._chunker = self.create_chunker(file, append=append, chunker_kwargs=chunker_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()


    @property
    def version(self) -> tuple[int, int]:
        return self._chunker.fileVersion


    @property
    def imageAttributes(self) -> ImageAttributes:
        """
        Attribute to get or set attributes of an `.nd2` file.

        See [`ImageAttributes`](attributes.md#limnd2.attributes.ImageAttributes) class for more information.

        In order to create an instance of `ImageAttributes` class from simple parameters, use [`ImageAttributes.create`](attributes.md#limnd2.attributes.ImageAttributes.create) method.
        """
        return self._chunker.imageAttributes

    @imageAttributes.setter
    def imageAttributes(self, val: ImageAttributes) -> None:
        if val != None:
            self._chunker.imageAttributes = val


    @property
    def experiment(self) -> ExperimentLevel:
        """
        Attribute to get or set experiments in an `.nd2` file.

        See [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel) class for more information.

        In order to create an instance of `ExperimentLevel` class, use [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) class.
        """
        return self._chunker.experiment

    @experiment.setter
    def experiment(self, val: ExperimentLevel) -> None:
        if val != None:
            self._chunker.experiment = val


    @property
    def pictureMetadata(self) -> PictureMetadata:
        """
        Attribute to get or set metadata of an `.nd2` file.

        See [`PictureMetadata`](metadata.md#limnd2.metadata.PictureMetadata) class for more information.

        In order to create an instance of `PictureMetadata` class, use [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) class.
        """
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
        """
        Seta image data using specified frame index.

        !!! warning
            You must manually keep track of the frame index and make sure that you are not
            overwriting the same frame multiple times and that images are written sequentially.
        """
        return self._chunker.setImage(seq_index, data)

    def finalize(self) -> None:
        """
        Expicitely finalize the file, this is not needed if you use `with` statement.
        """
        return self._chunker.finalize()

    def rollback(self) -> None:         # as Nd2 writer currently works only with mew images, rollback is not needed as there is nothing to roll back to
        return self._chunker.rollback()

def _create_chunker(file : FileLikeObject, *, readonly: bool = True, append: bool|None = None, chunker_kwargs: dict = {}):
    if isinstance(file, (str, Path)):
        if readonly:
            mode = "rb"
        else:
            if append is None:
                append = os.path.isfile(file)
            mode = "rb+" if append else "wb"

        if mode == "rb+":
            raise FileExistsError("This file already exists, can not open for writing.")

        fh = open(file, mode)
        chunker_kwargs.update(dict(filename=file))
        return LimBinaryIOChunker(fh, **chunker_kwargs)

    elif (hasattr(file, "read") or hasattr(file, "write")) and hasattr(file, "seek") and hasattr(file, "tell") and hasattr(file, "mode"):
        if readonly and "rb" != file.mode:
            raise ValueError("File handle passed to LimNd2Reader must have \"rb\" mode")
        elif not readonly and file.mode not in ("rb+", "wb"):
            raise ValueError("File handle passed to LimNd2Wrtier must have \"rb+\" or \"wb\" mode")
        return LimBinaryIOChunker(file, **chunker_kwargs)

def format_file_size(size: int) -> str:
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
