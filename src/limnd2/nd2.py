import datetime
import functools
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import limnd2

from .base import (
    BaseChunker,
    BinaryRasterMetadata,
    BinaryRleMetadata,
    FileLikeObject,
    ImageAttributes,
    Nd2LoggerEnabled,
    NumpyArrayLike,
    Store,
    FileStore,
    MemoryStore
)
from .custom_data import (
    CustomDescription,
    CustomDescriptionItemType,
    RecordedData,
    RecordedDataItem,
    RecordedDataType,
)
from .experiment import (
    ExperimentLevel,
    ExperimentLoopType,
    WellplateDesc,
    WellplateFrameInfo,
)
from .file_modern import LimBinaryIOChunker
from .file_legacy import LimJpeg2000Chunker, is_legacy_jpeg2000_source
from .metadata import PictureMetadata
from .results import (
    ResultItem,
    ResultPane,
    TableData,
    create_table_data_from_h5,
    read_results_from_h5,
)
from .textinfo import AppInfo, ImageTextInfo
from .variant import decode_var

if Nd2LoggerEnabled:
    import logging
    logger = logging.getLogger("limnd2")



class Nd2Reader:
    """
    High-level class for accessing `.nd2` data format. Especially attributes, metadata, properties, image, binary
    data as in NIS-Elements. This class relies on chunker classes (legacy JPEG2000 and modern ND2).
    The chunker classes are backed by stores abstracting the storage media (disk, memory).

    Also see [Quickstart](index.md#reading-nd2-files) for an
    example of how to use this class and how to read individual chunks, attributes, metadata and so on.
    """

    def _create_chunker(self, *args, **kwargs) -> BaseChunker:
        kwargs["readonly"] = True
        return _create_chunker(*args, **kwargs)

    @property
    def chunker(self):
        return self._chunker

    def __init__(self, file: FileLikeObject, *, chunker_kwargs: dict = {}) -> None:
        """
        Parameters
        -----------
        file : str | Path | Store | typing.BinaryIO | memoryview
            Filename of the ND2 file.
        chunker_kwargs
            Additional parameters for chunker.
        """
        super().__init__()
        self._chunker = self._create_chunker(file, chunker_kwargs=chunker_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    def finalize(self) -> None:
        return self._chunker.finalize()

    # METHODS AND PROPERTIES IMPLEMENTING Nd2ReaderProtocol -> those should be documented in protocols.py

    @property
    def version(self) -> tuple[int, int]:
        """
        Returns the version of the `.nd2` file as a tuple of two integers.

        (1, 0) - JPEG2000 legacy file format
        (2, 0) - ND2 metadata stored in Variant as xml
        (3, 0) - ND2 metadata stored in LiteVariant as proprietary binary format
        """
        return self.chunker.format_version

    @property
    def store(self) -> Store:
        """
        Returns the backing store (abstraction of the storage medium).
        """
        return self.chunker.store

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
    def experiment(self) -> ExperimentLevel | None:
        """
        Attribute to get experiments in an `.nd2` file.

        See [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel) class for more information.

        In order to create an instance of `ExperimentLevel` class, use [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) class.
        """
        return self._chunker.experiment

    @property
    def imageTextInfo(self) -> ImageTextInfo | None:
        return self._chunker.imageTextInfo

    @functools.cached_property
    def wellplateDesc(self) -> WellplateDesc | None:
        from .base import ND2_CHUNK_NAME_WellPlateDesc

        data = self.chunk(ND2_CHUNK_NAME_WellPlateDesc)
        return WellplateDesc.from_lv(data) if data is not None else None

    @functools.cached_property
    def wellplateFrameInfo(self) -> WellplateFrameInfo | None:
        from .base import ND2_CHUNK_NAME_WellPlateFrameInfo

        data = self.chunk(ND2_CHUNK_NAME_WellPlateFrameInfo)
        return WellplateFrameInfo.from_json(data) if data is not None else None

    @functools.cached_property
    def appInfo(self) -> AppInfo:
        from .base import ND2_CHUNK_NAME_AppInfo

        data = self.chunk(ND2_CHUNK_NAME_AppInfo)
        if data is None:
            return AppInfo()
        return AppInfo.from_var(data)

    @property
    def software(self) -> str:
        return self.appInfo.software

    @property
    def acqFrames(self) -> NumpyArrayLike | None:
        return self._chunker.acqFrames

    @property
    def acqTimes(self) -> NumpyArrayLike | None:
        return self._chunker.acqTimes

    @property
    def acqTimes2(self) -> NumpyArrayLike | None:
        return self._chunker.acqTimes2

    @property
    def compFrameRange(self) -> NumpyArrayLike:
        return self._chunker.compFrameRange

    @property
    def compRange(self) -> NumpyArrayLike:
        return self._chunker.compRange

    @functools.cached_property
    def imageDataRange(self) -> tuple[int, int]:
        return (
            (np.min(self.compRange[:, 0]), np.max(self.compRange[:, 1]))
            if self.isFloat
            else (0, 2**self.imageAttributes.uiBpcSignificant - 1)
        )

    @property
    def recordedData(self) -> RecordedData:
        from .base import ND2_CHUNK_NAME_CustomDataVar

        recData = RecordedData()
        if self.acqTimes is not None:
            strings = []
            for ms in self.acqTimes:
                sign = "-" if ms < 0 else ""
                total_ms = abs(ms)
                hh = int(total_ms // 3_600_000)
                total_ms -= hh * 3_600_000

                mm = int(total_ms // 60_000)
                total_ms -= mm * 60_000

                ss = int(total_ms // 1_000)
                total_ms -= ss * 1_000
                rem_ms = int(total_ms)
                strings.append(f"{sign}{hh}:{mm:02d}:{ss:02d}.{rem_ms:03d}")
            recData.append(
                RecordedDataItem(
                    ID="ACQTIME",
                    Desc="Time",
                    Unit="h:m:s.ms",
                    Type=RecordedDataType.eString,
                    Group=0,
                    Size=len(strings),
                    Data=np.array(strings),
                )
            )
        data = self.chunk(ND2_CHUNK_NAME_CustomDataVar)
        if data is not None:
            decoded = decode_var(data)
            desc = decoded.get("CustomTagDescription_v1.0", {})
            for i in range(len(desc)):
                itemDesc = desc.get(f"Tag{i}", None)
                if itemDesc is not None:
                    colData = self.chunk(
                        b"CustomData|%s!" % (itemDesc.get("ID").encode("utf-8"))
                    )
                    if colData is None:
                        continue
                    if isinstance(colData, memoryview):
                        colData = colData.tobytes()
                    recData.append(
                        RecordedDataItem.from_desc_and_data(itemDesc, colData)
                    )
        if 0 < len(recData):
            recData.insert(
                0,
                RecordedDataItem(
                    ID="INDEX",
                    Desc="Index",
                    Unit="",
                    Type=RecordedDataType.eInt,
                    Group=0,
                    Size=recData.rowCount,
                    Data=np.arange(1, recData.rowCount + 1),
                ),
            )
            recData.sort()
        return recData

    @property
    def binaryRleMetadata(self) -> BinaryRleMetadata:
        return self._chunker.binaryRleMetadata

    @property
    def binaryRasterMetadata(self) -> BinaryRasterMetadata | None:
        if self._chunker.binaryRasterMetadata is None:
            return None
        if 0 == len(self._chunker.binaryRasterMetadata) and 0 < len(
            self._chunker.binaryRleMetadata
        ):
            return self._chunker.binaryRleMetadata.makeRasterMetadata(
                self.imageAttributes.width, self.imageAttributes.height
            )
        else:
            return self._chunker.binaryRasterMetadata

    def dimensionSizes(self, skipSpectralLoop=True) -> dict[str, int]:
        """
        Returns a dictionary with dimension names as keys and their sizes as values.

        TODO: consider replacing it with fixed order tuple
        """
        if self.experiment is None:
            return {}

        shape = self.experiment.shape(skipSpectralLoop=skipSpectralLoop)
        names = self.experiment.dimnames(skipSpectralLoop=skipSpectralLoop)
        return dict(zip(names, shape))

    def generateLoopIndexes(self, named: bool = False) -> list:
        """
        Generates indexes for all loops in the experiment.

        TODO: consider using fixed order tuples [T, M, Z]
        """
        exp = self.experiment
        if exp is None:
            return []
        wp_desc = self.wellplateDesc
        wp_frameinfo = self.wellplateFrameInfo
        names, shape = (
            exp.dimnames(skipSpectralLoop=True),
            exp.shape(skipSpectralLoop=True),
        )
        if (
            isinstance(wp_desc, WellplateDesc)
            and isinstance(wp_frameinfo, WellplateFrameInfo)
            and "m" in names
            and len(wp_frameinfo)
        ):
            ret = []
            i = names.index("m")
            names = ("w",) + names
            mp_size, wp_size = shape[i], wp_frameinfo.nwells
            true_mp_size = mp_size // wp_size
            for indexes in exp.generateLoopIndexes(named=False):
                lst = list(indexes)
                windex, lst[i] = lst[i] // true_mp_size, lst[i] % true_mp_size
                lst = [windex] + lst
                ret.append(dict(zip(names, lst)) if named else lst)
            return ret

        else:
            return exp.generateLoopIndexes(named=named)

    def chunk(self, name: bytes | str) -> bytes | memoryview | None:
        """
        Returns data for specific chunk name

        Parameters
        ----------
        name : bytes|str
            Name of the chunk to retrieve.

        TODO: consider removing it from here too low-level
        """
        return self._chunker.chunk(name)

    def image(
        self,
        seq_index: int,
        *,
        rect: tuple[int, int, int, int] | None = None,
        downsample_level: int = 0
    ) -> NumpyArrayLike:
        """
        Returns specific 2D image frame as numpy.ndarray.

        Parameters
        ----------
        seqindex: int
            Zero-base sequence index of the frame.
        rect: tuple[int, int, int, int]|None
            Rectangle (x, y, w, h) specifying a portion of the resulting image.
        downsample_level: int
            Determines downsampling d = 2^downsample_level that produces
            lower level frames of size (w // d, h // d).
        """
        assert isinstance(seq_index, int) and 0 <= seq_index, "seq_index must be non-negative integer"
        assert isinstance(downsample_level, int) and 0 <= downsample_level, "down_size must be non-negative integer"
        return (
            self._chunker.image(seq_index, rect) if 0 == downsample_level else
            self._chunker.downsampledImage(
                seq_index,
                self.imageAttributes.lowerPowSizeList[downsample_level-1],
                rect)
        )

    def binaryRasterData(
        self,
        bin_id: int,
        seq_index: int,
        *,
        rect: tuple[int, int, int, int] | None = None,
        downsample_level: int = 0
    ) -> NumpyArrayLike:
        """
        Returns specific 2D binary data as numpy.ndarray.

        Parameters
        ----------
        bin_id: int
            Binary layer ID.
        seqindex: int
            Zero-base sequence index of the frame.
        rect: tuple[int, int, int, int]|None
            Rectangle (x, y, w, h) of the image (always in original resolution).
        downsample_level: int
            Determines downsampling d = 2^downsample_level that produces
            lower level frames of size (w // d, h // d).
        """
        assert isinstance(bin_id, int) and 0 < bin_id, "bin_id must be positive integer"
        assert isinstance(seq_index, int) and 0 <= seq_index, "seq_index must be non-negative integer"
        assert isinstance(downsample_level, int) and 0 <= downsample_level, "down_size must be non-negative integer"
        return (
            self._chunker.binaryRasterData(bin_id, seq_index, rect) if 0 == downsample_level else
            self._chunker.downsampledBinaryRasterData(
                bin_id,
                seq_index,
                self.imageAttributes.lowerPowSizeList[downsample_level-1],
                rect)
        )

    @functools.cached_property
    def results(self) -> dict[str, ResultItem]:
        """
        Returns a dictionary of all results in the accompanying `.h5` file.

        Each result potentially contains tabular results (tables, graphs, ...) and binary layers.
        """
        filename = self.store.filename
        if filename is None:
            return {}
        return read_results_from_h5(filename.replace(".nd2", ".h5"))

    # ADDITIONAL PROPERTIES AND METHODS NOT IN THE PROTOCOL SPECIFIC TO ND2Reader, THOSE SHOULD BE DOCUMENTED HERE

    @functools.cached_property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        """
        Returns 6D canonical data shape (T, M, Z, Y, X, C).
        """
        from .experiment import canonical_shape

        return canonical_shape(self.experiment) + self.imageAttributes.shape

    @functools.cached_property
    def calibration(self) -> tuple[float, float, float, float, float, float]:
        """
        Returns 6D canonical data calibration (T in ms, 0, Z in um, Y in um, X in um, 0) 0 is for uncalibrated.
        """
        from .experiment import canonical_calibration

        xy: float = 0.0
        if self.pictureMetadata is not None and self.pictureMetadata.bCalibrated:
            xy = self.pictureMetadata.dCalibration
        return canonical_calibration(self.experiment) + (xy, xy, 0)

    def delayedImageData(self, tiling: tuple[int, int] | None = None) -> Any:
        """
        Returns 6D canonical data shape (T, M, Z, Y, X, C) dask array with
        delayed chunks.

        Parameters
        ----------
        tiling : tuple[int, int] | None
            Optional frame tiling x, y.
        """
        try:
            import dask.array as da  # type: ignore
            from dask.delayed import delayed  # type: ignore

            def make_edges(n, step):
                edges = list(range(0, n, step))
                if edges[-1] != n:
                    edges.append(n)
                return edges

            def read_frame(
                nd2: Nd2Reader,
                index: int,
                rect: tuple[int, int, int, int] | None = None,
            ) -> np.ndarray:
                if rect is not None:
                    print(Rf"Reading frame {index} with rect {rect}.")
                else:
                    print(Rf"Reading frame {index}.")
                return nd2.image(index, rect=rect)

            nf = self.imageAttributes.frameCount
            nt, nm, nz, ny, nx, nc = self.shape

            edges: tuple[list[int], list[int]] | None = None
            if tiling is not None:
                edges = (make_edges(ny, tiling[1]), make_edges(nx, tiling[0]))

            def build_frame(i: int) -> da.Array:
                if edges is not None:
                    rows = []
                    for iy, (y0, y1) in enumerate(zip(edges[0][:-1], edges[0][1:])):
                        col_tiles = []
                        for ix, (x0, x1) in enumerate(zip(edges[1][:-1], edges[1][1:])):
                            d = delayed(read_frame)(self, i, (x0, y0, x1 - x0, y1 - y0))
                            a = da.from_delayed(
                                d,
                                shape=(y1 - y0, x1 - x0, nc),
                                dtype=self.imageAttributes.dtype,
                            )
                            col_tiles.append(a)
                        rows.append(da.concatenate(col_tiles, axis=1))
                    return da.concatenate(rows, axis=0)
                else:
                    return da.from_delayed(
                        delayed(read_frame)(self, i),
                        shape=self.imageAttributes.shape,
                        dtype=self.imageAttributes.dtype,
                    )

            if self.experiment:
                assert nf == nt * nm * nz, (
                    f"frameCount ({nf}) is not equal to the product of shape ({nt * nm * nz} = {nt} * {nm} * {nz})"
                )

                frames = [build_frame(i) for i in range(nf)]

                return da.stack(frames).reshape(nt, nm, nz, ny, nx, nc)

            else:
                frame: da.Array = build_frame(0)
                return frame.reshape(nt, nm, nz, ny, nx, nc)

        except ImportError:
            raise

    @functools.cached_property
    def customDescription(self) -> CustomDescription | None:
        from .base import ND2_CHUNK_NAME_CustomDescription

        data = self.chunk(ND2_CHUNK_NAME_CustomDescription)
        if data is None:
            return None
        return CustomDescription.from_lv(data)

    @functools.cached_property
    def smartExperimentDescription(self) -> dict[str, Any] | None:
        if self.customDescription is None or self.customDescription.name != "onepush":
            return None
        se_custom_data = {}
        for item in self.customDescription:
            if item.name in ["Assay", "Date", "Name", "Plate", "User", "Notes"]:
                if item.type == CustomDescriptionItemType.Date:
                    se_custom_data[item.name.lower()] = item.date.isoformat()
                else:
                    se_custom_data[item.name.lower()] = item.valueAsText
        return se_custom_data

    @property
    def chunkSize(self) -> tuple[int, int] | None:
        return None

    def crestDeepSimRawData(
        self, seqindex: int, component_index: int
    ) -> tuple[
        NumpyArrayLike, str, str, tuple[float, float], tuple[int, int], tuple[int, int]
    ]:
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

    def resultSizeOnDisk(self, result_name: str) -> int | None:
        """
        Returns size of the result.
        """
        raise NotImplementedError()

    def resultBinaryData(
        self, bin_id: int, seqindex: int, rect: tuple[int, int, int, int] | None = None
    ) -> NumpyArrayLike:
        pass
        return np.array([])

    def resultPrivateTable(
        self, result_name: str, pane: str, table_name: str
    ) -> TableData:
        the_pane: ResultPane | None = None
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
            raise KeyError(
                f"Table name {table_name} not found in H5 {result_name}/{pane} ."
            )

        filename = self.store.filename
        if filename is None:
            raise ValueError("Cannot read private table data, ND2 filename is None.")

        table_data: TableData = create_table_data_from_h5(
            filename.replace(".nd2", ".h5"), loc
        )
        the_pane.private_tables[table_name] = table_data

        return the_pane.private_tables[table_name]


class Nd2Writer:
    """
    ND2 file writer.

    Supports encoding of all image attributes, most commonly used experiments and most of image metadata.
    Currently does not support encoding of Well-plates, binary layers, ROIs and any custom data and text into chunk.

    !!! info
        Data is written in chunks, so you can write data in any order you want, image data however can
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

    def _create_chunker(self, *args, **kwargs) -> BaseChunker:
        kwargs["readonly"] = False
        return _create_chunker(*args, **kwargs)

    def __init__(
        self,
        file: FileLikeObject,
        *,
        append: bool | None = None,
        chunker_kwargs: dict = {},
    ) -> None:
        """
        Parameters
        -----------
        file : str | Path | Store | typing.BinaryIO | memoryview
            Filename of the ND2 file.
        chunker_kwargs
            Additional parameters for chunker.
        """
        super().__init__()
        self._chunker = self._create_chunker(
            file, append=append, chunker_kwargs=chunker_kwargs
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

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
    def experiment(self) -> ExperimentLevel | None:
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

    def setChunk(self, name: bytes | str, data: bytes | memoryview) -> None:
        return self._chunker.setChunk(name, data)

    def setImage(self, seq_index: int, data: NumpyArrayLike) -> None:
        """
        Set image data using specified frame index.

        !!! warning
            You must manually keep track of the frame index and make sure that you are not
            overwriting the same frame multiple times and that images are written sequentially.
        """
        return self._chunker.setImage(seq_index, data)

    def finalize(self) -> None:
        """
        Explicitly finalize the file, this is not needed if you use `with` statement.
        """
        return self._chunker.finalize()

    def rollback(self) -> None:
        return self._chunker.rollback()


def _create_chunker(
    file: FileLikeObject,
    *,
    readonly: bool = True,
    append: bool | None = None,
    uri: str | None = None,
    lastModified: datetime.datetime | None = None,
    chunker_kwargs: dict = {},
) -> BaseChunker:

    store: Store|None = None
    if isinstance(file, Store):
        store = file

    elif isinstance(file, (str, Path)):
        store = FileStore(file)

    elif isinstance(file, memoryview):
        assert readonly, "Writing memory store is not supported."
        store = MemoryStore(file, uri=uri, lastModified=lastModified)

    else:
        raise ValueError(
            f"argument 'file' expected to be 'str|Path|Store|typing.BinaryIO|memoryview' but was '{type(file).__name__}'"
        )

    # if the Store isn't open do it now
    if not store.isOpen:
        if readonly:
            mode = "rb"
        else:
            if append is None:
                append = store.isFile
            mode = "rb+" if append else "wb"
        store.open(mode)

    if is_legacy_jpeg2000_source(store):
        if readonly:
            return LimJpeg2000Chunker(store, **chunker_kwargs)
        raise RuntimeError("Writing legacy JPEG2000 ND2 files is not supported.")

    return LimBinaryIOChunker(store, **chunker_kwargs)

