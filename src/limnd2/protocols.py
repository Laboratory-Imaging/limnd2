from __future__ import annotations

import datetime
import functools
from pathlib import Path
from typing import Protocol, Any

from .attributes import ImageAttributes
from .base import (
    FileLikeObject,
    BinaryRleMetadata,
    BinaryRasterMetadata,
    NumpyArrayLike,
)
from .custom_data import RecordedData
from .experiment import ExperimentLevel, WellplateDesc, WellplateFrameInfo
from .file import LimBinaryIOChunker
from .metadata import PictureMetadata
from .results import ResultItem
from .textinfo import ImageTextInfo, AppInfo

class StorageInfoProtocol(Protocol):
    @property
    def filename(self) -> str | None:
        """
        Returns the filename of the file if available, otherwise `None`.
        """
        ...

    @property
    def url(self) -> str | None:
        """
        Returns the URL of the file if available, otherwise `None`.
        """
        ...

    @property
    def sizeOnDisk(self) -> int:
        """
        Returns the number of bytes the file takes on disk.
        """
        ...

    @property
    def lastModified(self) -> datetime.datetime:
        """
        Returns the modify time of the file on disk.
        """
        ...

class ResultsProtocol(Protocol):
    @property
    def storageInfo(self) -> StorageInfoProtocol:
        """
        Returns storage information.

        See [`StorageInfoProtocol`](protocols.md#limnd2.protocols.StorageInfoProtocol) for more information.
        """
        ...

    @property
    def items(self) -> dict[str, ResultItem]:
        """
        Returns a dictionary of all results in the accompanying `.h5` file.

        Each result potentially contains tabular results (tables, graphs, ...) and binary layers.
        """
        ...

class Nd2ReaderProtocol(Protocol):
    """
    Protocol for Nd2Reader instance for reading `.nd2` files and its attributes, metadata, properties, image data and so on.

    Also see [Quickstart](index.md#reading-nd2-files) for an
    example of how to use this class and how to read individual chunks, attributes, metadata and so on.
    """

    @property
    def version(self) -> tuple[int, int]:
        """
        Returns the version of the `.nd2` file as a tuple of two integers.
        """
        ...

    @property
    def storageInfo(self) -> StorageInfoProtocol:
        """
        Returns storage information.

        See [`StorageInfoProtocol`](#limnd2.protocols.StorageInfoProtocol) for more information.
        """
        ...

    @property
    def is3d(self) -> bool:
        """
        Returns `True` if the file contains valid z-stack, otherwise `False`.
        """
        ...

    @property
    def isMono(self) -> bool:
        """
        Returns `True` if the file contains only one component, otherwise `False`.
        """
        ...

    @property
    def isRgb(self) -> bool:
        """
        Returns `True` if the file contains RGB data, otherwise `False`.
        """
        ...

    @property
    def is8bitRgb(self) -> bool:
        """
        Returns `True` if the file contains 8-bit RGB data, otherwise `False`.
        """
        ...

    @property
    def isFloat(self) -> bool:
        """
        Returns `True` if the file data is 32-bit float, otherwise `False`.
        """
        ...

    @property
    def imageAttributes(self) -> ImageAttributes:
        """
        Attribute to get attributes of an `.nd2` file.

        See [`ImageAttributes`](attributes.md#limnd2.attributes.ImageAttributes) class for more information.

        In order to create an instance of `ImageAttributes` class from simple parameters, use [`ImageAttributes.create`](attributes.md#limnd2.attributes.ImageAttributes.create) method.
        """
        ...

    @property
    def pictureMetadata(self) -> PictureMetadata:
        """
        Attribute to get metadata of an `.nd2` file.

        See [`PictureMetadata`](metadata.md#limnd2.metadata.PictureMetadata) class for more information.

        In order to create an instance of `PictureMetadata` class, use [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) class.
        """
        ...

    @property
    def experiment(self) -> ExperimentLevel | None:
        """
        Attribute to get experiments in an `.nd2` file.

        See [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel) class for more information.

        In order to create an instance of `ExperimentLevel` class, use [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) class.
        """
        ...

    @property
    def imageTextInfo(self) -> ImageTextInfo:
        ...

    @property
    def wellplateDesc(self) -> WellplateDesc | None:
        ...

    @property
    def wellplateFrameInfo(self) -> WellplateFrameInfo | None:
        ...

    @property
    def appInfo(self) -> AppInfo:
        ...

    @property
    def software(self) -> str:
        ...

    @property
    def acqFrames(self) -> NumpyArrayLike:
        ...

    @property
    def acqTimes(self) -> NumpyArrayLike | None:
        ...

    @property
    def acqTimes2(self) -> NumpyArrayLike | None:
        ...

    @property
    def compFrameRange(self) -> NumpyArrayLike:
        ...

    @property
    def compRange(self) -> NumpyArrayLike:
        ...

    @property
    def imageDataRange(self) -> tuple[int, int]:
        ...

    @property
    def recordedData(self) -> RecordedData:
        ...

    @property
    def binaryRleMetadata(self) -> BinaryRleMetadata:
        ...

    @property
    def binaryRasterMetadata(self) -> BinaryRasterMetadata:
        ...

    def dimensionSizes(self, skipSpectralLoop: bool = True) -> dict[str, int]:
        """
        Returns a dictionary with dimension names as keys and their sizes as values.
        """
        ...

    def generateLoopIndexes(self, named: bool = False) -> list:
        """
        Generates indexes for all loops in the experiment.
        """
        ...

    def chunk(self, name: bytes | str) -> bytes | memoryview | None:
        """
        Returns data for specific chunk name

        Parameters
        ----------
        name : bytes|str
            Name of the chunk to retrieve.
        """
        ...

    def image(self, seqindex: int, rect: tuple[int, int, int, int] | None = None) -> NumpyArrayLike:
        """
        Get image data from specified frame as NumPy array.

        Parameters
        ----------
        seqindex: int
            Image sequence index you want to get.
        rect: tuple[int, int, int, int]|None
            Rectangle (x, y, w, h) of the image to get image to get.
        """
        ...

    def downsampledImage(self, seqindex: int, downsize: int, rect: tuple[int, int, int, int] | None = None) -> NumpyArrayLike:
        ...

    def binaryRasterData(self, bin_id: int, seqindex: int, rect: tuple[int, int, int, int] | None = None) -> NumpyArrayLike:
        ...

    def downsampledBinaryRasterData(self, bin_id: int, seqindex: int, downsize: int, rect: tuple[int, int, int, int] | None = None) -> NumpyArrayLike:
        ...

    @property
    def results(self) -> ResultsProtocol|None:
        """
        Returns a ResultsProtocol with all results in the accompanying `.h5` file.

        Each result potentially contains tabular results (tables, graphs, ...) and binary layers.
        """
        ...

class Nd2WriterProtocol(Protocol):
    """
    Experimental ND2 file writer.

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

    def create_chunker(self, *args, **kwargs) -> LimBinaryIOChunker:
        ...

    def __init__(self, file: FileLikeObject, *, append: bool | None = None, chunker_kwargs: dict = {}) -> None:
        """
        Parameters
        -----------
        file : str | Path | int | typing.BinaryIO
            Filename of the ND2 file.
        chunker_kwargs
            Additional parameters for chunker.
        """
        ...

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...

    @property
    def imageAttributes(self) -> ImageAttributes:
        """
        Attribute to get or set attributes of an `.nd2` file.

        See [`ImageAttributes`](attributes.md#limnd2.attributes.ImageAttributes) class for more information.

        In order to create an instance of `ImageAttributes` class from simple parameters, use [`ImageAttributes.create`](attributes.md#limnd2.attributes.ImageAttributes.create) method.
        """
        ...

    @imageAttributes.setter
    def imageAttributes(self, val: ImageAttributes) -> None:
        ...

    @property
    def experiment(self) -> ExperimentLevel:
        """
        Attribute to get or set experiments in an `.nd2` file.

        See [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel) class for more information.

        In order to create an instance of `ExperimentLevel` class, use [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) class.
        """
        ...

    @experiment.setter
    def experiment(self, val: ExperimentLevel) -> None:
        ...

    @property
    def pictureMetadata(self) -> PictureMetadata:
        """
        Attribute to get or set metadata of an `.nd2` file.

        See [`PictureMetadata`](metadata.md#limnd2.metadata.PictureMetadata) class for more information.

        In order to create an instance of `PictureMetadata` class, use [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) class.
        """
        ...

    @pictureMetadata.setter
    def pictureMetadata(self, val: PictureMetadata) -> None:
        ...

    @property
    def chunker(self):
        ...

    def setChunk(self, name: bytes | str, data: bytes | memoryview) -> None:
        ...

    def setImage(self, seq_index: int, data: NumpyArrayLike) -> None:
        """
        Seta image data using specified frame index.

        !!! warning
            You must manually keep track of the frame index and make sure that you are not
            overwriting the same frame multiple times and that images are written sequentially.
        """
        ...

    def finalize(self) -> None:
        """
        Explicitly finalize the file, this is not needed if you use `with` statement.
        """
        ...

    def rollback(self) -> None:
        ...
