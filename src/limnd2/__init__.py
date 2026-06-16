from importlib.metadata import PackageNotFoundError, version

__author__ = "Laboratory Imaging, s.r.o."
__email__ = "techsupp@lim.cz"

try:
    __version__ = version("limnd2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "__version__",
    "ImageAttributesCompression", "ImageAttributesPixelType", "ImageAttributes",
    "FileStore", "MemoryStore","BaseChunker", "FileLikeObject", "Nd2LoggerEnabled", "NumpyArrayLike", "NameNotInChunkmapError", "NotNd2Format", "UnsupportedChunkmapError", "UnexpectedCallError",
    "BinaryItemStateFlags", "BinaryItemColorMode", "BinaryRleMetadataItem", "BinaryRleMetadata", "BinaryRasterMetadataItem", "BinaryRasterMetadata", "BinaryRasterMetadataFactory",
    "CustomDescription", "RecordedData", "RecordedDataItem", "RecordedDataType",
    "ExperimentLoopType", "ExperimentTimeLoop", "ExperimentNETimeLoop", "ZStackType", "ExperimentZStackLoop", "ExperimentSpectralLoop", "ExperimentXYPosLoop", "WellplateDesc", "WellplateFrameInfoItem", "ExperimentLevel",
    "ExperimentFactory", "WellplateFactory",
    "frameExport", "seriesExport", "metadataAsJSON", "frameToRGB", "frame_to_rgb",
    "to_ome_zarr", "to_ome_types", "to_ome_xml", "to_ome_tiff",
    "gatherImageInformation", "imageInformationAsJSON", "imageInformationAsTXT", "imageInformationAsXLSX", "generalImageInfo",
    "Nd2Reader", "Nd2Writer",
    "ND2File",
    "PictureMetadata",
    "MetadataFactory",
    "ResultItem", "ResultPane", "TableData", "ResultPanesConfiguration",
    "ImageTextInfo",
]

from .attributes import ImageAttributesCompression, ImageAttributesPixelType, ImageAttributes
from .base import FileStore, MemoryStore, BaseChunker, FileLikeObject, NumpyArrayLike, Nd2LoggerEnabled, NameNotInChunkmapError, NotNd2Format, UnsupportedChunkmapError, UnexpectedCallError
from .binary import BinaryItemStateFlags, BinaryItemColorMode, BinaryRleMetadataItem, BinaryRleMetadata, BinaryRasterMetadataItem, BinaryRasterMetadata, BinaryRasterMetadataFactory
from .custom_data import CustomDescription, RecordedData, RecordedDataItem, RecordedDataType
from .experiment import ExperimentLoopType, ExperimentTimeLoop, ExperimentNETimeLoop, ZStackType, ExperimentZStackLoop, ExperimentSpectralLoop, ExperimentXYPosLoop, WellplateDesc, WellplateFrameInfoItem, ExperimentLevel
from .experiment_factory import ExperimentFactory
from .wellplate_factory import WellplateFactory
from .export import frameExport, frameToRGB, frame_to_rgb, seriesExport, metadataAsJSON
from .export_ome_tiff import to_ome_tiff, to_ome_types, to_ome_xml
from .export_ome_zarr import to_ome_zarr
from .image_info import gatherImageInformation, imageInformationAsJSON, imageInformationAsTXT, imageInformationAsXLSX, generalImageInfo
from .nd2 import Nd2Reader, Nd2Writer
from .nd2file import ND2File
from .metadata import PictureMetadata
from .metadata_factory import MetadataFactory
from .results import ResultItem, ResultPane, TableData, ResultPanesConfiguration
from .textinfo import ImageTextInfo
