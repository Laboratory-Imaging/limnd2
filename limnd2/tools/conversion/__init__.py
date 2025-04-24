__author__ = "Laboratory Imaging, s.r.o."
__email__ = "techsupp@lim.cz"
__all__ = [
    "FileCrawler", "TiffReader", "tiff_to_NIS"
]

from .crawler import FileCrawler
from .tiff_reader import TiffReader
from .tiff_to_NIS import tiff_to_NIS