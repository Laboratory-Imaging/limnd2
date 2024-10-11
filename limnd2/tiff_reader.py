import tifffile
import logging
from pathlib import Path
from typing import Type, Any, Union

import numpy as np
import warnings
import math

from .attributes import ImageAttributes

warnings.filterwarnings("ignore", category=RuntimeWarning, module="tifffile")
logging.getLogger('tifffile').setLevel(logging.ERROR)

class TiffPageMetadata:
    page_index: int
    shape: tuple[int, ...]
    dtype: Type[np.dtype]
    number_of_dimensions: int
    tags: dict[str, Any]
    color_space: str            # color space (e.g., RGB, GRAY, CMYK).
    compression: str            # compression used for the image (e.g., LZW, JPEG, NONE).

    def __init__(self, page: tifffile.TiffPage, page_index: int) -> None:
        self.page_index = page_index
        self.shape = page.shape
        self.dtype = page.dtype
        self.number_of_dimensions = len(page.shape)
        self.tags = {tag.name: tag.value for tag in page.tags.values()}
        self.color_space = page.photometric.name if page.photometric else 'Unknown'
        self.compression = page.compression.name if page.compression else 'None'

    def __str__(self):
        return f"\n\t\tPage number: {self.page_index}, Shape: {self.shape}, Type: {self.dtype}, Color space: {self.color_space}" #+ "".join(f"\n\t\t\t{k}: {v}" for k, v in self.tags.items())

class TiffPagesMetadata:
    pages: list[TiffPageMetadata]
    same_dimensions: bool

    def __init__(self, pages: tifffile.TiffPages):
        self.pages = []
        self.same_dimensions = True

        first_shape = None

        for page_index, page in enumerate(pages):
            page_obj = TiffPageMetadata(page, page_index)
            self.pages.append(page_obj)
            if not first_shape:
                first_shape = page_obj.shape
            
            if self.same_dimensions:
                if first_shape != page_obj.shape:
                    self.same_dimensions = False
    
    def __str__(self):
        return f"\n\tNumber of pages: {len(self.pages)}, same shape: {self.same_dimensions}" + "".join(str(page) for page in self.pages)

class TiffReader:
    path: Path
    pages_metadata: TiffPagesMetadata
    number_of_pages: int

    def __init__(self, path: Union[Path, str]) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        with tifffile.TiffFile(path) as tif:
            self.pages_metadata = TiffPagesMetadata(tif.pages)
        self.number_of_pages = len(self.pages_metadata.pages)
            
    def __str__(self) -> str:
        return str(self.path) + str(self.pages_metadata)
    
    def get_array(self, page_index: int = 0) -> np.array:
        if page_index < 0 or page_index >= self.number_of_pages:
            raise ValueError(f"Invalid page index, requested index: {page_index}, valid indices: 0 - {self.number_of_pages - 1}")
        
        with tifffile.TiffFile(self.path) as tif:
            return tif.pages[page_index].asarray()
    
    def get_nd2_attributes(self, page_index: int = 0):
        if page_index < 0 or page_index >= self.number_of_pages:
            raise ValueError(f"Invalid page index, requested index: {page_index}, valid indices: 0 - {self.number_of_pages - 1}")
        page = self.pages_metadata.pages[page_index]
        shape = page.shape
        bits = page.dtype.itemsize * 8
        return ImageAttributes(
            uiWidth = shape[1],
            uiWidthBytes= shape[1] * (1 if len(shape) <= 2 else shape[2]),
            uiHeight = shape[0],
            uiComp = 1 if len(shape) <= 2 else shape[2],                   
            uiBpcInMemory = bits if bits % 8 == 0 else math.ceil(bits / 8) * 8,   
            uiBpcSignificant = bits,                                                 
            uiSequenceCount = 1,
            uiTileWidth = shape[1],                                  
            uiTileHeight = shape[0],                                             
            uiVirtualComponents = 1 if len(shape) <= 2 else shape[2]
        )
    
