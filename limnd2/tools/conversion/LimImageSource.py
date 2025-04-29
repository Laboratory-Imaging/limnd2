from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from pathlib import Path

import numpy as np
from limnd2.attributes import ImageAttributes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from limnd2.tools.conversion.tiff_to_NIS_argparser import PathParserArgs


class LimImageSource(ABC):
    """Abstract class for reading images from different sources."""
    filename: Path
    _is_rgb: bool
    _additional_dimensions: dict

    def update_axis_result(information_dict: dict, full_category: str, size: int):
        # function to update values in the information dictionary defined below
        if full_category in information_dict["axis_parsed"]:
            raise ValueError(f"Error: {full_category} already exists in axis_parsed.")

        information_dict["axis_parsed"].append(full_category)
        information_dict["shape"].append(size)

    DEFAULT_INFORMATION_TEMPLATE = {
        "axis_parsed": [],
        "shape" : []
    }

    @abstractmethod
    def __init__(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.is_file():
            raise FileNotFoundError(f"{filename} is not a file.")
        self.filename = filename

        #following properties are only calculated when needed
        self._is_rgb = None
        self._additional_dimensions = None

    @abstractmethod
    def read(self) -> np.ndarray:
        """Read the image into numpy array writeable by limnd2 library."""
        raise NotImplementedError("read not implemented for this abstract image source.")


    @property
    def additional_information(self) -> dict:
        """Get additional dimensions from the image source (useful for multipage and OME tiff files)."""
        return copy.deepcopy(self.DEFAULT_INFORMATION_TEMPLATE)

    def _calculate_additional_information(self) -> dict:
        """
        Return additional information from the image source (extra dimensions, checks if image is RGB).

        Used mainly for multipage and OME tiff files.
        In other file formats, this will return default dictionary.
        """
        return {}

    def get_file_dimensions(self) -> dict[str, int]:
        """
        Return dimensions inside just the file.
        In OME TIFF file, this will return OME dimensions within file.
        In multipage file, this will return unknown dimension in the file.
        In other file formats, this will return empty dictionary.
        """
        return {}

    @property
    @abstractmethod
    def is_rgb(self) -> bool:
        """Check if the image is RGB."""
        raise NotImplementedError("is_rgb not implemented for this abstract image source.")

    def parse_additional_dimensions(self, sources: dict[list["LimImageSource"], tuple], original_dimensions: dict[str, int], unknown_dimension_type: str = None) \
        -> tuple[list["LimImageSource"], dict[str, int]]:
        """
        Parse additional dimensions from the image source.
        In OME TIFF file, this will parse OME dimensions within file.
        In multipage file, this will parse unknown dimension in the file.
        In other file formats, this will do nothing.
        """
        return sources, original_dimensions

    @abstractmethod
    def nd2_attributes(self, *, sequence_count = 1) -> ImageAttributes:
        """Get ND2 attributes from the image source."""
        raise NotImplementedError("nd2_attributes not implemented for this abstract image source.")

    def parse_additional_metadata(self, parsed_args: PathParserArgs):
        """
        Parse additional metadata from the image source.
        In OME-tiff file this will parse OME metadata and add it to the parsed_args.
        In other file formats this will do nothing.
        """
        return
