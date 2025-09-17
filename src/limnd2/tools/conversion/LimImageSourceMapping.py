from __future__ import annotations

import enum
import os
from pathlib import Path
import re


from .LimImageSourceJpeg import LimImageSourceJpeg
from .LimImageSourcePng import LimImageSourcePng
from .LimImageSourceTiff import LimImageSourceTiff

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .LimImageSource import LimImageSource

class ImageFormat(enum.Enum):
    TIFF = enum.auto()
    PNG = enum.auto()
    JPEG = enum.auto()

EXTENSION_MAP = {
    ImageFormat.TIFF: [".tiff", ".tif", ".btf"],
    ImageFormat.PNG: [".png"],
    ImageFormat.JPEG: [".jpeg", ".jpg"]
}

EXTENSION_TO_FORMAT = {ext: fmt for fmt, exts in EXTENSION_MAP.items() for ext in exts}

READER_CLASS_MAP = {
    ImageFormat.TIFF: LimImageSourceTiff,
    ImageFormat.PNG: LimImageSourcePng,
    ImageFormat.JPEG: LimImageSourceJpeg
}

def open_lim_image_source(filename: str | Path) -> LimImageSource:
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist.")
    if not filename.is_file():
        raise FileNotFoundError(f"{filename} is not a file.")

    extension = filename.suffix.lower()
    if extension in EXTENSION_TO_FORMAT:
        image_format = EXTENSION_TO_FORMAT[extension]
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Supported extensions are: {', '.join(EXTENSION_TO_FORMAT.keys())}.")

    image_source_class = READER_CLASS_MAP[image_format]
    return image_source_class(filename)

def image_format_from_regexp(regexp_str: re.Pattern | str) -> ImageFormat | None:
    if isinstance(regexp_str, re.Pattern):
        regexp_str = regexp_str.pattern

    path = regexp_str.removesuffix(".*")           # remove the last .* from the regexp string
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in EXTENSION_TO_FORMAT:
        return EXTENSION_TO_FORMAT[ext]
    elif ext == "":
        raise ValueError(f"File extension not found. Make sure file extension is part of regular expression matching files.")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are: {', '.join(EXTENSION_TO_FORMAT.keys())}.")

