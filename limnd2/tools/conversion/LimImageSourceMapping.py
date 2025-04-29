import enum
import os
from pathlib import Path
import re

from .LimImageSource import LimImageSource
from .LimImageSourceJpeg import LimImageSourceJpeg
from .LimImageSourcePng import LimImageSourcePng
from .LimImageSourceTiff import LimImageSourceTiff

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

def image_format_from_regexp(regexp_str: re.Pattern) -> ImageFormat | None:

    # TODO: possibly improve this so that input regexp doesnt have to contain the file extension,
    # maybe add a flag to the parser saying what type of files to parse (tiff / png / jpeg) and if
    # it is not provided, use the extension from the regexp

    path = regexp_str.pattern.removesuffix(".*")           # remove the last .* from the regexp string
    _, ext = os.path.splitext(path)
    if ext in EXTENSION_TO_FORMAT:
        return EXTENSION_TO_FORMAT[ext]
    elif ext == "":
        raise ValueError(f"File extension not found. Make sure file extension is part of regular expression matching files.")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are: {', '.join(EXTENSION_TO_FORMAT.keys())}.")

