from __future__ import annotations

import enum
import importlib
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .LimImageSource import LimImageSource

class ImageFormat(enum.Enum):
    TIFF = enum.auto()
    LSM = enum.auto()
    CZI = enum.auto()
    OIF_OIB = enum.auto()
    PNG = enum.auto()
    JPEG = enum.auto()
    ND2 = enum.auto()

_FORMAT_REGISTRY: dict[ImageFormat, dict[str, object]] = {
    ImageFormat.TIFF: {
        "module": ".LimImageSourceTiff",
        "class": "LimImageSourceTiff",
        "extensions": [".tiff", ".tif", ".btf"],
        "extra": "commonff",            # specify which extra dependency is needed for given format
    },
    ImageFormat.LSM: {
        "module": ".LimImageSourceLsm",
        "class": "LimImageSourceLsm",
        "extensions": [".lsm"],
        "extra": "commonff",
    },
    ImageFormat.CZI: {
        "module": ".LimImageSourceCzi",
        "class": "LimImageSourceCzi",
        "extensions": [".czi"],
        "extra": "czi",
    },
    ImageFormat.OIF_OIB: {
        "module": ".LimImageSourceOifOib",
        "class": "LimImageSourceOifOib",
        "extensions": [".oib", ".oif"],
        "extra": "olympus",
    },
    ImageFormat.PNG: {
        "module": ".LimImageSourcePng",
        "class": "LimImageSourcePng",
        "extensions": [".png"],
        "extra": "commonff",
    },
    ImageFormat.JPEG: {
        "module": ".LimImageSourceJpeg",
        "class": "LimImageSourceJpeg",
        "extensions": [".jpeg", ".jpg"],
        "extra": "commonff",
    },
    ImageFormat.ND2: {
        "module": ".LimImageSourceNd2",
        "class": "LimImageSourceNd2",
        "extensions": [".nd2"],
        "extra": None,
    },
}

EXTENSION_MAP: dict[ImageFormat, list[str]] = {}
READER_CLASS_MAP: dict[ImageFormat, type["LimImageSource"]] = {}
_OPTIONAL_EXTENSION_TO_EXTRA: dict[str, str] = {}

for fmt, config in _FORMAT_REGISTRY.items():
    extensions = config["extensions"]  # type: ignore[assignment]
    extra = config["extra"]  # type: ignore[assignment]
    if extra:
        for ext in extensions:
            _OPTIONAL_EXTENSION_TO_EXTRA[ext] = extra

    try:
        module = importlib.import_module(config["module"], package=__package__)
        reader_cls = getattr(module, config["class"])
    except Exception:
        continue

    EXTENSION_MAP[fmt] = list(extensions)
    READER_CLASS_MAP[fmt] = reader_cls

EXTENSION_TO_FORMAT = {
    ext: fmt for fmt, exts in EXTENSION_MAP.items() for ext in exts
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
    elif extension in _OPTIONAL_EXTENSION_TO_EXTRA:
        extra = _OPTIONAL_EXTENSION_TO_EXTRA[extension]
        raise ImportError(
            f'Loading "{extension}" files requires the optional extra '
            f'"{extra}". Install it with `pip install "limnd2[{extra}]"`.'
        )
    else:
        supported = ", ".join(EXTENSION_TO_FORMAT.keys()) or "none"
        raise ValueError(
            f"Unsupported file extension: {extension}. Supported extensions are: {supported}."
        )

    try:
        image_source_class = READER_CLASS_MAP[image_format]
    except KeyError:
        extra = _FORMAT_REGISTRY[image_format]["extra"]
        if extra:
            raise ImportError(
                f'The reader for "{image_format.name}" is unavailable. Install '
                f'`limnd2[{extra}]` to enable it.'
            ) from None
        raise ImportError(
            f'The reader for "{image_format.name}" is unavailable.'
        ) from None

    return image_source_class(filename)

def image_format_from_regexp(regexp_str: re.Pattern | str) -> ImageFormat | None:
    if isinstance(regexp_str, re.Pattern):
        regexp_str = regexp_str.pattern

    path = regexp_str.removesuffix(".*")           # remove the last .* from the regexp string
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in EXTENSION_TO_FORMAT:
        return EXTENSION_TO_FORMAT[ext]
    elif ext in _OPTIONAL_EXTENSION_TO_EXTRA:
        extra = _OPTIONAL_EXTENSION_TO_EXTRA[ext]
        raise ImportError(
            f'Reading "{ext}" sources requires the optional extra '
            f'"{extra}". Install it with `pip install "limnd2[{extra}]"`.'
        )
    elif ext == "":
        raise ValueError("File extension not found. Make sure the extension is part of the regular expression.")
    else:
        supported = ", ".join(EXTENSION_TO_FORMAT.keys()) or "none"
        raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are: {supported}.")
