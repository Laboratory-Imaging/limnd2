from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import json
from pathlib import Path
from copy import deepcopy
from dataclasses import is_dataclass, asdict
import re

import numpy as np
import sys
from limnd2.attributes import ImageAttributes

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
    from limnd2.tools.conversion.LimConvertUtils import ConvertSequenceArgs


class LimImageSource(ABC):
    """
    Abstract class for reading images of different formats.
    Currently supported formats are:

    - TIFF and its subformats (`.ome.tiff`, `.btf`, `.tif`, `.tiff`)
    - PNG (`.png`)
    - JPEG (`.jpg`, `.jpeg`)

    This class provides a common interface for reading images and converting them to ND2 format,
    as well as for extracting additional information from the images.

    !!! warning
        This class is not intended to be used directly, but rather as a base class for specific image source classes.
        In order to correctly create object of this class, **use the [`open()`](convert_image_source.md#limnd2.tools.conversion.LimImageSource.LimImageSource.open)
        method**, which will automatically detect the file type and create the appropriate LimImageSource object.

    """
    filename: Path
    supports_streaming_read: bool = False
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
        # Get additional dimensions from the image source (useful for multipage and OME tiff files).
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

    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSource", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSource", list[int]], dict[str, int]]:

        # Parse additional dimensions from the image source.
        # In OME TIFF file, this will parse OME dimensions within file.
        # In multipage file, this will parse unknown dimension in the file.
        # In other file formats, this will do nothing.

        return sources, original_dimensions

    @abstractmethod
    def nd2_attributes(self, *, sequence_count = 1) -> ImageAttributes:
        """
        Get ND2 attributes from the image source.

        !!! note
            If you use those attributes when converting file sequence to ND2,
            you need to replace the sequence count with the number of files in the sequence.
            You must also set component count if you use convert multichannel ND2 image.

        """
        raise NotImplementedError("nd2_attributes not implemented for this abstract image source.")

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):

        # Parse additional metadata from the image source.
        # In OME-tiff file this will parse OME metadata and add it to the metadata_storage.
        # In other file formats this will do nothing.
        return

    def metadata_as_pattern_settings(self) -> dict:
        """
        Return metadata as a dictionary for internal usage in QML.
        """
        return {}

    @staticmethod
    def open(filename: str | Path) -> LimImageSource:
        """
        Create a LimImageSource object from a filename.
        This will automatically detect the file type and create the appropriate LimImageSource object (`LimImageSourceTiff`, `LimImageSourcePng`, ...).
        """

        from limnd2.tools.conversion.LimImageSourceMapping import open_lim_image_source

        return open_lim_image_source(filename)


def _single_file_channel_label_sort_key(label: str) -> tuple:
    text = str(label).strip()
    if text == "":
        return (3, "")

    text = re.sub(r"[XY]+$", "", text).strip()
    if "-" in text:
        token, chan = text.split("-", 1)
        token = token.strip()
        chan = chan.strip()

        if token.isdigit():
            token_key = (0, int(token))
        else:
            m = re.search(r"(\d+)$", token)
            token_key = (1, token.casefold(), int(m.group(1)) if m else -1)

        chan_idx = re.match(r"(?i)^channel[_ ]?(\d+)$", chan)
        if chan_idx:
            chan_key = (0, int(chan_idx.group(1)))
        else:
            first_num = re.search(r"(\d+)", chan)
            if first_num:
                chan_key = (1, int(first_num.group(1)), chan.casefold())
            else:
                chan_key = (2, chan.casefold())

        return (0, token_key, chan_key)

    first_num = re.search(r"(\d+)", text)
    if first_num:
        return (1, int(first_num.group(1)), text.casefold())
    return (2, text.casefold())


def _sort_single_file_qml_channels(qml_settings: dict) -> dict:
    channels = qml_settings.get("channels")
    if not isinstance(channels, list):
        return qml_settings
    indexed = list(enumerate(channels))
    indexed.sort(
        key=lambda item: (
            _single_file_channel_label_sort_key(item[1][0]) if isinstance(item[1], list) and item[1] else (4, ""),
            item[0],
        )
    )
    qml_settings["channels"] = [row for _, row in indexed]
    return qml_settings



def get_file_dimensions_as_json(file_path: Path | None = None):
    if file_path is None:
        file_path = Path(sys.argv[1])

    image_source = LimImageSource.open(file_path)

    try:
        dims: dict[str, int] = image_source.get_file_dimensions() or {}
        result: dict[str, int | bool | str | dict | list] = dict(dims)

        result["is_rgb"] = bool(getattr(image_source, "is_rgb", False))
        result["has_file_info"] = bool(dims)

        if hasattr(image_source, "metadata_as_pattern_settings"):
            try:
                result["qml_settings"] = image_source.metadata_as_pattern_settings() or {}
                result["qml_settings"] = _sort_single_file_qml_channels(result["qml_settings"])
            except Exception:
                result["qml_settings"] = {}
        else:
            result["qml_settings"] = {}

        result["has_qml_settings"] = bool(result["qml_settings"])

        result["error"] = False
        result["error_message"] = ""

    except Exception as e:
        result = {
            "error": True,
            "error_message": str(e),
        }

    print(json.dumps(result, indent=2))

_UNSET_CALIB = -1.0

def _coalesce_number(a_val, b_val):
    return a_val if a_val is not None else b_val

def _plane_name(plane):
    return getattr(plane, "name", "") or ""

def _is_missing_setting(val):
    if val is None:
        return True
    if isinstance(val, (int, float)):
        return val <= 0
    if isinstance(val, str):
        return val.strip() == ""
    return False

def _merge_metadata_factory(a_meta, b_meta, *, keep_planes: bool = True):
    """
    Conservative merge for MetadataFactory:
      - If A.metadata is None → take B.metadata
      - Else (both exist):
          * pixel_calibration: fill from B if A is unset (-1.0/None) and B has a real value
          * other_settings: fill missing/empty values in A from B
          * planes: if A has none → copy all from B
                    else append only planes from B whose names don't exist in A
    """
    if a_meta is None:
        return deepcopy(b_meta) if b_meta is not None else None
    if b_meta is None:
        return a_meta

    # pixel_calibration
    if getattr(a_meta, "pixel_calibration", _UNSET_CALIB) in (None, _UNSET_CALIB):
        b_cal = getattr(b_meta, "pixel_calibration", _UNSET_CALIB)
        if b_cal not in (None, _UNSET_CALIB):
            a_meta.pixel_calibration = b_cal

    # other_settings
    a_settings = getattr(a_meta, "_other_settings", None)
    b_settings = getattr(b_meta, "_other_settings", None)
    if b_settings:
        if a_settings is None:
            a_meta._other_settings = deepcopy(b_settings)
        else:
            for key, value in b_settings.items():
                if _is_missing_setting(value):
                    continue
                if _is_missing_setting(a_settings.get(key, None)):
                    a_settings[key] = deepcopy(value)

    # planes
    if keep_planes:
        a_planes = getattr(a_meta, "planes", None)
        b_planes = getattr(b_meta, "planes", None)

        if not a_planes and b_planes:
            a_meta.planes = deepcopy(b_planes)
        elif a_planes and b_planes:
            existing = {_plane_name(p) for p in a_planes}
            for p in b_planes:
                if _plane_name(p) not in existing:
                    a_planes.append(deepcopy(p))

    return a_meta

def merge_four_fields(a, b):
    """
    Merge only: time_step, z_step, channels, metadata.
    Rules:
      - Keep A as source of truth.
      - For numbers: fill A.<field> from B only if A.<field> is None.
      - For channels: if A.channels already has anything (non-empty), LEAVE IT AS-IS.
                      Only if A.channels is None/empty, copy B.channels.
      - For metadata: if A.metadata is None, take B.metadata; otherwise fill unset
                      pixel_calibration and append planes missing by name.
    Mutates and returns 'a'.
    """
    if b is None:
        return a

    # ---- numbers ----
    if hasattr(a, "time_step") or hasattr(b, "time_step"):
        a.time_step = _coalesce_number(getattr(a, "time_step", None), getattr(b, "time_step", None))

    if hasattr(a, "z_step") or hasattr(b, "z_step"):
        a.z_step = _coalesce_number(getattr(a, "z_step", None), getattr(b, "z_step", None))

    # ---- channels (replace ONLY if A has none) ----
    a_has_channels = False
    if hasattr(a, "channels"):
        a_channels = getattr(a, "channels", None)
        b_channels = getattr(b, "channels", None)

        a_has_channels = bool(a_channels) and isinstance(a_channels, dict) and len(a_channels) > 0
        b_has_channels = bool(b_channels) and isinstance(b_channels, dict) and len(b_channels) > 0

        if not a_has_channels and b_has_channels:
            # Replace entirely (copy) only when A omitted channels
            a.channels = deepcopy(b_channels)
        # else: keep A.channels exactly as-is (do nothing)

    # ---- metadata (safe merge) ----
    if hasattr(a, "metadata"):
        a_meta = getattr(a, "metadata", None)
        b_meta = getattr(b, "metadata", None)
        a.metadata = _merge_metadata_factory(a_meta, b_meta, keep_planes=not a_has_channels)

    return a
