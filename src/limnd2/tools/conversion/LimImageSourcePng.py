from pathlib import Path

from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.tools.conversion.LimConvertUtils import logprint
from .LimImageSource import LimImageSource

import numpy as np

_COMMON_FF_HINT = (
    '[commonff] extra not installed. Install it with `pip install "limnd2[commonff]"`.'
)


def _missing_convert_dependency(package: str) -> ImportError:
    msg = (
        f'Missing optional dependency "{package}" required for PNG conversion. '
        f"{_COMMON_FF_HINT}"
    )
    return ImportError(msg)


def _require_pillow() -> type["Image"]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise _missing_convert_dependency("Pillow") from exc
    return Image


Image = _require_pillow()


class LimImageSourcePng(LimImageSource):
    """Class for reading images from PNG files."""

    def __init__(self, filename: str | Path):
        super().__init__(filename)

    def read(self) -> np.ndarray:
        """Read the image into numpy array writeable by limnd2 library."""
        with Image.open(self.filename) as img:
            if img.mode in ["L", "P"]:
                return np.array(img, dtype=np.uint8)

            elif img.mode in ["RGB"]:
                return np.array(img, dtype=np.uint8)[..., ::-1]

            elif img.mode in ["RGBA"]:
                # Convert RGBA to RGB by dropping the alpha channel
                img = img.convert("RGB")
                return np.array(img, dtype=np.uint8)[..., ::-1]

            elif img.mode == "I":
                return np.array(img, dtype=np.int32)

            elif img.mode == "F":
                return np.array(img, dtype=np.float32)

            else:
                # default for other modes:
                img = img.convert("RGB")
                return np.array(img, dtype=np.uint8)[..., ::-1]

    @property
    def is_rgb(self) -> bool:
        """Check if the image is RGB."""
        if self._is_rgb is None:
            with Image.open(self.filename) as img:
                if img.mode in ["L", "P", "I", "F"]:
                    self._is_rgb = False

                elif img.mode in ["RGB", "RGBA"]:
                    self._is_rgb = True

                else:
                    # default for other modes (those are converted to RGB):
                    self._is_rgb = True

        return self._is_rgb

    def nd2_attributes(self, *, sequence_count=1):
        """Get the attributes of the image for ND2 file."""
        with Image.open(self.filename) as img:
            if img.mode == "L":
                comps = 1
                bpc = 8
                pixel_type = ImageAttributesPixelType.pxtUnsigned

            elif img.mode == "P":
                logprint("WARNING: PNG file is in palette mode, colors may be incorrect.", type="warning")
                comps = 1
                bpc = 8
                pixel_type = ImageAttributesPixelType.pxtUnsigned

            elif img.mode == "I":
                comps = 1
                bpc = 32
                pixel_type = ImageAttributesPixelType.pxtSigned

            elif img.mode == "F":
                comps = 1
                bpc = 32
                pixel_type = ImageAttributesPixelType.pxtReal

            elif img.mode == "RGB":
                comps = 3
                bpc = 8
                pixel_type = ImageAttributesPixelType.pxtUnsigned

            elif img.mode == "RGBA":
                logprint("WARNING: RGBA PNG file, converting to RGB.", type="warning")
                comps = 3
                bpc = 8
                pixel_type = ImageAttributesPixelType.pxtUnsigned

            else:
                logprint(f"WARNING: Unsorrted PNG file mode: {img.mode}, converting to RGB.", type="warning")
                comps = 3
                bpc = 8
                pixel_type = ImageAttributesPixelType.pxtUnsigned

        width_bytes = ImageAttributes.calcWidthBytes(img.width, bpc, comps)
        return ImageAttributes(
            uiWidth = img.width,
            uiWidthBytes = width_bytes,
            uiHeight = img.height,
            uiComp = comps,
            uiBpcInMemory = bpc,
            uiBpcSignificant = bpc,
            uiSequenceCount = sequence_count,
            uiTileWidth = img.width,
            uiTileHeight = img.height,
            uiVirtualComponents = comps,
            ePixelType = pixel_type
        )

