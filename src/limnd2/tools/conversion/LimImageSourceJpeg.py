from __future__ import annotations

from pathlib import Path

from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.tools.conversion.LimConvertUtils import logprint

from .LimImageSource import LimImageSource

import numpy as np


class LimImageSourceJpeg(LimImageSource):
    """Class for reading images from JPEG files."""

    def __init__(self, filename: str | Path):
        super().__init__(filename)

    def read(self) -> np.ndarray:
        """Read the image into numpy array writeable by limnd2 library."""
        from PIL import Image, ImageOps
        with Image.open(self.filename) as img:
            img = ImageOps.exif_transpose(img)
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
        from PIL import Image, ImageOps
        if self._is_rgb is None:
            with Image.open(self.filename) as img:
                img = ImageOps.exif_transpose(img)
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
        from PIL import Image, ImageOps
        with Image.open(self.filename) as img:
            img = ImageOps.exif_transpose(img)
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

        return ImageAttributes(
            uiWidth = img.width,
            uiWidthBytes = img.width * comps * bpc,
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

    def supposed_orientation(self) -> tuple[int, str]:
        """
        Return the EXIF orientation (1-8) and a human-readable description
        such as 'Rotated 90° CW'.
        """
        ORIENTATION_TAG = 274  # EXIF Orientation

        orientation_map = {
            1: "Normal",
            2: "Mirrored horizontally",
            3: "Rotated 180°",
            4: "Mirrored vertically",
            5: "Rotated 90° CW and mirrored horizontally",
            6: "Rotated 90° CW",
            7: "Rotated 90° CCW and mirrored horizontally",
            8: "Rotated 90° CCW",
        }

        from PIL import Image
        with Image.open(self.filename) as img:
            w, h = img.size
            orientation = 1

            try:
                exif = img.getexif()
                if exif:
                    orientation = int(exif.get(ORIENTATION_TAG, 1))
            except Exception:
                pass

        exif_desc = orientation_map.get(orientation, "Unknown")
        description = f"{exif_desc}"

        return orientation, description

