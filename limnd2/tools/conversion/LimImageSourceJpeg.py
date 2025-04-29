from pathlib import Path
from .LimImageSource import LimImageSource


class LimImageSourceJpeg(LimImageSource):
    """Class for reading images from JPEG files."""

    def __init__(self, filename: str | Path):
        raise NotImplementedError("JPEG source not implemented yet.")
