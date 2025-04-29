from pathlib import Path
from .LimImageSource import LimImageSource


class LimImageSourcePng(LimImageSource):
    """Class for reading images from PNG files."""

    def __init__(self, filename: str | Path):
        raise NotImplementedError("PNG source not implemented yet.")