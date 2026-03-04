from __future__ import annotations

from contextlib import contextmanager
import copy
import itertools
import math
from pathlib import Path

import numpy as np
import zarr

from limnd2.attributes import ImageAttributes, ImageAttributesPixelType

from .LimConvertUtils import ConversionSettings, ConvertSequenceArgs, logprint
from .LimImageSource import LimImageSource

_COMMON_FF_HINT = (
    '[commonff] extra not installed. Install it with `pip install "limnd2[commonff]"`.'
)


def _missing_convert_dependency(package: str) -> ImportError:
    msg = (
        f'Missing optional dependency "{package}" required for TIFF conversion. '
        f"{_COMMON_FF_HINT}"
    )
    return ImportError(msg)


def _require_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise _missing_convert_dependency("tifffile") from exc
    return tifffile


def _require_ome_types():
    try:
        import ome_types
    except ImportError as exc:
        raise _missing_convert_dependency("ome-types") from exc
    return ome_types


tifffile = _require_tifffile()

_SAMPLE_DIMS = (1, 2, 3, 4)


def _height_width_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        if shape[-1] in _SAMPLE_DIMS:
            return shape[0], shape[1]
        if shape[0] in _SAMPLE_DIMS:
            return shape[1], shape[2]
        return shape[-2], shape[-1]
    if shape[-1] in _SAMPLE_DIMS:
        return shape[-3], shape[-2]
    return shape[-2], shape[-1]


def _components_from_shape(shape: tuple[int, ...]) -> int:
    if len(shape) <= 2:
        return 1
    if len(shape) == 3:
        if shape[-1] in _SAMPLE_DIMS:
            return int(shape[-1])
        if shape[0] in _SAMPLE_DIMS:
            return int(shape[0])
        return 1
    if shape[-1] in _SAMPLE_DIMS:
        return int(shape[-1])
    if shape[-3] in _SAMPLE_DIMS:
        return int(shape[-3])
    return 1


def _normalize_to_yxs(tile: np.ndarray) -> np.ndarray:
    """Normalize to (Y,X) or (Y,X,S)."""
    if tile.ndim == 2:
        return tile
    if tile.ndim == 3:
        # planar (S,Y,X) -> (Y,X,S)
        if tile.shape[0] in _SAMPLE_DIMS and tile.shape[-1] not in _SAMPLE_DIMS:
            tile = np.moveaxis(tile, 0, -1)
        return tile
    raise ValueError(f"Unsupported ndim {tile.ndim} for shape {tile.shape}")


def _tile_index(arr, xx0: int, xx1: int, yy0: int, yy1: int):
    """
    Index ROI without exploding leading dims:
      - (Y,X)
      - (Y,X,S)
      - (S,Y,X)
      - (...,Y,X) and (...,Y,X,S) -> pins leading dims to 0
    """
    shape = arr.shape
    ndim = len(shape)

    if ndim == 2:
        return (slice(yy0, yy1), slice(xx0, xx1))

    if ndim == 3:
        # (Y,X,S)
        if shape[-1] in _SAMPLE_DIMS:
            return (slice(yy0, yy1), slice(xx0, xx1), slice(None))
        # (S,Y,X)
        if shape[0] in _SAMPLE_DIMS:
            return (slice(None), slice(yy0, yy1), slice(xx0, xx1))
        # fallback: treat as (Z,Y,X) and pick Z=0
        return (0, slice(yy0, yy1), slice(xx0, xx1))

    # ndim >= 4
    if shape[-1] in _SAMPLE_DIMS:
        # (..., Y, X, S)
        lead = (0,) * (ndim - 4)
        return lead + (slice(yy0, yy1), slice(xx0, xx1), slice(None))

    # (..., Y, X)
    lead = (0,) * (ndim - 3)
    return lead + (slice(yy0, yy1), slice(xx0, xx1))


def _slice_yx(arr, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    idx = _tile_index(arr, x0, x1, y0, y1)
    tile = np.asarray(arr[idx])
    tile = _normalize_to_yxs(tile)
    if tile.ndim == 3 and tile.shape[-1] == 3:
        tile = tile[..., ::-1]
    return tile


class LimImageSourceTiffBase(LimImageSource):
    """Image source reading from TIFF (supports ROI reads)."""

    idf: int
    channel_index: int | None
    _idf_page_keys: list[int] | None

    def __init__(self, filename: str | Path, idf: int = 0, channel_index: int | None = None):
        super().__init__(filename)
        self.idf = idf
        self.channel_index = int(channel_index) if channel_index is not None else None
        # Optional mapping from logical frame index to concrete TIFF page index.
        self._idf_page_keys = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filename}, {self.idf}, channel_index={self.channel_index})"

    @staticmethod
    def _read_image_description(path: str | Path) -> str | None:
        try:
            with tifffile.TiffFile(path) as tif:
                desc_tag = tif.pages[0].tags.get("ImageDescription")
                if desc_tag is None:
                    return None
                value = desc_tag.value
        except Exception:
            return None

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value
        return str(value)

    @classmethod
    def has_ome_metadata(cls, path: str | Path) -> bool:
        text = cls._read_image_description(path)
        if not text:
            return False
        text = text.lstrip()
        return (
            (text.startswith("<?xml") and "<OME" in text)
            or "http://www.openmicroscopy.org/Schemas/OME" in text
        )

    @classmethod
    def has_meta_metadata(cls, path: str | Path) -> bool:
        text = cls._read_image_description(path)
        if not text:
            return False
        return "<MetaData>" in text and "<PlaneInfo>" in text and "<prop" in text

    @property
    def additional_information(self) -> dict:
        if self._additional_dimensions is None:
            self._additional_dimensions = self._calculate_additional_information()
        return copy.deepcopy(self._additional_dimensions)

    @property
    def supports_tile_read(self) -> bool:
        return True

    def _page_key_for_idf(self, idf: int) -> int:
        idx = int(idf)
        if self._idf_page_keys is None:
            return idx
        if idx < 0 or idx >= len(self._idf_page_keys):
            raise IndexError(
                f"TIFF page index {idx} is out of range for mapped pages ({len(self._idf_page_keys)})."
            )
        return int(self._idf_page_keys[idx])

    def read(self) -> np.ndarray:
        page_key = self._page_key_for_idf(self.idf)
        with tifffile.TiffFile(self.filename) as tif:
            arr = tif.asarray(key=page_key)

        arr = _normalize_to_yxs(np.asarray(arr))
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[..., ::-1]
        return arr

    def read_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        with self.open_tile_reader() as r:
            return np.array(r(x, y, w, h), copy=True)

    @contextmanager
    def open_tile_reader(self):
        """
        Open TIFF and return a callable read_tile(x,y,w,h)->ndarray
        without reopening TIFF / rebuilding Zarr per tile.
        """
        tif = tifffile.TiffFile(self.filename)
        store = None
        try:
            page_key = self._page_key_for_idf(self.idf)
            page = tif.pages[page_key]

            # Fast path: memmap when possible
            try:
                arr = tifffile.memmap(self.filename, page=page_key)
            except Exception:
                store = page.aszarr(chunkmode="strile")
                root = zarr.open(store, mode="r")
                # unwrap groups
                arr = root
                while not hasattr(arr, "shape"):
                    key = "0" if "0" in arr else sorted(arr.keys())[0]
                    arr = arr[key]

            height, width = _height_width_from_shape(arr.shape)

            def _read_tile(x: int, y: int, w: int, h: int) -> np.ndarray:
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + int(w), y0 + int(h)

                xx0, xx1 = max(0, x0), min(width, x1)
                yy0, yy1 = max(0, y0), min(height, y1)

                if xx0 >= xx1 or yy0 >= yy1:
                    if len(arr.shape) >= 3 and (arr.shape[-1] in _SAMPLE_DIMS or arr.shape[0] in _SAMPLE_DIMS):
                        sample_count = arr.shape[-1] if arr.shape[-1] in _SAMPLE_DIMS else arr.shape[0]
                        return np.empty((0, 0, sample_count), dtype=arr.dtype)
                    return np.empty((0, 0), dtype=arr.dtype)

                return _slice_yx(arr, xx0, xx1, yy0, yy1)

            yield _read_tile

        finally:
            try:
                tif.close()
            finally:
                if store is not None:
                    try:
                        store.close()
                    except Exception:
                        pass

    def get_file_dimensions(self) -> dict[str, int]:
        file_info = self.additional_information
        new_dimensions: dict[str, int] = {}
        for dimension_type, count in zip(file_info["axis_parsed"], file_info["shape"]):
            new_dimensions[dimension_type] = count
        return new_dimensions

    def _calculate_additional_information(self) -> dict:
        """
        Map axes character codes to dimension names (from tifffile source code):
        - X : width          (image width)
        - Y : height         (image length)
        - Z : depth          (image depth)
        - S : sample         (color space and extra samples)
        - I : sequence       (generic sequence of images, frames, planes, pages)
        - T : time           (time series)
        - C : channel        (acquisition path or emission wavelength)
        - A : angle          (OME)
        - P : phase          (OME. In LSM, P maps to position)
        - R : tile           (OME. Region, position, or mosaic)
        - H : lifetime       (OME. Histogram)
        - E : lambda         (OME. Excitation wavelength)
        - Q : other          (OME)
        - L : exposure       (FluoView)
        - V : event          (FluoView)
        - M : mosaic         (LSM 6)
        - J : column         (NDTiff)
        - K : row            (NDTiff)
        """

        grouped_axis_categories = {
            ("multipoint", "m"): "RPMJK",
            ("timeloop", "t"): "TVL",
            ("zstack", "z"): "ZA",
            ("channel", "c"): "CSEH",
            ("unknown", "unknown"): "IQ",
        }

        dimensional_axis = "XY"

        axis_category_by_letter = {
            letter: (full, short)
            for (full, short), letters in grouped_axis_categories.items()
            for letter in letters
        }

        result = copy.deepcopy(LimImageSource.DEFAULT_INFORMATION_TEMPLATE)

        with tifffile.TiffReader(self.filename) as tiff:
            pm_name = tiff.pages[0].photometric.name

            if pm_name in ("RGB", "YCBCR"):
                result["is_rgb"] = True
                self._is_rgb = True
            else:
                result["is_rgb"] = False
                self._is_rgb = False

            if len(tiff.series) == 1:
                self._idf_page_keys = None
                for axis, size in zip(tiff.series[0].axes, tiff.series[0].shape):
                    if axis not in axis_category_by_letter and axis not in dimensional_axis:
                        raise ValueError(f"Error: {axis} is not a known axis type.")
                    if axis in dimensional_axis:
                        continue

                    full_category, _ = axis_category_by_letter[axis]
                    if full_category == "channel" and self.is_rgb:
                        continue
                    LimImageSource.update_axis_result(result, full_category, size)
            else:
                # Allow multi-series TIFF file only if all series have same resolution
                resolutions = [s.shape[-2:] for s in tiff.series if len(s.shape) >= 2]
                same_res = all(r == resolutions[0] for r in resolutions)
                if same_res:
                    self._idf_page_keys = None
                    total_images = sum(len(s.pages) for s in tiff.series)
                    LimImageSource.update_axis_result(result, "unknown", total_images)
                else:
                    candidates = [s for s in tiff.series if len(s.shape) >= 2 and len(s.pages) > 0]
                    if not candidates:
                        raise ValueError("Multi-page TIFF file does not contain readable image series.")

                    primary = max(
                        candidates,
                        key=lambda s: (int(s.shape[-2]) * int(s.shape[-1]), len(s.pages)),
                    )
                    self._idf_page_keys = [
                        int(getattr(page, "index", idx))
                        for idx, page in enumerate(primary.pages)
                    ]
                    LimImageSource.update_axis_result(result, "unknown", len(self._idf_page_keys))
                    logprint(
                        "Multi-page TIFF has mixed resolutions. Using primary-resolution series only.",
                        type="warning",
                    )
        return result

    @property
    def is_rgb(self) -> bool:
        if self._is_rgb is None:
            with tifffile.TiffReader(self.filename) as tiff:
                pm_name = tiff.pages[0].photometric.name
                self._is_rgb = pm_name in ("RGB", "YCBCR")
        return self._is_rgb

    def parse_additional_dimensions(
        self,
        sources: dict["LimImageSourceTiffBase", list[int]],
        original_dimensions: dict[str, int],
        unknown_dimension_type: str | None = None,
        preserve_duplicate_dimension_names: bool = False,
        respect_per_file_channel_count: bool = False,
    ) -> tuple[dict["LimImageSourceTiffBase", list[int]], dict[str, int]]:
        """
        Parse additional dimensions from the TIFF source.
        Adds IDF to tiff file based on dimensions found in the TIFF file.
        Returns a list of new sources and a dictionary of new dimensions.
        """
        new_files: dict[LimImageSourceTiffBase, list[int]] = {}
        new_dimension = self.get_file_dimensions()

        if not new_dimension:
            return sources, original_dimensions

        ranges = [range(r) for r in new_dimension.values()]
        index_tuples = list(itertools.product(*ranges))
        idf_values: list[int]
        if (
            len(new_dimension) == 1
            and "unknown" in new_dimension
            and self._idf_page_keys is not None
            and len(self._idf_page_keys) >= len(index_tuples)
        ):
            idf_values = self._idf_page_keys[: len(index_tuples)]
        else:
            idf_values = list(range(len(index_tuples)))

        index_to_idf = [(idx_tuple, idf_values[idf]) for idf, idx_tuple in enumerate(index_tuples)]
        channel_axis_index = list(new_dimension.keys()).index("channel") if "channel" in new_dimension else None

        for file, dims in sources.items():
            for ome_dims, idf in index_to_idf:
                file_copy = copy.deepcopy(file)
                file_copy.idf = idf
                if channel_axis_index is not None:
                    file_copy.channel_index = int(ome_dims[channel_axis_index])
                else:
                    file_copy.channel_index = None
                new_files[file_copy] = dims + list(ome_dims)

        new_dims = original_dimensions.copy()
        for dim, size in new_dimension.items():
            target_dim = dim
            if preserve_duplicate_dimension_names and target_dim in new_dims:
                suffix_index = 2
                while f"{dim}__dup{suffix_index}" in new_dims:
                    suffix_index += 1
                target_dim = f"{dim}__dup{suffix_index}"
            new_dims[target_dim] = size

        if new_dims.get("unknown", 0) > 1:
            if unknown_dimension_type is None:
                logprint(
                    "WARNING: File contains unknown dimension, but no unknown dimension type was provided. "
                    "This may lead to data loss.",
                    type="warning",
                )
            else:
                target_dim = unknown_dimension_type
                if target_dim in new_dims and target_dim != "unknown":
                    suffix_index = 2
                    while f"{target_dim}__dup{suffix_index}" in new_dims:
                        suffix_index += 1
                    target_dim = f"{target_dim}__dup{suffix_index}"

                remapped_dims: dict[str, int] = {}
                for dim_name, size in new_dims.items():
                    if dim_name == "unknown":
                        remapped_dims[target_dim] = size
                    else:
                        remapped_dims[dim_name] = size
                new_dims = remapped_dims

        return new_files, new_dims

    @staticmethod
    def calculate_bpc_significant(
        numpy_bits_memory: int,
        tiff_bits_memory: int | tuple,
        max_sample_value: int | tuple,
    ) -> int:
        """
        Calculate significant bits from TIFF tags.
        """

        if isinstance(tiff_bits_memory, tuple):
            if len(set(tiff_bits_memory)) != 1:
                raise ValueError(f"TIFF bits memory is a tuple with different values: {tiff_bits_memory}.")
            tiff_bits_memory = tiff_bits_memory[0]

        if isinstance(max_sample_value, tuple):
            if len(set(max_sample_value)) != 1:
                raise ValueError(f"Max sample value is a tuple with different values: {max_sample_value}.")
            max_sample_value = max_sample_value[0]

        if numpy_bits_memory != tiff_bits_memory:
            raise ValueError(f"Mismatch between numpy dtype bits ({numpy_bits_memory}) and TIFF bits ({tiff_bits_memory}).")

        if max_sample_value <= 0:
            return tiff_bits_memory

        bpc_significant = max_sample_value.bit_length()
        if bpc_significant <= 8 and tiff_bits_memory > 8:
            return tiff_bits_memory
        return bpc_significant

    def nd2_attributes(self, *, sequence_count=1) -> ImageAttributes:
        with tifffile.TiffReader(self.filename) as tiff:
            page = tiff.pages[self.idf]

            shape = page.shape
            dtype = page.dtype
            tags = {tag.name: tag.value for tag in page.tags.values()}

        height, width = _height_width_from_shape(shape)
        components = int(tags.get("SamplesPerPixel", 0) or 0)
        if components <= 0:
            components = _components_from_shape(shape)
        if components not in _SAMPLE_DIMS:
            components = _components_from_shape(shape)
        numpy_bits = dtype.itemsize * 8

        tiff_bits = tags.get("BitsPerSample", numpy_bits)
        max_value = tags.get("MaxSampleValue", -1)

        bpc_significant = self.calculate_bpc_significant(numpy_bits, tiff_bits, max_value)

        bpc_memory = bpc_significant if bpc_significant % 8 == 0 else math.ceil(bpc_significant / 8) * 8
        width_bytes = ImageAttributes.calcWidthBytes(width, bpc_memory, components)

        if dtype in (np.int8, np.int16, np.int32):
            pixel_type = ImageAttributesPixelType.pxtSigned
        elif dtype in (np.uint8, np.uint16, np.uint32):
            pixel_type = ImageAttributesPixelType.pxtUnsigned
        elif dtype in (np.float16, np.float32, np.float64):
            pixel_type = ImageAttributesPixelType.pxtReal
        else:
            raise ValueError("TIFF file has unsupported pixel type.")

        return ImageAttributes(
            uiWidth=width,
            uiWidthBytes=width_bytes,
            uiHeight=height,
            uiComp=components,
            uiBpcInMemory=bpc_memory,
            uiBpcSignificant=bpc_significant,
            uiSequenceCount=sequence_count,
            uiTileWidth=width,
            uiTileHeight=height,
            uiVirtualComponents=components,
            ePixelType=pixel_type,
        )

    def parse_additional_metadata(self, metadata_storage: ConvertSequenceArgs | ConversionSettings):
        return

    def metadata_as_pattern_settings(self) -> dict:
        return {}
