from __future__ import annotations

import datetime
import io
import os
import struct
import threading
import typing

import numpy as np

from .attributes import (
    ImageAttributes,
    ImageAttributesCompression,
    ImageAttributesPixelType,
    NumpyArrayLike,
)
from .base import (
    BaseChunker,
    BinaryIdNotFountError,
    JP2_MAGIC,
    NameNotInChunkmapError,
    UnexpectedCallError,
    Store
)
from .metadata import PictureMetadata, PictureMetadataPicturePlanes
from .metadata_factory import MetadataFactory

try:
    import imagecodecs
except ModuleNotFoundError:
    imagecodecs = None

try:
    from functools import cached_property
except ImportError:  # pragma: no cover - for very old Python
    cached_property = property  # type: ignore[misc,assignment]


LEGACY_CHUNKMAP_SIGNATURE = b"LABORATORY IMAGING ND BOX MAP 00"

# Trailer at the very end of the file: signature + offset of the chunk map.
STRUCT_TRAILER = struct.Struct("<32sQ")

# Legacy chunk map entry: JP2 box type, LIM/ND2 type, file offset.
STRUCT_MAP_ENTRY = struct.Struct("<4s4sI")

# JP2 box header: big-endian length and 4-byte box type.
STRUCT_BOX_HEADER = struct.Struct(">I4s")

# Correct JP2 Image Header (IHDR) payload: 14 bytes total
# HEIGHT (4) | WIDTH (4) | NC (2) | BPC (1) | C (1) | UnkC (1) | IPR (1)
STRUCT_IHDR = struct.Struct(">IIHBBBB")


def _parse_jpeg2000_codestream_header(payload: bytes) -> dict[str, int]:
    """
    Parse a JPEG 2000 codestream header (SIZ marker) to obtain
    image dimensions and bit depth.

    Assumes the codestream starts with:
        SOC (0xFF4F)
        SIZ (0xFF51)
    which is the baseline JPEG 2000 layout.
    """
    mv = memoryview(payload)

    # Minimum length for SOC + SIZ (marker + Lsiz + base SIZ fields)
    if len(mv) < 2 + 2 + 2 + 36:
        raise RuntimeError("JPEG2000 codestream too short to contain SOC/SIZ header.")

    # SOC marker
    if mv[0] != 0xFF or mv[1] != 0x4F:
        raise RuntimeError("JPEG2000 codestream does not start with SOC marker (0xFF4F).")

    pos = 2

    # SIZ marker
    if mv[pos] != 0xFF or mv[pos + 1] != 0x51:
        marker = (mv[pos] << 8) | mv[pos + 1]
        raise RuntimeError(f"Expected SIZ marker (0xFF51) after SOC, found 0x{marker:04X}.")
    pos += 2

    # Lsiz: length of SIZ segment (includes itself, excludes marker)
    if len(mv) < pos + 2:
        raise RuntimeError("Truncated SIZ marker segment.")
    Lsiz = (mv[pos] << 8) | mv[pos + 1]
    pos += 2

    if len(mv) < pos + (Lsiz - 2):
        raise RuntimeError("Truncated SIZ marker payload.")

    # Parse SIZ fields (see JPEG2000 Part 1 spec)
    # Rsiz (2), Xsiz (4), Ysiz (4), XOsiz (4), YOsiz (4),
    # XTsiz (4), YTsiz (4), XTOsiz (4), YTOsiz (4), Csiz (2)
    Rsiz = (mv[pos] << 8) | mv[pos + 1]
    pos += 2

    def _u32() -> int:
        nonlocal pos
        v = int.from_bytes(mv[pos : pos + 4], "big")
        pos += 4
        return v

    Xsiz = _u32()
    Ysiz = _u32()
    XOsiz = _u32()
    YOsiz = _u32()
    XTsiz = _u32()
    YTsiz = _u32()
    XTOsiz = _u32()
    YTOsiz = _u32()

    Csiz = (mv[pos] << 8) | mv[pos + 1]
    pos += 2

    # Component parameters: 3 bytes per component (Ssiz, XRsiz, YRsiz)
    if len(mv) < pos + 3 * Csiz:
        raise RuntimeError("Truncated component parameters in SIZ marker.")

    # We only need the first component's precision
    Ssiz0 = mv[pos]
    precision_minus_one = Ssiz0 & 0x7F
    bits_per_component = precision_minus_one + 1

    # Image size: width = Xsiz - XOsiz, height = Ysiz - YOsiz
    width = Xsiz - XOsiz
    height = Ysiz - YOsiz

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid JPEG2000 SIZ dimensions parsed from codestream.")

    # We return "compression" as 0 to map to lossless in _build_image_attributes,
    # since true lossy/lossless cannot be inferred from SIZ alone.
    return {
        "rows": height,
        "columns": width,
        "channels": Csiz,
        "bits_per_component": bits_per_component,
        "compression": 0,
    }


def _read_legacy_chunkmap(fh: typing.BinaryIO) -> dict[bytes, list[int]]:
    """
    Read the legacy ND2 chunk map from the end of the file.

    Each entry describes a JP2 box and an ND2/LIM logical type.
    For a few special boxes, we key the map by box_type (e.g. 'jP  ', 'ftyp', 'jp2h'),
    otherwise by the LIM type (e.g. 'LUNK', 'ACAL', 'VCAL', etc.).
    """
    current = fh.tell()
    fh.seek(0, os.SEEK_END)
    file_size = fh.tell()
    if file_size < STRUCT_TRAILER.size:
        raise RuntimeError("Legacy ND2 file is too small to contain a chunk map.")

    # Read trailer
    fh.seek(-STRUCT_TRAILER.size, os.SEEK_END)
    sig, map_start = STRUCT_TRAILER.unpack(fh.read(STRUCT_TRAILER.size))
    if sig != LEGACY_CHUNKMAP_SIGNATURE:
        raise RuntimeError("Missing legacy ND2 chunk map signature.")
    if map_start <= STRUCT_TRAILER.size or map_start > file_size:
        raise RuntimeError("Invalid legacy ND2 chunk map offset.")

    # Jump to the beginning of the chunk map
    fh.seek(-map_start, os.SEEK_END)
    count_bytes = fh.read(4)
    if len(count_bytes) != 4:
        raise RuntimeError("Could not read legacy chunk count.")
    chunk_count = int.from_bytes(count_bytes, "big", signed=False)

    entries_size = chunk_count * STRUCT_MAP_ENTRY.size
    entries = fh.read(entries_size)
    if len(entries) != entries_size:
        raise RuntimeError("Legacy ND2 chunk map is truncated.")

    chunkmap: dict[bytes, list[int]] = {}
    for idx in range(chunk_count):
        box_type, lim_type, offset = STRUCT_MAP_ENTRY.unpack_from(
            entries, idx * STRUCT_MAP_ENTRY.size
        )
        # For core JP2 boxes we key by box_type; for everything else, use LIM type.
        key = box_type if box_type in {b"jP  ", b"ftyp", b"jp2h"} else lim_type
        chunkmap.setdefault(key, []).append(offset)

    fh.seek(current, os.SEEK_SET)
    return chunkmap


def _coerce_datetime(value: datetime.datetime | str | None) -> datetime.datetime | None:
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None

def _apply_legacy_vimd_to_picture_metadata(
    self,
    pm: PictureMetadata,
    frame0_meta: dict,
    planes: dict,
) -> None:
    """
    Enrich an existing limnd2.metadata.PictureMetadata instance using legacy VIMD
    information.

    We try to fill, where possible:

      * global calibration and objective fields on PictureMetadata
      * per-plane name (sDescription) and color (uiColor) from PicturePlanes.Plane
      * per-plane component count (uiCompCount)
      * per-plane pinhole diameter (from global PinholeRadius)
      * objective info into SampleSettings.pObjectiveSetting

    Everything uses object.__setattr__ because the dataclasses are frozen.
    Any failure is swallowed so this never breaks basic metadata.
    """

    # ------------------------------------------------------------------
    # Global calibration: VIMD["Calibration"] → PictureMetadata.dCalibration
    # ------------------------------------------------------------------
    cal: str | None = frame0_meta.get("Calibration", None)
    cal_val: float | None
    try:
        cal_val = float(cal)        # type: ignore
    except (TypeError, ValueError):
        cal_val = None

    if cal_val is not None and cal_val > 0:
        try:
            object.__setattr__(pm, "dCalibration", cal_val)
            object.__setattr__(pm, "bCalibrated", True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Objective / microscope-like fields on PictureMetadata
    # ------------------------------------------------------------------
    obj_name = frame0_meta.get("ObjectiveName")
    if obj_name:
        try:
            object.__setattr__(pm, "wsObjectiveName", obj_name)
        except Exception:
            pass

    for src_key, dst_attr in (
        ("ObjectiveMag", "dObjectiveMag"),
        ("ObjectiveNA", "dObjectiveNA"),
        ("RefractIndex1", "dRefractIndex1"),
        ("ProjectiveMag", "dZoom"),
    ):
        val = frame0_meta.get(src_key)
        try:
            f = float(val)      # type: ignore
        except (TypeError, ValueError):
            f = None
        if f is not None and f > 0:
            try:
                object.__setattr__(pm, dst_attr, f)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Access picture planes container (sPicturePlanes) and plane list.
    # ------------------------------------------------------------------
    planes_block = getattr(pm, "sPicturePlanes", None)
    if not isinstance(planes_block, PictureMetadataPicturePlanes):
        return

    plane_list = getattr(planes_block, "sPlaneNew", None)
    if not isinstance(plane_list, list) or not plane_list:
        return

    # ------------------------------------------------------------------
    # Per-plane enrichment from VIMD["PicturePlanes"]["Plane"]
    #   - name  → PicturePlaneDesc.sDescription
    #   - color → PicturePlaneDesc.uiColor
    #   - comps → PicturePlaneDesc.uiCompCount
    # ------------------------------------------------------------------
    if planes:
        try:
            vimd_plane_items = sorted(planes.items(), key=lambda kv: int(kv[0]))
        except Exception:
            vimd_plane_items = list(planes.items())

        for idx, (_p_idx, pmeta) in enumerate(vimd_plane_items):
            if idx >= len(plane_list):
                break
            desc = plane_list[idx]

            # Channel name from OpticalConfigName
            name = pmeta.get("OpticalConfigName") or pmeta.get("Name")
            if name:
                try:
                    object.__setattr__(desc, "sDescription", name)
                except Exception:
                    pass

            # Channel color from legacy Color (ABGR). Lower 24 bits are BGR,
            # which matches uiColor layout in PicturePlaneDesc.
            color_val = pmeta.get("Color")
            if isinstance(color_val, int) and color_val != 0:
                try:
                    object.__setattr__(desc, "uiColor", color_val & 0xFFFFFF)
                except Exception:
                    pass

            # Per-plane component count, if present
            comp_val = pmeta.get("CompCount")
            if isinstance(comp_val, int) and comp_val > 0:
                try:
                    object.__setattr__(desc, "uiCompCount", comp_val)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Pinhole: frame0_meta["PinholeRadius"] → plane.dPinholeDiameter
    # (PictureMetadata.__post_init__ already handles unknown dPinholeRadius,
    # but here we can fill it explicitly from VIMD if present.)
    # ------------------------------------------------------------------
    pin_rad = frame0_meta.get("PinholeRadius")
    try:
        pin_rad_val = float(pin_rad)        # type: ignore
    except (TypeError, ValueError):
        pin_rad_val = None

    if pin_rad_val is not None and pin_rad_val > 0:
        diameter = 2.0 * pin_rad_val
        for desc in plane_list:
            try:
                object.__setattr__(desc, "dPinholeDiameter", diameter)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Keep uiCount / uiCompCount consistent with the planes we now have.
    # ------------------------------------------------------------------
    try:
        total_comp = 0
        for desc in plane_list:
            c = getattr(desc, "uiCompCount", 1)
            try:
                c_int = int(c)
            except Exception:
                c_int = 1
            if c_int <= 0:
                c_int = 1
            total_comp += c_int

        object.__setattr__(planes_block, "uiCount", len(plane_list))
        object.__setattr__(planes_block, "uiCompCount", total_comp)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Push objective info down into SampleSettings.pObjectiveSetting so that
    # high-level helpers (objectiveName(), objectiveMagnification(), …)
    # return useful values.
    # ------------------------------------------------------------------
    sample_settings = getattr(planes_block, "sSampleSetting", None)
    if not isinstance(sample_settings, list) or not sample_settings:
        return

    wsObjectiveName = getattr(pm, "wsObjectiveName", "")
    dObjectiveMag = getattr(pm, "dObjectiveMag", None)
    dObjectiveNA = getattr(pm, "dObjectiveNA", None)
    dRefractIndex1 = getattr(pm, "dRefractIndex1", None)

    for setting in sample_settings:
        try:
            obj = setting.pObjectiveSetting
        except Exception:
            continue

        if wsObjectiveName:
            try:
                object.__setattr__(obj, "wsObjectiveName", wsObjectiveName)
            except Exception:
                pass

        for src_attr_pm, dst_attr_obj in (
            ("dObjectiveMag", "dObjectiveMag"),
            ("dObjectiveNA", "dObjectiveNA"),
            ("dRefractIndex1", "dRefractIndex"),
        ):
            val = getattr(pm, src_attr_pm, None)
            try:
                f = float(val)          # type: ignore
            except (TypeError, ValueError):
                f = None
            if f is not None and f > 0:
                try:
                    object.__setattr__(obj, dst_attr_obj, f)
                except Exception:
                    pass


class LimJpeg2000Chunker(BaseChunker):
    """
    Chunker for legacy JPEG2000-based ND2 files (JP2 magic).

    The legacy container stores image planes as JPEG2000 codestreams organized in a
    custom chunk map located at the end of the file. Only read-only scenarios are
    supported; writing back to the legacy format is intentionally not implemented.
    """

    def __init__(self, store: Store, **kwargs) -> None:
        assert isinstance(store, Store), f"argument 'store' expected to be 'Store' but was '{type(file).__name__}'"
        assert store.isOpen is not None, f"argument 'store' expected to be opened'"

        self._store: Store = store
        self._lock = threading.RLock()

        # Load and freeze the chunk map
        self._chunkmap = _read_legacy_chunkmap(self._store.io)
        self._chunkmap_offsets = {
            name: tuple(offsets) for name, offsets in self._chunkmap.items()
        }
        self._version = (1, 0)

        # Read basic image header info (dimensions, channels, bits)
        header = self._read_header()

        # Try to get logical channel count and component count from PicturePlanes
        # in the VIMD XML, mirroring LegacyReader.attributes().
        chan_count_meta, comp_count_meta = self._probe_picture_plane_counts()

        # Logical channel count: prefer metadata "Count", fall back to JP2 header.
        self._channel_count = (
            chan_count_meta
            if chan_count_meta and chan_count_meta > 0
            else max(1, header.get("channels", 1))
        )
        # Keep the PicturePlanes CompCount as a hint for uiComp in image attributes.
        self._component_count_meta: int | None = comp_count_meta

        # Image planes are stored as JPEG2000 codestreams under LUNK.
        self._image_offsets = self._chunkmap_offsets.get(b"LUNK", ())
        if not self._image_offsets:
            raise RuntimeError("Legacy ND2 file does not contain LUNK image data.")
        if len(self._image_offsets) % self._channel_count != 0:
            raise RuntimeError("Legacy ND2 image chunk count does not match channels.")

        # Frames: LegacyReader uses len(chunkmap[b"VCAL"]) as sequenceCount.
        # Prefer that if it matches the LUNK-based count, else fall back to LUNK.
        seq_from_lunk = len(self._image_offsets) // self._channel_count
        vcal_offsets = self._chunkmap_offsets.get(b"VCAL", ())
        if vcal_offsets and len(vcal_offsets) == seq_from_lunk:
            self._sequence_count = len(vcal_offsets)
        else:
            self._sequence_count = seq_from_lunk

        image_attrs = self._build_image_attributes(
            header,
            component_count_hint=self._component_count_meta,
        )
        picture_metadata = self._build_default_picture_metadata(image_attrs.uiComp)
        super().__init__(
            with_image_attributes=image_attrs,
            with_picture_metadata=picture_metadata,
        )

        # placeholder for ExperimentLevel cache
        self._experiment: typing.Any = None

        # Ensure we start reading from the beginning for subsequent operations.
        self._store.io.seek(0)

    # -------------------------------------------------------------------------
    # Small helpers to read legacy XML blocks (VIMD, ARTT, …) the same way the
    # LegacyReader does.
    # -------------------------------------------------------------------------

    def _decode_legacy_xml(self, name: bytes) -> dict:
        """
        Decode a legacy XML-based chunk (ARTT, VIMD, TINF, AIM1, AIMD, ...).
        Returns a nested dict or {} if parsing fails.
        """
        chunk = self.chunk(name)
        if not chunk:
            return {}
        if isinstance(chunk, memoryview):
            chunk = chunk.tobytes()

        try:
            from nd2._parse._legacy_xml import parse_xml_block  # type: ignore
        except Exception:
            return {}

        try:
            return parse_xml_block(chunk)
        except Exception:
            return {}

    def _probe_picture_plane_counts(self) -> tuple[int | None, int | None]:
        """
        Best-effort extraction of PicturePlanes Count (logical channels)
        and CompCount (total components) from the VIMD chunk.

        Returns (channelCount, componentCount) or (None, None) on failure.
        """
        meta = self._decode_legacy_xml(b"VIMD")
        if not meta:
            return None, None

        planes = meta.get("PicturePlanes") or {}
        try:
            nC = int(planes.get("Count", 0) or 0)
            comp = int(planes.get("CompCount", 0) or 0)
        except Exception:
            return None, None

        if nC <= 0 or comp <= 0:
            return None, None

        return nC, comp

    # frame-0 metadata, like LegacyReader._frame0_meta()
    @cached_property
    def _frame0_meta(self) -> dict:
        meta = self._decode_legacy_xml(b"VIMD")
        return meta if isinstance(meta, dict) else {}

    def _read_header(self) -> dict[str, int]:
        """
        Read basic header information (rows, columns, channels, bits_per_component).

        Logic:
          1. If we have a 'jp2h' entry in the chunk map, read IHDR from that box
             (standard JP2 header path).
          2. Otherwise, fall back to parsing the JPEG2000 codestream SIZ marker
             from the first LUNK image codestream.
        """
        # --- Path 1: use jp2h/ihdr if present in the chunk map ---
        offsets = self._chunkmap_offsets.get(b"jp2h")
        if offsets:
            jp2h_offset = offsets[0]
            with self._lock:
                # Read jp2h box header
                self._store.io.seek(jp2h_offset)
                header = self._store.io.read(STRUCT_BOX_HEADER.size)
                if len(header) != STRUCT_BOX_HEADER.size:
                    raise RuntimeError("Legacy ND2 jp2h box header is truncated.")
                length, box_type = STRUCT_BOX_HEADER.unpack(header)
                if box_type != b"jp2h":
                    raise RuntimeError("Chunk map jp2h entry does not point to jp2h box.")

                # Read first child box of jp2h, which should be IHDR
                ihdr_header = self._store.io.read(STRUCT_BOX_HEADER.size)
                if len(ihdr_header) != STRUCT_BOX_HEADER.size:
                    raise RuntimeError("Legacy ND2 ihdr box header is truncated.")
                ihdr_length, ihdr_type = STRUCT_BOX_HEADER.unpack(ihdr_header)
                if ihdr_type != b"ihdr":
                    raise RuntimeError("Legacy ND2 file does not contain ihdr chunk.")

                data = self._store.io.read(STRUCT_IHDR.size)

            if len(data) != STRUCT_IHDR.size:
                raise RuntimeError("Legacy ND2 ihdr chunk is truncated.")

            height, width, channels, bpc, compression, _unkC, _ipr = STRUCT_IHDR.unpack(
                data
            )
            return {
                "rows": height,
                "columns": width,
                "channels": channels,
                "bits_per_component": bpc + 1,
                # Keep raw 'compression' field; mapping is handled later.
                "compression": compression,
            }

        # --- Path 2: no jp2h in chunk map -> parse codestream SIZ from first LUNK ---
        # We only need the first image plane to derive dimensions.
        lunk_offsets = self._chunkmap_offsets.get(b"LUNK")
        if not lunk_offsets:
            raise RuntimeError(
                "Legacy ND2 file has no jp2h header and no LUNK image data to probe."
            )

        first_offset = lunk_offsets[0]
        with self._lock:
            # Read entire box payload for simplicity; payload is the JPEG2000 codestream.
            self._store.io.seek(first_offset, os.SEEK_SET)
            box_header = self._store.io.read(STRUCT_BOX_HEADER.size)
            if len(box_header) != STRUCT_BOX_HEADER.size:
                raise RuntimeError("Legacy ND2 LUNK box header is truncated.")
            length, box_type = STRUCT_BOX_HEADER.unpack(box_header)
            payload_length = max(0, length - STRUCT_BOX_HEADER.size)
            payload = self._store.io.read(payload_length)

        if len(payload) != payload_length:
            raise RuntimeError("Legacy ND2 LUNK codestream payload is truncated.")

        header = _parse_jpeg2000_codestream_header(payload)
        return header

    def _build_image_attributes(
        self,
        header: dict[str, int],
        *,
        component_count_hint: int | None = None,
    ) -> ImageAttributes:
        """
        Build ImageAttributes using the same information and decisions as
        nd2's LegacyReader.attributes(), but mapped into our ImageAttributes
        dataclass.
        """
        width = header["columns"]
        height = header["rows"]

        # Legacy: bpcim = head["bits_per_component"]
        bpc_in_memory = header["bits_per_component"]

        # Legacy: bpcs = _advanced_image_attributes.get("SignificantBits", bpcim)
        bits_significant: int
        try:
            advanced = (
                self._decode_legacy_xml(b"ARTT").get("AdvancedImageAttributes") or {}
            )
            bits_significant = int(
                advanced.get("SignificantBits", bpc_in_memory) or bpc_in_memory
            )
        except Exception:
            bits_significant = bpc_in_memory

        # Legacy: compCount = picplanes["CompCount"], fallback to channels/1
        component_count = component_count_hint or header.get("channels", 1)
        if component_count <= 0:
            component_count = 1

        # Same row-width math as ImageAttributes.create() / LegacyReader
        width_bytes = ImageAttributes.calcWidthBytes(
            width, bpc_in_memory, component_count
        )

        # Legacy: compressionLevel=head.get("compression")
        compression_flag = header.get("compression", 0)
        compression = (
            ImageAttributesCompression.ictLossLess
            if compression_flag == 0
            else ImageAttributesCompression.ictLossy
        )

        # Match LegacyReader semantics: 32 significant bits -> float, else unsigned
        pixel_type = (
            ImageAttributesPixelType.pxtReal
            if bits_significant == 32
            else ImageAttributesPixelType.pxtUnsigned
        )

        return ImageAttributes(
            uiWidth=width,
            uiWidthBytes=width_bytes,
            uiHeight=height,
            uiComp=component_count,
            uiBpcInMemory=bpc_in_memory,
            uiBpcSignificant=bits_significant,
            uiSequenceCount=self._sequence_count,
            uiTileWidth=width,
            uiTileHeight=height,
            eCompression=compression,
            dCompressionParam=float(compression_flag),
            ePixelType=pixel_type,
            uiVirtualComponents=component_count,
        )

    def _build_default_picture_metadata(self, component_count: int) -> PictureMetadata:
        metadata = PictureMetadata()
        metadata.makeValid(component_count)
        return metadata

    # -------------------------------------------------------------------------
    # Basic chunker properties
    # -------------------------------------------------------------------------

    @property
    def filename(self) -> str | None:
        return self._store.filename

    @property
    def size_on_disk(self) -> int:
        return self._store.sizeOnDisk

    @property
    def last_modified(self) -> datetime.datetime:
        return self._store.lastModified

    @property
    def format_version(self) -> tuple[int, int]:
        return self._version

    @property
    def chunker_name(self) -> str:
        return "legacy_jpeg2000"

    @property
    def is_readonly(self) -> bool:
        return True

    @property
    def chunk_names(self) -> list[bytes]:
        return list(self._chunkmap_offsets.keys())

    # Provide "channels" like other chunkers / factory expect.
    @property
    def channels(self) -> int:
        return self._channel_count

    @property
    def pixel_size_um(self) -> float:
        """
        Approximate pixel size in microns from legacy Calibration, if available.
        """
        meta = self._frame0_meta
        cal = meta.get("Calibration", None)
        try:
            cal_f = float(cal)      # type: ignore
        except Exception:
            cal_f = 1.0
        if cal_f <= 0:
            cal_f = 1.0
        return cal_f

    # -------------------------------------------------------------------------

    def _read_box_payload(self, offset: int) -> bytes:
        with self._lock:
            self._store.io.seek(offset, os.SEEK_SET)
            header = self._store.io.read(STRUCT_BOX_HEADER.size)
            if len(header) != STRUCT_BOX_HEADER.size:
                raise RuntimeError("Legacy ND2 chunk header is truncated.")
            length, _box_type = STRUCT_BOX_HEADER.unpack(header)
            payload = self._store.io.read(length - STRUCT_BOX_HEADER.size)
        if len(payload) != length - STRUCT_BOX_HEADER.size:
            raise RuntimeError("Legacy ND2 chunk payload is truncated.")
        return payload

    def chunk(self, name: bytes | str) -> bytes | memoryview | None:
        if isinstance(name, str):
            name = name.encode("ascii")
        if not BaseChunker._is_chunk_data(name):
            raise UnexpectedCallError("chunk", name)
        offsets = self._chunkmap_offsets.get(name)
        if not offsets:
            return None
        return self._read_box_payload(offsets[0])

    def setChunk(self, name: bytes | str, data: bytes | memoryview) -> None:
        raise PermissionError("Legacy JPEG2000 chunker is read-only.")

    def _ensure_frame_index(self, seqindex: int) -> None:
        if seqindex < 0 or seqindex >= self._sequence_count:
            raise IndexError(f"Sequence index {seqindex} out of bounds.")

    # -------------------------------------------------------------------------
    # Experiment reconstruction from AIM1/AIMD (LegacyReader logic)
    # -------------------------------------------------------------------------

    @property
    def experiment(self) -> "ExperimentLevel | None":  # type: ignore[override]
        """Return ExperimentLevel decoded from legacy AIM1/AIMD XML, if present."""
        # Already parsed? just return the cached value
        if self._experiment is not None:
            return self._experiment  # type: ignore[return-value]

        # Legacy experiment description chunks (AIM1 preferred, AIMD fallback)
        xml_chunk = self.chunk(b"AIM1") or self.chunk(b"AIMD")
        if not xml_chunk:
            return None

        if isinstance(xml_chunk, memoryview):
            xml_chunk = xml_chunk.tobytes()

        # Local import to avoid hard dependency at module import time
        try:
            from nd2._parse._legacy_xml import parse_xml_block  # type: ignore
        except Exception:
            # No XML parser available – cannot reconstruct experiment
            return None

        try:
            meta = parse_xml_block(xml_chunk)
        except Exception:
            # Malformed XML or unexpected legacy format
            return None

        # ----- normalize legacy experiment dict (like LegacyReader._raw_exp_loops) -----
        version = ""
        meta.pop("UnknownData", None)
        while len(meta) == 1:
            key, val = next(iter(meta.items()))
            if "_V" in key:
                version = key.split("_V")[1]
            meta = val
            meta.pop("UnknownData", None)
        meta["Version"] = version

        # ----- helper: parse dims from legacy description text (T, S, …) -----
        def _legacy_dims_from_description() -> dict[str, int]:
            desc = ""
            try:
                tinf_chunk = self.chunk(b"TINF")
                if not tinf_chunk:
                    return {}
                if isinstance(tinf_chunk, memoryview):
                    tinf_chunk = tinf_chunk.tobytes()
                tinf = parse_xml_block(tinf_chunk)
                for item in tinf.get("TextInfoItem", []):
                    text = item.get("Text", "")
                    if text.startswith("Metadata:"):
                        desc = text
                        break
            except Exception:
                return {}
            if not desc:
                return {}

            import re

            m = re.search(r"Dimensions:\s?([^\r]+)\r?\n", desc)
            if not m:
                return {}
            dims_str = m.group(1)
            # match LegacyReader: λ -> channel axis, XY -> position axis
            dims_str = dims_str.replace("λ", "C").replace("XY", "S")
            dimsize = re.compile(r"(\w+)'?\((\d+)\)")
            return {k: int(v) for k, v in dimsize.findall(dims_str)}

        ddim = _legacy_dims_from_description()

        # ----- flatten legacy loop chain into a list of RawExperimentLoop dicts -----
        raw_levels: list[dict] = []
        meta_copy = dict(meta)  # mutate a shallow copy only

        if "LoopNo00" in meta_copy:
            # "old style" files: multiple LoopNoXX entries at top level
            for key, value in meta_copy.items():
                if key == "Version":
                    continue
                if isinstance(value, dict):
                    raw_levels.append(value)
        else:
            # "new style": single loop with nested NextLevelEx chain
            while meta_copy:
                if len(meta_copy) == 1:
                    only_val = next(iter(meta_copy.values()))
                    if isinstance(only_val, dict):
                        meta_copy = only_val
                        meta_copy.pop("UnknownData", None)
                    elif isinstance(only_val, list) and only_val:
                        meta_copy = only_val[0]
                        meta_copy.pop("UnknownData", None)
                    else:
                        break
                raw_levels.append(meta_copy)
                next_level = meta_copy.get("NextLevelEx")
                if not isinstance(next_level, dict):
                    break
                meta_copy = next_level

        if not raw_levels:
            self._experiment = None
            return None

        from .experiment import ExperimentLevel, ExperimentLoopType
        import typing as _t

        exp_levels: list[ExperimentLevel] = []

        for meta_level in raw_levels:
            type_id = int(meta_level.get("Type", 0) or 0)
            loop_pars = meta_level.get("LoopPars") or {}

            # ----- XY multipoint (Type == 2) -----
            if type_id == 2:
                params = _t.cast(dict, loop_pars)
                pos_x = params.get("PosX") or {}
                pos_y = params.get("PosY") or {}
                pos_z = params.get("PosZ") or {}
                pfs = params.get("PFSOffset") or {}
                use_z = bool(params.get("UseZ", False))

                poscount = len(pos_x)
                target_count = ddim.get("S") or params.get("Count") or poscount
                target_count = int(target_count)

                keys = sorted(pos_x.keys())
                # LegacyReader uses reversed order (poscount - i - 1)
                sel_keys = [keys[poscount - i - 1] for i in range(min(target_count, poscount))]

                points: list[dict[str, float | str]] = []
                for k in sel_keys:
                    points.append(
                        {
                            "dPosX": float(pos_x.get(k, 0.0)),
                            "dPosY": float(pos_y.get(k, 0.0)),
                            "dPosZ": float(pos_z.get(k, 0.0)),
                            "dPFSOffset": float(pfs.get(k, 0.0)),
                            "dPosName": "",
                        }
                    )

                uLoopPars = {
                    "uiCount": target_count,
                    "bUseZ": use_z,
                    "Points": points,
                }
                eType = ExperimentLoopType.eEtXYPosLoop

            # ----- Time loop (Type == 8) – pick period that matches T-dim if possible -----
            elif type_id == 8:
                params = _t.cast(dict, loop_pars)
                periods = params.get("Period") or {}
                per = None
                count_from_desc = ddim.get("T")

                if count_from_desc is not None:
                    for p in periods.values():
                        if p.get("Count") == count_from_desc:
                            per = p
                            break

                if per is None and periods:
                    per = next(iter(periods.values()))

                if not per:
                    continue  # nothing usable

                uLoopPars = {
                    "uiCount": int(per.get("Count", 0)),
                    "dStart": float(per.get("Start", 0.0)),
                    "dPeriod": float(per.get("Period", 0.0)),
                    "dDuration": float(per.get("Duration", 0.0)),
                    "dMinPeriodDiff": float(per.get("MinPeriodDiff", 0.0)),
                    "dMaxPeriodDiff": float(per.get("MaxPeriodDiff", 0.0)),
                    "dAvgPeriodDiff": float(
                        per.get("AvgPeriodDiff", per.get("Period", 0.0))
                    ),
                }
                eType = ExperimentLoopType.eEtTimeLoop

            # ----- Z-stack (Type == 4) -----
            elif type_id == 4:
                params = _t.cast(dict, loop_pars)
                uLoopPars = {
                    "uiCount": int(params.get("Count", 0)),
                    "dZLow": float(params.get("ZLow", 0.0)),
                    "dZLowPFSOffset": float(params.get("ZLowPFSOffset", 0.0)),
                    "dZHigh": float(params.get("ZHigh", 0.0)),
                    "dZHighPFSOffset": float(params.get("ZHighPFSOffset", 0.0)),
                    "dZHome": float(params.get("ZHome", 0.0)),
                    "dZStep": float(params.get("ZStep", 0.0)),
                    "bAbsolute": bool(params.get("Absolute", False)),
                    "bTriggeredPiezo": bool(params.get("TriggeredPiezo", False)),
                }
                eType = ExperimentLoopType.eEtZStackLoop

            else:
                # LegacyReader ignores other loop types (e.g. channel loop type 6)
                continue

            level = ExperimentLevel(
                eType=eType,
                uLoopPars=uLoopPars,  # dict – ExperimentLevel.__post_init__ converts it
                wsApplicationDesc=meta_level.get("ApplicationDesc", ""),
                wsUserDesc=meta_level.get("UserDesc", ""),
                aMeasProbesBase64=meta_level.get("MeasProbesBase64", b""),
                pItemValid=meta_level.get("ItemValid"),
                sAutoFocusBeforeLoop=meta_level.get("AutoFocusBeforeLoop", {}),
                wsCommandBeforeLoop=meta_level.get("CommandBeforeLoop", ""),
                wsCommandBeforeCapture=meta_level.get("CommandBeforeCapture", ""),
                wsCommandAfterCapture=meta_level.get("CommandAfterCapture", ""),
                wsCommandAfterLoop=meta_level.get("CommandAfterLoop", ""),
                bControlShutter=bool(meta_level.get("ControlShutter", False)),
                bControlLight=bool(meta_level.get("ControlLight", False)),
                bUsePFS=bool(meta_level.get("UsePFS", False)),
                uiRepeatCount=int(meta_level.get("RepeatCount", 1)),
                uiNextLevelCount=0,
                ppNextLevelEx=None,
            )
            exp_levels.append(level)

        if not exp_levels:
            self._experiment = None
            return None

        # ----- chain levels like ExperimentFactory._create_experiment -----
        root = exp_levels[0]
        current = root
        for nxt in exp_levels[1:]:
            object.__setattr__(current, "ppNextLevelEx", [nxt])
            object.__setattr__(current, "uiNextLevelCount", 1)
            current = nxt

        self._experiment = root if root.valid else None
        return self._experiment

    # -------------------------------------------------------------------------
    # Picture metadata (LV / VAR) with legacy VIMD + attributes + experiment
    # -------------------------------------------------------------------------

    def _metadata_from_slx_block(self) -> PictureMetadata | None:
        """
        Try to decode SLxPictureMetadata from an embedded ImageMetadataSeq* block.
        Returns None if nothing decodes cleanly.
        """
        raw = getattr(self, "_get_metadata_bytes", None)
        if callable(raw):
            raw = raw()
        else:
            raw = getattr(self, "_metadata_bytes", None)

        if not raw:
            return None

        data = bytes(raw)       # type: ignore

        candidates: list[bytes] = []
        idx = data.find(b"SLxPictureMetadata")
        if idx != -1:
            candidates.append(data[idx:])  # start at the struct itself
        candidates.append(data)  # full block as a fallback

        for candidate in candidates:
            # Try LiteVariant first (modern ND2)
            try:
                return PictureMetadata.from_lv(candidate)
            except Exception:
                pass

            # Then try classic Variant encoding (older ND2)
            try:
                return PictureMetadata.from_var(candidate)
            except Exception:
                pass

        return None

    def _metadata_from_legacy_vimd(self) -> PictureMetadata | None:
        """
        Build PictureMetadata using legacy VIMD frame-0 metadata, image attributes
        and experiment – conceptually similar to LegacyReader._load_metadata,
        but funneled through MetadataFactory and then enriched in-place.

        This is the "pure legacy" path when no SLxPictureMetadata block is present.
        """
        meta = self._frame0_meta
        if not meta:
            return None

        # --- XY pixel calibration from VIMD.Calibration ---
        cal = meta.get("Calibration", None)
        try:
            px_cal = float(cal)     # type: ignore
        except Exception:
            px_cal = 1.0
        if px_cal <= 0:
            px_cal = 1.0

        # --- Channels from PicturePlanes ---
        picture_planes = meta.get("PicturePlanes") or {}
        planes = picture_planes.get("Plane") or {}
        n_channels = len(planes) or self._channel_count

        # --- Heuristic RGB flag (same idea as before) ---
        is_rgb = n_channels in (3, 4) and self.imageAttributes.uiComp in (3, 4)

        # Build a base metadata object using MetadataFactory (existing behavior).
        pm = self._fallback_metadata_from_factory(
            pixel_calibration=px_cal,
            n_channels=n_channels,
            is_rgb=is_rgb,
        )
        if pm is None:
            return None

        # Try to enrich the PictureMetadata in-place using VIMD + experiment.
        # All mutations are guarded by hasattr so this is safe even if your
        # PictureMetadata schema differs.
        try:
            _apply_legacy_vimd_to_picture_metadata(self, pm, meta, planes)
        except Exception:
            # Any error in enrichment should not break basic metadata
            pass

        return pm


    @property
    def pictureMetadata(self) -> PictureMetadata:
        """
        Return picture-level metadata.

        Order of preference:
          1. Decode SLxPictureMetadata LV/VAR block (if present).
          2. Synthesize from legacy VIMD + attributes + experiment
             (similar spirit to LegacyReader._load_metadata).
          3. Generic MetadataFactory-based fallback.
        """
        # 1) Modern / LV path
        pm = self._metadata_from_slx_block()
        if pm is not None:
            return pm

        # 2) Legacy path using frame-0 VIMD and experiment
        pm = self._metadata_from_legacy_vimd()
        if pm is not None:
            return pm

        # 3) Last-resort generic metadata
        return self._fallback_metadata_from_factory()


    @pictureMetadata.setter
    def pictureMetadata(self, val: PictureMetadata) -> None:
        raise PermissionError("Legacy JPEG2000 chunker is read-only.")

    def _fallback_metadata_from_factory(
        self,
        *,
        pixel_calibration: float | None = None,
        n_channels: int | None = None,
        is_rgb: bool | None = None,
    ) -> PictureMetadata:
        """
        Best-effort minimal metadata when the embedded ND2 block is missing
        or unreadable. Uses MetadataFactory and whatever basic information
        the chunker exposes (channel count, RGB / non-RGB, pixel size).
        """

        # pixel size
        if pixel_calibration is None:
            # Prefer VIMD.Calibration if available
            meta = self._frame0_meta
            cal = meta.get("Calibration", None)
            try:
                v = float(cal)      # type: ignore
            except Exception:
                v = -1.0
            if v > 0:
                pixel_calibration = v
            else:
                # fall back to whatever attribute the reader may expose
                pixel_calibration = float(
                    getattr(
                        self,
                        "pixel_size_um",
                        getattr(self, "pixel_calibration", 1.0),
                    )
                )

        # channel count
        if n_channels is None:
            n_channels = getattr(self, "channels", None)
            if not isinstance(n_channels, int) or n_channels <= 0:
                n_channels = self._channel_count

        # rgb flag
        if is_rgb is None:
            is_rgb = bool(getattr(self, "is_rgb", False))
            if not is_rgb:
                is_rgb = n_channels in (3, 4)

        factory = MetadataFactory(pixel_calibration=pixel_calibration)
        return factory.createMetadata(
            number_of_channels_fallback=n_channels,
            is_rgb_fallback=is_rgb,
        )

    # -------------------------------------------------------------------------
    # Image data
    # -------------------------------------------------------------------------

    def image(
        self, seqindex: int, rect: tuple[int, int, int, int] | None = None
    ) -> NumpyArrayLike:
        self._ensure_frame_index(seqindex)
        if imagecodecs is None:  # pragma: no cover - dependency is part of install
            raise ModuleNotFoundError(
                'imagecodecs is required to read legacy JPEG2000 ND2 files. '
                'Install it with `pip install "limnd2[legacy]"`.'
            )

        planes: list[np.ndarray] = []
        for channel in range(self._channel_count):
            chunk_index = seqindex * self._channel_count + channel
            payload = self._read_box_payload(self._image_offsets[chunk_index])
            plane = imagecodecs.jpeg2k_decode(payload)
            planes.append(plane)

        data = np.stack(planes, axis=-1)

        if rect is not None:
            x, y, width, height = rect
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(data.shape[1], x0 + max(0, width))
            y1 = min(data.shape[0], y0 + max(0, height))
            data = data[y0:y1, x0:x1, :]

        target_dtype = self.imageAttributes.dtype
        if data.dtype != target_dtype:
            data = data.astype(target_dtype, copy=False)
        return data

    def setImage(self, seqindex: int, image: NumpyArrayLike, acqtime: float = -1.0) -> None:
        raise PermissionError("Legacy JPEG2000 chunker is read-only.")

    def readDownsampledImage(
        self, seqindex: int, downsize: int, rect: tuple[int, int, int, int] | None = None
    ) -> NumpyArrayLike:
        raise NameNotInChunkmapError(f"DownsampledColorData_{downsize}")

    def setDownsampledImage(
        self, seqindex: int, downsize: int, image: NumpyArrayLike
    ) -> None:
        raise PermissionError("Legacy JPEG2000 chunker is read-only.")

    def binaryRasterData(
        self, binid: int, seqindex: int, rect: tuple[int, int, int, int] | None = None
    ) -> NumpyArrayLike:
        raise BinaryIdNotFountError(binid)

    def setBinaryRasterData(
        self, binid: int, seqindex: int, binimage: NumpyArrayLike
    ) -> None:
        raise PermissionError("Legacy JPEG2000 chunker is read-only.")

    def readDownsampledBinaryRasterData(
        self,
        binid: int,
        seqindex: int,
        downsize: int,
        rect: tuple[int, int, int, int] | None = None,
    ) -> NumpyArrayLike:
        raise NameNotInChunkmapError(f"DownsampledBinary_{downsize}")

    def setDownsampledBinaryRasterData(
        self, binid: int, seqindex: int, downsize: int, binimage: NumpyArrayLike
    ) -> None:
        raise PermissionError("Legacy JPEG2000 chunker is read-only.")

    def finalize(self) -> None:
        self._store.close()

    def rollback(self) -> None:
        # Legacy files are read-only – there is nothing to roll back.
        return


def is_legacy_jpeg2000_source(store: Store) -> bool:
    """
    Return True if the provided binary source appears to be a legacy JPEG2000 ND2 file.

    The detection logic matches the JP2 magic used by Nikon's legacy ND2 container.
    """

    if store.mem is not None and 4 <= len(store.mem):
        magic = int.from_bytes(store.mem[:4], "little", signed=False)
        return magic == JP2_MAGIC

    elif store.io is not None:
        try:
            pos = store.io.tell()
            store.io.seek(0)
            header = store.io.read(4)
            store.io.seek(pos)
        except (OSError, AttributeError):
            return False
        return struct.unpack("<I", header)[0] == JP2_MAGIC if len(header) == 4 else False

    else:
        return False