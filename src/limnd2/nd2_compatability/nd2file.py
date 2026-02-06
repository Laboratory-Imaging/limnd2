"""
This file serves as a compatibility layer between the
[limnd2](https://github.com/Laboratory-Imaging/limnd2) library by [Laboratory Imaging s.r.o.](https://github.com/Laboratory-Imaging)
and the [nd2](https://github.com/tlambert03/nd2) library by Harvard Medical School microscopist [Talley Lambert](https://github.com/tlambert03).

It is designed for users familiar with Talley Lambert's `nd2` library, providing a seamless transition by mimicking its
interface while utilizing the `limnd2` library as the underlying implementation.

We extend our heartfelt thanks to Talley Lambert for his outstanding work on the `nd2` library, which has greatly benefited the imaging community,
and for his invaluable input in helping us develop the `limnd2` library.

!!! warning "Under Development"
    This feature is under development. Only some methods are currently supported and sometimes only partially.
"""

from __future__ import annotations

from functools import cached_property
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, SupportsInt, cast
import json
import math
import re
import threading
import warnings

import numpy as np

from limnd2.attributes import ImageAttributesCompression, ImageAttributesPixelType
from limnd2.base import (
    BaseChunker,
    BinaryIdNotFountError,
    NameNotInChunkmapError,
    ND2_CHUNK_FORMAT_ImageMetadataLV_1p,
    ND2_CHUNK_FORMAT_ImageMetadata_1p,
    ND2_CHUNK_NAME_CustomDataVar,
    ND2_CHUNK_NAME_CustomDataVarLI,
    ND2_CHUNK_MAGIC,
    JP2_MAGIC,
)
from limnd2.experiment import (
    ExperimentLoopType,
    ExperimentNETimeLoop,
    ExperimentTimeLoop,
    ExperimentXYPosLoop,
    ExperimentZStackLoop,
    ZStackType,
)
from limnd2.lite_variant import decode_lv
from limnd2.metadata import PictureMetadata, PicturePlaneModalityFlags
from limnd2.nd2 import Nd2Reader
from limnd2.variant import decode_var

from .nd2file_types import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ome_types


class AXIS:
    X = "X"
    Y = "Y"
    Z = "Z"
    CHANNEL = "C"
    RGB = "S"
    TIME = "T"
    POSITION = "P"
    UNKNOWN = "U"

    _MAP: dict[str, str] = {
        "Unknown": UNKNOWN,
        "TimeLoop": TIME,
        "XYPosLoop": POSITION,
        "ZStackLoop": Z,
        "NETimeLoop": TIME,
        "CustomLoop": UNKNOWN,
    }

    @classmethod
    def frame_coords(cls) -> set[str]:
        return {cls.X, cls.Y, cls.CHANNEL, cls.RGB}


def _convert_records_to_dict_of_lists(
    records: list[dict[str, Any]], null_val: Any = float("nan")
) -> Mapping[str, list[Any]]:
    col_names: dict[str, None] = {column: None for r in records for column in r}
    output: dict[str, list[Any]] = {col_name: [] for col_name in col_names}
    for record, col_name in product(records, col_names):
        output[col_name].append(record.get(col_name, null_val))
    return output


def _convert_records_to_dict_of_dicts(
    records: list[dict[str, Any]], null_val: Any = float("nan")
) -> Mapping[str, dict[int, Any]]:
    col_names: dict[str, None] = {column: None for r in records for column in r}
    output: dict[str, dict[int, Any]] = {col_name: {} for col_name in col_names}
    for (idx, record), col_name in product(enumerate(records), col_names):
        output[col_name][idx] = record.get(col_name, null_val)
    return output


def _strip_prefix_key(key: str) -> str:
    i = 0
    while i < len(key) and key[i].islower():
        i += 1
    return key[i:] if i else key


def _strip_prefix_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return { _strip_prefix_key(str(k)): _strip_prefix_dict(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _strip_prefix_dict(v) for v in obj ]
    return obj


_TEXTINFO_DIM_RE = re.compile(
    "([A-Za-z\\u03bb]+)\\s*[\\'\\u2019\\u2032]*\\s*\\((\\d+)\\)"
)
_TEXTINFO_TIME_RE = re.compile(r"Time Loop:\\s*(\\d+)")
_TEXTINFO_Z_RE = re.compile(r"Z Stack Loop:\\s*(\\d+)")


def _parse_textinfo_dims(desc: str) -> dict[str, int]:
    if not desc:
        return {}
    dims: dict[str, int] = {}
    lines = desc.splitlines()
    for line in lines:
        if "Dimensions:" not in line:
            continue
        payload = line.split("Dimensions:", 1)[1]
        for token in payload.split("x"):
            match = _TEXTINFO_DIM_RE.search(token)
            if not match:
                continue
            label = re.sub(r"[^A-Za-z\\u03bb]", "", match.group(1)).upper()
            count = int(match.group(2))
            if not label or count <= 0:
                continue
            if label.startswith("T"):
                dims[AXIS.TIME] = count
            elif label.startswith("Z"):
                dims[AXIS.Z] = count
            elif label.startswith("P"):
                dims[AXIS.POSITION] = count
            elif label in ("L", "\u039b"):
                dims[AXIS.CHANNEL] = count
        break

    time_match = _TEXTINFO_TIME_RE.search(desc)
    if time_match:
        dims.setdefault(AXIS.TIME, int(time_match.group(1)))
    z_match = _TEXTINFO_Z_RE.search(desc)
    if z_match:
        dims.setdefault(AXIS.Z, int(z_match.group(1)))
    return dims


class ND2File:
    limnd2: Nd2Reader
    _path: FileOrBinaryIO

    def __init__(
        self,
        path: FileOrBinaryIO,
        *,
        validate_frames: bool = False,
        search_window: int = 100,
    ) -> None:
        self._path = path
        self._source_fh = path if hasattr(path, "read") else None
        self._lock = threading.RLock()
        self._error_radius: int | None = (
            search_window * 1000 if validate_frames else None
        )
        self._source = self._coerce_source(path)
        self.limnd2 = Nd2Reader(self._source)  # type: ignore[arg-type]
        self._closed = False

    def _coerce_source(self, path: FileOrBinaryIO):
        if isinstance(path, (str, Path)):
            return path
        if hasattr(path, "read"):
            fh = cast(Any, path)
            try:
                pos = fh.tell()
            except Exception:
                pos = None
            data = fh.read()
            if pos is not None:
                try:
                    fh.seek(pos)
                except Exception:
                    pass
            return memoryview(data)
        return path

    @staticmethod
    def is_supported_file(path: StrOrPath) -> bool:
        if hasattr(path, "read"):
            fh = cast(Any, path)
            pos = fh.tell()
            fh.seek(0)
            magic = fh.read(4)
            fh.seek(pos)
        else:
            with open(path, "rb") as fh:
                magic = fh.read(4)
        return magic in (ND2_CHUNK_MAGIC.to_bytes(4, "little"), JP2_MAGIC.to_bytes(4, "little"))

    @cached_property
    def version(self) -> tuple[int, ...]:
        """works"""
        self._ensure_open()
        return self.limnd2.version

    @property
    def path(self) -> str:
        """works"""
        if isinstance(self._path, (str, Path)):
            return str(self._path)
        if hasattr(self._path, "name"):
            return str(getattr(self._path, "name"))
        return "<memory>"

    @property
    def is_legacy(self) -> bool:
        self._ensure_open()
        return tuple(self.limnd2.version) == (1, 0)

    def open(self) -> None:
        if self.closed:
            self.limnd2 = Nd2Reader(self._source)  # type: ignore[arg-type]
            self._closed = False

    def close(self) -> None:
        if not self.closed:
            self.limnd2.finalize()
            self._closed = True

    @property
    def closed(self) -> bool:
        if self._source_fh is not None and getattr(self._source_fh, "closed", False):
            return True
        return self._closed

    def __enter__(self) -> "ND2File":
        self.open()
        return self

    def __del__(self) -> None:
        if not getattr(self, "closed", True):
            warnings.warn(
                "ND2File file not closed before garbage collection. "
                "Please use `with ND2File(...):` context or call `.close()`.",
                stacklevel=2,
            )
            self.close()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("limnd2", None)
        state.pop("_lock", None)
        state.pop("sizes", None)
        state["_closed"] = self.closed
        return state

    def __setstate__(self, d: dict[str, Any]) -> None:
        _was_closed = d.pop("_closed", False)
        self.__dict__ = d
        self._lock = threading.RLock()
        self.limnd2 = Nd2Reader(self._source)  # type: ignore[arg-type]
        self._closed = False
        if _was_closed:
            self.close()

    def _ensure_open(self) -> None:
        if self.closed:
            self.open()

    @cached_property
    def attributes(self) -> Attributes:
        """works"""
        self._ensure_open()
        att = self.limnd2.imageAttributes
        compression_level = att.dCompressionParam
        if att.eCompression == ImageAttributesCompression.ictLossy:
            compression = "lossy"
        elif att.eCompression == ImageAttributesCompression.ictLossLess:
            compression = "lossless"
        else:
            compression = None
            compression_level = None

        if att.ePixelType == ImageAttributesPixelType.pxtUnsigned:
            pixel_data_type = "unsigned"
        elif att.ePixelType == ImageAttributesPixelType.pxtReal:
            pixel_data_type = "float"
        else:
            pixel_data_type = "unsigned"

        channel_count: int | None = None
        component_count: int | None = None

        pm = getattr(self.limnd2, "pictureMetadata", None)
        try:
            if pm is not None and hasattr(pm, "sPicturePlanes"):
                candidate = pm.sPicturePlanes.uiCount
                if candidate and candidate > 0:
                    channel_count = candidate
        except Exception:
            channel_count = None

        if channel_count is None or channel_count <= 0:
            if att.uiComp in (3, 4):
                channel_count = 1
            else:
                channel_count = max(1, att.uiComp)
        component_count = max(1, att.uiComp)
        if channel_count == 1 and component_count > 1 and component_count not in (3, 4):
            channel_count = component_count

        return Attributes(
            bitsPerComponentInMemory=att.uiBpcInMemory,
            bitsPerComponentSignificant=att.uiBpcSignificant,
            componentCount=component_count,
            heightPx=att.uiHeight,
            pixelDataType=pixel_data_type,
            sequenceCount=att.uiSequenceCount,
            widthBytes=att.uiWidthBytes,
            widthPx=att.uiWidth,
            compressionLevel=compression_level,
            compressionType=compression,
            tileHeightPx=None if att.uiTileHeight == att.uiHeight else att.uiTileHeight,
            tileWidthPx=None if att.uiTileWidth == att.uiWidth else att.uiTileWidth,
            channelCount=channel_count,
        )

    @cached_property
    def text_info(self) -> TextInfo:
        self._ensure_open()
        info = self.limnd2.imageTextInfo
        if info is None:
            return {}
        return info.to_dict()

    @cached_property
    def rois(self) -> dict[int, ROI]:
        # ROI metadata is not currently parsed in limnd2; return empty mapping.
        return {}

    @cached_property
    def experiment(self) -> list[ExpLoop]:
        """works for T, M, Z experiments"""
        self._ensure_open()
        if self.limnd2.experiment is None:
            return []

        exps: list[ExpLoop] = []
        count = 0
        for exp in self.limnd2.experiment:
            if exp.eType == ExperimentLoopType.eEtTimeLoop and isinstance(
                exp.uLoopPars, ExperimentTimeLoop
            ):
                ep = TimeLoopParams(
                    startMs=exp.uLoopPars.dStart,
                    periodMs=exp.uLoopPars.dPeriod,
                    durationMs=exp.uLoopPars.dDuration,
                    periodDiff=PeriodDiff(
                        avg=exp.uLoopPars.dAvgPeriodDiff,
                        max=exp.uLoopPars.dMaxPeriodDiff,
                        min=exp.uLoopPars.dMinPeriodDiff,
                    ),
                )
                e = TimeLoop(
                    count=exp.uLoopPars.uiCount,
                    nestingLevel=count,
                    parameters=ep,
                )
                count += 1
                exps.append(e)

            elif exp.eType == ExperimentLoopType.eEtNETimeLoop and isinstance(
                exp.uLoopPars, ExperimentNETimeLoop
            ):
                periods = []
                for period in exp.uLoopPars.pPeriod:
                    params = TimeLoopParams(
                        startMs=period.dStart,
                        periodMs=period.dPeriod,
                        durationMs=period.dDuration,
                        periodDiff=PeriodDiff(
                            avg=period.dAvgPeriodDiff,
                            max=period.dMaxPeriodDiff,
                            min=period.dMinPeriodDiff,
                        ),
                    )
                    periods.append(Period(count=period.uiCount, **params.__dict__))
                ep = NETimeLoopParams(periods=periods)
                e = NETimeLoop(
                    count=exp.uLoopPars.uiCount,
                    nestingLevel=count,
                    parameters=ep,
                )
                count += 1
                exps.append(e)

            elif exp.eType == ExperimentLoopType.eEtZStackLoop and isinstance(
                exp.uLoopPars, ExperimentZStackLoop
            ):
                ep = ZStackLoopParams(
                    bottomToTop=exp.uLoopPars.iType
                    in (
                        ZStackType.zstBottomToTopFixedTop,
                        ZStackType.zstBottomToTopFixedBottom,
                        ZStackType.zstSymmetricRangeFixedHomeBottomToTop,
                        ZStackType.zstAsymmetricRangeFixedHomeBottomToTop,
                    ),
                    homeIndex=exp.uLoopPars.homeIndex,
                    stepUm=exp.uLoopPars.dZStep,
                    deviceName=exp.uLoopPars.wsZDevice,
                )
                e = ZStackLoop(
                    count=exp.uLoopPars.uiCount,
                    nestingLevel=count,
                    parameters=ep,
                )
                count += 1
                exps.append(e)

            elif exp.eType == ExperimentLoopType.eEtXYPosLoop and isinstance(
                exp.uLoopPars, ExperimentXYPosLoop
            ):
                points = []
                for index, pos in enumerate(exp.uLoopPars.Points):
                    if (not exp.pItemValid) or (
                        exp.pItemValid and exp.pItemValid[index]
                    ):
                        points.append(
                            Position(
                                stagePositionUm=StagePosition(
                                    x=pos.dPosX,
                                    y=pos.dPosY,
                                    z=pos.dPosZ if exp.uLoopPars.bUseZ else 0.0,
                                ),
                                pfsOffset=pos.dPFSOffset,
                                name=pos.dPosName if pos.dPosName else None,
                            )
                        )
                ep = XYPosLoopParams(isSettingZ=exp.uLoopPars.bUseZ, points=points)
                e = XYPosLoop(count=len(points), nestingLevel=count, parameters=ep)
                count += 1
                exps.append(e)
            elif exp.eType == ExperimentLoopType.eEtSpectLoop:
                pass  # skipped on purpose
            else:
                print(__file__, f"Experiment {exp.name} not implemented.")
        return exps

    def events(
        self,
        *,
        orient: Literal["records", "list", "dict"] = "records",
        null_value: Any = float("nan"),
    ) -> ListOfDicts | DictOfLists | DictOfDicts:
        if orient not in ("records", "dict", "list"):
            raise ValueError("orient must be one of 'records', 'dict', or 'list'")
        self._ensure_open()
        rec = self.limnd2.recordedData
        if not rec:
            records: list[dict[str, Any]] = []
        else:
            records = [
                {item.ID: item.data[i] if i < len(item.data) else null_value for item in rec}
                for i in range(rec.rowCount)
            ]
        if orient == "records":
            return records
        if orient == "list":
            return _convert_records_to_dict_of_lists(records, null_value)
        return _convert_records_to_dict_of_dicts(records, null_value)

    def unstructured_metadata(
        self,
        *,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        out: dict[str, Any] = {}
        chunker = self.limnd2.chunker
        for name in chunker.chunk_names:
            if not BaseChunker._is_chunk_data(name):
                continue
            if BaseChunker.isSkipChunk(name):
                continue
            name_str = name.decode("ascii", errors="ignore")
            if include is not None and name_str not in include:
                continue
            if exclude is not None and name_str in exclude:
                continue
            data = chunker.chunk(name)
            if data is None:
                continue
            if isinstance(data, memoryview):
                data = data.tobytes()
            decoded: Any | None = None
            try:
                decoded = decode_lv(data)
            except Exception:
                try:
                    decoded = decode_var(data)
                except Exception:
                    try:
                        decoded = json.loads(data.decode("utf-8"))
                    except Exception:
                        decoded = data
            if strip_prefix:
                decoded = _strip_prefix_dict(decoded)
            out[name_str] = decoded
        return out

    @cached_property
    def metadata(self) -> Metadata:
        self._ensure_open()
        attrs = self.attributes
        pm = self.limnd2.pictureMetadata
        channel_count = attrs.channelCount or 1
        contents = Contents(channelCount=channel_count, frameCount=attrs.sequenceCount)
        channels: list[Channel] = []
        loops = self._loop_indices_for_metadata()

        if pm is not None:
            for idx, plane in enumerate(pm.channels):
                ch_name = (
                    "RGB" if plane.uiCompCount == 3 else (plane.sDescription or f"Channel {idx}")
                )
                color = Color.from_abgr_u4(plane.uiColor)
                emission = plane.emissionWavelengthNm or None
                excitation = plane.excitationWavelengthNm or None
                channel_meta = ChannelMeta(
                    name=ch_name,
                    index=idx,
                    color=color,
                    emissionLambdaNm=emission,
                    excitationLambdaNm=excitation,
                )

                sample = pm.sampleSettings(plane)
                microscope = Microscope(
                    objectiveMagnification=(
                        sample.objectiveMagnification if sample else pm.dObjectiveMag
                    ),
                    objectiveName=(sample.objectiveName if sample else pm.wsObjectiveName),
                    objectiveNumericalAperture=(
                        sample.objectiveNumericAperture if sample else pm.dObjectiveNA
                    ),
                    zoomMagnification=(sample.dRelayLensZoom if sample else pm.dZoom),
                    immersionRefractiveIndex=(
                        sample.refractiveIndex if sample else pm.dRefractIndex1
                    ),
                    projectiveMagnification=None,
                    pinholeDiameterUm=(plane.dPinholeDiameter if plane.dPinholeDiameter > 0 else None),
                    modalityFlags=PicturePlaneModalityFlags.to_str_list(plane.uiModalityMask),
                )

                z_count = self.sizes.get(AXIS.Z, 1)
                voxel = self.voxel_size()
                volume = Volume(
                    axesCalibrated=(pm.bCalibrated, pm.bCalibrated, z_count > 1),
                    axesCalibration=(voxel.x, voxel.y, voxel.z),
                    axesInterpretation=("distance", "distance", "distance"),
                    bitsPerComponentInMemory=attrs.bitsPerComponentInMemory,
                    bitsPerComponentSignificant=attrs.bitsPerComponentSignificant,
                    cameraTransformationMatrix=(
                        pm.dStgLgCT11,
                        pm.dStgLgCT21,
                        pm.dStgLgCT12,
                        pm.dStgLgCT22,
                    ),
                    componentCount=self.components_per_channel,
                    componentDataType=attrs.pixelDataType,
                    voxelCount=(attrs.widthPx or 0, attrs.heightPx, z_count),
                    componentMaxima=(
                        self._component_maxima() if attrs.componentCount else None
                    ),
                    componentMinima=(
                        self._component_minima() if attrs.componentCount else None
                    ),
                    pixelToStageTransformationMatrix=None,
                )

                channels.append(
                    Channel(
                        channel=channel_meta,
                        loops=loops,
                        microscope=microscope,
                        volume=volume,
                    )
                )

        return Metadata(contents=contents, channels=channels)

    def frame_metadata(self, seq_index: int | tuple) -> FrameMetadata | dict:
        self._ensure_open()
        idx = cast(
            "int",
            (
                self._seq_index_from_coords(seq_index)
                if isinstance(seq_index, tuple)
                else seq_index
            ),
        )
        pm = self._picture_metadata_for_seq(idx)
        base_meta = self.metadata
        loop_map = self.loop_indices[idx] if idx < len(self.loop_indices) else {}
        loops = LoopIndices(
            NETimeLoop=None,
            TimeLoop=loop_map.get(AXIS.TIME),
            XYPosLoop=loop_map.get(AXIS.POSITION),
            ZStackLoop=loop_map.get(AXIS.Z),
        )

        position = Position(
            stagePositionUm=StagePosition(
                x=pm.dXPos,
                y=pm.dYPos,
                z=pm.dZPos,
            ),
            pfsOffset=None,
            name=None,
        )
        time = TimeStamp(
            absoluteJulianDayNumber=pm.dTimeAbsolute,
            relativeTimeMs=pm.dTimeMSec,
        )

        channels = []
        for ch in base_meta.channels or []:
            channels.append(
                FrameChannel(
                    channel=ch.channel,
                    loops=loops,
                    microscope=ch.microscope,
                    volume=ch.volume,
                    position=position,
                    time=time,
                )
            )
        return FrameMetadata(contents=base_meta.contents or Contents(0, 0), channels=channels)

    @cached_property
    def custom_data(self) -> dict[str, Any]:
        self._ensure_open()
        out: dict[str, Any] = {}
        for name in (ND2_CHUNK_NAME_CustomDataVar, ND2_CHUNK_NAME_CustomDataVarLI):
            data = self.limnd2.chunk(name)
            if data is None:
                continue
            if isinstance(data, memoryview):
                data = data.tobytes()
            try:
                decoded = decode_var(data)
            except Exception:
                decoded = {}
            out[name.decode("ascii", errors="ignore")] = decoded
        if self.limnd2.customDescription is not None:
            out["CustomDescription"] = [item.valueAsText for item in self.limnd2.customDescription]
        if self.limnd2.smartExperimentDescription is not None:
            out["SmartExperiment"] = self.limnd2.smartExperimentDescription
        return out

    def jobs(self):
        """Return JOBS metadata if available, else None.

        limnd2 compatibility layer does not currently parse JOBS metadata,
        so this returns None for now.
        """
        return None

    @cached_property
    def ndim(self) -> int:
        return len(self.shape)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return self._coord_shape + self._frame_shape

    def _coord_info(self) -> list[tuple[int, str, int]]:
        return [(i, x.type, x.count) for i, x in enumerate(self.experiment)]

    @cached_property
    def sizes(self) -> Mapping[str, int]:
        attrs = self.attributes
        dims = {AXIS._MAP[c[1]]: c[2] for c in self._coord_info()}
        seq_count = attrs.sequenceCount or 1
        coord_dims = {k: v for k, v in dims.items() if k != AXIS.CHANNEL}
        coord_product = int(np.prod(list(coord_dims.values()))) if coord_dims else 1
        if coord_product != seq_count:
            text_dims = _parse_textinfo_dims(
                self.text_info.get("description", "") if self.text_info else ""
            )
            if text_dims:
                for axis in (AXIS.TIME, AXIS.Z, AXIS.POSITION):
                    if axis in text_dims and text_dims[axis] > 0:
                        dims[axis] = text_dims[axis]
                coord_dims = {k: v for k, v in dims.items() if k != AXIS.CHANNEL}
                coord_product = (
                    int(np.prod(list(coord_dims.values()))) if coord_dims else 1
                )
            if coord_product != seq_count and AXIS.Z not in dims:
                if coord_product > 0 and seq_count % coord_product == 0:
                    inferred = seq_count // coord_product
                    if inferred > 1:
                        dims[AXIS.Z] = inferred
        dims[AXIS.CHANNEL] = (
            dims.pop(AXIS.CHANNEL)
            if AXIS.CHANNEL in dims
            else (attrs.channelCount or 1)
        )
        dims[AXIS.Y] = attrs.heightPx
        dims[AXIS.X] = attrs.widthPx or -1
        if self.components_per_channel == 3:
            dims[AXIS.RGB] = self.components_per_channel
        else:
            dims[AXIS.CHANNEL] = attrs.componentCount
        return MappingProxyType({k: v for k, v in dims.items() if v != 1})

    @property
    def is_rgb(self) -> bool:
        return self.components_per_channel in (3, 4)

    @property
    def components_per_channel(self) -> int:
        attrs = self.attributes
        channel_count = attrs.channelCount or 1
        return attrs.componentCount // channel_count

    @property
    def size(self) -> int:
        return int(math.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @cached_property
    def dtype(self) -> np.dtype:
        self._ensure_open()
        return np.dtype(self.limnd2.imageAttributes.dtype)

    def voxel_size(self, channel: int = 0) -> VoxelSize:
        self._ensure_open()
        pm = self.limnd2.pictureMetadata
        dx = dy = 1.0
        if pm is not None and pm.bCalibrated and pm.dCalibration > 0:
            dx = dy = pm.dCalibration
        dz = 1.0
        if self.limnd2.experimentZStackLoop is not None:
            dz = self.limnd2.experimentZStackLoop.dZStep
        return VoxelSize(dx, dy, dz)

    def asarray(self, position: int | None = None) -> np.ndarray:
        final_shape = list(self.shape)
        if position is None:
            seqs: Sequence[int] = range(self._frame_count)
        else:
            if isinstance(position, str):
                try:
                    position = self._position_names().index(position)
                except ValueError as e:
                    raise ValueError(f"{position!r} is not a valid position name") from e
            try:
                pidx = list(self.sizes).index(AXIS.POSITION)
            except ValueError as exc:
                if position > 0:
                    raise IndexError(
                        f"Position {position} is out of range. "
                        f"Only 1 position available"
                    ) from exc
                seqs = range(self._frame_count)
            else:
                if position >= self.sizes[AXIS.POSITION]:
                    raise IndexError(
                        f"Position {position} is out of range. "
                        f"Only {self.sizes[AXIS.POSITION]} positions available"
                    )

                ranges: list[range | tuple] = [range(x) for x in self._coord_shape]
                ranges[pidx] = (position,)
                coords = list(zip(*product(*ranges)))
                seqs = self._seq_index_from_coords(coords)  # type: ignore
                final_shape[pidx] = 1

        arr: np.ndarray = np.stack([self.read_frame(i) for i in seqs])
        return arr.reshape(final_shape)

    def __array__(self) -> np.ndarray:
        return self.asarray()

    def write_tiff(
        self,
        dest: str | PathLike,
        *,
        include_unstructured_metadata: bool = True,
        progress: bool = False,
        on_frame: Callable[[int, int, dict[str, int]], None] | None | None = None,
        modify_ome: Callable[[ome_types.OME], None] | None = None,
    ) -> None:
        try:
            import tifffile as tf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Missing optional dependency "tifffile" required for TIFF export.'
            ) from exc

        dest_path = Path(dest)
        output_ome = ".ome." in dest_path.name

        sizes = dict(self.sizes)
        n_positions = sizes.pop(AXIS.POSITION, 1)
        if sizes:
            axes, shape = zip(*sizes.items())
        else:
            axes, shape = (), ()
        metadata = {"axes": "".join(axes).upper().replace(AXIS.UNKNOWN, "Q")}

        ome_xml: bytes | None = None
        if output_ome:
            if self.is_legacy:
                warnings.warn(
                    "Cannot write OME metadata for legacy nd2 files. "
                    "Please use a different file extension to avoid confusion",
                    stacklevel=2,
                )
            else:
                ome = self.ome_metadata(
                    include_unstructured=include_unstructured_metadata,
                    tiff_file_name=dest_path.name,
                )
                if modify_ome:
                    modify_ome(ome)
                ome_xml_str = ome.to_xml(exclude_unset=True)
                # TIFF ImageDescription requires 7-bit ASCII.
                try:
                    ome_xml_str.encode("ascii")
                except UnicodeEncodeError:
                    ome_xml_str = ome_xml_str.encode(
                        "ascii", "xmlcharrefreplace"
                    ).decode("ascii")
                ome_xml = ome_xml_str.encode("utf-8")

        total = self._frame_count
        loop_indices = self.loop_indices
        if len(loop_indices) != total:
            axes_for_frames = [
                k for k in self.sizes.keys() if k not in {AXIS.Y, AXIS.X, AXIS.RGB}
            ]
            ranges = [range(self.sizes[k]) for k in axes_for_frames]
            loop_indices = tuple(
                dict(zip(axes_for_frames, idx)) for idx in product(*ranges)
            )

        if progress:
            try:
                from tqdm import tqdm  # type: ignore
            except Exception:
                tqdm = None
        else:
            tqdm = None

        pbar = tqdm(total=total, desc=f"Exporting {self.path}") if tqdm else None

        p_groups: dict[int, list[tuple[int, dict[str, int]]]] = {}
        for f_num, f_index in enumerate(loop_indices):
            p_groups.setdefault(f_index.get(AXIS.POSITION, 0), []).append(
                (f_num, f_index)
            )
        if sum(len(v) for v in p_groups.values()) != total:
            p_groups = {0: [(i, {}) for i in range(total)]}
            n_positions = 1

        def position_iter(p: int):
            for f_num, f_index in p_groups.get(p, []):
                if on_frame is not None:
                    on_frame(f_num, total, f_index)
                yield self.read_frame(f_num)
                if pbar is not None:
                    pbar.set_description(repr(f_index))
                    pbar.update()

        tf_ome = False if ome_xml else None
        pixelsize = self.voxel_size().x
        photometric = tf.PHOTOMETRIC.RGB if self.is_rgb else tf.PHOTOMETRIC.MINISBLACK
        try:
            with tf.TiffWriter(dest_path, bigtiff=True, ome=tf_ome) as tif:
                for p in range(n_positions):
                    tif.write(
                        iter(position_iter(p)),
                        shape=shape,
                        dtype=self.dtype,
                        resolution=(1 / pixelsize, 1 / pixelsize),
                        resolutionunit=tf.RESUNIT.MICROMETER,
                        photometric=photometric,
                        metadata=metadata,
                        description=ome_xml,
                    )
        finally:
            if pbar is not None:
                pbar.close()

    def to_dask(self, wrapper: bool = True, copy: bool = True):
        try:
            from dask.array.core import map_blocks
        except ImportError as exc:  # pragma: no cover
            raise ImportError("dask is required for to_dask()") from exc

        chunks = [(1,) * x for x in self._coord_shape]
        chunks += [(x,) for x in self._frame_shape]
        dask_arr = map_blocks(
            self._dask_block,
            copy=copy,
            chunks=chunks,
            dtype=self.dtype,
        )
        if wrapper:
            try:
                from resource_backed_dask_array import ResourceBackedDaskArray
            except Exception:
                return dask_arr
            return ResourceBackedDaskArray.from_array(dask_arr, self)
        return dask_arr

    _NO_IDX = -1

    def _seq_index_from_coords(self, coords: Sequence) -> Sequence[int] | SupportsInt:
        if not self._coord_shape:
            return self._NO_IDX
        return np.ravel_multi_index(coords, self._coord_shape)  # type: ignore

    def _dask_block(self, copy: bool, block_id: tuple[int]) -> np.ndarray:
        if isinstance(block_id, np.ndarray):
            return None
        with self._lock:
            was_closed = self.closed
            if self.closed:
                self.open()
            try:
                ncoords = len(self._coord_shape)
                idx = self._seq_index_from_coords(block_id[:ncoords])

                if idx == self._NO_IDX:
                    if any(block_id):
                        raise ValueError(
                            f"Cannot get chunk {block_id} for single frame image."
                        )
                    idx = 0
                data = self.read_frame(int(idx))  # type: ignore
                data = data.copy() if copy else data
                return data[(np.newaxis,) * ncoords]
            finally:
                if was_closed:
                    self.close()

    def to_xarray(
        self,
        delayed: bool = True,
        squeeze: bool = True,
        position: int | None = None,
        copy: bool = True,
    ):
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover
            raise ImportError("xarray is required for to_xarray()") from exc

        data = self.to_dask(copy=copy) if delayed else self.asarray(position)
        dims = list(self.sizes)
        coords = self._expand_coords(squeeze)
        if not squeeze:
            for missing_dim in set(coords).difference(dims):
                dims.insert(0, missing_dim)
            missing_axes = len(dims) - data.ndim
            if missing_axes > 0:
                data = data[(np.newaxis,) * missing_axes]

        if position is not None and not delayed and AXIS.POSITION in coords:
            coords[AXIS.POSITION] = [coords[AXIS.POSITION][position]]

        x = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            attrs={
                "metadata": {
                    "metadata": self.metadata,
                    "experiment": self.experiment,
                    "attributes": self.attributes,
                    "text_info": self.text_info,
                }
            },
        )
        if delayed and position is not None and AXIS.POSITION in coords:
            x = x.isel({AXIS.POSITION: [position]})
        return x.squeeze() if squeeze else x

    @property
    def _raw_frame_shape(self) -> tuple[int, int, int, int]:
        attr = self.attributes
        return (
            attr.heightPx,
            attr.widthPx or -1,
            attr.channelCount or 1,
            self.components_per_channel,
        )

    @property
    def _frame_shape(self) -> tuple[int, ...]:
        return tuple(v for k, v in self.sizes.items() if k in AXIS.frame_coords())

    @cached_property
    def _coord_shape(self) -> tuple[int, ...]:
        return tuple(v for k, v in self.sizes.items() if k not in AXIS.frame_coords())

    @property
    def _frame_count(self) -> int:
        if hasattr(self.limnd2, "_seq_count"):
            return cast("int", self.limnd2._seq_count())
        return int(np.prod(self._coord_shape)) if self._coord_shape else 1

    def _get_frame(self, index: SupportsInt) -> np.ndarray:  # pragma: no cover
        warnings.warn(
            'Use of "_get_frame" is deprecated, use the public "read_frame" instead.',
            stacklevel=2,
        )
        return self.read_frame(index)

    def read_frame(self, frame_index: SupportsInt) -> np.ndarray:
        self._ensure_open()
        frame = self.limnd2.image(int(frame_index))
        frame = np.asarray(frame)
        raw_shape = self._raw_frame_shape
        if frame.size == int(np.prod(raw_shape)):
            frame = frame.reshape(raw_shape)
            frame = frame.transpose((2, 0, 1, 3)).squeeze()
            return frame
        if frame.ndim == 3:
            h = self.attributes.heightPx
            w = self.attributes.widthPx or frame.shape[1]
            ch = self.attributes.channelCount or 1
            comp = self.attributes.componentCount
            comps_per_channel = max(1, comp // ch)
            # Squeeze trailing singleton channel axis
            if frame.shape[-1] == 1:
                frame = frame[..., 0]
                if frame.ndim == 2:
                    return frame

            # RGB or multi-component per channel -> prefer channel-last (Y,X,S)
            if comps_per_channel in (3, 4):
                if frame.shape[0] == comps_per_channel and frame.shape[1] == h and frame.shape[2] == w:
                    return np.moveaxis(frame, 0, -1)
                if frame.shape[0] == h and frame.shape[1] == comps_per_channel and frame.shape[2] == w:
                    return np.moveaxis(frame, 1, -1)
                if frame.shape[0] == h and frame.shape[1] == w and frame.shape[2] == comps_per_channel:
                    return frame
                if frame.shape[0] == w and frame.shape[1] == h and frame.shape[2] == comps_per_channel:
                    return frame.transpose(1, 0, 2)
                return frame

            # Multi-channel grayscale -> channel-first (C,Y,X)
            if ch > 1:
                # H x W x C
                if frame.shape[0] == h and frame.shape[1] == w and frame.shape[2] == ch:
                    return np.moveaxis(frame, -1, 0)
                # H x C x W
                if frame.shape[0] == h and frame.shape[1] == ch and frame.shape[2] == w:
                    return frame.transpose(1, 0, 2)
                # W x C x H
                if frame.shape[0] == w and frame.shape[1] == ch and frame.shape[2] == h:
                    return frame.transpose(1, 2, 0)
                # C x H x W (already)
                if frame.shape[0] == ch and frame.shape[1] == h and frame.shape[2] == w:
                    return frame

            # Single-channel grayscale
            if frame.shape[0] == h and frame.shape[1] == w:
                return frame
            if frame.shape[0] == w and frame.shape[1] == h:
                return frame.transpose(1, 0, 2)

        return frame.squeeze()

    @cached_property
    def loop_indices(self) -> tuple[dict[str, int], ...]:
        axes = [AXIS._MAP[x.type] for x in self.experiment]
        indices = product(*(range(x.count) for x in self.experiment))
        return tuple(dict(zip(axes, x)) for x in indices)

    def _expand_coords(self, squeeze: bool = True) -> dict:
        dx, dy, dz = self.voxel_size()

        coords: dict[str, Any] = {
            AXIS.Y: np.arange(self.attributes.heightPx) * dy,
            AXIS.X: np.arange(self.attributes.widthPx or 1) * dx,
            AXIS.CHANNEL: self._channel_names,
            AXIS.POSITION: ["XYPos:0"],
        }

        for c in self.experiment:
            if squeeze and c.count <= 1:
                continue
            if c.type == "ZStackLoop":
                coords[AXIS.Z] = np.arange(c.count) * c.parameters.stepUm
            elif c.type == "TimeLoop":
                coords[AXIS.TIME] = np.arange(c.count) * (
                    c.parameters.periodDiff.avg
                    if c.parameters.periodDiff.avg is not None
                    else c.parameters.periodMs
                )
            elif c.type == "NETimeLoop":
                pers = [
                    np.arange(p.count)
                    * (
                        p.periodDiff.avg if p.periodDiff.avg is not None else p.periodMs
                    )
                    for p in c.parameters.periods
                ]
                coords[AXIS.TIME] = np.hstack(pers)
            elif c.type == "XYPosLoop":
                coords[AXIS._MAP["XYPosLoop"]] = self._position_names(c)

        if self.components_per_channel > 1:
            coords[AXIS.RGB] = ["Red", "Green", "Blue", "alpha"][
                : self.components_per_channel
            ]

        if AXIS.Z in self.sizes and AXIS.Z not in coords:
            coords[AXIS.Z] = np.arange(self.sizes[AXIS.Z]) * dz

        if squeeze:
            coords = {k: v for k, v in coords.items() if len(v) > 1}
        return coords

    def _position_names(self, loop: XYPosLoop | None = None) -> list[str]:
        if loop is None:
            for c in self.experiment:
                if c.type == "XYPosLoop":
                    loop = c
                    break
        if loop is None:
            return ["XYPos:0"]
        return [p.name or f"XYPos:{i}" for i, p in enumerate(loop.parameters.points)]

    @property
    def _channel_names(self) -> list[str]:
        return [c.channel.name for c in self.metadata.channels or []]

    def __repr__(self) -> str:
        try:
            details = " (closed)" if self.closed else f" {self.dtype}: {self.sizes!r}"
            extra = f": {Path(self.path).name!r}{details}"
        except Exception:
            extra = ""
        return f"<ND2File at {hex(id(self))}{extra}>"

    @cached_property
    def binary_data(self):
        self._ensure_open()
        meta = self.limnd2.binaryRasterMetadata
        if not meta or len(meta) == 0:
            meta = self.limnd2.binaryRleMetadata
        if not meta or len(meta) == 0:
            return None

        chunk_names = set(self.limnd2.chunker.chunk_names)
        rle_present: set[tuple[int, int]] = set()
        raster_present: set[tuple[int, int]] = set()
        if hasattr(self.limnd2, "binaryRleMetadata"):
            regex_dict = self.limnd2.binaryRleMetadata.dataChunkNameRegexDict
            for name in chunk_names:
                if hasattr(BaseChunker, "isBinaryRleDataChunk"):
                    hit = BaseChunker.isBinaryRleDataChunk(regex_dict, name)
                else:
                    hit = BaseChunker.isBinaryRleData(regex_dict, name)  # type: ignore[attr-defined]
                if hit:
                    rle_present.add(hit)
        for name in chunk_names:
            hit = BaseChunker.isBinaryRasterData(name)
            if hit:
                raster_present.add((hit[0], hit[1]))

        seq_count = self.attributes.sequenceCount
        coord_shape = self._coord_shape
        if coord_shape and int(np.prod(coord_shape)) != seq_count:
            coord_shape = (seq_count,)

        layers: list[BinaryLayer] = []
        for item in meta:
            data_list: list[np.ndarray | None] = []
            for seq in range(seq_count):
                present = (item.id, seq) in raster_present or (item.id, seq) in rle_present
                if not present:
                    data_list.append(None)
                    continue
                try:
                    arr = self.limnd2.binaryRasterData(item.id, seq)
                except (BinaryIdNotFountError, NameNotInChunkmapError):
                    data_list.append(None)
                    continue
                data_list.append(np.asarray(arr))
            layers.append(
                BinaryLayer(
                    data=data_list,
                    name=item.name,
                    file_tag=getattr(item, "strFileTag", ""),
                    comp_name=getattr(item, "binComp", None),
                    comp_order=getattr(item, "binCompOrder", None),
                    color=getattr(item, "binColor", None),
                    color_mode=getattr(item, "binColorMode", None),
                    state=getattr(item, "binState", None),
                    layer_id=item.id,
                    coordinate_shape=coord_shape,
                )
            )

        return BinaryLayers(layers)

    def ome_metadata(
        self, *, include_unstructured: bool = True, tiff_file_name: str | None = None
    ):
        try:
            import ome_types.model as m  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ome-types is required for ome_metadata()") from exc

        if self.is_legacy:
            raise NotImplementedError("OME metadata is not available for legacy files")

        sizes = dict(self.sizes)
        size_x = sizes.get(AXIS.X, 1)
        size_y = sizes.get(AXIS.Y, 1)
        size_z = sizes.get(AXIS.Z, 1)
        size_t = sizes.get(AXIS.TIME, 1)
        size_c = sizes.get(AXIS.CHANNEL, 1)

        voxel = self.voxel_size()
        channels = []
        for ch in self.metadata.channels or []:
            channel = m.Channel(
                id=f"Channel:{ch.channel.index}",
                name=ch.channel.name,
                samples_per_pixel=sizes.get(AXIS.RGB, 1),
            )
            if not self.is_rgb:
                channel.color = m.Color(ch.channel.color)
                if ch.channel.emissionLambdaNm is not None:
                    channel.emission_wavelength = ch.channel.emissionLambdaNm
                    channel.emission_wavelength_unit = m.UnitsLength.NANOMETER
                if ch.channel.excitationLambdaNm is not None:
                    channel.excitation_wavelength = ch.channel.excitationLambdaNm
                    channel.excitation_wavelength_unit = m.UnitsLength.NANOMETER
            channels.append(channel)

        planes = []
        tiff_blocks = []
        if tiff_file_name is not None:
            import uuid
            uuid_ = f"urn:uuid:{uuid.uuid4()}"

        ifd = 0
        for t in range(size_t):
            for z in range(size_z):
                for c in range(size_c):
                    planes.append(m.Plane(the_z=z, the_t=t, the_c=c))
                    if tiff_file_name is not None:
                        tiff_blocks.append(
                            m.TiffData(
                                uuid=m.TiffData.UUID(value=uuid_, file_name=tiff_file_name),
                                ifd=ifd,
                                first_c=c,
                                first_t=t,
                                first_z=z,
                                plane_count=1,
                            )
                        )
                        ifd += 1

        pixels = m.Pixels(
            id="Pixels:0",
            channels=channels,
            planes=planes,
            tiff_data_blocks=tiff_blocks,
            dimension_order=m.Pixels_DimensionOrder.XYCZT,
            type=str(self.dtype),
            significant_bits=self.attributes.bitsPerComponentSignificant,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_c=size_c,
            size_t=size_t,
            physical_size_x=voxel.x,
            physical_size_y=voxel.y,
            physical_size_x_unit=m.UnitsLength.MICROMETER,
            physical_size_y_unit=m.UnitsLength.MICROMETER,
        )
        if AXIS.Z in sizes:
            pixels.physical_size_z = voxel.z
            pixels.physical_size_z_unit = m.UnitsLength.MICROMETER

        image = m.Image(id="Image:0", name=Path(self.path).stem, pixels=pixels)
        ome = m.OME(images=[image], creator="limnd2 ND2File")

        if include_unstructured:
            all_meta = m.MapAnnotation(
                description="ND2 unstructured metadata, encoded as a JSON string.",
                namespace="https://github.com/Laboratory-Imaging/limnd2",
                value={
                    k: json.dumps(v, default=str)
                    for k, v in self.unstructured_metadata().items()
                },
            )
            ome.structured_annotations = m.StructuredAnnotations(map_annotations=[all_meta])

        return ome

    def _picture_metadata_for_seq(self, seq_index: int) -> PictureMetadata:
        data = self.limnd2.chunk(ND2_CHUNK_FORMAT_ImageMetadataLV_1p % (seq_index))
        if data is not None:
            return PictureMetadata.from_lv(data)
        data = self.limnd2.chunk(ND2_CHUNK_FORMAT_ImageMetadata_1p % (seq_index))
        if data is not None:
            return PictureMetadata.from_var(data)
        return self.limnd2.pictureMetadata

    def _loop_indices_for_metadata(self) -> LoopIndices | None:
        if not self.experiment:
            return None
        loops = {}
        for idx, loop in enumerate(self.experiment):
            if loop.type == "TimeLoop":
                loops["TimeLoop"] = idx
            elif loop.type == "NETimeLoop":
                loops["NETimeLoop"] = idx
            elif loop.type == "ZStackLoop":
                loops["ZStackLoop"] = idx
            elif loop.type == "XYPosLoop":
                loops["XYPosLoop"] = idx
        return LoopIndices(**loops) if loops else None

    def _component_minima(self) -> list[float]:
        try:
            return [float(x) for x in self.limnd2.compRange[:, 0]]
        except Exception:
            return []

    def _component_maxima(self) -> list[float]:
        try:
            return [float(x) for x in self.limnd2.compRange[:, 1]]
        except Exception:
            return []


class BinaryLayer:
    def __init__(
        self,
        *,
        data: list[np.ndarray | None],
        name: str,
        file_tag: str,
        comp_name: str | None,
        comp_order: int | None,
        color: int | None,
        color_mode: int | None,
        state: int | None,
        layer_id: int | None,
        coordinate_shape: tuple[int, ...],
    ) -> None:
        self.data = data
        self.name = name
        self.file_tag = file_tag
        self.comp_name = comp_name
        self.comp_order = comp_order
        self.color = color
        self.color_mode = color_mode
        self.state = state
        self.layer_id = layer_id
        self.coordinate_shape = coordinate_shape

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int) -> np.ndarray | None:
        return self.data[key]

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return next((s.shape for s in self.data if s is not None), (0, 0))

    def __array__(self) -> np.ndarray:
        ary = self.asarray()
        return ary if ary is not None else np.ndarray([])

    def asarray(self) -> np.ndarray | None:
        frame_shape = self.frame_shape
        if frame_shape == (0, 0):
            return None
        d = [i if i is not None else np.zeros(frame_shape, dtype="uint16") for i in self.data]
        return cast("np.ndarray", np.stack(d).reshape(self.coordinate_shape + frame_shape))


class BinaryLayers(Sequence[BinaryLayer]):
    def __init__(self, data: list[BinaryLayer]) -> None:
        self._data = data

    def __getitem__(self, key: int | slice) -> BinaryLayer | list[BinaryLayer]:
        return self._data[key]

    def __iter__(self) -> Iterable[BinaryLayer]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} with {len(self)} layers>"

    def __array__(self) -> np.ndarray:
        return self.asarray()

    def asarray(self) -> np.ndarray:
        out = []
        for bin_layer in self._data:
            d = bin_layer.asarray()
            if d is not None:
                out.append(d)
        return np.stack(out)
