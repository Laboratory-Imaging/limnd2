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

# ORIGINAL TYPES FOR ND2

from __future__ import annotations

from functools import cached_property
from typing import Callable, SupportsInt

import numpy as np

from limnd2.experiment import ExperimentLoopType, ExperimentTimeLoop, ZStackType
from .nd2file_types import *

# IMPORTS FOR LIMND2
from limnd2.attributes import ImageAttributesCompression, ImageAttributesPixelType
from limnd2.nd2 import Nd2Reader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ome_types

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
        self.limnd2 = Nd2Reader(path)

    @staticmethod
    def is_supported_file(path: StrOrPath) -> bool:
        raise NotImplementedError("Method is_supported_file not implemented")

    @cached_property
    def version(self) -> tuple[int, ...]:
        """works"""
        return self.limnd2.version

    @property
    def path(self) -> str:
        """works"""
        return str(self._path)

    @property
    def is_legacy(self) -> bool:
        return False

    def open(self) -> None:
        return

    def close(self) -> None:
        return

    @property
    def closed(self) -> bool:
        raise NotImplementedError("Method closed not implemented")

    def __enter__(self) -> "ND2File":
        return self

    def __del__(self) -> None:
        pass

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        raise NotImplementedError("Method __getstate__ not implemented")

    def __setstate__(self, d: dict[str, Any]) -> None:
        raise NotImplementedError("Method __setstate__ not implemented")

    @cached_property
    def attributes(self) -> Attributes:
        """works"""
        att = self.limnd2.imageAttributes
        compressionLevel = att.dCompressionParam
        if att.eCompression == ImageAttributesCompression.ictLossy:
            compression = "lossy"
        elif att.eCompression == ImageAttributesCompression.ictLossLess:
            compression = "lossless"
        else:
            compression = None
            compressionLevel = None

        channel_count = self.limnd2.pictureMetadata.sPicturePlanes.uiCount

        return Attributes(
            bitsPerComponentInMemory = att.uiBpcInMemory,
            bitsPerComponentSignificant = att.uiBpcSignificant,
            componentCount = att.uiComp,
            heightPx = att.uiHeight,
            pixelDataType = "unsigned" if att.ePixelType == ImageAttributesPixelType.pxtUnsigned else "float",
            sequenceCount = att.uiSequenceCount,
            widthBytes = att.uiWidthBytes,
            widthPx = att.uiWidth,
            compressionLevel = compressionLevel,
            compressionType = compression,
            tileHeightPx = None if att.uiTileHeight == att.uiHeight else att.uiTileHeight,
            tileWidthPx = None if att.uiTileWidth == att.uiWidth else att.uiTileWidth,
            channelCount = channel_count
        )

    @cached_property
    def text_info(self) -> TextInfo:
        raise NotImplementedError("Method text_info not implemented")

    @cached_property
    def rois(self) -> dict[int, ROI]:
        raise NotImplementedError("Method rois not implemented")

    @cached_property
    def experiment(self) -> list[ExpLoop]:
        """works for T, M, Z experiments"""
        if self.limnd2.experiment == None:
            return []

        exps = []
        count = 0
        for exp in self.limnd2.experiment:
            if exp.eType == ExperimentLoopType.eEtTimeLoop:
                ep = TimeLoopParams(
                    startMs = exp.uLoopPars.dStart,
                    periodMs = exp.uLoopPars.dPeriod,
                    durationMs = exp.uLoopPars.dDuration,
                    periodDiff = PeriodDiff(
                        avg = exp.uLoopPars.dAvgPeriodDiff,
                        max = exp.uLoopPars.dMaxPeriodDiff,
                        min = exp.uLoopPars.dMinPeriodDiff
                    )
                )
                e = TimeLoop(
                    count = exp.uLoopPars.uiCount,
                    nestingLevel = count,
                    parameters = ep
                )
                count += 1
                exps.append(e)

            elif exp.eType == ExperimentLoopType.eEtZStackLoop:
                ep = ZStackLoopParams(
                    bottomToTop = exp.uLoopPars.iType in (ZStackType.zstBottomToTopFixedTop,
                                                          ZStackType.zstBottomToTopFixedBottom,
                                                          ZStackType.zstSymmetricRangeFixedHomeBottomToTop,
                                                          ZStackType.zstAsymmetricRangeFixedHomeBottomToTop),
                    homeIndex = exp.uLoopPars.homeIndex,
                    stepUm = exp.uLoopPars.dZStep,
                    deviceName = exp.uLoopPars.wsZDevice
                )
                e = ZStackLoop(
                    count = exp.uLoopPars.uiCount,
                    nestingLevel = count,
                    parameters = ep
                )
                count += 1
                exps.append(e)

            elif exp.eType == ExperimentLoopType.eEtXYPosLoop:
                points = []
                for index, pos in enumerate(exp.uLoopPars.Points):
                    if (not exp.pItemValid) or (exp.pItemValid and exp.pItemValid[index]):
                        points.append(Position(
                            stagePositionUm = StagePosition(x = pos.dPosX,
                                                            y = pos.dPosY,
                                                            z = pos.dPosZ if exp.uLoopPars.bUseZ else 0.0),
                            pfsOffset = pos.dPFSOffset,
                            name = pos.dPosName if pos.dPosName else None
                        ))
                ep = XYPosLoopParams(
                    isSettingZ = exp.uLoopPars.bUseZ,
                    points = points
                )
                e = XYPosLoop(
                    count = len(points),
                    nestingLevel = count,
                    parameters = ep
                )
                count += 1
                exps.append(e)
            elif exp.eType == ExperimentLoopType.eEtSpectLoop:
                pass    # skipped on purpose
            else:
                print(__file__, f"Experiment {exp.name} not implemented.")
        return exps


    def events(
        self,
        *,
        orient: Literal["records", "list", "dict"] = "records",
        null_value: Any = float("nan"),
        ) -> ListOfDicts | DictOfLists | DictOfDicts:
        raise NotImplementedError("Method events not implemented")

    def unstructured_metadata(
        self,
        *,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Method unstructured_metadata not implemented")

    @cached_property
    def metadata(self) -> Metadata:
        raise NotImplementedError("Method metadata not implemented")

    def frame_metadata(self, seq_index: int | tuple) -> FrameMetadata | dict:
        raise NotImplementedError("Method frame_metadata not implemented")

    @cached_property
    def custom_data(self) -> dict[str, Any]:
        raise NotImplementedError("Method custom_data not implemented")

    @cached_property
    def ndim(self) -> int:
        raise NotImplementedError("Method ndim not implemented")

    @cached_property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError("Method shape not implemented")

    def _coord_info(self) -> list[tuple[int, str, int]]:
        raise NotImplementedError("Method _coord_info not implemented")

    @cached_property
    def sizes(self) -> Mapping[str, int]:
        raise NotImplementedError("Method sizes not implemented")

    @property
    def is_rgb(self) -> bool:
        raise NotImplementedError("Method is_rgb not implemented")

    @property
    def components_per_channel(self) -> int:
        raise NotImplementedError("Method components_per_channel not implemented")

    @property
    def size(self) -> int:
        raise NotImplementedError("Method size not implemented")

    @property
    def nbytes(self) -> int:
        raise NotImplementedError("Method nbytes not implemented")

    @cached_property
    def dtype(self) -> np.dtype:
        raise NotImplementedError("Method dtype not implemented")

    def voxel_size(self, channel: int = 0) -> VoxelSize:
        raise NotImplementedError("Method voxel_size not implemented")

    def asarray(self, position: int | None = None) -> np.ndarray:
        raise NotImplementedError("Method asarray not implemented")

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
        raise NotImplementedError("Method write_tiff not implemented")

    def to_dask(self):
        raise NotImplementedError("Method to_dask not implemented")

    def to_xarray(self):
        raise NotImplementedError("Method to_xarray not implemented")

    def _raw_frame_shape(self):
        raise NotImplementedError("Method _raw_frame_shape not implemented")

    def _frame_shape(self):
        raise NotImplementedError("Method _frame_shape not implemented")

    def _coord_shape(self):
        raise NotImplementedError("Method _coord_shape not implemented")

    def _frame_count(self):
        raise NotImplementedError("Method _frame_count not implemented")

    def _get_frame(self):
        raise NotImplementedError("Method _get_frame not implemented")

    def read_frame(self, frame_index: SupportsInt) -> np.ndarray:
        raise NotImplementedError("Method read_frame not implemented")

    @cached_property
    def loop_indices(self) -> tuple[dict[str, int], ...]:
        raise NotImplementedError("Method loop_indices not implemented")

    def _expand_coords(self):
        raise NotImplementedError("Method _expand_coords not implemented")

    def _position_names(self):
        raise NotImplementedError("Method _position_names not implemented")

    def _channel_names(self):
        raise NotImplementedError("Method _channel_names not implemented")

    def __repr__(self) -> str:
        raise NotImplementedError("Method __repr__ not implemented")

    def binary_data(self):
        raise NotImplementedError("Method binary_data not implemented")

    def ome_metadata(self):
        raise NotImplementedError("Method ome_metadata not implemented")
