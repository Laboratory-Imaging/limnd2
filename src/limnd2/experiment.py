"""
This module stores classes and functions for handling experiment data in `.nd2` file.

Experiments in .nd2 files define how image sequences are organized and looped. The most common types of loops include:

- **Time Loop** (timeloop): A sequence of images captured over time.
- **Z-Stack** (zstack): Frames stacked along the z-axis, representing different focal planes.
- **Multi-Point** (multipoint): Images captured at multiple specified locations (points) with known coordinates.

An image can have no experiment, a single experiment, or a combination of multiple experiments.

This experiment data is stored in the [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel) class,
you can get instance of this class by getting [`experiment`](nd2.md#limnd2.nd2.Nd2Reader.experiment) peoperty of [`Nd2Reader`](nd2.md#limnd2.nd2.Nd2Reader.experiment) object.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, cast, overload
import collections, enum, itertools, json, math, zlib
from dataclasses import MISSING, asdict, dataclass, fields

from .metadata import PictureMetadataPicturePlanes, PicturePlaneDesc
from .lite_variant import decode_lv, encode_lv, LVSerializable, ELxLiteVariantType as LVType, LV_field
from .variant import decode_var

class ExperimentLoopType(enum.IntEnum):
    """
    Enum specifying which experiment type was used in [`ExperimentLevel`](experiment.md#limnd2.experiment.ExperimentLevel)
    Attributes
    ----------
    eEtTimeLoop : int
        Timeloop experiment (see [`ExperimentTimeLoop`](experiment.md#limnd2.experiment.ExperimentTimeLoop))
    eEtNETimeLoop : int
        Non-equidistant timeloop experiment (see [`ExperimentNETimeLoop`](experiment.md#limnd2.experiment.ExperimentNETimeLoop))
    eEtXYPosLoop : int
        Multipoint experiment (see [`ExperimentXYPosLoop`](experiment.md#limnd2.experiment.ExperimentXYPosLoop))
    eEtZStackLoop : int
        Z-stack experiment (see [`ExperimentZStackLoop`](experiment.md#limnd2.experiment.ExperimentZStackLoop))
    eEtSpectLoop : int
        Spectral experiment (see [`ExperimentSpectralLoop`](experiment.md#limnd2.experiment.ExperimentSpectralLoop))
    """
    eEtUnknown              = 0
    eEtTimeLoop             = 1
    eEtXYPosLoop            = 2
    eEtXYDiscrLoop          = 3
    eEtZStackLoop           = 4
    eEtPolarLoop            = 5
    eEtSpectLoop            = 6
    eEtCustomLoop           = 7
    eEtNETimeLoop           = 8
    eEtManTimeLoop          = 9
    eEtZStackLoopAccurate   = 10        # not used yet

    @staticmethod
    def toName(eType: ExperimentLoopType|int):
        names = [ '?', 't', 'm', 'm', 'z', '!', 's', 'c', 't', '!', '!' ]
        return names[eType]

    @staticmethod
    def toLongName(eType: ExperimentLoopType|int):
        """
        Returns long name for experiment type.
        """
        names = [ 'Unknown', 'Time', 'Multipoint', 'Multipoint', 'Z-Stack', 'Polar', 'Spectral', 'Custom', 'Time', 'Time', 'Z-Stack' ]
        return names[eType]

    @staticmethod
    def toShortName(eType: ExperimentLoopType|int):
        """
        Returns short name for experiment type.
        """
        names = [ '?', 'T', 'XY', 'XY', 'Z', 'P', 'λ', 'C', 'T', 'T', 'Z' ]
        return names[eType]

class ExperimentType(enum.IntFlag):
    eEtDefault              = 0
    eEtStimulation          = 1
    eEtBleaching            = 2
    eEtIncubation           = 2048
    eEtLiquidHandling       = 4096

def _format_time(ms) -> str:
    # whole seconds
    if 0 == (ms % 1_000) and (seconds := (ms / 1_000)) < 60:
        return f"{seconds:.0f} sec"
    if 0 == (ms % 60_000) and (minutes := (ms / 60_000)) < 60:
        return f"{minutes:.0f} min"
    elif 0 == (ms % 3_600_000) and (hours := (ms / 3_600_000)) < 60:
        return f"{hours:.0f} hr"
    else:
        ss = int((ms/1_000)%60)
        mm = int((ms/(60_000))%60)
        hh = int(ms/(3_600_000))
        ms -= (3_600_000)*hh + (60_000)*mm + 60*ss
        return "%d:%02d:%02d.%03d" % (hh, mm, ss, ms % 1000)

@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentLoop(LVSerializable):
    uiCount: int                    = LV_field(0,                         LVType.UINT32)

    @staticmethod
    def createExperimentLoop(eType: ExperimentLoopType, uLoopPars: dict):
        uLoopPars = uLoopPars[0] if type(uLoopPars) == list else uLoopPars
        if eType == ExperimentLoopType.eEtTimeLoop:
            return ExperimentTimeLoop(**uLoopPars)
        if eType == ExperimentLoopType.eEtNETimeLoop:
            return ExperimentNETimeLoop(**uLoopPars)
        elif eType == ExperimentLoopType.eEtZStackLoop:
            return ExperimentZStackLoop(**uLoopPars)
        elif eType == ExperimentLoopType.eEtXYPosLoop:
            return ExperimentXYPosLoop(**uLoopPars)
        elif eType == ExperimentLoopType.eEtSpectLoop:
            return ExperimentSpectralLoop(**uLoopPars)
        else:
            raise NotImplementedError(f'Loop type {eType} not implemented yet')

    @property
    def step(self) -> float|None:
        return None

    @property
    def stepUnit(self) -> str|None:
        return None

    @property
    def info(self) -> list[dict[str, Any]]:
        #override in child classes
        return []

@dataclass(frozen=True, kw_only=True)
class ExperimentTimeLoop(ExperimentLoop, LVSerializable):
    """
    Dataclass storing parameters about timeloop experiment in the image.

    **Attributes:**
    !!! note
        Only selected attributes are listed, for full list of attributes see class definition.

    Attributes
    ----------
    uiCount : int
        Number of frames in the experiment
    dStart : float
        Start time of the experiment
    dPeriod : float
        Time interval between frames in miliseconds.
    dDuration : float
        Duration of the experiment in miliseconds. If value is non-zero, dPeriod must be zero and the experiment is captured as fast as possible for dDuration ms.
    dMinPeriodDiff : float
        Minimum difference between periods.
    dMaxPeriodDiff : float
        Maximum difference between periods.
    dAvgPeriodDiff : float
        Average difference between periods.
    """
    dStart: float                   = LV_field(0,                         LVType.DOUBLE)
    dPeriod: float                  = LV_field(0,                         LVType.DOUBLE)
    dDuration: float                = LV_field(0,                         LVType.DOUBLE)
        # if dDuration is nonzero, then dPeriod must be zero and
        # means experiment is captured as fast as possible for dDuration ms

    dMinPeriodDiff: float           = LV_field(0,                         LVType.DOUBLE)
    dMaxPeriodDiff: float           = LV_field(0,                         LVType.DOUBLE)
    dAvgPeriodDiff: float           = LV_field(0,                         LVType.DOUBLE)
    wsPhaseName: str                = LV_field("",                        LVType.STRING)
    sAutoFocusBeforePeriod: dict    = LV_field(dict,                      LVType.ENCODING_NOT_IMPLEMENTED)
    sAutoFocusBeforeCapture: dict   = LV_field(dict,                      LVType.ENCODING_NOT_IMPLEMENTED)
    uiLoopType: ExperimentType      = LV_field(ExperimentType.eEtDefault, LVType.UINT32)
    # 0..default type, 1..Stimulation, 2..Bleaching, 32..Incubation

    uiGroup: int                    = LV_field(0,                         LVType.UINT32)
    # 0..no group, HIWORD(uiGroup)..Index of group from 1, LOWORD(uiGroup)..Index inside group

    uiStimulationCount: int         = LV_field(0,                         LVType.UINT32)
    bDurationPref: bool             = LV_field(None,                      LVType.BOOL)
    # If true, time loop will stop at the dDuration time regardless uiCount

    pIncubationData: bytes          = LV_field(b'',                       LVType.UNKNOWN)       # looks unused
    # parameters for incubation device

    uiIncubationDataSize: int       = LV_field(0,                         LVType.UNKNOWN)       # looks unused
    wsInterfaceName: str            = LV_field("",                        LVType.STRING)
    uiTreatment: int                = LV_field(0,                         LVType.UINT32)
    dIncubationDuration: float      = LV_field(-1.0,                      LVType.DOUBLE)

    def __post_init__(self):
        object.__setattr__(self, 'uiLoopType', ExperimentType(self.uiLoopType))

    @property
    def formattedInterval(self) -> str:
        """
        Returns formatted time interval between frames.
        """
        return _format_time(self.dPeriod) if 0.0 < self.dPeriod else 'No Delay'

    @property
    def formattedDuration(self) -> str:
        """
        Returns formatted duration of the experiment.
        """
        return _format_time(self.dDuration) if 0.0 < self.dDuration else 'Continuous'

    @property
    def step(self) -> float|None:
        return self.dAvgPeriodDiff if 0 <= self.dAvgPeriodDiff else self.dPeriod

    @property
    def stepUnit(self) -> str|None:
        return "ms"

    @property
    def info(self) -> list[dict[str, Any]]:
        """
        Returns information about timeloop experiment.
        """
        return [ dict(Phase='#1', Interval=self.formattedInterval, Duration=self.formattedDuration, Loops=self.uiCount) ]

    def __str__(self):
        return f"Timeloop experiment({self.uiCount} frames, interval: {self.formattedInterval}, duration: {self.formattedDuration})"


@dataclass(frozen=True, kw_only=True)
class ExperimentNETimeLoop(ExperimentLoop, LVSerializable):
    """
    Dataclass for storing parameters about nonequidistant time loop experiment.
    This is done by storing a list of [`ExperimentTimeLoop`](experiment.md#limnd2.experiment.ExperimentTimeLoop)
    instances (each one is called a period).

    **Attributes:**
    !!! note
        Only selected attributes are listed, for full list of attributes see class definition.

    Attributes
    ----------
    uiCount : int
        Number of frames in the experiment (total from frames in all periods)
    uiPeriodCount : int
        Number of periods in the experiment
    pPeriod : list[ExperimentTimeLoop]
        List of periods in the nonequidistant time loop experiment
    """
    uiPeriodCount: int                                      = LV_field(0,               LVType.UINT32)
    pPeriod: list[ExperimentTimeLoop]                       = LV_field(list,            LVType.LEVEL)
    pSubLoops: dict | None                                  = LV_field(None,            LVType.ENCODING_NOT_IMPLEMENTED)
    # list (if this list is empty, use ppNextLevelEx list from experiment for each time phase)

    sAutoFocusBeforePeriod: dict                            = LV_field(dict,            LVType.ENCODING_NOT_IMPLEMENTED)
    sAutoFocusBeforeCapture: dict                           = LV_field(dict,            LVType.ENCODING_NOT_IMPLEMENTED)
    wsCommandBeforePeriod: dict                             = LV_field(dict,            LVType.ENCODING_NOT_IMPLEMENTED)
    wsCommandAfterPeriod: dict                              = LV_field(dict,            LVType.ENCODING_NOT_IMPLEMENTED)
    pPeriodValid: bytes                                     = LV_field(bytes,           LVType.BYTEARRAY)

    def __post_init__(self):
        if isinstance(self.pPeriod, dict):
            periods = []
            for period in self.pPeriod.values():
                periods.append(ExperimentTimeLoop(**period))
            object.__setattr__(self, 'pPeriod', periods)

    @property
    def info(self) -> list[dict[str, Any]]:
        """
        Returns information about nonequidistant time loop experiment.
        """
        return [ period.info[0] for period in self.pPeriod ]


class ZStackType(enum.IntEnum):
    """
    Enumeration of Z stack movement types.

    Attributes
    ----------
    zstBottomToTopFixedTop : int
        Bottom -> Top stack with a fixed Top position.
    zstBottomToTopFixedBottom : int
        Bottom -> Top stack with a fixed Bottom position.
    zstSymmetricRangeFixedHomeBottomToTop : int
        Symmetric range around a fixed Home position (Bottom -> Top).
    zstAsymmetricRangeFixedHomeBottomToTop : int
        Asymmetric range around a fixed Home position (Bottom -> Top).
    zstTopToBottomFixedTop : int
        Top -> Bottom stack with a fixed Top position.
    zstTopToBottomFixedBottom : int
        Top -> Bottom stack with a fixed Bottom position.
    zstSymmetricRangeFixedHomeTopToBottom : int
        Symmetric range around a fixed Home position (Top -> Bottom).
    zstAsymmetricRangeFixedHomeTopToBottom : int
        Asymmetric range around a fixed Home position (Top -> Bottom).
    """
    zstBottomToTopFixedTop                  = 0
    zstBottomToTopFixedBottom               = 1
    zstSymmetricRangeFixedHomeBottomToTop   = 2
    zstAsymmetricRangeFixedHomeBottomToTop  = 3
    zstTopToBottomFixedTop                  = 4
    zstTopToBottomFixedBottom               = 5
    zstSymmetricRangeFixedHomeTopToBottom   = 6
    zstAsymmetricRangeFixedHomeTopToBottom  = 7

@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentZStackLoop(ExperimentLoop, LVSerializable):
    """
    Dataclass for storing parameters about Z-stack experiment in the image.

    **Attributes:**
    !!! note
        Only selected attributes are listed, for full list of attributes see class definition.

    Attributes
    ----------
    uiCount : int
        Number of frames in the experiment (total from frames in all periods)
    dZLow : float
        The lowest Z position in the experiment.
    dZLowPFSOffset : float
        Offset applied to the lowest Z position for Perfect Focus System (PFS).
    dZHigh : float
        The highest Z position in the experiment.
    dZHighPFSOffset : float
        Offset applied to the highest Z position for Perfect Focus System (PFS).
    dZHome : float
        The home (central) Z position in the experiment.
    dZStep : float
        Step size between Z positions in the experiment.
    """
    dZLow: float                            = LV_field(0.0,   LVType.DOUBLE)
    dZLowPFSOffset: float                   = LV_field(0.0,   LVType.DOUBLE)
    dZHigh: float                           = LV_field(0.0,   LVType.DOUBLE)
    dZHighPFSOffset: float                  = LV_field(0.0,   LVType.DOUBLE)
    dZHome: float                           = LV_field(0.0,   LVType.DOUBLE)
    dZStep: float                           = LV_field(0.0,   LVType.DOUBLE)
    dReferencePosition: float               = LV_field(0.0,   LVType.DOUBLE)
    dTIRFPosition: float                    = LV_field(0.0,   LVType.DOUBLE)
    dTIRFPFSOffset: float                   = LV_field(0.0,   LVType.DOUBLE)
    iType: ZStackType                       = LV_field(0,     LVType.INT32)
    bAbsolute: bool                         = LV_field(False, LVType.BOOL)
    bTriggeredPiezo: bool                   = LV_field(False, LVType.BOOL)
    bZInverted: bool                        = LV_field(False, LVType.BOOL)
    bTIRF: bool                             = LV_field(False, LVType.BOOL)
    wsZDevice: str                          = LV_field("",    LVType.STRING)
    wsCommandBeforeCapture: str             = LV_field("",    LVType.STRING)
    wsCommandAfterCapture: str              = LV_field("",    LVType.STRING)

    def __post_init__(self):

        if 'dZLow#1' in self._unknown_fields:
            self._unknown_fields.pop('dZLow#1')

        if "sCommandAfterCapture" in self._unknown_fields:
            object.__setattr__(self, 'wsCommandBeforeCapture', self._unknown_fields.pop('sCommandAfterCapture'))

        if "sCommandBeforeCapture" in self._unknown_fields:
            object.__setattr__(self, 'wsCommandBeforeCapture', self._unknown_fields.pop('sCommandBeforeCapture'))

        if "sZDevice" in self._unknown_fields:
            object.__setattr__(self, 'wsZDevice', self._unknown_fields.pop('sZDevice'))

        object.__setattr__(self, 'iType', ZStackType(self.iType))

    @property
    def homeIndex(self):
        """
        Returns index of the frame with home position.
        """
        tol = 0.05
        range = abs(self.dZHigh - self.dZLow)
        homeRangeF = abs(self.dZLow - self.dZHome)
        homeRangeI = abs(self.dZHigh - self.dZHome)
        if self.iType in (ZStackType.zstSymmetricRangeFixedHomeBottomToTop, ZStackType.zstAsymmetricRangeFixedHomeBottomToTop):
            if self.dZStep <= 0.0:
                return min(int((self.uiCount - 1) * (homeRangeI if self.bZInverted else homeRangeF) / range), self.uiCount - 1)
            else:
                return min(int(abs(math.ceil(((homeRangeI if self.bZInverted else homeRangeF) - tol*self.dZStep) / self.dZStep))), self.uiCount - 1)
        elif self.iType in (ZStackType.zstSymmetricRangeFixedHomeTopToBottom, ZStackType.zstAsymmetricRangeFixedHomeTopToBottom):
            if self.dZStep <= 0.0:
                return min(int((self.uiCount - 1) * (homeRangeF if self.bZInverted else homeRangeI) / range), self.uiCount - 1)
            else:
                return min(int(abs(math.ceil(((homeRangeF if self.bZInverted else homeRangeI) - tol*self.dZStep) / self.dZStep))), self.uiCount - 1)
        else:
            return (self.uiCount - 1) // 2

    @property
    def step(self) -> float|None:
        """
        Returns step size between Z positions in the experiment in micrometers.
        """
        dStep = self.dZStep
        uiCount = max(self.uiCount, 2)
        uiHome = self.homeIndex
        if self.iType in (ZStackType.zstSymmetricRangeFixedHomeBottomToTop, ZStackType.zstAsymmetricRangeFixedHomeBottomToTop, ZStackType.zstSymmetricRangeFixedHomeTopToBottom, ZStackType.zstAsymmetricRangeFixedHomeTopToBottom):
            dStep = abs(self.dZHome - self.dZLow) / uiHome if 0 < uiHome else 0
        else:
            dStep = abs(self.dZHigh - self.dZLow) / (uiCount - 1)
        return dStep

    @property
    def top(self):
        """
        Returns the highest Z position in the experiment.
        """
        if self.iType in (ZStackType.zstSymmetricRangeFixedHomeBottomToTop, ZStackType.zstAsymmetricRangeFixedHomeBottomToTop, ZStackType.zstSymmetricRangeFixedHomeTopToBottom, ZStackType.zstAsymmetricRangeFixedHomeTopToBottom):
            return self.dZHigh - self.dZHome
        else:
            return -self.dZLow + self.dReferencePosition if self.bZInverted else self.dZHigh + self.dReferencePosition

    @property
    def bottom(self):
        """
        Returns the lowest Z position in the experiment.
        """
        if self.iType in (ZStackType.zstSymmetricRangeFixedHomeBottomToTop, ZStackType.zstAsymmetricRangeFixedHomeBottomToTop, ZStackType.zstSymmetricRangeFixedHomeTopToBottom, ZStackType.zstAsymmetricRangeFixedHomeTopToBottom):
            return self.dZLow - self.dZHome
        else:
            return -self.dZHigh + self.dReferencePosition if self.bZInverted else self.dZLow + self.dReferencePosition

    @property
    def stepUnit(self) -> str|None:
        return "µm"

    @property
    def info(self) -> list[dict[str, Any]]:
        """
        Returns information about zstack experiment.
        """
        return [ dict(Step=self.step, Top=self.top, Bottom=self.bottom, Count=self.uiCount, Drive=self.wsZDevice)]

    def __str__(self):
        return f"Z-Stack experiment({self.uiCount} frames, step: {self.dZStep})"

@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentSpectralLoopPoint(LVSerializable):
    pAutoFocus: dict                            = LV_field(dict,                            LVType.ENCODING_NOT_IMPLEMENTED)
    pZStackPos: int                             = LV_field(0,                               LVType.INT32)
    pdOffset: float                             = LV_field(0.0,                             LVType.DOUBLE)
    wsCommandBeforeCapture: str                 = LV_field("",                              LVType.STRING)
    wsCommandAfterCapture: str                  = LV_field("",                              LVType.STRING)

    pass

    """
    Atributes found in XML variant, but not in LV
    pPlaneDesc: PicturePlaneDesc                = LV_field(PicturePlaneDesc,                LVType.LEVEL)             # TODO
    """

    def __post_init__(self):
        if "pPlaneDesc" in self._unknown_fields:
            self._unknown_fields.pop("pPlaneDesc")


@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentSpectralLoop(ExperimentLoop, LVSerializable):
    """
    Dataclass for storing parameters about spectral loop experiment in the image.
    """
    pPlanes: PictureMetadataPicturePlanes       = LV_field(PictureMetadataPicturePlanes,    LVType.LEVEL)
    iOffsetReference: int                       = LV_field(0,                               LVType.INT32)
    bMergeCameras: bool                         = LV_field(False,                           LVType.BOOL)
    bWaitForPFS: bool                           = LV_field(False,                           LVType.BOOL)
    bAskForFilter: bool                         = LV_field(False,                           LVType.BOOL)
    Points: list[ExperimentSpectralLoopPoint]   = LV_field(list,                            LVType.LEVEL)

    def __post_init__(self):
        if isinstance(self.Points, dict):
            points = [ExperimentSpectralLoopPoint(**point) for point in self.Points.values()]
            object.__setattr__(self, "Points", points)

        object.__setattr__(self, "pPlanes", PictureMetadataPicturePlanes(**self.pPlanes))


        uf = self._unknown_fields
        d = uf.get("pAutoFocus", None) or uf.get("szCommandBeforeCapture", None) or uf.get("szCommandAfterCapture", None) or uf.get("pZStackPos", None)
        if d:
            points = []
            for key in sorted(d.keys()):
                point = ExperimentSpectralLoopPoint(pAutoFocus = uf.get("pAutoFocus", {}).get(key, {}),
                                                    pZStackPos = uf.get("pZStackPos", {}).get(key, 0),
                                                    wsCommandBeforeCapture = uf.get("szCommandBeforeCapture", {}).get(key, ""),
                                                    wsCommandAfterCapture = uf.get("szCommandAfterCapture", {}).get(key, ""))
                points.append(point)
                object.__setattr__(self, "Points", points)

            if "pAutoFocus" in uf: uf.pop("pAutoFocus")
            if "szCommandBeforeCapture" in uf: uf.pop("szCommandBeforeCapture")
            if "szCommandAfterCapture" in uf: uf.pop("szCommandAfterCapture")
            if "pZStackPos" in uf: uf.pop("pZStackPos")

        if "pPlaneDesc" in self._unknown_fields:
            planes: dict = self._unknown_fields.pop("pPlaneDesc")
            planes_dict = {k: PicturePlaneDesc(**v) for k, v in planes.items() }
            object.__setattr__(self.pPlanes, "sPlaneNew", planes_dict)





    @property
    def info(self) -> list[dict[str, Any]]:
        """
        Returns information about spectral loop experiment.
        """
        ret = []
        for i, plane in enumerate(self.pPlanes.sPlaneNew):
            idx = f'#{i+1}'
            ocs = self.pPlanes.sSampleSetting[plane.uiSampleIndex].sOpticalConfigs if plane.uiSampleIndex < len(self.pPlanes.sSampleSetting) else []
            ret.append(dict(Index=idx, Name=plane.sDescription, OC=', '.join([oc.sOpticalConfigName for oc in ocs]), Color=plane.colorAsHtmlString))
        return ret

    def replacePlanes(self, picturePlanes: PictureMetadataPicturePlanes) -> None:
        object.__setattr__(self, 'pPlanes', picturePlanes)

@dataclass(frozen=True, kw_only=True)
class ExperimentXYPosLoopPoint(LVSerializable):
    """
    Dataclass for storing infomartion about a single point in multipoint experiment.

    Attributes
    ----------
    dPosX : float
        The X-coordinate of the position.
    dPosY : float
        The Y-coordinate of the position.
    dPosZ : float
        The Z-coordinate of the position.
    dPFSOffset : float
        The offset applied for Perfect Focus System (PFS).
    dPosName : str
        A descriptive name for the position.
    """
    dPosX: float                    = LV_field(0.0,               LVType.DOUBLE)
    dPosY: float                    = LV_field(0.0,               LVType.DOUBLE)
    dPosZ: float                    = LV_field(0.0,               LVType.DOUBLE)
    dPFSOffset: float               = LV_field(0.0,               LVType.DOUBLE)
    dPosName: str                   = LV_field("",                LVType.STRING)

    @staticmethod
    def create_points(points: dict):
        points_list = []
        for point in points.values():
            points_list.append(ExperimentXYPosLoopPoint(**point))
        return points_list

    @staticmethod
    def create_points_XML(xdict: dict[str, float],
                          ydict: dict[str, float],
                          zdict: dict[str, float],
                          offdict: dict[str, float],
                          namedict: dict[str, str]):
        points = []
        for key in xdict:
            points.append(ExperimentXYPosLoopPoint(dPosX=xdict[key],
                                                   dPosY=ydict[key],
                                                   dPosZ=zdict[key],
                                                   dPFSOffset=offdict[key],
                                                   dPosName=namedict[key]))
        return points

    def __str__(self):
        return f"[{self.dPosX:.1f}, {self.dPosY:.1f}]"



@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentXYPosLoop(ExperimentLoop, LVSerializable):
    """
    Dataclass for storing parameters about multipoint experiment in the image.

    Attributes
    ----------

    bUseZ : bool
        Whether Z-axis positions are used in the experiment.
    bRelativeXY : bool
        Whether XY coordinates are defined relative to a reference point.
    dReferenceX : float
        The X-coordinate of the reference point.
    dReferenceY : float
        The Y-coordinate of the reference point.
    bRedefineAfterPFS : bool
        Whether to redefine points after using the Perfect Focus System (PFS).
    bRedefineAfterAutoFocus : bool
        Whether to redefine points after performing autofocus.
    bKeepPFSOn : bool
        Whether to keep the Perfect Focus System (PFS) active during the experiment.
    bSplitMultipoints : bool
        Whether to split multipoints into separate groups or sequences.
    bUseAFPlane : bool
        Whether to use the autofocus plane for determining Z positions.
    bZEnabled : bool
        Whether the Z-axis is enabled in the experiment.
    sZDevice : str
        The name of the Z-axis device used in the experiment.
    sAFBefore : dict
        Settings for autofocus performed before the experiment.
    Points : list of ExperimentXYPosLoopPoint
        A list of points defining the XY positions for the experiment.
    """
    bUseZ: bool                             = LV_field(False,             LVType.BOOL)
    bRelativeXY: bool                       = LV_field(True,              LVType.BOOL)
    dReferenceX: float                      = LV_field(0.0,               LVType.DOUBLE)
    dReferenceY: float                      = LV_field(0.0,               LVType.DOUBLE)
    bRedefineAfterPFS: bool                 = LV_field(False,             LVType.BOOL)
    bRedefineAfterAutoFocus: bool           = LV_field(False,             LVType.BOOL)
    bKeepPFSOn: bool                        = LV_field(False,             LVType.BOOL)
    bSplitMultipoints: bool                 = LV_field(False,             LVType.BOOL)
    bUseAFPlane: bool                       = LV_field(False,             LVType.BOOL)
    bZEnabled: bool                         = LV_field(False,             LVType.BOOL)
    sZDevice: str                           = LV_field("",                LVType.STRING)
    sAFBefore: dict                         = LV_field(dict,              LVType.ENCODING_NOT_IMPLEMENTED)
    Points: list[ExperimentXYPosLoopPoint]  = LV_field(None,              LVType.LEVEL)

    pass
    """
    uiRelativeIdx: object                   = LV_field(None,              LVType.DO_NOT_ENCODE)       #TODO
    """

    def __post_init__(self):
        if self.Points and isinstance(self.Points, dict):
            object.__setattr__(self, 'Points', ExperimentXYPosLoopPoint.create_points(self.Points))

        if self.Points and isinstance(self.Points, list):
            object.__setattr__(self, 'Points', ExperimentXYPosLoopPoint.create_points({i: p for i, p in enumerate(self.Points)}))

        elif all(key in self._unknown_fields for key in ("dPosX", "dPosY", "dPosZ", "dPFSOffset", "pPosName")):
            object.__setattr__(self, 'Points', ExperimentXYPosLoopPoint.create_points_XML(self._unknown_fields.pop("dPosX"),
                                                                                          self._unknown_fields.pop("dPosY"),
                                                                                          self._unknown_fields.pop("dPosZ"),
                                                                                          self._unknown_fields.pop("dPFSOffset"),
                                                                                          self._unknown_fields.pop("pPosName")))

        if "sAutoFocusBeforeCapture" in self._unknown_fields:
            object.__setattr__(self, 'sAFBefore', self._unknown_fields.pop("sAutoFocusBeforeCapture"))

        if "uiRelativeIdx" in self._unknown_fields:
            self._unknown_fields.pop("uiRelativeIdx")


    @property
    def info(self) -> list[dict[str, Any]]:
        """
        Returns information about multipoint experiment.
        """
        ret = []
        for i in range(self.uiCount):
            name = self.Points[i].dPosName if i < len(self.Points) and self.Points[i].dPosName else f"#{i}"
            d = dict(Name=name, X=self.Points[i].dPosX, Y=self.Points[i].dPosY)
            if self.bUseZ and self.uiCount == len(self.Points):
                d['Z'] = self.Points[i].dPosZ
            ret.append(d)
        return ret

    def __str__(self):
        coords = ", ".join([str(point) for point in self.Points])
        return f"Multipoint experiment({self.uiCount} frames, point coordinates: {coords})"


@dataclass(frozen=True, kw_only=True)
class WellplateDesc:
    name: str = ""
    rows: int = 0
    columns: int = 0
    rowNaming: str = ''
    columnNaming: str = ''

    @staticmethod
    def from_lv(data: bytes|memoryview) -> WellplateDesc:
        decoded = decode_lv(data)
        return WellplateDesc(**decoded.get('PlateDesc', {}))

@dataclass(init=False, frozen=True)
class WellplateFrameInfoItem:
    plateIndex: int = 0
    plateUuid: str = ""
    seqIndex: int = 0
    wellIndex: int = 0
    wellName: str = ""
    wellColIndex: int = 0
    wellRowIndex: int = 0

    def __init__(self,
        *,
        plateIndex: int = 0,
        plateUuid: str = "",
        seqIndex: int = 0,
        wellIndex: int = 0,
        wellName: str = "",
        wellColIndex: int = 0,
        wellRowIndex: int = 0,
        **kwargs):
            object.__setattr__(self, 'plateIndex', plateIndex)
            object.__setattr__(self, 'plateUuid', plateUuid)
            object.__setattr__(self, 'seqIndex', seqIndex)
            object.__setattr__(self, 'wellIndex', wellIndex)
            object.__setattr__(self, 'wellName', wellName)
            object.__setattr__(self, 'wellColIndex', wellColIndex)
            object.__setattr__(self, 'wellRowIndex', wellRowIndex)
            if 'wellCompactName' in kwargs and not self.wellName:
                object.__setattr__(self, 'wellName', kwargs['wellCompactName'])



class WellplateFrameInfo(collections.UserList):
    def __init__(self, iterable):
        super().__init__(WellplateFrameInfoItem(**item) if type(item) == dict else item  for item in iterable)

    @property
    def nwells(self) -> int:
        return len(set([item.wellIndex for item in self.data]))

    @staticmethod
    def from_json(data: bytes|memoryview) -> WellplateFrameInfo:
        if type(data) == memoryview:
            data = data.tobytes()
        decoded = json.loads(zlib.decompress(data).decode('utf-8'))
        return WellplateFrameInfo(decoded)


class ExperimentIterator:
    def __init__(self, explevels: list[ExperimentLevel]):
        self.explevels: list[ExperimentLevel] = explevels

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.explevels.pop(0)
        except IndexError:
            raise StopIteration


@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentLevel(LVSerializable):
    """
    This class stores information about experiments used in an image, this class stores information about one experiment level directly and
    it may contain nested experiments (similar to linked list data structure).

    This nesting of experimtents is used to define the order of experiments in the image.

    To iterate over all experiments in the image, iterate over this class instance in for loop like this:

    ```py
    for exp in experiment:
        print(exp.name)
        print(exp.count)
        # another experiment specific instructions
    ```

    Each experiment level has a type, which is defined by [`ExperimentLoopType`](experiment.md#limnd2.experiment.ExperimentLoopType) enum stored in `eType` attribute,
    this type defines which experiment parameters are used in given level, those parameters are stored in `uLoopPars` attribute.

    In order to get experiment of specific type, use [`findLevel`](experiment.md#limnd2.experiment.ExperimentLevel.findLevel) method with an
    instance of [`ExperimentLoopType`](experiment.md#limnd2.experiment.ExperimentLoopType) enum as argument.

    This will allow you to get attributes for specific experiment type, for example, to get z-stack loop experiment attributes, use this code:

    ```py
    zstack = experiment.findLevel(limnd2.ExperimentLoopType.eEtZStackLoop)
    ```

    **Attributes:**
    !!! note
        Only selected attributes are listed, for full list of attributes see class definition.

    Attributes
    ----------
    eType : ExperimentLoopType
        Experiment type used in the image, also defines which parameters are used in this level.
    uLoopPars: ExperimentLoop
        Parameters of the current level loop, the structure depends on the content of `eType` member variable.
    ppNextLevelEx: list[ExperimentLevel]
        List of nested ExperimentLevel instances.
    uiNextLevelCount: int
        Number of nested experiments.
    """
    eType: ExperimentLoopType               = LV_field(ExperimentLoopType.eEtUnknown,     LVType.UINT32)
    # Type of the current loop, determines the union member to be used

    wsApplicationDesc: str                  = LV_field("",                                LVType.STRING)
    # Unique identification of the application which created the image (experiment)

    wsUserDesc: str                         = LV_field("",                                LVType.STRING)
    aMeasProbesBase64: str                  = LV_field("",                                LVType.STRING)
    # Time measurement probes definition

    wsCameraName: str                       = LV_field("",                                LVType.STRING)
    uLoopPars: ExperimentLoop               = LV_field(ExperimentLoop,                    LVType.LEVEL)
    # A specification of parameters of the current level loop.
    # The structure depends on the content of eType member variable.


    pItemValid: bytes                       = LV_field(None,                              LVType.BYTEARRAY)
    # A list of bools specifying whether the items are branched in a next level.
    # This is the only possibility how to break experiment orthogonality.
    # Default value is None, it means all items are used and the experiment is fully orthogonal.

    sAutoFocusBeforeLoop: dict              = LV_field(dict,                              LVType.ENCODING_NOT_IMPLEMENTED)
    wsCommandBeforeLoop: str                = LV_field("",                                LVType.STRING)
    wsCommandBeforeCapture: str             = LV_field("",                                LVType.STRING)
    wsCommandAfterCapture: str              = LV_field("",                                LVType.STRING)
    wsCommandAfterLoop: str                 = LV_field("",                                LVType.STRING)
    bControlShutter: bool                   = LV_field(False,                             LVType.BOOL)
    bControlLight: bool                     = LV_field(False,                             LVType.BOOL)
    bUsePFS: bool                           = LV_field(False,                             LVType.BOOL)
    bUseWatterSupply: bool                  = LV_field(False,                             LVType.BOOL)
    bUseHWSequencer: bool                   = LV_field(False,                             LVType.BOOL)
    bUseTiRecipe: bool                      = LV_field(False,                             LVType.BOOL)
    bUseIntenzityCorrection: bool           = LV_field(False,                             LVType.BOOL)
    bUseTriggeredAcquisition: bool          = LV_field(False,                             LVType.BOOL)
    bTriggeredStimulation: bool             = LV_field(False,                             LVType.BOOL)
    # Ax + GalvoXY triggered sequential stimulation = Ax se triggeruje na end
    # TTL Outu na galvu a galvo se triggeruje na pulzik na TTL Inu vygenerovany Axem.

    bKeepObject: bool                       = LV_field(False,                             LVType.BOOL)
    RecordAllData: bool                     = LV_field(False,                             LVType.BOOL)
    # used in JOBS via expression. Determines (in job only) whether all data should be recorded

    vectStimulationConfigurations: list     = LV_field(list,                              LVType.ENCODING_NOT_IMPLEMENTED)
    # phase, list SC

    vectStimulationConfigurationsSize: int  = LV_field(0,                                 LVType.INT32)
    uiRepeatCount: int                      = LV_field(1,                                 LVType.UINT32)
    # Number of repeatings (how many times must be the current subexperiment repeated)

    uiNextLevelCount: int                   = LV_field(0,                                 LVType.UINT32)
    ppNextLevelEx: list[ExperimentLevel]    = LV_field(None,                              LVType.LEVEL)
    # A list of subloops

    pLargeImage: dict | None                = LV_field(None,                              LVType.ENCODING_NOT_IMPLEMENTED)
    # A pointer to a LargeImage description structure or None, if the LargeImage is not defined in this loop

    pNIExperiment: dict | None              = LV_field(None,                              LVType.ENCODING_NOT_IMPLEMENTED)
    # A pointer to a NICard description structure or None, if the NICard description is not defined in this loop

    pRecordedData: dict | None              = LV_field(None,                              LVType.ENCODING_NOT_IMPLEMENTED)
    # A pointer to a Recorded Data description structure or None, if Recorded Data is not defined in this loop

    sParallelExperiment: dict | None        = LV_field(None,                              LVType.ENCODING_NOT_IMPLEMENTED)
    # Parallel experiments description

    pExternalData: dict | None              = LV_field(None,                              LVType.ENCODING_NOT_IMPLEMENTED)
    # External data (liquid handling)

    iRecipeDSCPort: int                     = LV_field(None,                              LVType.INT32)

    def __post_init__(self):
        if isinstance(self.pItemValid, dict):
            object.__setattr__(self, 'pItemValid', bytes(self.pItemValid.values()))
        object.__setattr__(self, 'eType', ExperimentLoopType(self.eType))
        object.__setattr__(self, 'ppNextLevelEx', ExperimentLevel.createExperimentLevels(self.ppNextLevelEx))
        object.__setattr__(self, 'uLoopPars', ExperimentLoop.createExperimentLoop(self.eType, self.uLoopPars)) # type: ignore

        if not self.pLargeImage and "pLargeImageEx" in self._unknown_fields:
            object.__setattr__(self, 'pLargeImage', self._unknown_fields.pop("pLargeImageEx"))
        if "pLargeImageEx" in self._unknown_fields:
            self._unknown_fields.pop("pLargeImageEx")

    def __iter__(self):                     # type: ignore[override]        error ignored by design, this class implements Mapping type, which should iterate over keys, but we want to iterate over experiments
        return ExperimentIterator(self._allLevels())

    @property
    def dims(self) -> dict[str, int]:
        """
        Returns a dictionary mapping each experiment to number of frames in that experiment.
        """
        return { exp_loop.typeName: exp_loop.count for exp_loop in self }

    @property
    def valid(self) -> bool:
        """
        Checks if experiment and all subexperiments are valid.
        """
        return (
            self.eType != ExperimentLoopType.eEtUnknown
            and 0 < self.uLoopPars.uiCount
            and ((self.ppNextLevelEx is None) or all(exp.valid for exp in self.ppNextLevelEx))
        )

    @property
    def isLambda(self) -> bool:
        """
        Checks if experiment is spectral loop experiment.
        """
        return self.eType == ExperimentLoopType.eEtSpectLoop

    @property
    def count(self) -> int:
        """
        Returns number of frames in the experiment.
        """
        return len([item for item in self.pItemValid if item]) if self.pItemValid and len(self.pItemValid) else self.uLoopPars.uiCount

    @property
    def name(self) -> str:
        """
        Returns name of the experiment.
        """
        return ExperimentLoopType.toLongName(self.eType)

    @property
    def shortName(self) -> str:
        """
        Returns short name of the experiment.
        """
        return ExperimentLoopType.toShortName(self.eType)

    @property
    def typeName(self) -> str:
        """
        Returns type name of the experiment.
        """
        return ExperimentLoopType.toName(self.eType)

    def loopTypes(self, *, skipSpectralLoop: bool = True) -> tuple[ExperimentLoopType, ...]:
        """
        Returns tuple with `ExperimentLoopType` instances of all experiments in the image.
        """
        ret = tuple() if self.isLambda and skipSpectralLoop else (self.eType, )
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].loopTypes(skipSpectralLoop=skipSpectralLoop)
        return ret

    def indexOfLoop(self, loopType: ExperimentLoopType, *, skipSpectralLoop: bool = True) -> int | None:
        """
        Returns index of specified loop type or None.
        """
        return self.loopTypes(skipSpectralLoop=skipSpectralLoop).index(loopType)

    def findLevel(self, loopType: ExperimentLoopType) -> ExperimentLevel | None:
        """
        Find and returns experiment of specified type or None.
        """
        if self.eType == loopType:
            return self
        if 0 < len(nl := self._nextLevels()):
            ret = nl[0].findLevel(loopType)
            if ret is not None:
                return ret
        return None

    def ndim(self, skipSpectralLoop: bool = True) -> int:
        """
        Returns number of experiments in the experiment level.
        """
        ret = 0 if self.isLambda and skipSpectralLoop else 1
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].ndim(skipSpectralLoop=skipSpectralLoop)
        return ret

    def shape(self, *, skipSpectralLoop: bool = True) -> tuple[int, ...]:
        """
        Returns shape of the experiment. (number of frames in each experiment)
        """
        ret = tuple() if self.isLambda and skipSpectralLoop else (self.count, )
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].shape(skipSpectralLoop=skipSpectralLoop)
        return ret

    def dimnames(self, *, skipSpectralLoop: bool = True) -> tuple[str, ...]:
        """
        Returns names of nested dimensions.
        """
        ret = tuple() if self.isLambda and skipSpectralLoop else (ExperimentLoopType.toName(self.eType),)
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].dimnames(skipSpectralLoop=skipSpectralLoop)
        return tuple(ret)

    @overload
    def generateLoopIndexes(self, *, named: Literal[True]) -> list[dict[str, int]]: ...
    @overload
    def generateLoopIndexes(self, *, named: Literal[False] = ...) -> list[tuple[int, ...]]: ...
    @overload
    def generateLoopIndexes(self, *, named: bool) -> list[tuple[int, ...]] | list[dict[str, int]]: ...

    def generateLoopIndexes(self, *, named: bool = False) -> list[tuple[int, ...]] | list[dict[str, int]]:
        """
        Generate list of indexes for all experiments in the image.
        """
        ranges = [list(range(dim)) for dim in self.shape(skipSpectralLoop=True)]
        loopindexes = itertools.product(*ranges)
        if named:
            names = self.dimnames(skipSpectralLoop=True)
            return [ dict(zip(names, item)) for item in loopindexes]
        else:
            return list(loopindexes)

    def _allLevels(self) -> list[ExperimentLevel]:
        ret: list[ExperimentLevel] = [cast(ExperimentLevel, self)]
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0]._allLevels()
        return ret

    def _nextLevels(self) -> list[ExperimentLevel]:
        ret = []
        if (self.eType == ExperimentLoopType.eEtNETimeLoop
            and isinstance(self.uLoopPars, ExperimentNETimeLoop)
            and self.uLoopPars.pSubLoops is not None
            and self.uLoopPars.uiPeriodCount == len(self.uLoopPars.pSubLoops)):
            for i in range(self.uLoopPars.uiPeriodCount):
                if (self.uLoopPars.pPeriodValid is None) or True == self.uLoopPars.pPeriodValid[i]:
                    ret += self.uLoopPars.pSubLoops[i]
        else:
            for i in range(self.uiNextLevelCount):
                ret.append(self.ppNextLevelEx[i])
        return ret

    def to_lv(self) -> bytes:
        return encode_lv({"SLxExperiment" : self.to_serializable_dict()})

    @staticmethod
    def createExperimentLevels(ppNextLevelEx: list[ExperimentLevel]|dict):
        if ppNextLevelEx is None:
            return None
        if type(ppNextLevelEx) == dict:
            levels = []
            for i in range(len(ppNextLevelEx)):
                level = ppNextLevelEx.get(f'i{i:010d}', None)
                if level is None:
                    raise TypeError()
                if 1 == len(level) and 'SLxExperiment' in level:
                    level = level['SLxExperiment']
                levels.append(ExperimentLevel(**level))
            return levels
        elif type(ppNextLevelEx) == list and all(type(item) == dict for item in ppNextLevelEx):
            levels = []
            for exp in ppNextLevelEx:
                levels.append(ExperimentLevel(**exp))
            return levels
        elif type(ppNextLevelEx) == list and all(isinstance(item, ExperimentLevel) for item in ppNextLevelEx):
            return ppNextLevelEx
        else:
            raise TypeError("Unexpected type")

    @staticmethod
    def from_lv(data: bytes|memoryview) -> ExperimentLevel:
        decoded = decode_lv(data)
        return ExperimentLevel(**decoded.get('SLxExperiment', {}))

    @staticmethod
    def from_var(data: bytes|memoryview) -> ExperimentLevel:
        decoded = decode_var(data)
        return ExperimentLevel(**decoded[0]) # type: ignore

    def __str__(self):
        all = self._allLevels()
        return ", ".join([str(x.uLoopPars) for x in all])
