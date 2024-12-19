from __future__ import annotations

import collections, enum, itertools, json, math, zlib
from dataclasses import MISSING, dataclass, fields

from .metadata import PictureMetadataPicturePlanes, PicturePlaneDesc
from .lite_variant import decode_lv, encode_lv, LVSerializable, ELxLiteVariantType as LVType, LV_field
from .variant import decode_var
from .treeview_helper import get_format_fn

class ExperimentLoopType(enum.IntEnum):
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
        names = [ 'Unknown', 'Time', 'Multipoint', 'Multipoint', 'Z-Stack', 'Polar', 'Spectral', 'Custom', 'Time', 'Time', 'Z-Stack' ]
        return names[eType]

    @staticmethod
    def toShortName(eType: ExperimentLoopType|int):
        names = [ '?', 'T', 'XY', 'XY', 'Z', 'P', 'λ', 'C', 'T', 'T', 'Z' ]
        return names[eType]

class ExperimentType(enum.IntFlag):
    eEtDefault              = 0,
    eEtStimulation          = 1,
    eEtBleaching            = 2,
    eEtIncubation           = 2048,
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

@dataclass(frozen=True, kw_only=True)
class ExperimentTimeLoop(ExperimentLoop, LVSerializable):
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
        return _format_time(self.dPeriod) if 0.0 < self.dPeriod else 'No Delay'

    @property
    def formattedDuration(self) -> str:
        return _format_time(self.dDuration) if 0.0 < self.dDuration else 'Continuous'

    @property
    def info(self) -> list[dict[str, any]]:
        return [ dict(Phase='#1', Interval=self.formattedInterval, Duration=self.formattedDuration, Loops=self.uiCount) ]


@dataclass(frozen=True, kw_only=True)
class ExperimentNETimeLoop(ExperimentLoop, LVSerializable):
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
    def info(self) -> list[dict[str, any]]:
        return [ period.info[0] for period in self.pPeriod ]


class ZStackType(enum.IntEnum):
    zstBottomToTopFixedTop                  = 0 # Bottom -> Top stack with fixed Top position
    zstBottomToTopFixedBottom               = 1 # Bottom -> Top stack with fixed Bottom position
    zstSymmetricRangeFixedHomeBottomToTop   = 2 # Symmetric Range around fixed Home position (Bottom -> Top)
    zstAsymmetricRangeFixedHomeBottomToTop  = 3 # Asymmetric Range around fixed Home position (Bottom -> Top)
    zstTopToBottomFixedTop                  = 4 # Top -> Bottom stack with fixed Top position
    zstTopToBottomFixedBottom               = 5 # Top -> Bottom stack with fixed Bottom position
    zstSymmetricRangeFixedHomeTopToBottom   = 6 # Symmetric Range around fixed Home position (Top -> Bottom)
    zstAsymmetricRangeFixedHomeTopToBottom  = 7 # Asymmetric Range around fixed Home position (Top -> Bottom)

@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentZStackLoop(ExperimentLoop, LVSerializable):
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

    # probably error in NIS elements, should be duplicate of dZLow
    #dZLow_1: float                          = LV_field("",    LVType.DO_NOT_ENCODE)                # DONE

    """
    Atributes found in XML variant, but not in LV

    #sZDevice: object                        = LV_field(None,  LVType.ENCODING_NOT_IMPLEMENTED)     # DONE
    #sCommandAfterCapture: str               = LV_field("",    LVType.ENCODING_NOT_IMPLEMENTED)     # DONE
    #sCommandBeforeCapture: str              = LV_field("",    LVType.ENCODING_NOT_IMPLEMENTED)     # DONE
    """

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
    def step(self):
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
         if self.iType in (ZStackType.zstSymmetricRangeFixedHomeBottomToTop, ZStackType.zstAsymmetricRangeFixedHomeBottomToTop, ZStackType.zstSymmetricRangeFixedHomeTopToBottom, ZStackType.zstAsymmetricRangeFixedHomeTopToBottom):
            return self.dZHigh - self.dZHome
         else:
            return -self.dZLow + self.dReferencePosition if self.bZInverted else self.dZHigh + self.dReferencePosition

    @property
    def bottom(self):
         if self.iType in (ZStackType.zstSymmetricRangeFixedHomeBottomToTop, ZStackType.zstAsymmetricRangeFixedHomeBottomToTop, ZStackType.zstSymmetricRangeFixedHomeTopToBottom, ZStackType.zstAsymmetricRangeFixedHomeTopToBottom):
            return self.dZLow - self.dZHome
         else:
            return -self.dZHigh + self.dReferencePosition if self.bZInverted else self.dZLow + self.dReferencePosition

    @property
    def info(self) -> list[dict[str, any]]:
        return [ dict(Step=self.step, Top=self.top, Bottom=self.bottom, Count=self.uiCount, Drive=self.wsZDevice)]

@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentSpectralLoopPoint(LVSerializable):
    pAutoFocus: dict                            = LV_field(dict,                            LVType.ENCODING_NOT_IMPLEMENTED)
    pZStackPos: int                             = LV_field(0,                               LVType.INT32)
    pdOffset: float                             = LV_field(0.0,                             LVType.DOUBLE)
    wsCommandBeforeCapture: str                 = LV_field("",                              LVType.STRING)
    wsCommandAfterCapture: str                  = LV_field("",                              LVType.STRING)

    """
    Atributes found in XML variant, but not in LV
    pPlaneDesc: PicturePlaneDesc                = LV_field(PicturePlaneDesc,                LVType.LEVEL)             # TODO
    """

    def __post_init__(self):
        if "pPlaneDesc" in self._unknown_fields:
            self._unknown_fields.pop("pPlaneDesc")


@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentSpectralLoop(ExperimentLoop, LVSerializable):
    pPlanes: PictureMetadataPicturePlanes       = LV_field(PictureMetadataPicturePlanes,    LVType.LEVEL)
    iOffsetReference: int                       = LV_field(0,                               LVType.INT32)
    bMergeCameras: bool                         = LV_field(False,                           LVType.BOOL)
    bWaitForPFS: bool                           = LV_field(False,                           LVType.BOOL)
    bAskForFilter: bool                         = LV_field(False,                           LVType.BOOL)
    Points: list[ExperimentSpectralLoopPoint]   = LV_field(list,                            LVType.LEVEL)

    """
    Atributes found in XML variant, but not in LV
    pPlaneDesc: PicturePlaneDesc                = LV_field(None,                            LVType.DO_NOT_ENCODE)                     # DONE
    szCommandBeforeCapture: object              = LV_field(None,                            LVType.ENCODING_NOT_IMPLEMENTED)          # DONE
    szCommandAfterCapture: object               = LV_field(None,                            LVType.ENCODING_NOT_IMPLEMENTED)          # DONE
    pZStackPos: object                          = LV_field(None,                            LVType.ENCODING_NOT_IMPLEMENTED)          # DONE
    pAutoFocus: dict                            = LV_field(None,                            LVType.ENCODING_NOT_IMPLEMENTED)          # DONE
    """

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
    def info(self) -> list[dict[str, any]]:
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




@dataclass(frozen=True, kw_only=True, init=False)
class ExperimentXYPosLoop(ExperimentLoop, LVSerializable):
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

    """
    Atributes found in XML variant, but not in LV
    dPosX: object                           = LV_field(None,              LVType.DO_NOT_ENCODE)       #DONE
    dPosY: object                           = LV_field(None,              LVType.DO_NOT_ENCODE)       #DONE
    dPosZ: object                           = LV_field(None,              LVType.DO_NOT_ENCODE)       #DONE
    dPFSOffset: object                      = LV_field(None,              LVType.DO_NOT_ENCODE)       #DONE
    pPosName: object                        = LV_field(None,              LVType.DO_NOT_ENCODE)       #DONE
    sAutoFocusBeforeCapture: object         = LV_field(None,              LVType.DO_NOT_ENCODE)       #DONE

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
    def info(self) -> list[dict[str, any]]:
        ret = []
        for i in range(self.uiCount):
            name = self.Points[i].dPosName if i < len(self.Points) and self.Points[i].dPosName else f"#{i}"
            d = dict(Name=name, X=self.Points[i].dPosX, Y=self.Points[i].dPosY)
            if self.bUseZ and self.uiCount == len(self.Points):
                d['Z'] = self.Points[i].dPosZ
            ret.append(d)
        return ret


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

    """
    Atributes found in XML variant, but not in LV
    pLargeImageEx: object                   = LV_field(None,                              LVType.ENCODING_NOT_IMPLEMENTED)        # DONE
    """

    def __post_init__(self):
        if isinstance(self.pItemValid, dict):
            object.__setattr__(self, 'pItemValid', bytes(self.pItemValid.values()))
        object.__setattr__(self, 'eType', ExperimentLoopType(self.eType))
        object.__setattr__(self, 'ppNextLevelEx', ExperimentLevel.createExperimentLevels(self.ppNextLevelEx))
        object.__setattr__(self, 'uLoopPars', ExperimentLoop.createExperimentLoop(self.eType, self.uLoopPars))

        if not self.pLargeImage and "pLargeImageEx" in self._unknown_fields:
            object.__setattr__(self, 'pLargeImage', self._unknown_fields.pop("pLargeImageEx"))
        if "pLargeImageEx" in self._unknown_fields:
            self._unknown_fields.pop("pLargeImageEx")

    def __iter__(self):
        return ExperimentIterator(self._allLevels())

    @property
    def dims(self) -> dict[str, int]:
        return { exp_loop.typeName: exp_loop.count for exp_loop in self }

    @property
    def valid(self) -> bool:
        return (
            self.eType != ExperimentLoopType.eEtUnknown
            and 0 < self.uLoopPars.uiCount
            and ((self.ppNextLevelEx is None) or all(exp.valid for exp in self.ppNextLevelEx))
        )

    @property
    def isLambda(self) -> bool:
        return self.eType == ExperimentLoopType.eEtSpectLoop

    @property
    def count(self) -> int:
        return len([item for item in self.pItemValid if item]) if self.pItemValid and len(self.pItemValid) else self.uLoopPars.uiCount

    @property
    def name(self) -> str:
        return ExperimentLoopType.toLongName(self.eType)

    @property
    def shortName(self) -> str:
        return ExperimentLoopType.toShortName(self.eType)

    @property
    def typeName(self) -> str:
        return ExperimentLoopType.toName(self.eType)

    def loopTypes(self, *, skipSpectralLoop: bool = True) -> tuple[ExperimentLoopType]:
        ret = tuple() if self.isLambda and skipSpectralLoop else (self.eType, )
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].loopTypes(skipSpectralLoop=skipSpectralLoop)
        return ret

    def indexOfLoop(self, loopType: ExperimentLoopType, *, skipSpectralLoop: bool = True) -> int:
        return self.loopTypes(skipSpectralLoop=skipSpectralLoop).index(loopType)

    def findLevel(self, loopType: ExperimentLoopType) -> ExperimentLevel|None:
        if self.eType == loopType:
            return self
        if 0 < len(nl := self._nextLevels()):
            ret = nl[0].findLevel(loopType)
            if ret is not None:
                return ret
        return None

    def ndim(self, skipSpectralLoop: bool = True) -> int:
        ret = 0 if self.isLambda and skipSpectralLoop else 1
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].ndim(skipSpectralLoop=skipSpectralLoop)
        return ret

    def shape(self, *, skipSpectralLoop: bool = True) -> tuple[int]:
        ret = tuple() if self.isLambda and skipSpectralLoop else (self.count, )
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].shape(skipSpectralLoop=skipSpectralLoop)
        return ret

    def dimnames(self, *, skipSpectralLoop: bool = True) -> tuple[str]:
        ret = tuple() if self.isLambda and skipSpectralLoop else (ExperimentLoopType.toName(self.eType),)
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0].dimnames(skipSpectralLoop=skipSpectralLoop)
        return tuple(ret)

    def generateLoopIndexes(self, *, named: bool = False) -> list[tuple]:
        ranges = [list(range(dim)) for dim in self.shape(skipSpectralLoop=True)]
        loopindexes = itertools.product(*ranges)
        if named:
            names = self.dimnames(skipSpectralLoop=True)
            return [ dict(zip(names, item)) for item in loopindexes]
        else:
            return list(loopindexes)

    def _allLevels(self) -> list[ExperimentLevel]:
        ret = [ self ]
        if 0 < len(nl := self._nextLevels()):
            ret += nl[0]._allLevels()
        return ret

    def _nextLevels(self) -> list[ExperimentLevel]:
        ret = []
        if (self.eType == ExperimentLoopType.eEtNETimeLoop
            and self.uLoopPars.pSubLoops is not None
            and self.uLoopPars.uiPeriodCount == len(self.uLoopPars.pSubLoops)):
            for i in range(self.uLoopPars.uiPeriodCount):
                if (self.uLoopPars.pPeriodValid is None) or True == self.uLoopPars.pPeriodValid[i]:
                    ret += self.uLoopPars.pSubLoops[i]
        else:
            for i in range(self.uiNextLevelCount):
                ret.append(self.ppNextLevelEx[i])
        return ret

    def to_table(self):
        right_align = {'text-align': 'right'}
        css_style = { 'X': right_align, 'Y': right_align, 'Z': right_align, 'Bottom': right_align, 'Count': right_align, 'Step': right_align, 'Top': right_align, 'Interval': right_align, 'Duration': right_align, 'Loops': right_align }
        min_width = { 'X': '100px', 'Y': '100px', 'Z': '100px', 'Bottom': '80px', 'Count': '60px', 'Step': '80px', 'Top': '80px', 'Interval': '100px', 'Duration': '100px', 'Loops': '100px' }
        format_fn = { 'X': get_format_fn(2), 'Y': get_format_fn(2), 'Z': get_format_fn(3), 'Bottom': get_format_fn(3), 'Step': get_format_fn(3), 'Top': get_format_fn(3), 'Color': '(coldef) => { coldef.fmtfn = function(val) { return val === "#ffffff" ? "Brightfield" : "" }; };' }
        style_fn = { 'Color':  '(coldef) => { coldef.stylefn = function(val) { return  val === "#ffffff" ? { "background": "linear-gradient(0.25turn, rgba(255,0,0,0.3), rgba(0,255,0,0.3), rgba(0,0,255,0.3))" } : { "background-color": `${val}ee` }; } };' }
        replace = { 'X': 'X Pos [µm]', 'Y': 'Y Pos [µm]', 'Z': 'Z Pos [µm]', 'OC': 'Opt. conf.', 'Bottom': 'Bottom [µm]', 'Drive': 'Z Drive', 'Step': 'Z Step [µm]', 'Top': 'Top [µm]' }
        col_defs = []
        for k in self.uLoopPars.info[0].keys():
            s = css_style.get(k, {})
            d = dict(id=k, title=replace.get(k, k), headerStyle=s, style=s)
            fmt_fn_code = format_fn.get(k, None)
            if fmt_fn_code is not None:
                d["fmtfncode"] = fmt_fn_code
            style_fn_code = style_fn.get(k, None)
            if style_fn_code is not None:
                d["stylefncode"] = style_fn_code
            min_w = min_width.get(k, None)
            if min_w is not None:
                d["minwidth"] = min_w
            col_defs.append(d)

        return dict(coldefs=col_defs, rowdata=self.uLoopPars.info)

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
        return ExperimentLevel(**decoded[0])
