from __future__ import annotations

import collections, enum, itertools, json, math, zlib
from dataclasses import dataclass, field
from .metadata import PictureMetadataPicturePlanes
from .lite_variant import decode_lv
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
    eEtZStackLoopAccurate   = 10 # not used yet

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

@dataclass(frozen=True, kw_only=True)
class ExperimentLoop:
    uiCount: int = 0

    @staticmethod
    def createExperimentLoop(eType, uLoopPars):
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

@dataclass(init=False, frozen=True)
class ExperimentTimeLoop(ExperimentLoop):
    dStart: float                   = 0
    dPeriod: float                  = 0
    dDuration: float                = 0                                 # if dDuration is nonzero, then dPeriod must be zero and
                                                                        # means experiment is captured as fast as possible for dDuration ms
    dMinPeriodDiff: float           = 0
    dMaxPeriodDiff: float           = 0
    dAvgPeriodDiff: float           = 0
    wsPhaseName: str                = ""
    sAutoFocusBeforePeriod: dict    = field(default_factory=dict)
    sAutoFocusBeforeCapture: dict   = field(default_factory=dict)
    uiLoopType: ExperimentType      = ExperimentType.eEtDefault         # 0..default type, 1..Stimulation, 2..Bleaching, 32..Incubation
    uiGroup: int                    = 0                                 # 0..no group, HIWORD(uiGroup)..Index of group from 1, LOWORD(uiGroup)..Index inside group
    uiStimulationCount: int         = 0
    bDurationPref: bool             = False                             # If true, time loop will stop at the dDuration time regardless uiCount
    pIncubationData: bytes          = b''                               # parameters for incubation device
    uiIncubationDataSize: int       = 0
    wsInterfaceName: str            = ""
    uiTreatment: int                = 0
    dIncubationDuration: float      = -1

    def __init__(   self,
                    *,
                    uiCount: int = 0,
                    dStart: float= 0,
                    dPeriod: float = 0,
                    dDuration: float = 0,
                    dMinPeriodDiff: float = 0,
                    dMaxPeriodDiff: float = 0,
                    dAvgPeriodDiff: float = 0,
                    wsPhaseName: str = "",
                    sAutoFocusBeforePeriod: dict = {},
                    sAutoFocusBeforeCapture: dict = {},
                    uiLoopType: ExperimentType|int = ExperimentType.eEtDefault,
                    uiGroup: int = 0,
                    uiStimulationCount: int = 0, 
                    bDurationPref: bool = False,
                    pIncubationData: bytes = b'',
                    uiIncubationDataSize: int = 0,
                    wsInterfaceName: str = "",
                    uiTreatment: int = 0,
                    dIncubationDuration: float= -1,
                    **kwargs):
        super().__init__(uiCount=uiCount)
        object.__setattr__(self, 'dStart', dStart)
        object.__setattr__(self, 'dPeriod', dPeriod)
        object.__setattr__(self, 'dDuration', dDuration)
        object.__setattr__(self, 'dMinPeriodDiff', dMinPeriodDiff)
        object.__setattr__(self, 'dMaxPeriodDiff', dMaxPeriodDiff)
        object.__setattr__(self, 'dAvgPeriodDiff', dAvgPeriodDiff)
        object.__setattr__(self, 'wsPhaseName', wsPhaseName)
        object.__setattr__(self, 'sAutoFocusBeforePeriod', sAutoFocusBeforePeriod)
        object.__setattr__(self, 'sAutoFocusBeforeCapture', sAutoFocusBeforeCapture)
        object.__setattr__(self, 'uiLoopType', uiLoopType)
        object.__setattr__(self, 'uiGroup', uiGroup)
        object.__setattr__(self, 'uiStimulationCount', uiStimulationCount)
        object.__setattr__(self, 'bDurationPref', bDurationPref)
        object.__setattr__(self, 'pIncubationData', pIncubationData)
        object.__setattr__(self, 'uiIncubationDataSize', uiIncubationDataSize)
        object.__setattr__(self, 'wsInterfaceName', wsInterfaceName)
        object.__setattr__(self, 'uiTreatment', uiTreatment)
        object.__setattr__(self, 'dIncubationDuration', dIncubationDuration)

    @property
    def formattedInterval(self) -> str:
        return _format_time(self.dPeriod) if 0.0 < self.dPeriod else 'No Delay'
    
    @property
    def formattedDuration(self) -> str:
        return _format_time(self.dDuration) if 0.0 < self.dDuration else 'Continuous'    

    @property
    def info(self) -> list[dict[str, any]]:
        return [ dict(Phase='#1', Interval=self.formattedInterval, Duration=self.formattedDuration, Loops=self.uiCount) ]

@dataclass(init=False, frozen=True)
class ExperimentNETimeLoop(ExperimentLoop):
    uiPeriodCount: int                  = 0
    pPeriod: list[ExperimentTimeLoop]   = field(default_factory=list)
    pSubLoops: list[list[ExperimentLevel]]|None = None                  # list (if this list is empty, use ppNextLevelEx list from 
                                                                        #        experiment for each time phase)
    sAutoFocusBeforePeriod: dict        = field(default_factory=dict)
    sAutoFocusBeforeCapture: dict       = field(default_factory=dict)
    wsCommandBeforePeriod: list[str]    = field(default_factory=list)
    wsCommandAfterPeriod: list[str]     = field(default_factory=list)
    pPeriodValid: list[bool]            = field(default_factory=list)

    def __init__(   self,
                    *,
                    uiCount: int = 0,
                    uiPeriodCount: int = 0,
                    pPeriod: list[ExperimentTimeLoop]|dict = [],
                    pSubLoops: list[list[ExperimentLevel]]|dict[dict] = None,
                    sAutoFocusBeforePeriod: dict = {},
                    sAutoFocusBeforeCapture: dict = {},
                    wsCommandBeforePeriod: list[str]|dict = [],
                    wsCommandAfterPeriod: list[str]|dict = [],
                    pPeriodValid: list[bool] = [],
                    **kwargs):
        super().__init__(uiCount=uiCount)
        if type(pPeriodValid) == bytes:
            pPeriodValid = [item != 0 for item in pPeriodValid]
        if type(wsCommandBeforePeriod) == dict and 0 < len(wsCommandBeforePeriod):
            strs = []
            for i in range(len(wsCommandBeforePeriod)):
                strs.append(wsCommandBeforePeriod.get(f'i{i:010d}', ""))
            wsCommandBeforePeriod = strs
        if type(wsCommandAfterPeriod) == dict and 0 < len(wsCommandBeforePeriod):
            strs = []
            for i in range(len(wsCommandAfterPeriod)):
                strs.append(wsCommandAfterPeriod.get(f'i{i:010d}', ""))
            wsCommandAfterPeriod = strs
        object.__setattr__(self, 'uiPeriodCount', uiPeriodCount)
        if 0 < self.uiPeriodCount:
            periods = []
            if self.uiPeriodCount != len(pPeriodValid):
                raise ValueError(f"Mismatch in len(pPeriodValid)={len(pPeriodValid)} and uiPeriodCount={uiPeriodCount}")            
            if self.uiPeriodCount != len(pPeriod):
                raise ValueError(f"Mismatch in len(pPeriod)={len(pPeriod)} and uiPeriodCount={uiPeriodCount}")
            if type(pPeriod) == dict:
                for i in range(self.uiPeriodCount):
                    period = pPeriod.get(f'i{i:010d}', {})
                    periods.append(ExperimentTimeLoop(**period))
                object.__setattr__(self, 'pPeriod', periods)
            elif type(pPeriod) == list and all(isinstance(item, ExperimentLevel) for item in pPeriod):
                object.__setattr__(self, 'pPeriod', pPeriod)
            else:
                raise TypeError()
        else:
            object.__setattr__(self, 'pPeriod', [])
        if type(pSubLoops) in (list, dict) and len(pSubLoops) == self.uiPeriodCount:
            if type(pSubLoops) == dict:
                subexps = []
                for i in range(self.uiPeriodCount):
                    periodexp = pSubLoops.get(f'i{i:010d}', {})
                    if 'uiNextLevelCount' in periodexp and 'ppNextLevelEx' in periodexp:
                        subexps.append(ExperimentLevel.createExperimentLevels(periodexp['ppNextLevelEx']))
                    else:
                        raise TypeError()
                pSubLoops = subexps
            elif type(pSubLoops) == list and all(type(item) == list and all(isinstance(el, ExperimentLevel) for el in item) for item in pSubLoops):
                pass
            else:
                raise TypeError()
            object.__setattr__(self, 'pSubLoops', pSubLoops)
        else:
            object.__setattr__(self, 'pSubLoops', None)
        object.__setattr__(self, 'sAutoFocusBeforePeriod', sAutoFocusBeforePeriod)
        object.__setattr__(self, 'sAutoFocusBeforeCapture', sAutoFocusBeforeCapture)
        object.__setattr__(self, 'wsCommandBeforePeriod', wsCommandBeforePeriod)
        object.__setattr__(self, 'wsCommandAfterPeriod', wsCommandAfterPeriod)
        object.__setattr__(self, 'pPeriodValid', pPeriodValid)

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

@dataclass(init=False, frozen=True)
class ExperimentZStackLoop(ExperimentLoop):
    dZLow: float = 0.0
    dZLowPFSOffset: float = 0.0
    dZHigh: float = 0.0
    dZHighPFSOffset: float = 0.0
    dZHome: float = 0.0
    dZStep: float = 0.0
    dReferencePosition: float = 0.0
    dTIRFPosition: float = 0.0
    dTIRFPFSOffset: float = 0.0
    iType: ZStackType = 0
    bAbsolute: bool = False
    bTriggeredPiezo: bool = False
    bZInverted: bool = False
    bTIRF: bool = False
    wsZDevice: str = ""
    wsCommandBeforeCapture: str = ""
    wsCommandAfterCapture: str = ""

    def __init__(   self,
                    *,
                    uiCount: int = 0,
                    dZLow: float = 0.0,
                    dZLowPFSOffset: float = 0.0,
                    dZHigh: float = 0.0,
                    dZHighPFSOffset: float = 0.0,
                    dZHome: float = 0.0,
                    dZStep: float = 0.0,
                    dReferencePosition: float = 0.0,
                    dTIRFPosition: float = 0.0,
                    dTIRFPFSOffset: float = 0.0,
                    iType: ZStackType|int = 0,
                    bAbsolute: bool = False,
                    bTriggeredPiezo: bool = False,
                    bZInverted: bool = False,
                    bTIRF: bool = False,
                    wsZDevice: str = "",
                    wsCommandBeforeCapture: str = "",
                    wsCommandAfterCapture: str = "",                  
                    **kwargs):
        super().__init__(uiCount=uiCount)
        object.__setattr__(self, 'dZLow', dZLow)
        object.__setattr__(self, 'dZLowPFSOffset', dZLowPFSOffset)
        object.__setattr__(self, 'dZHigh', dZHigh)
        object.__setattr__(self, 'dZHighPFSOffset', dZHighPFSOffset)
        object.__setattr__(self, 'dZHome', dZHome)
        object.__setattr__(self, 'dZStep', dZStep)
        object.__setattr__(self, 'dReferencePosition', dReferencePosition)
        object.__setattr__(self, 'dTIRFPosition', dTIRFPosition)
        object.__setattr__(self, 'dTIRFPFSOffset', dTIRFPFSOffset)
        object.__setattr__(self, 'iType', ZStackType(iType) if type(iType) == int else iType)
        object.__setattr__(self, 'bAbsolute', bAbsolute)
        object.__setattr__(self, 'bTriggeredPiezo', bTriggeredPiezo)
        object.__setattr__(self, 'bZInverted', bZInverted)
        object.__setattr__(self, 'bTIRF', bTIRF)
        object.__setattr__(self, 'wsZDevice', wsZDevice)
        object.__setattr__(self, 'wsCommandBeforeCapture', wsCommandBeforeCapture)
        object.__setattr__(self, 'wsCommandAfterCapture', wsCommandAfterCapture)

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
        return [ dict(Step=self.step, Top=self.top, Bottom=self.bottom, Count=self.uiCount, Drive=self.wsZDevice) ]

@dataclass(init=False, frozen=True)
class ExperimentSpectralLoop(ExperimentLoop):
    pPlanes: PictureMetadataPicturePlanes = field(default_factory=PictureMetadataPicturePlanes)
    pAutoFocus: dict = field(default_factory=dict)
    pZStackPos: list[int] = field(default_factory=list)
    wsCommandBeforeCapture: list[str] = field(default_factory=list)
    wsCommandAfterCapture: list[str] = field(default_factory=list)
    pdOffset: list[float] = field(default_factory=list)
    iOffsetReference: int = 0
    bMergeCameras: bool = False
    bWaitForPFS: bool = False
    bAskForFilter: bool = False

    def __init__(   self,
                    *,
                    uiCount: int = 0,
                    pPlanes: PictureMetadataPicturePlanes|dict = {},
                    pAutoFocus: dict = {},
                    pZStackPos: list[int] = [],
                    wsCommandBeforeCapture: list[str] = [],
                    wsCommandAfterCapture: list[str] = [],
                    pdOffset: list[float] = [],
                    iOffsetReference: int = 0,
                    bMergeCameras: bool = False,
                    bWaitForPFS: bool = False,
                    bAskForFilter: bool = False,
                    **kwargs):
        pPlanes_ = PictureMetadataPicturePlanes(**pPlanes)
        uiCount = pPlanes_.uiCount
        if 'Points' in kwargs:
            points = kwargs['Points']
            wsCommandBeforeCapture, wsCommandAfterCapture = [""]*uiCount, [""]*uiCount
            pZStackPos, pdOffset = [0]*uiCount, [0.0]*uiCount
            if len(points) != uiCount:
                ValueError(f"Mismatch in len(Poins)={len(points)} and uiCount={uiCount}")
            for i in range(uiCount):
                point = points.get(f'i{i:010d}', {})
        super().__init__(uiCount=uiCount)
        object.__setattr__(self, 'pPlanes', pPlanes_)
        object.__setattr__(self, 'pAutoFocus', pAutoFocus)
        object.__setattr__(self, 'pZStackPos', pZStackPos)
        object.__setattr__(self, 'wsCommandBeforeCapture', wsCommandBeforeCapture)
        object.__setattr__(self, 'wsCommandAfterCapture', wsCommandAfterCapture)
        object.__setattr__(self, 'pdOffset', pdOffset)
        object.__setattr__(self, 'iOffsetReference', iOffsetReference)
        object.__setattr__(self, 'bMergeCameras', bMergeCameras)
        object.__setattr__(self, 'bWaitForPFS', bWaitForPFS)
        object.__setattr__(self, 'bAskForFilter', bAskForFilter)

    @property
    def info(self) -> list[dict[str, any]]:
        ret = []
        for i, plane in enumerate(self.pPlanes.sPlane):
            idx = f'#{i+1}'
            ocs = self.pPlanes.sSampleSetting[plane.uiSampleIndex].sOpticalConfigs if plane.uiSampleIndex < len(self.pPlanes.sSampleSetting) else []
            ret.append(dict(Index=idx, Name=plane.sDescription, OC=', '.join([oc.sOpticalConfigName for oc in ocs]), Color=plane.colorAsHtmlString))
        return ret
        

@dataclass(init=False, frozen=True)
class ExperimentXYPosLoop(ExperimentLoop):
    dPosX: list[float]              = field(default_factory=list)
    dPosY: list[float]              = field(default_factory=list)
    bUseZ: bool                     = False
    dPosZ: list[float]              = field(default_factory=list)
    dPFSOffset: list[float]         = field(default_factory=list)
    bRelativeXY: bool               = True
    dReferenceX: float              = 0
    dReferenceY: float              = 0
    bRedefineAfterPFS: bool         = False
    bRedefineAfterAutoFocus: bool   = False
    bKeepPFSOn: bool                = False
    bSplitMultipoints: bool         = False
    bUseAFPlane: bool               = False
    bZEnabled: bool                 = False
    pPosName: list[str]             = field(default_factory=list)
    sZDevice: str                   = ""
    sAutoFocusBeforeCapture: dict   = field(default_factory=dict)

    def __init__(   self,
                    *,
                    uiCount: int = 0,
                    dPosX: list[float] = [],
                    dPosY: list[float] = [],
                    bUseZ: bool = False,
                    dPosZ: list[float] = [],
                    dPFSOffset: list[float] = [],
                    bRelativeXY: bool = True,
                    dReferenceX: float = 0,
                    dReferenceY: float = 0,
                    bRedefineAfterPFS: bool = False,
                    bRedefineAfterAutoFocus: bool = False,
                    bKeepPFSOn: bool = False,
                    bSplitMultipoints: bool = False,
                    bUseAFPlane: bool = False,
                    bZEnabled: bool = False,
                    pPosName: list[str] = [],
                    sZDevice: str = "",
                    sAutoFocusBeforeCapture: dict = {},
                    **kwargs):
        super().__init__(uiCount=uiCount)
        if 'Points' in kwargs:
            points = kwargs['Points']
            if len(points) != uiCount:
                ValueError(f"Mismatch in len(Poins)={len(points)} and uiCount={uiCount}")
            dPosX, dPosY, dPosZ, dPFSOffset = [0.0]*uiCount, [0.0]*uiCount, [0.0]*uiCount, [-1.0]*uiCount
            pPosName = [""]*uiCount
            for i in range(uiCount):
                point = points.get(f'i{i:010d}', {})
                dPosX[i], dPosY[i] = point.get("dPosX", 0.0), point.get("dPosY", 0.0)
                dPosZ[i] = point.get("dPosZ", 0.0)
                dPFSOffset[i] = point.get("dPFSOffset", -1.0)
                pPosName[i] = point.get("dPosName", "")
        sAutoFocusBeforeCapture = kwargs.get("sAFBefore", sAutoFocusBeforeCapture)
        object.__setattr__(self, 'dPosX', dPosX)
        object.__setattr__(self, 'dPosY', dPosY)
        object.__setattr__(self, 'bUseZ', bUseZ)
        object.__setattr__(self, 'dPosZ', dPosZ)
        object.__setattr__(self, 'dPFSOffset', dPFSOffset)
        object.__setattr__(self, 'bRelativeXY', bRelativeXY)
        object.__setattr__(self, 'dReferenceX', dReferenceX)
        object.__setattr__(self, 'dReferenceY', dReferenceY)
        object.__setattr__(self, 'bRedefineAfterPFS', bRedefineAfterPFS)
        object.__setattr__(self, 'bRedefineAfterAutoFocus', bRedefineAfterAutoFocus)
        object.__setattr__(self, 'bKeepPFSOn', bKeepPFSOn)
        object.__setattr__(self, 'bSplitMultipoints', bSplitMultipoints)
        object.__setattr__(self, 'bUseAFPlane', bUseAFPlane)
        object.__setattr__(self, 'bZEnabled', bZEnabled)
        object.__setattr__(self, 'pPosName', pPosName)
        object.__setattr__(self, 'sZDevice', sZDevice)
        object.__setattr__(self, 'sAutoFocusBeforeCapture', sAutoFocusBeforeCapture)

    @property
    def info(self) -> list[dict[str, any]]:
        ret = []
        for i in range(self.uiCount):
            name = self.pPosName[i] if i < len(self.pPosName) and self.pPosName[i] else f"#{i}"
            d = dict(Name=name, X=self.dPosX[i], Y=self.dPosY[i])
            if self.bUseZ and self.uiCount == len(self.dPosZ):
                d['Z'] = self.dPosZ[i]
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


@dataclass(init=False, frozen=True)
class ExperimentLevel:
    eType: ExperimentLoopType       = ExperimentLoopType.eEtUnknown     # Type of the current loop, determines the union member to be used
    wsApplicationDesc: str          = ""                                # Unique identification of the application which created the image (experiment)
    wsUserDesc: str                 = ""
    wsMeasProbes: str               = ""                                # Time measurement probes definition
    wsCameraName: str               = ""
    uLoopPars: ExperimentLoop       = field(default_factory=ExperimentLoop) # A specification of parameters of the current level loop.   
                                                                        #   The structure depends on the content of eType member variable.
    pItemValid: list[bool]|None     = None                              # A list of bools specifying whether the items are branched in a next level.
                                                                        #   This is the only possibility how to break experiment orthogonality.
                                                                        #   Default value is None, it means all items are used and the experiment
                                                                        #   is fully orthogonal.
    sAutoFocusBeforeLoop: dict      = field(default_factory=dict)
    wsCommandBeforeLoop: str        = ""
    wsCommandBeforeCapture: str     = ""
    wsCommandAfterCapture: str      = ""
    wsCommandAfterLoop: str         = ""
    bControlShutter: bool           = False
    bControlLight: bool             = False
    bUsePFS: bool                   = False
    bUseWatterSupply: bool          = False
    bUseHWSequencer: bool           = False
    bUseTiRecipe: bool              = False
    bUseIntenzityCorrection: bool   = False
    bUseTriggeredAcquisition: bool  = False
    bTriggeredStimulation: bool     = False                             # Ax + GalvoXY triggered sequential stimulation = Ax se triggeruje na end TTL Outu na galvu a galvo se triggeruje na pulzik na TTL Inu vygenerovany Axem.
    bKeepObject: bool               = False
    bRecordAllAvailableData: bool   = False                             # used in JOBS via expression. Determines (in job only) whether all data should be recorded    
    vectStimulationConfigurations: list = field(default_factory=list)   # phase, list SC
    uiRepeatCount: int              = 1                                 # Number of repeatings (how many times must be the current subexperiment repeated)
    uiNextLevelCount: int           = 0
    ppNextLevelEx: list[ExperimentLevel] = field(default_factory=list)  # A list of subloops   
    pLargeImage: dict|None          = None                              # A pointer to a LargeImage description structure or None, if the LargeImege is not defined in this loop
    pNIExperiment: dict|None        = None                              # A pointer to a NICard description structure or None, if the NICard description is not defined in this loop   
    pRecordedData: dict|None        = None                              # A pointer to a Recorded Data description structure or None, if Recorded Data is not defined in this loop   
    sParallelExperiment: dict|None  = None                              # Parallel experiments description
    pExternalData: dict|None        = None                              # External data (liquid handling)

    def __init__(   self,
                    *,
                    eType: ExperimentLoopType|int = ExperimentLoopType.eEtUnknown,
                    wsApplicationDesc: str = "",
                    wsUserDesc: str = "",
                    wsMeasProbes: str = "",
                    wsCameraName: str = "", 
                    uLoopPars: ExperimentLoop|dict = {},
                    pItemValid: list[bool]|bytes|None = None,
                    sAutoFocusBeforeLoop: dict = {},
                    wsCommandBeforeLoop: str = "",
                    wsCommandBeforeCapture: str = "",
                    wsCommandAfterCapture: str = "",
                    wsCommandAfterLoop: str = "",
                    bControlShutter: bool = False,
                    bControlLight: bool = False,
                    bUsePFS: bool = False,
                    bUseWatterSupply: bool = False,
                    bUseHWSequencer: bool= False,
                    bUseTiRecipe: bool = False,
                    bUseIntenzityCorrection: bool = False,
                    bUseTriggeredAcquisition: bool = False,
                    bTriggeredStimulation: bool= False,
                    bKeepObject: bool = False,
                    bRecordAllAvailableData: bool = False,
                    vectStimulationConfigurations: list = [],
                    uiRepeatCount: int = 1,
                    uiNextLevelCount: int = 0,
                    ppNextLevelEx: list[ExperimentLevel]|dict = [],
                    pLargeImage: dict|None= None,
                    pNIExperiment: dict|None = None,
                    pRecordedData: dict|None = None,
                    sParallelExperiment: dict|None = None,
                    pExternalData: dict|None = None,
                    **kwargs):
        wsMeasProbes = kwargs.get('aMeasProbesBase64', wsMeasProbes)
        bRecordAllAvailableData = kwargs.get('RecordAllData', bRecordAllAvailableData)
        if 'vectStimulationConfigurations' in kwargs:
            pass
        if type(pItemValid) == bytes:
            pItemValid = [ item != 0 for item in pItemValid ]
        if pItemValid == {}:
            pItemValid = None
        object.__setattr__(self, 'eType', ExperimentLoopType(eType) if type(eType) == int else eType)
        object.__setattr__(self, 'wsApplicationDesc', wsApplicationDesc)
        object.__setattr__(self, 'wsUserDesc', wsUserDesc)
        object.__setattr__(self, 'wsMeasProbes', wsMeasProbes)
        object.__setattr__(self, 'wsCameraName', wsCameraName)
        object.__setattr__(self, 'uLoopPars', ExperimentLoop.createExperimentLoop(self.eType, uLoopPars))
        object.__setattr__(self, 'pItemValid', pItemValid)
        object.__setattr__(self, 'sAutoFocusBeforeLoop', sAutoFocusBeforeLoop)
        object.__setattr__(self, 'wsCommandBeforeLoop', wsCommandBeforeLoop)
        object.__setattr__(self, 'wsCommandBeforeCapture', wsCommandBeforeCapture)
        object.__setattr__(self, 'wsCommandAfterCapture', wsCommandAfterCapture)
        object.__setattr__(self, 'wsCommandAfterLoop', wsCommandAfterLoop)
        object.__setattr__(self, 'bControlShutter', bControlShutter)
        object.__setattr__(self, 'bControlLight', bControlLight)
        object.__setattr__(self, 'bUsePFS', bUsePFS)
        object.__setattr__(self, 'bUseWatterSupply', bUseWatterSupply)
        object.__setattr__(self, 'bUseHWSequencer', bUseHWSequencer)
        object.__setattr__(self, 'bUseTiRecipe', bUseTiRecipe)
        object.__setattr__(self, 'bUseIntenzityCorrection', bUseIntenzityCorrection)
        object.__setattr__(self, 'bUseTriggeredAcquisition', bUseTriggeredAcquisition)
        object.__setattr__(self, 'bTriggeredStimulation', bTriggeredStimulation)
        object.__setattr__(self, 'bKeepObject', bKeepObject)
        object.__setattr__(self, 'bRecordAllAvailableData', bRecordAllAvailableData)
        object.__setattr__(self, 'vectStimulationConfigurations', vectStimulationConfigurations)
        object.__setattr__(self, 'uiRepeatCount', uiRepeatCount)
        object.__setattr__(self, 'uiNextLevelCount', uiNextLevelCount)
        object.__setattr__(self, 'ppNextLevelEx', ExperimentLevel.createExperimentLevels(ppNextLevelEx))
        object.__setattr__(self, 'pLargeImage', pLargeImage)
        object.__setattr__(self, 'pNIExperiment', pNIExperiment)
        object.__setattr__(self, 'pRecordedData', pRecordedData)
        object.__setattr__(self, 'sParallelExperiment', sParallelExperiment)
        object.__setattr__(self, 'pExternalData', pExternalData)

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
            and all(exp.valid for exp in self.ppNextLevelEx)
        )
    
    @property
    def isLambda(self) -> bool:
        return self.eType == ExperimentLoopType.eEtSpectLoop
    
    @property
    def count(self) -> int:
        return len([item for item in self.pItemValid if item]) if self.pItemValid is not None else self.uLoopPars.uiCount
    
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
        raise NotImplementedError()
    
    @staticmethod
    def createExperimentLevels(ppNextLevelEx: list[ExperimentLevel]|dict):
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
