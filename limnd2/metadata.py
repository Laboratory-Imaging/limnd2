from __future__ import annotations

import datetime, enum
from dataclasses import dataclass, field
from .litevariant import decode_lv
from .variant import decode_var

def jdn_now():
   result = datetime.datetime.utcnow().timestamp()
   result /= 86_400         # seconds per day
   result += 2_440_587.5    # JDN EPOCH
   return result;    

class PictureMetadataTimeSourceType(enum.IntEnum):
    etsSW = 0
    etsNIDAQ = 1

class PictureMetadataAxisDescription(enum.IntEnum):
    eaxdX = 0
    eaxdY = 1
    eaxdT = 2
    eaxdZ = 3
    eaxdPoint = 4 # confocal point scan has both ePictureXAxis and ePictureYAxis set to this value

class PictureMetadataPicturePlanesRepresentation(enum.IntEnum):
    eRepDefault = 0
    eRepHDR     = 2

class PicturePlaneModality(enum.IntEnum):
    eModWidefieldFluo       = 0
    eModBrightfield         = 1
    eModLaserScanConfocal   = 2
    eModSpinDiskConfocal    = 3
    eModSweptFieldConfocal  = 4
    eModMultiPhotonFluo     = 5
    eModPhaseContrast       = 6
    eModDIContrast          = 7
    eModSpectralConfocal    = 8
    eModVAASConfocal        = 9
    eModVAASConfocalIF      = 10
    eModVAASConfocalNF      = 11
    eModDSDConfocal         = 12

class PicturePlaneModalityFlags(enum.IntFlag):
    modFluorescence                     = 0x0000000000000001
    modBrightfield                      = 0x0000000000000002
    modDarkfield                        = 0x0000000000000004
    modMaskLight                        = (modFluorescence|modBrightfield|modDarkfield)
    modPhaseContrast                    = 0x0000000000000010
    modDIContrast                       = 0x0000000000000020
    modNAMC                             = 0x0000000000000008
    modMaskContrast                     = (modPhaseContrast|modDIContrast|modNAMC)
    modCamera                           = 0x0000000000000100
    modLaserScanConfocal                = 0x0000000000000200
    modSpinDiskConfocal                 = 0x0000000000000400
    modSweptFieldConfocalSlit           = 0x0000000000000800
    modSweptFieldConfocalPinholes       = 0x0000000000001000
    modDSDConfocal                      = 0x0000000000002000
    modSIM                              = 0x0000000000004000
    modISIM                             = 0x0000000000008000
    modRCM                              = 0x0000000000000040
    modSora                             = 0x0000000040000000
    modLiveSR                           = 0x0000000000040000
    modLightSheet                       = 0x0000000000080000
    modDeepSIM                          = 0x0000002000000000
    modMaskAcqHWType                    = (modCamera|modLaserScanConfocal|modSpinDiskConfocal|modSweptFieldConfocalSlit|modSweptFieldConfocalPinholes|modDSDConfocal|modRCM|modDeepSIM|modISIM|modSora|modLiveSR|modLightSheet)
    modMultiPhotonFluo                  = 0x0000000000010000
    modTIRF                             = 0x0000000000020000
    modPMT                              = 0x0000000000100000
    modSpectral                         = 0x0000000000200000
    modVAAS_IF                          = 0x0000000000400000
    modVAAS_NF                          = 0x0000000000800000
    modTransmitDetector                 = 0x0000000001000000
    modNonDescannedDetector             = 0x0000000002000000
    modVirtualFilter                    = 0x0000000004000000
    modGaAsP                            = 0x0000000008000000
    modRemainder                        = 0x0000000010000000
    modAUX                              = 0x0000000020000000
    modCustomDescChannel                = 0x0000000080000000
    modSTED                             = 0x0000000100000000
    modGalvano                          = 0x0000000200000000
    modResonant                         = 0x0000000400000000
    modAX                               = 0x0000000800000000
    modStorm                            = 0x0000001000000000
    modNSPARCDetector                   = 0x0000004000000000
    modPMT_IRGaAsP                      = 0x0000008000000000
    modPMT_GaAs                         = 0x0000010000000000
    modMaskDetector                     = (modSpectral|modVAAS_IF|modVAAS_NF|modTransmitDetector|modNonDescannedDetector|modVirtualFilter|modAUX|modNSPARCDetector)

    @staticmethod
    def from_modality(mod: PicturePlaneModality) -> PicturePlaneModalityFlags:
        return {
            PicturePlaneModality.eModWidefieldFluo:     PicturePlaneModalityFlags.modModFluorescence|PicturePlaneModalityFlags.modCamera,
            PicturePlaneModality.eModBrightfield:       PicturePlaneModalityFlags.modBrightfield|PicturePlaneModalityFlags.modCamera,
            PicturePlaneModality.eModLaserScanConfocal: PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modLaserScanConfocal,
            PicturePlaneModality.eModSpinDiskConfocal:  PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modSpinDiskConfocal,
            PicturePlaneModality.eModSweptFieldConfocal:PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modSweptFieldConfocalSlit,
            PicturePlaneModality.eModMultiPhotonFluo:   PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modMultiPhotonFluo|PicturePlaneModalityFlags.modLaserScanConfocal,
            PicturePlaneModality.eModPhaseContrast:     PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modPhaseContrast,
            PicturePlaneModality.eModDIContrast:        PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modDIContrast,
            PicturePlaneModality.eModSpectralConfocal:  PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modLaserScanConfocal|PicturePlaneModalityFlags.modSpectral,
            PicturePlaneModality.eModVAASConfocal:      PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modLaserScanConfocal,
            PicturePlaneModality.eModVAASConfocalIF:    PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modLaserScanConfocal|PicturePlaneModalityFlags.modVAAS_IF,
            PicturePlaneModality.eModVAASConfocalNF:    PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modLaserScanConfocal|PicturePlaneModalityFlags.modVAAS_NF,
            PicturePlaneModality.eModDSDConfocal:       PicturePlaneModalityFlags.modDSDConfocal
        }.get(mod, PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modCamera)

@dataclass(init=False, frozen=True)
class PictureMetadataPicturePlanesPlaneDesc:
    uiCompCount: int = 1
    uiSampleIndex: int = 0                  # Specifies a sample relation of this instance. SLxPicturePlaneDesc instances are
                                            #   grouped by this index. Index also determines the sample settings for this instance
                                            #   (see SLxSampleSetting).
    dObjCalibration1to1: float = 0          # Calibration and camera chip used for acquisition
    sizeObjFullChip: tuple[int, int] = (0, 0)
    uiModalityMask: PicturePlaneModalityFlags = PicturePlaneModalityFlags.modFluorescence
    pFluorescentProbe: dict = field(default_factory=dict)
                                            # Spectrum of the fluorescence fluorophore used. It can be specified by the user and
                                            #   can be used for any calculations (spectral unmixing etc.)
    pFilterPath: dict = field(default_factory=dict)                  
                                            # Filter path description, comes from devices. It can contain information about all
                                            #   elements influencing the spectral properties in the optical path. Optionally there
                                            #   can be lamps (incl. spectra), filters (incl. CCD sensitivity) and shutters.
                                            #   It must enable a description of an experiment using e.g.: exc. filterwheel, ems. fw,
                                            #   mirrors, shutter, DIA lamp for DIC    
    dLampVoltage: float = 0
    dFadingCorr: float = 0                  # The coefficient used for fluorescence fading correction.
    uiColor: int = 0x00ffffff               # The colour used for representation of the plane and optionally for look-up table
                                            #   creation. By default same as excitation filter.
    sDescription: str = ""                  # name
    dAcqTime: float = 0                     # acquistion time of one single image plane (value can be different for more planes inside one picture)
    dPinholeDiameter: float = -1            # pinhole size in um
    iChannelSeriesIndex: int = -1           # channel series index
    iCapturedPlaneIndex: int = -1           # index of picture plane at the capture time,
                                            #   used to access correct camerasettings items corresponding to this picture plane    
    
    def __init__(   self,
                    *,
                    uiCompCount: int = 1,
                    uiSampleIndex: int = 0,
                    dObjCalibration1to1: float = 0,
                    sizeObjFullChip: tuple[int, int] = (0, 0),
                    uiModalityMask: PicturePlaneModalityFlags = PicturePlaneModalityFlags.modFluorescence,
                    pFluorescentProbe: dict = {},
                    pFilterPath: dict = {},
                    dLampVoltage: float = 0,
                    dFadingCorr: float = 0,
                    uiColor: int = 0x00ffffff,
                    sDescription: str = "",
                    dAcqTime: float = 0,
                    dPinholeDiameter: float = -1,
                    iChannelSeriesIndex: int = -1,
                    iCapturedPlaneIndex: int = -1,
                    **kwargs):        
        sizeObjFullChip = (kwargs.get('sizeObjFullChip.cx', sizeObjFullChip[0]), kwargs.get('sizeObjFullChip.cy', sizeObjFullChip[1]))
        uiModalityMask = PicturePlaneModalityFlags.from_modality(kwargs['eModality']) if 'eModality' in kwargs else uiModalityMask
        object.__setattr__(self, 'uiCompCount', uiCompCount)
        object.__setattr__(self, 'uiSampleIndex', uiSampleIndex)
        object.__setattr__(self, 'dObjCalibration1to1', dObjCalibration1to1)
        object.__setattr__(self, 'sizeObjFullChip', sizeObjFullChip)
        object.__setattr__(self, 'uiModalityMask', uiModalityMask)
        object.__setattr__(self, 'pFluorescentProbe', pFluorescentProbe)
        object.__setattr__(self, 'pFilterPath', pFilterPath)
        object.__setattr__(self, 'dLampVoltage', dLampVoltage)
        object.__setattr__(self, 'dFadingCorr', dFadingCorr)
        object.__setattr__(self, 'uiColor', uiColor)
        object.__setattr__(self, 'sDescription', sDescription)
        object.__setattr__(self, 'dAcqTime', dAcqTime)
        object.__setattr__(self, 'dPinholeDiameter', dPinholeDiameter)
        object.__setattr__(self, 'iChannelSeriesIndex', iChannelSeriesIndex)
        object.__setattr__(self, 'iCapturedPlaneIndex', iCapturedPlaneIndex)

@dataclass(init=False, frozen=True)
class PictureMetadataPicturePlanes:
    uiCount: int = 0                         # == len(sPlane)
    uiCompCount: int = 0                     # the sum of uiCompCount of all sPlane members
    sPlane: list[PictureMetadataPicturePlanesPlaneDesc] = field(default_factory=list)
    uiSampleCount: int = 0                   # the count of different samples, each sample may have 1 or more planes
    sSampleSetting: list[dict] = field(default_factory=list)
                                            #list[PictureMetadataPicturePlanesSampleSettings] = [] # camera setting, microscope setting,...
    sDescription: str = ""
    eRepresentation: PictureMetadataPicturePlanesRepresentation = PictureMetadataPicturePlanesRepresentation.eRepDefault
    iExperimentSettingsCount: int = 0
    sExperimentSetting: list[dict] = field(default_factory=list)
    iStimulationSettingsCount: int = 0
    sStimulationSetting: list[dict] = field(default_factory=list)

    def __init__(   self, 
                    *,
                    uiCount: int = 0,
                    uiCompCount: int = 0,
                    sPlane: list[PictureMetadataPicturePlanesPlaneDesc]|dict|None = None, 
                    sPlaneNew: dict[str, dict]|None = None, 
                    uiSampleCount: int = 0,
                    sSampleSetting: list[dict] = [],
                    sDescription: str = "",
                    eRepresentation: PictureMetadataPicturePlanesRepresentation = PictureMetadataPicturePlanesRepresentation.eRepDefault,
                    iExperimentSettingsCount: int = 0,
                    sExperimentSetting: list[dict] = [],
                    iStimulationSettingsCount: int = 0,
                    sStimulationSetting: list[dict] = []):
        object.__setattr__(self, 'uiCount', uiCount)
        object.__setattr__(self, 'uiCompCount', uiCompCount)
        sPlane_ = []
        if sPlaneNew is not None:
            for _, item in sPlaneNew.items():
                if type(item) == dict:
                    sPlane_.append(PictureMetadataPicturePlanesPlaneDesc(**item))
                else:
                    raise TypeError()
        elif type(sPlane) == dict:
            for _, item in sPlane.items():
                if type(item) == dict:
                    sPlane_.append(PictureMetadataPicturePlanesPlaneDesc(**item))
                else:
                    raise TypeError()
        elif type(sPlane) == list:
            for item in sPlane:
                if type(item) == dict:
                    sPlane_.append(PictureMetadataPicturePlanesPlaneDesc(**item))
                elif isinstance(item, PictureMetadataPicturePlanesPlaneDesc):
                    sPlane_.append(item)
                else:
                    raise TypeError()
        object.__setattr__(self, 'sPlane', sPlane_)
        object.__setattr__(self, 'uiSampleCount', uiSampleCount)
        object.__setattr__(self, 'sSampleSetting', sSampleSetting)
        object.__setattr__(self, 'sDescription', sDescription)
        object.__setattr__(self, 'eRepresentation', eRepresentation)
        object.__setattr__(self, 'iExperimentSettingsCount', iExperimentSettingsCount)
        object.__setattr__(self, 'sExperimentSetting', sExperimentSetting)
        object.__setattr__(self, 'iStimulationSettingsCount', iStimulationSettingsCount)
        object.__setattr__(self, 'sStimulationSetting', sStimulationSetting)

    @property
    def valid(self) -> bool:
        return 0 < self.uiCount and self.uiCount <= self.uiCompCount and self.uiCount == len(self.sPlane)
    
    def makeValid(self, comps: int, **kwargs) -> None:
        if comps not in (1, 3):
            raise ValueError()        
        args = dict(uiCompCount=comps, 
                    uiModalityMask=PicturePlaneModalityFlags.modBrightfield if comps == 3 else PicturePlaneModalityFlags.modFluorescence, 
                    sDescription="RGB" if comps == 3 else "Mono")
        args.update(kwargs)
        object.__setattr__(self, 'uiCount', 1)
        object.__setattr__(self, 'uiCompCount', comps)
        object.__setattr__(self, 'sPlane', [ PictureMetadataPicturePlanesPlaneDesc(**args) ])


@dataclass(frozen=True, kw_only=True)
class PictureMetadataPhysicalQuantity:
    wsName: str = ""
    uiIntepretation: int = 0
    dValue: float = 0
    
@dataclass(init=False, frozen=True)
class PictureMetadata:
    dTimeAbsolute: float = -1.0             # time specification when the picture was captured [Julian Day Number]
    dTimeMSec: float = 0.0                  # time offset of captured frame [ms]
    eTimeSource: PictureMetadataTimeSourceType = PictureMetadataTimeSourceType.etsSW
    dXPos: float = 0.0
    dYPos: float = 0.0
    uiRow: int = 0
    uiCol: int = 0
    dZPos: float = 0.0
    bZPosAbsolute: bool = False 
    dAngle: float = 0.0
    sPicturePlanes: PictureMetadataPicturePlanes = field(default_factory=PictureMetadataPicturePlanes)
    dTemperK: float = 293                   # temperature (in Kelvins)
    dCalibration: float = -1                # microns to pixel
    dAspect: float = -1                     # pixel aspect ratio
    dCalibPrecision: float = -1             # calibration precision in microns
    bCalibrated: bool = False               # is calibration valid 
    wsObjectiveName: str = ""
    dObjectiveMag: float = -1
    dObjectiveNA: float = -1
    dRefractIndex1: float = -1
    dRefractIndex2: float = -1
    dZoom: float = -1
    pPhysicalVar: list[PictureMetadataPhysicalQuantity] = field(default_factory=list)
    uiPhysicalVarCount: int = 0             # == len(pPhysicalVar)
    wsCustomData: str = ""
    ePictureXAxis: PictureMetadataAxisDescription = PictureMetadataAxisDescription.eaxdX
    ePictureYAxis: PictureMetadataAxisDescription = PictureMetadataAxisDescription.eaxdY
    dTimeAxisCalibration: float = -1        # valid when there is eaxdT axis, in ms
    dZAxisCalibration: float = -1           # valid when there is eaxdZ axis
    dStgLgCT11: float = 1                   # transformation matrix, more general than dAngle
    dStgLgCT21: float = 0
    dStgLgCT12: float = 0
    dStgLgCT22: float = 1
    baOpticalPathsCorrections: bytes = b''  # Inter-modality registration support - CLxOpticalPathsCorrectionTable serialized to LiteVariant

    def __init__(   self, 
                    *,
                    dTimeAbsolute: float = jdn_now(),
                    dTimeMSec: float = 0.0,
                    eTimeSource: PictureMetadataTimeSourceType = PictureMetadataTimeSourceType.etsSW,
                    dXPos: float = 0.0,
                    dYPos: float = 0.0,
                    uiRow: int = 0,
                    uiCol: int = 0,
                    dZPos: float = 0.0,
                    bZPosAbsolute: bool = False,
                    dAngle: float = 0.0,
                    sPicturePlanes: PictureMetadataPicturePlanes|dict = {},
                    dTemperK: float = 293,
                    dCalibration: float = -1,
                    dAspect: float = -1,
                    dCalibPrecision: float = -1,
                    bCalibrated: bool = False,
                    wsObjectiveName: str = "",
                    dObjectiveMag: float = -1,
                    dObjectiveNA: float = -1,
                    dRefractIndex1: float = -1,
                    dRefractIndex2: float = -1,
                    dZoom: float = -1,
                    pPhysicalVar: list[PictureMetadataPhysicalQuantity] = [],
                    uiPhysicalVarCount: int = 0,
                    wsCustomData: str = "",
                    ePictureXAxis: PictureMetadataAxisDescription = PictureMetadataAxisDescription.eaxdX,
                    ePictureYAxis: PictureMetadataAxisDescription = PictureMetadataAxisDescription.eaxdY,
                    dTimeAxisCalibration: float = -1,
                    dZAxisCalibration: float = -1,
                    dStgLgCT11: float = 1,
                    dStgLgCT21: float = 0,
                    dStgLgCT12: float = 0,
                    dStgLgCT22: float = 1,
                    baOpticalPathsCorrections: bytes = b'',
                    **kwargs):
        uiCol = kwargs.get("uiCon20(L", uiCol)
        object.__setattr__(self, 'dTimeAbsolute', dTimeAbsolute)
        object.__setattr__(self, 'dTimeMSec', dTimeMSec)
        object.__setattr__(self, 'eTimeSource', eTimeSource)
        object.__setattr__(self, 'dXPos', dXPos)
        object.__setattr__(self, 'dYPos', dYPos)
        object.__setattr__(self, 'uiRow', uiRow if uiRow != 0xffffffff else 0)
        object.__setattr__(self, 'uiCol', uiCol if uiCol != 0xffffffff else 0)
        object.__setattr__(self, 'dZPos', dZPos)
        object.__setattr__(self, 'bZPosAbsolute', bZPosAbsolute)
        object.__setattr__(self, 'dAngle', dAngle)
        if type(sPicturePlanes) == dict:
            object.__setattr__(self, 'sPicturePlanes', PictureMetadataPicturePlanes(**sPicturePlanes))
        elif isinstance(sPicturePlanes, PictureMetadataPicturePlanes):
            object.__setattr__(self, 'sPicturePlanes', sPicturePlanes)
        else:
            raise TypeError()
        object.__setattr__(self, 'dTemperK', dTemperK)
        object.__setattr__(self, 'dCalibration', dCalibration)
        object.__setattr__(self, 'dAspect', dAspect)
        object.__setattr__(self, 'dCalibPrecision', dCalibPrecision)
        object.__setattr__(self, 'bCalibrated', bCalibrated)
        object.__setattr__(self, 'wsObjectiveName', wsObjectiveName)
        object.__setattr__(self, 'dObjectiveMag', dObjectiveMag)
        object.__setattr__(self, 'dObjectiveNA', dObjectiveNA)
        object.__setattr__(self, 'dRefractIndex1', dRefractIndex1)
        object.__setattr__(self, 'dRefractIndex2', dRefractIndex2)
        object.__setattr__(self, 'dZoom', dZoom)
        object.__setattr__(self, 'pPhysicalVar', pPhysicalVar)
        object.__setattr__(self, 'uiPhysicalVarCount', uiPhysicalVarCount)
        object.__setattr__(self, 'wsCustomData', wsCustomData)
        object.__setattr__(self, 'ePictureXAxis', ePictureXAxis)
        object.__setattr__(self, 'ePictureYAxis', ePictureYAxis)
        object.__setattr__(self, 'dTimeAxisCalibration', dTimeAxisCalibration)
        object.__setattr__(self, 'dZAxisCalibration', dZAxisCalibration)
        object.__setattr__(self, 'dStgLgCT11', dStgLgCT11)
        object.__setattr__(self, 'dStgLgCT21', dStgLgCT21)
        object.__setattr__(self, 'dStgLgCT12', dStgLgCT12)
        object.__setattr__(self, 'dStgLgCT22', dStgLgCT22)
        object.__setattr__(self, 'baOpticalPathsCorrections', baOpticalPathsCorrections)


    @property
    def valid(self) -> bool:
        return self.sPicturePlanes.valid
    
    def makeValid(self, comps: int, **kwargs) -> None:
        self.sPicturePlanes.makeValid(comps, **kwargs)

    @property
    def isRgb(self) -> bool:
        return 1 == self.sPicturePlanes.uiCount and 3 == self.sPicturePlanes.uiCompCount

    @property
    def componentNames(self) -> list[str]:
        ret = []
        for plane in self.sPicturePlanes.sPlane:
            if 3 == plane.uiCompCount:
                ret.append("Blue")
                ret.append("Green")
                ret.append("Red")
            else:
                ret.append(plane.sDescription)
        return ret
    
    @property
    def componentColors(self) -> list[tuple[float, float, float]]:
        def color_as_tuple(color):
            b = ((color >> 16) & 0xFF) / 255.0
            g = ((color >> 8) & 0xFF) / 255.0
            r = (color & 0xFF) / 255.0
            return (r, g, b)
        ret = []
        for plane in self.sPicturePlanes.sPlane:
            if 3 == plane.uiCompCount:
                ret.append((0, 0, 1))
                ret.append((0, 1, 0))
                ret.append((1, 0, 0))
            else:
                ret.append(color_as_tuple(plane.uiColor))
        return ret        

      
    def to_lv(self) -> bytes:
        raise NotImplementedError()
    
    @staticmethod
    def from_lv(data: bytes|memoryview) -> PictureMetadata:
        decoded = decode_lv(data)
        return PictureMetadata(**decoded.get('SLxPictureMetadata', {}))
    
    @staticmethod
    def from_var(data: bytes|memoryview) -> PictureMetadata:
        decoded = decode_var(data)
        return PictureMetadata(**decoded.get('SLxPictureMetadata', {}))