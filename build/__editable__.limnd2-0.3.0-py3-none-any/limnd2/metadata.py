from __future__ import annotations

import datetime, enum, numpy as np, operator
from dataclasses import dataclass, field
from .lite_variant import decode_lv
from .variant import decode_var

def jdn_now():
   result = datetime.datetime.now(datetime.UTC).timestamp()
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

class OpticalFilterPlacement(enum.IntEnum):
    eOfpNoFilter = 0
    eOfpExcitation = 1      # excitation filter (position by a lamp)
    eOfpEmission = 2        # emission filter (position by a camera)
    eOfpFilterTurret = 3    # filter block (mainly fluorescence)
    eOfpLamp = 4            # spectrum of a lamp
    eOfnCameraChip = 5      # camera chip sensitivity
    eOfpUserOverride = 6    # user defined emission wavelength

class OpticalFilterNature(enum.IntFlag):
    eOfnGeneric = 0x0000    # wide-band or unspecified spectra
    eOfnRGB     = 0x0001    # triple-band filter suitable for RGB cameras (or naked eye)
    eOfnRed		= 0x0002
    eOfnGreen   = 0x0004
    eOfnBlue    = 0x0008

class OpticalFilterSpectType(enum.IntEnum):
    eOftBandpass            = 1 # specified by lower (raising edge) and higher (falling) wavelength
    eOftNarrowBandpass      = 2 # specified by one wavelength (peak)
    eOftLowpass             = 3 # specified by one wavelength (falling edge)
    eOftHighpass            = 4 # specified by one wavelength (raising edge)
    eOftBarrier             = 5 # specified by lower (falling edge) and higher (raising) wavelength
    eOftMultiplepass        = 6 # specified by a few edges
    eOftFull                = 7 # full position of a filterwheel
    eOftEmpty               = 8 # empty position of a filterwheel        

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
            PicturePlaneModality.eModWidefieldFluo:     PicturePlaneModalityFlags.modFluorescence|PicturePlaneModalityFlags.modCamera,
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
    
    @staticmethod
    def to_str_list(flags : PicturePlaneModalityFlags) -> list[str]:
        ret: list[str] = []
        # light
        if flags & PicturePlaneModalityFlags.modFluorescence:
            ret.append("Fluorescence")
        if flags & PicturePlaneModalityFlags.modBrightfield:
            ret.append("Brightfield")
        if flags & PicturePlaneModalityFlags.modDarkfield:
            ret.append("Darkfield")
        # contrast
        if flags & PicturePlaneModalityFlags.modPhaseContrast:
            ret.append("Phase")
        if flags & PicturePlaneModalityFlags.modDIContrast:
            ret.append("DIC")
        if flags & PicturePlaneModalityFlags.modNAMC:
            ret.append("NAMC")
        # HW
        if flags & PicturePlaneModalityFlags.modCamera:
            ret.append("Camera")
        if flags & PicturePlaneModalityFlags.modLaserScanConfocal:
            ret.append("LaserScanConfocal")
        if flags & PicturePlaneModalityFlags.modSpinDiskConfocal:
            ret.append("SpinDiskConfocal")
        if flags & PicturePlaneModalityFlags.modSweptFieldConfocalSlit:
            ret.append("SweptFieldConfocalSlit")
        if flags & PicturePlaneModalityFlags.modSweptFieldConfocalPinholes:
            ret.append("SweptFieldConfocalPinholes")
        if flags & PicturePlaneModalityFlags.modDSDConfocal:
            ret.append("DSDConfocal")
        if flags & PicturePlaneModalityFlags.modSIM:
            ret.append("SIM")
        if flags & PicturePlaneModalityFlags.modISIM:
            ret.append("ISIM")
        if flags & PicturePlaneModalityFlags.modRCM:
            ret.append("RCM")
        if flags & PicturePlaneModalityFlags.modSora:
            ret.append("SORA")
        if flags & PicturePlaneModalityFlags.modLiveSR:
            ret.append("LiveSR")
        if flags & PicturePlaneModalityFlags.modLightSheet:
            ret.append("LightSheet")
        if flags & PicturePlaneModalityFlags.modDeepSIM:
            ret.append("DeepSIM")
        if flags & PicturePlaneModalityFlags.modMultiPhotonFluo:
            ret.append("MultiPhotonFluo")
        if flags & PicturePlaneModalityFlags.modTIRF:
            ret.append("TIRF")
        if flags & PicturePlaneModalityFlags.modPMT:
            ret.append("PMT")
        if flags & PicturePlaneModalityFlags.modSpectral:
            ret.append("Spectral")
        if flags & PicturePlaneModalityFlags.modVAAS_IF:
            ret.append("VAASIF")
        if flags & PicturePlaneModalityFlags.modVAAS_NF:
            ret.append("VAASNF")
        if flags & PicturePlaneModalityFlags.modTransmitDetector:
            ret.append("TransmitDetector")
        if flags & PicturePlaneModalityFlags.modNonDescannedDetector:
            ret.append("NonDescannedDetector")
        if flags & PicturePlaneModalityFlags.modVirtualFilter:
            ret.append("VirtualFilter")
        if flags & PicturePlaneModalityFlags.modGaAsP:
            ret.append("GaAsP")
        if flags & PicturePlaneModalityFlags.modRemainder:
            ret.append("Remainder")
        if flags & PicturePlaneModalityFlags.modAUX:
            ret.append("AUX")
        if flags & PicturePlaneModalityFlags.modCustomDescChannel:
            ret.append("CustomDescChannel")
        if flags & PicturePlaneModalityFlags.modGalvano:
            ret.append("Galvano")
        if flags & PicturePlaneModalityFlags.modResonant:
            ret.append("Resonant")
        if flags & PicturePlaneModalityFlags.modAX:
            ret.append("AX")
        if flags & PicturePlaneModalityFlags.modStorm:
            ret.append("Storm")
        if flags & PicturePlaneModalityFlags.modNSPARCDetector:
            ret.append("NSPARC")
        if flags & PicturePlaneModalityFlags.modPMT_IRGaAsP:
            ret.append("PMTIRGaAsP")
        if flags & PicturePlaneModalityFlags.modPMT_GaAs:
            ret.append("PMTGaAs")
        return ret


class OpticalSpectrumPointType(enum.IntEnum):
    eSptInvalid = 0
    eSptPoint = 1
    eSptRaisingEdge = 2
    eSptFallingEdge = 3
    eSptPeak = 4
    eSptRange = 5


@dataclass(frozen=True, kw_only=True)
class OpticalSpectrumPoint:
    eType: OpticalSpectrumPointType = 0
    dWavelength: float = 0.0
    dTValue: float = 0.0

@dataclass(init=False, frozen=True)
class OpticalSpectrum:
    bPoints: bool = False
    pPoint: list[OpticalSpectrumPoint] = field(default_factory=list)

    def __init__(   self,
                    *,
                    bPoints: bool = False,
                    pPoint: dict|list[OpticalSpectrumPoint] = [],
                    **kwargs):
        
        object.__setattr__(self, 'bPoints', bPoints)
        if type(pPoint) == dict and 'uiCount' in kwargs:
            pPoint_ = []
            for i in range(kwargs['uiCount']):
                pPoint_.append(OpticalSpectrumPoint(**pPoint[f"Point{i}"]))
            object.__setattr__(self, 'pPoint', pPoint_)
        elif type(pPoint) == list and all(isinstance(item, OpticalSpectrumPoint) for item in pPoint):
            object.__setattr__(self, 'pPoint', pPoint)
        else:
            object.__setattr__(self, 'pPoint', [])

    @property
    def isValid(self) -> bool:
        return 0 < len(self.pPoint)
    
    @property
    def count(self) -> int:
        return len(self.pPoint)
    
    def findmaxtvalue(self) -> tuple[int, float]:
        return max(enumerate(pt.dTValue for pt in self.pPoint), key=operator.itemgetter(1))
    
    def peakAndFWHM(self) -> tuple[float, float, float]:
        if self.bPoints:
            if not self.isValid:
               raise ValueError
            ifirst, tmax = self.findmaxtvalue()
            ilast = ifirst + 1
            halftmax = tmax / 2
            while tmax == self.pPoint[ilast].dTValue:
               ilast += 1
            ilast -= 1
            peak = (self.pPoint[ifirst].wavelength + self.pPoint[ilast].wavelength) / 2;
            while ilast < len(self.pPoint) and halftmax < self.pPoint[ilast].dTValue:
                ilast += 1
            while 0 <= ifirst and halftmax < self.pPoint[ifirst].dTValue:
                ifirst -= 1                
            if not (0 <= ifirst and ilast < len(self.pPoint)):
               raise ValueError            
            return peak, self.pPoint[ifirst].dWavelength, self.pPoint[ilast].dWavelength               
        else:
            for index, pt in enumerate(self.pPoint):
                if pt.eType == OpticalSpectrumPointType.eSptRaisingEdge and index + 1 < len(self.pPoint) and self.pPoint[index + 1].eType == OpticalSpectrumPointType.eSptFallingEdge:
                   return (pt.wavelength + self.pPoint[index + 1].wavelength) / 2, pt.dWavelength, self.pPoint[index + 1].wavelength
                elif pt.eType == OpticalSpectrumPointType.eSptPeak:
                    return pt.dWavelength, pt.dWavelength, pt.dWavelength
                
    def singleWavelength(self) -> float:
        try:
            peak, _, _ = self.peakAndFWHM()
            return peak
        except ValueError:
            pass
        if 1 == len(self.pPoint):
            return self.pPoint[0].dWavelength        
        num, denom = 0.0, 0.0
        for pt in self.pPoint:
            num += pt.dWavelength * pt.dTValue
            denom += pt.dTValue
        return num / denom if 0 < denom else 0

    def wavelengthRange(self) -> tuple[float, float]:
        if not self.isValid:
            raise ValueError
        dWlMin = self.pPoint[ 0].dWavelength
        dWlMax = self.pPoint[-1].dWavelength
        if not self.bPoints:
            if self.pPoint[0].eType in (OpticalSpectrumPointType.eSptRaisingEdge, OpticalSpectrumPointType.eSptPeak):
                dWlMin = self.pPoint[ 0].dWavelength - 1.0
            if self.pPoint[-1].eType in (OpticalSpectrumPointType.eSptFallingEdge, OpticalSpectrumPointType.eSptPeak):                
                dWlMin = self.pPoint[-1].dWavelength + 1.0
        return dWlMin, dWlMax
    
    @staticmethod
    def combine(a: OpticalSpectrum, b: OpticalSpectrum) -> OpticalSpectrum:
        ret, tol = OpticalSpectrum._combine_low(a, b, 0)
        if 0 < tol:
            ret, _ = OpticalSpectrum._combine_low(a, b, tol)
        return ret

    @staticmethod
    def _combine_low(a: OpticalSpectrum, b: OpticalSpectrum, tol: float) -> tuple[OpticalSpectrum, float]:
        if 0 == a.count:
            return b, 0
        result = []
        ia, ib, mindiff = 0, 0, 0.0
        awl = [pt.dWavelength for pt in a.pPoint]
        bwl = [pt.dWavelength for pt in b.pPoint]
        transpa = a.pPoint[0].eType == OpticalSpectrumPointType.eSptFallingEdge
        transpb = b.pPoint[0].eType == OpticalSpectrumPointType.eSptFallingEdge
        while ia < a.count and ib < b.count:
            if awl[ia] < bwl[ib]:
                if a.pPoint[ia].eType in (OpticalSpectrumPointType.eSptFallingEdge, OpticalSpectrumPointType.eSptPeak) and b.pPoint[ib].eType in (OpticalSpectrumPointType.eSptRaisingEdge, OpticalSpectrumPointType.eSptPeak):
                    diff = bwl[ib] - awl[ia]
                    if diff < mindiff:
                        mindiff = diff
            if awl[ia] > bwl[ib]:
                if a.pPoint[ia].eType in (OpticalSpectrumPointType.eSptRaisingEdge, OpticalSpectrumPointType.eSptPeak) and b.pPoint[ib].eType in (OpticalSpectrumPointType.eSptFallingEdge, OpticalSpectrumPointType.eSptPeak):
                    diff = awl[ia] - bwl[ib]
                    if diff < mindiff:
                        mindiff = diff
            if 0 < tol:
                if awl[ia] < bwl[ib]:
                    if a.pPoint[ia].eType in (OpticalSpectrumPointType.eSptFallingEdge, OpticalSpectrumPointType.eSptPeak) and b.pPoint[ib].eType in (OpticalSpectrumPointType.eSptRaisingEdge, OpticalSpectrumPointType.eSptPeak):
                        diff = bwl[ib] - awl[ia]
                        if diff <= tol:
                            bwl[ib] = awl[ia] - 1
                if awl[ia] > bwl[ib]:
                    if a.pPoint[ia].eType in (OpticalSpectrumPointType.eSptRaisingEdge, OpticalSpectrumPointType.eSptPeak) and b.pPoint[ib].eType in (OpticalSpectrumPointType.eSptFallingEdge, OpticalSpectrumPointType.eSptPeak):
                        diff = awl[ia] - bwl[ib]
                        if diff <= tol:
                            awl[ia] = bwl[ib] - 1
            doa = awl[ia] < bwl[ib] or (awl[ia] <= bwl[ib] and b.pPoint[ib].eType == OpticalSpectrumPointType.eSptFallingEdge) or ib == b.count
            idx = ia if doa else ib
            spect = a if doa else b
            point: OpticalSpectrumPoint = a[ia] if doa else b[ib]
            point.dWavelength = awl[ia] if doa else bwl[ib]
            def set_transposethis(x):
                nonlocal transpa, transpb
                if doa: 
                    transpa=x 
                else:
                    transpb=x
            transposeother = lambda: transpb if doa else transpa
            if point.eType == OpticalSpectrumPointType.eSptPoint:
                if transposeother():
                    result.append(point)
                    mindiff = 0
                set_transposethis(idx + 1 < spect.count)
            elif point.eType == OpticalSpectrumPointType.eSptPeak:
                if transposeother():
                    result.append(point)
                    mindiff = 0
            elif point.eType == OpticalSpectrumPointType.eSptRaisingEdge:
                if transposeother():
                    result.append(point)
                    mindiff = 0
                set_transposethis(True)           
            elif point.eType == OpticalSpectrumPointType.eSptFallingEdge:
                if transposeother():
                    result.append(point)
                    mindiff = 0
                set_transposethis(False)
            if doa:
                ia += 1
            else:
                ib += 1
        return OpticalSpectrum(bPoints=a.bPoints, pPoint=result)


@dataclass(init=False, frozen=True)
class FluorescentProbe:
    m_sName: str = ""
    m_uiColor: int = 0
    m_ExcitationSpectrum: OpticalSpectrum = field(default_factory=OpticalSpectrum)
    m_EmissionSpectrum: OpticalSpectrum = field(default_factory=OpticalSpectrum)

    def __init__(   self,
                    *,
                    m_sName: str = "",
                    m_uiColor: int = 0,
                    m_ExcitationSpectrum: OpticalSpectrum|dict = {},
                    m_EmissionSpectrum: OpticalSpectrum|dict = {},
                    **kwargs):
        object.__setattr__(self, 'm_sName', m_sName)
        object.__setattr__(self, 'm_uiColor', m_uiColor)
        object.__setattr__(self, 'm_ExcitationSpectrum', OpticalSpectrum(**m_ExcitationSpectrum) if type(m_ExcitationSpectrum) == dict else m_ExcitationSpectrum)
        object.__setattr__(self, 'm_EmissionSpectrum', OpticalSpectrum(**m_EmissionSpectrum) if type(m_EmissionSpectrum) == dict else m_EmissionSpectrum)

@dataclass(init=False, frozen=True)
class OpticalFilter:
    m_sName: str = ""
    m_sUserName: str = ""
    m_ePlacement: OpticalFilterPlacement = 0
    m_eNature: OpticalFilterNature = 0
    m_eSpctType: OpticalFilterSpectType = 0
    m_uiColor: int = 0
    m_ExcitationSpectrum: OpticalSpectrum = field(default_factory=OpticalSpectrum)
    m_EmissionSpectrum: OpticalSpectrum = field(default_factory=OpticalSpectrum)
    m_MirrorSpectrum: OpticalSpectrum = field(default_factory=OpticalSpectrum)

    def __init__(   self,
                    *,
                    m_sName: str = "",
                    m_sUserName: str = "",
                    m_ePlacement: OpticalFilterPlacement = 0,
                    m_eNature: OpticalFilterNature = 0,
                    m_eSpctType: OpticalFilterSpectType = 0,
                    m_uiColor: int = 0,
                    m_ExcitationSpectrum: OpticalSpectrum|dict,
                    m_EmissionSpectrum: OpticalSpectrum|dict,
                    m_MirrorSpectrum: OpticalSpectrum|dict,
                    **kwargs):
        object.__setattr__(self, 'm_sName', m_sName)
        object.__setattr__(self, 'm_sUserName', m_sUserName)
        object.__setattr__(self, 'm_ePlacement', m_ePlacement)
        object.__setattr__(self, 'm_eNature', m_eNature)
        object.__setattr__(self, 'm_eSpctType', m_eSpctType)
        object.__setattr__(self, 'm_uiColor', m_uiColor)                
        object.__setattr__(self, 'm_ExcitationSpectrum', OpticalSpectrum(**m_ExcitationSpectrum) if type(m_ExcitationSpectrum) == dict else m_ExcitationSpectrum)
        object.__setattr__(self, 'm_EmissionSpectrum', OpticalSpectrum(**m_EmissionSpectrum) if type(m_EmissionSpectrum) == dict else m_EmissionSpectrum)    
        object.__setattr__(self, 'm_MirrorSpectrum', OpticalSpectrum(**m_MirrorSpectrum) if type(m_MirrorSpectrum) == dict else m_MirrorSpectrum)  

@dataclass(init=False, frozen=True)
class OpticalFilterPath:    
    m_sDescr: str = ""
    m_pFilter: list[OpticalFilter] = field(default_factory=list)

    def __init__(   self,
                    *,
                    m_sDescr: str = "",
                    m_pFilter: list[OpticalFilter]|dict = {},
                    **kwargs):
        object.__setattr__(self, 'm_sDescr', m_sDescr)
        if type(m_pFilter) == dict:
            m_pFilter_ = []
            for _, item in m_pFilter.items():
                m_pFilter_.append(OpticalFilter(**item))            
            object.__setattr__(self, 'm_pFilter', m_pFilter_)
        elif type(m_pFilter) == list and all(isinstance(item, OpticalFilter) for item in m_pFilter):
            object.__setattr__(self, 'm_pFilter', m_pFilter)
        else:
            object.__setattr__(self, 'm_pFilter', [])

    @property
    def isValid(self):
        return 0 < len(self.m_pFilter)
    
    def meanEmissionWavelength(self) -> float:
        def em2pt(s: OpticalSpectrum) -> float:
            if s.pPoint[0].eType == OpticalSpectrumPointType.eSptRaisingEdge and s.pPoint[1].eType == OpticalSpectrumPointType.eSptFallingEdge:
                return (s.pPoint[0].dWavelength + s.pPoint[1].dWavelength) / 2
            elif s.pPoint[0].eType == OpticalSpectrumPointType.eSptPeak and s.pPoint[1].eType == OpticalSpectrumPointType.eSptRange:
                return s.pPoint[0].dWavelength

        for flt in self.m_pFilter:
            if OpticalFilterPlacement.eOfpUserOverride == flt.m_ePlacement and 1 == flt.m_EmissionSpectrum.count:
                return flt.m_EmissionSpectrum.pPoint[0].dWavelength
        for flt in self.m_pFilter:
            if OpticalFilterPlacement.eOfpEmission == flt.m_ePlacement and 1 == flt.m_EmissionSpectrum.count:
                return flt.m_EmissionSpectrum.pPoint[0].dWavelength
        for flt in self.m_pFilter:
            if OpticalFilterPlacement.eOfpEmission == flt.m_ePlacement and 2 == flt.m_EmissionSpectrum.count:
                return em2pt(flt.m_EmissionSpectrum)
        for flt in self.m_pFilter:
            if OpticalFilterPlacement.eOfpFilterTurret == flt.m_ePlacement:
                dMin, dMax = flt.m_EmissionSpectrum.wavelengthRange()
                if 1 == flt.m_EmissionSpectrum.count and 0 < (dMin + dMax) / 2:
                    return flt.m_EmissionSpectrum.pPoint[0].dWavelength
        for flt in self.m_pFilter:
            if OpticalFilterPlacement.eOfpFilterTurret == flt.m_ePlacement and 2 == flt.m_EmissionSpectrum.count:
                return em2pt(flt.m_EmissionSpectrum)
            
        exSpectrum: OpticalSpectrum = None
        for flt in self.m_pFilter:
            if 0 < flt.m_ExcitationSpectrum.count:
                exSpectrum = OpticalSpectrum.combine(exSpectrum, flt.m_ExcitationSpectrum)
        emSpectrum: OpticalSpectrum = None
        for flt in self.m_pFilter:
            dMin, dMax = flt.m_EmissionSpectrum.wavelengthRange()
            if 0 < flt.m_EmissionSpectrum.count and 0 < (dMin + dMax) / 2:
                emSpectrum = OpticalSpectrum.combine(emSpectrum, flt.m_EmissionSpectrum)
        ex = None 
        for pt in exSpectrum.pPoint:
            if pt.eType in (OpticalSpectrumPointType.eSptPeak, OpticalSpectrumPointType.eSptFallingEdge):
                ex = pt
        if ex is not None:
            for j, pt in enumerate(emSpectrum.pPoint):
                if pt.eType in (OpticalSpectrumPointType.eSptPeak, OpticalSpectrumPointType.eSptRaisingEdge) and ex.dWavelength < pt.dWavelength:
                    if pt.eType == OpticalSpectrumPointType.eSptPeak:
                        return pt.dWavelength
                    elif j + 1 < emSpectrum.count and pt.eType == OpticalSpectrumPointType.eSptRaisingEdge:
                        return (pt.dWavelength + emSpectrum.pPoint[j+1].dWavelength) / 2
                    else:
                        return pt.dWavelength
        if 0 < emSpectrum.count:
            if emSpectrum.pPoint[0].eType in (OpticalSpectrumPointType.eSptPoint, OpticalSpectrumPointType.eSptPeak):
                return emSpectrum.pPoint[0].dWavelength
            if 1 < emSpectrum.count:
                return (emSpectrum.pPoint[0].dWavelength + emSpectrum.pPoint[1].dWavelength) / 2
        return 0


    def closestExcitationWavelength(self, emission: float) -> float:
        for flt in self.m_pFilter:
            if flt.m_ePlacement == OpticalFilterPlacement.eOfpUserOverride and 1 == flt.m_ExcitationSpectrum.count:
                return flt.m_ExcitationSpectrum.pPoint[0].dWavelength
        closestWL, dist = None, 1e300
        for flt in self.m_pFilter:
            if flt.m_ePlacement in (OpticalFilterPlacement.eOfpExcitation, OpticalFilterPlacement.eOfpFilterTurret, OpticalFilterPlacement.eOfpLamp) and flt.m_ExcitationSpectrum.isValid:
                if flt.m_ExcitationSpectrum.pPoint[0].eType == OpticalSpectrumPointType.eSptRaisingEdge and 2 == flt.m_ExcitationSpectrum.count:
                    wl = (flt.m_ExcitationSpectrum.pPoint[0] + flt.m_ExcitationSpectrum.pPoint[1]) / 2
                    d = 10000.0 if emission < wl else 0.0
                    d += abs(d-emission)
                    if d < dist:
                        closestWL, dist = wl, d
                else:
                    for pt in flt.m_ExcitationSpectrum.pPoint:
                        if pt.eType == OpticalSpectrumPointType.eSptPeak:
                            d = 10000.0 if emission < pt.dWavelength else 0.0
                            d += abs(d-emission)
                            if d < dist:
                                closestWL, dist = pt.dWavelength, d
        return closestWL

@dataclass(init=False, frozen=True)
class PicturePlaneDesc:
    uiCompCount: int = 1
    uiSampleIndex: int = 0                  # Specifies a sample relation of this instance. SLxPicturePlaneDesc instances are
                                            #   grouped by this index. Index also determines the sample settings for this instance
                                            #   (see SLxSampleSetting).
    dObjCalibration1to1: float = 0          # Calibration and camera chip used for acquisition
    sizeObjFullChip: tuple[int, int] = (0, 0)
    uiModalityMask: PicturePlaneModalityFlags = PicturePlaneModalityFlags.modFluorescence
    pFluorescentProbe: FluorescentProbe = field(default_factory=FluorescentProbe)
                                            # Spectrum of the fluorescence fluorophore used. It can be specified by the user and
                                            #   can be used for any calculations (spectral unmixing etc.)
    pFilterPath: OpticalFilterPath = field(default_factory=OpticalFilterPath)                  
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
    emissionWavelengthNm: float = 0
    excitationWavelengthNm: float = 0
    
    def __init__(   self,
                    *,
                    uiCompCount: int = 1,
                    uiSampleIndex: int = 0,
                    dObjCalibration1to1: float = 0,
                    sizeObjFullChip: tuple[int, int] = (0, 0),
                    uiModalityMask: PicturePlaneModalityFlags = PicturePlaneModalityFlags.modFluorescence,
                    pFluorescentProbe: dict|FluorescentProbe = FluorescentProbe,
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

        if type(pFluorescentProbe) == dict:
            object.__setattr__(self, 'pFluorescentProbe', FluorescentProbe(**pFluorescentProbe))
        elif isinstance(pFluorescentProbe, FluorescentProbe):
            object.__setattr__(self, 'pFluorescentProbe', pFluorescentProbe)
        else:
            object.__setattr__(self, 'pFluorescentProbe', FluorescentProbe())

        if type(pFilterPath) == dict:
            object.__setattr__(self, 'pFilterPath', OpticalFilterPath(**pFilterPath))            
        elif isinstance(pFilterPath, OpticalFilterPath):
            object.__setattr__(self, 'pFilterPath', pFilterPath)
        else:
            object.__setattr__(self, 'pFilterPath', OpticalFilterPath())

        object.__setattr__(self, 'dLampVoltage', dLampVoltage)
        object.__setattr__(self, 'dFadingCorr', dFadingCorr)
        object.__setattr__(self, 'uiColor', uiColor)
        object.__setattr__(self, 'sDescription', sDescription)
        object.__setattr__(self, 'dAcqTime', dAcqTime)
        object.__setattr__(self, 'dPinholeDiameter', dPinholeDiameter)
        object.__setattr__(self, 'iChannelSeriesIndex', iChannelSeriesIndex)
        object.__setattr__(self, 'iCapturedPlaneIndex', iCapturedPlaneIndex)

        if self.pFluorescentProbe.m_EmissionSpectrum.isValid:
            object.__setattr__(self, 'emissionWavelengthNm', self.pFluorescentProbe.m_EmissionSpectrum.singleWavelength())
        if self.pFluorescentProbe.m_ExcitationSpectrum.isValid:
            object.__setattr__(self, 'excitationWavelengthNm', self.pFluorescentProbe.m_ExcitationSpectrum.singleWavelength())
        if self.emissionWavelengthNm is None and self.pFilterPath.isValid:
            object.__setattr__(self, 'emissionWavelengthNm', self.pFilterPath.meanEmissionWavelength())
        if self.excitationWavelengthNm is None and self.emissionWavelengthNm is not None and self.pFilterPath.isValid:
            object.__setattr__(self, 'excitationWavelengthNm', self.pFilterPath.closestExcitationWavelength(self.emissionWavelengthNm))

    @property
    def isBrightfield(self) -> bool:
        return (self.uiModalityMask & PicturePlaneModalityFlags.modBrightfield)
    
    @property
    def isDarkfield(self) -> bool:
        return (self.uiModalityMask & PicturePlaneModalityFlags.modDarkfield)    
    
    @property
    def isFluorescence(self) -> bool:
        return (self.uiModalityMask & PicturePlaneModalityFlags.modFluorescence)
    
    @property
    def isContrast(self) -> bool:
        return (self.uiModalityMask & PicturePlaneModalityFlags.modMaskContrast)
    
    @property
    def modalityList(self) -> list[str]:
        return PicturePlaneModalityFlags.to_str_list(self.uiModalityMask)
    
    @property
    def colorAsTuple(self):
        b = (self.uiColor >> 16) & 0xFF
        g = (self.uiColor >> 8) & 0xFF
        r = self.uiColor & 0xFF
        return (r, g, b)
    
    @property
    def colorAsClampedTuple(self):
        b = ((self.uiColor >> 16) & 0xFF) / 255.0
        g = ((self.uiColor >> 8) & 0xFF) / 255.0
        r = (self.uiColor & 0xFF) / 255.0
        return (r, g, b)    
    
    @property
    def colorAsHtmlString(self):
        b = (self.uiColor >> 16) & 0xFF
        g = (self.uiColor >> 8) & 0xFF
        r = self.uiColor & 0xFF
        return f'#{r:02x}{g:02x}{b:02x}'


@dataclass(frozen=True)
class SampleSettingsOC:
        uiOCTypeKey: int = 0
        sOpticalConfigName: str = ""

@dataclass(frozen=True, kw_only=True)
class ObjectiveSetting:
    wsObjectiveName: str = ""
    wsObjectiveCode: str = ""
    dObjectiveMag: float = 0.0
    dObjectiveNA: float = 0.0
    dRefractIndex: float = 0.0
    bTiltingNosepiece: bool = False
    dHorizontalAngle: float = 0.0
    dVerticalAngle: float = 0.0
    dOpticalAxis: float = 0.0


@dataclass(init=False, frozen=True)
class SampleSettings:
    pCameraSetting: dict = field(default_factory=dict)
    pDeviceSetting: dict = field(default_factory=dict)
    pObjectiveSetting: ObjectiveSetting = field(default_factory=ObjectiveSetting)
    sOpticalConfigs: list[SampleSettingsOC] = field(default_factory=list)
    sSpecSettings: str = ""
    uiModeFQ: int = 0
    baScanArea: bytes = field(default_factory=bytes)
    matCameraToStage: np.ndarray = field(default_factory=lambda: np.eye(2, 2))
    dExposureTime: float = 0.0    
    dScalingToIntensity: float = 0.0    
    dRelayLensZoom: float = 1.0
    dObjectiveToPinholeZoom: float = 1.0

    def __init__(   self, 
                    *,
                    pCameraSetting: dict = {},
                    pDeviceSetting: dict = {},
                    pObjectiveSetting: dict = {},
                    sOpticalConfigs: list = [], 
                    sSpecSettings: str = "",
                    uiModeFQ: int = 0,
                    baScanArea: bytes = b"",
                    matCameraToStage: np.ndarray = np.eye(2, 2),
                    dExposureTime: float = 0.0,
                    dScalingToIntensity: float = 0.0,
                    dRelayLensZoom: float = 1.0,
                    dObjectiveToPinholeZoom: float = 1.0,
                    **kwargs):
        
        object.__setattr__(self, 'pCameraSetting', pCameraSetting)
        object.__setattr__(self, 'pDeviceSetting', pDeviceSetting)

        if type(pObjectiveSetting) == dict:
            object.__setattr__(self, 'pObjectiveSetting', ObjectiveSetting(**pObjectiveSetting))
        elif isinstance(pObjectiveSetting, ObjectiveSetting):
            object.__setattr__(self, 'pObjectiveSetting', pObjectiveSetting)

        sOpticalConfigs_: list = []
        if type(sOpticalConfigs) == dict:
            for _, item in sOpticalConfigs.items():
                sOpticalConfigs_.append(SampleSettingsOC(**item))
        elif type(sOpticalConfigs) == list and all(isinstance(item, SampleSettingsOC) for item in sOpticalConfigs):
            sOpticalConfigs = sOpticalConfigs_

        object.__setattr__(self, 'sOpticalConfigs', sOpticalConfigs_)
        object.__setattr__(self, 'sSpecSettings', sSpecSettings)
        object.__setattr__(self, 'uiModeFQ', uiModeFQ)
        object.__setattr__(self, 'baScanArea', baScanArea)
        object.__setattr__(self, 'matCameraToStage', matCameraToStage)
        object.__setattr__(self, 'dExposureTime', dExposureTime)
        object.__setattr__(self, 'dScalingToIntensity', dScalingToIntensity)
        object.__setattr__(self, 'dRelayLensZoom', dRelayLensZoom)
        object.__setattr__(self, 'dObjectiveToPinholeZoom', dObjectiveToPinholeZoom)

    @property
    def cameraName(self) -> str:
        return self.pCameraSetting.get("CameraUserName", "")

    @property
    def microscopeName(self) -> str:
        return self.pDeviceSetting.get("m_sMicroscopeFullName", "")
    
    @property
    def objectiveName(self) -> str:
        return self.pObjectiveSetting.wsObjectiveName

    @property
    def objectiveCode(self) -> str:
        return self.pObjectiveSetting.wsObjectiveCode
    
    @property
    def objectiveMagnification(self) -> float:
        return self.pObjectiveSetting.dObjectiveMag

    @property
    def objectiveNumericAperture(self) -> float:
        return self.pObjectiveSetting.dObjectiveNA
    
    @property
    def refractiveIndex(self) -> float:
        return self.pObjectiveSetting.dRefractIndex
    
    @property
    def opticalConfigurations(self) -> list[str]:
        return [item.sOpticalConfigName for item in self.sOpticalConfigs]


@dataclass(init=False, frozen=True)
class PictureMetadataPicturePlanes:
    uiCount: int = 0                         # == len(sPlane)
    uiCompCount: int = 0                     # the sum of uiCompCount of all sPlane members
    sPlane: list[PicturePlaneDesc] = field(default_factory=list)
    sSampleSetting: list[SampleSettings] = field(default_factory=list)
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
                    sPlane: list[PicturePlaneDesc]|dict|None = None, 
                    sPlaneNew: dict[str, dict]|None = None, 
                    sSampleSetting: list[dict] = [],
                    sDescription: str = "",
                    eRepresentation: PictureMetadataPicturePlanesRepresentation = PictureMetadataPicturePlanesRepresentation.eRepDefault,
                    iExperimentSettingsCount: int = 0,
                    sExperimentSetting: list[dict] = [],
                    iStimulationSettingsCount: int = 0,
                    sStimulationSetting: list[dict] = [],
                    **kwargs):
        object.__setattr__(self, 'uiCount', uiCount)
        object.__setattr__(self, 'uiCompCount', uiCompCount)
        sPlane_ = []
        if sPlaneNew is not None:
            for _, item in sPlaneNew.items():
                if type(item) == dict:
                    sPlane_.append(PicturePlaneDesc(**item))
                else:
                    raise TypeError()
        elif type(sPlane) == dict:
            for _, item in sPlane.items():
                if type(item) == dict:
                    sPlane_.append(PicturePlaneDesc(**item))
                else:
                    raise TypeError()
        elif type(sPlane) == list:
            for item in sPlane:
                if type(item) == dict:
                    sPlane_.append(PicturePlaneDesc(**item))
                elif isinstance(item, PicturePlaneDesc):
                    sPlane_.append(item)
                else:
                    raise TypeError()
                
        object.__setattr__(self, 'sPlane', sPlane_)

        sSampleSetting_ = []
        if type(sSampleSetting) == dict:
            for _, item in sSampleSetting.items():
                sSampleSetting_.append(SampleSettings(**item))
        elif type(sSampleSetting) == list and all(isinstance(item, SampleSettings) for item in sSampleSetting):
            sSampleSetting_ = sSampleSetting

        object.__setattr__(self, 'sSampleSetting', sSampleSetting_)
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
        object.__setattr__(self, 'sPlane', [ PicturePlaneDesc(**args) ])


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
    def channels(self) -> list[PicturePlaneDesc]:
        return self.sPicturePlanes.sPlane

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

    def sampleSettings(self, plane: int = 0) -> SampleSettings|None:
        try:
            return self.sPicturePlanes.sSampleSetting[self.sPicturePlanes.sPlane[plane].uiSampleIndex]
        except (AttributeError, IndexError) as _:
            return None
    
    def cameraName(self, plane: int = 0) -> str:
        try:
            return self.sampleSettings(plane).cameraName
        except (AttributeError, IndexError) as _:
            return ""
    
    def microscopeName(self, plane: int = 0) -> str:
        try:
            return self.sampleSettings(plane).microscopeName
        except (AttributeError, IndexError):
            return ""
        
    def objectiveName(self, plane: int = 0) -> str:
        try:
            return self.sampleSettings(plane).objectiveName
        except (AttributeError, IndexError):
            return ""        

    def opticalConfigurations(self, plane: int = 0) -> list[str]:
        try:        
            return self.sampleSettings(plane).opticalConfigurations
        except (AttributeError, IndexError):
            return []        
      
    def to_lv(self) -> bytes:
        raise NotImplementedError()
    
    @staticmethod
    def from_lv(data: bytes|memoryview) -> PictureMetadata:
        decoded = decode_lv(data)
        return PictureMetadata(**decoded.get('SLxPictureMetadata', {}))
    
    @staticmethod
    def from_var(data: bytes|memoryview) -> PictureMetadata:
        decoded = decode_var(data)
        return PictureMetadata(**decoded[0])