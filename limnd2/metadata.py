from __future__ import annotations

import datetime, enum, numpy as np, operator
import typing
from functools import cached_property
from typing import Any
from dataclasses import MISSING, dataclass, field, fields

from .lite_variant import decode_lv, ELxLiteVariantType as LVType, LV_field, LVSerializable, encode_lv
from .variant import decode_var
from .treeview_helper import create_treeview_grouping

def jdn_now():
   result = datetime.datetime.now(datetime.UTC).timestamp()
   result /= 86_400         # seconds per day
   result += 2_440_587.5    # JDN EPOCH
   return result

def calculateColor(color_string: str) -> int:
        """
        Calculates channel color integer (used as uiColor).

        Parameters
        ----------
        color_string : str
            Color to be converted, either as hex string ("#ff0000"), common colors are also supported ("Red").

        Returns
        -------
        int
            converted color value
        """
        COLOR_NAME_TO_HTML = {
            "Red": "#ff0000",
            "Green": "#00ff00",
            "Blue": "#0000ff",
            "Yellow": "#ffff00",
            "Cyan": "#00ffff",
            "Magenta": "#ff00ff",
            "Black": "#000000",
            "White": "#ffffff",
            "Gray": "#808080",
            "Orange": "#ffa500",
            "Pink": "#ffc0cb",
            "Purple": "#800080",
            "Brown": "#a52a2a",
        }

        hex_color = COLOR_NAME_TO_HTML.get(color_string.capitalize(), color_string)

        if hex_color.startswith('#'):
            hex_color = hex_color[1:]

        if len(hex_color) != 6 or not all(char in "0123456789abcdefABCDEF" for char in hex_color):
            raise ValueError(f"Invalid HTML color string: '{color_string}'")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return (b << 16) | (g << 8) | r

class PictureMetadataTimeSourceType(enum.IntEnum):
    etsSW: typing.Final    = 0
    etsNIDAQ: typing.Final = 1

class PictureMetadataAxisDescription(enum.IntEnum):
    eaxdX: typing.Final         = 0
    eaxdY: typing.Final         = 1
    eaxdT: typing.Final         = 2
    eaxdZ: typing.Final         = 3
    eaxdPoint : typing.Final    = 4 # confocal point scan has both ePictureXAxis and ePictureYAxis set to this value

class PictureMetadataPicturePlanesRepresentation(enum.IntEnum):
    eRepDefault: typing.Final   = 0
    eRepHDR: typing.Final       = 2

class OpticalFilterPlacement(enum.IntEnum):
    eOfpNoFilter: typing.Final          = 0
    eOfpExcitation: typing.Final        = 1         # excitation filter (position by a lamp)
    eOfpEmission: typing.Final          = 2         # emission filter (position by a camera)
    eOfpFilterTurret: typing.Final      = 3         # filter block (mainly fluorescence)
    eOfpLamp: typing.Final              = 4         # spectrum of a lamp
    eOfnCameraChip: typing.Final        = 5         # camera chip sensitivity
    eOfpUserOverride: typing.Final      = 6         # user defined emission wavelength

class OpticalFilterNature(enum.IntFlag):
    eOfnGeneric: typing.Final           = 0x0000    # wide-band or unspecified spectra
    eOfnRGB: typing.Final               = 0x0001    # triple-band filter suitable for RGB cameras (or naked eye)
    eOfnRed: typing.Final 	            = 0x0002
    eOfnGreen: typing.Final             = 0x0004
    eOfnBlue: typing.Final              = 0x0008

class OpticalFilterSpectType(enum.IntEnum):
    eOftBandpass: typing.Final          = 1         # specified by lower (raising edge) and higher (falling) wavelength
    eOftNarrowBandpass: typing.Final    = 2         # specified by one wavelength (peak)
    eOftLowpass: typing.Final           = 3         # specified by one wavelength (falling edge)
    eOftHighpass: typing.Final          = 4         # specified by one wavelength (raising edge)
    eOftBarrier: typing.Final           = 5         # specified by lower (falling edge) and higher (raising) wavelength
    eOftMultiplepass: typing.Final      = 6         # specified by a few edges
    eOftFull: typing.Final              = 7         # full position of a filterwheel
    eOftEmpty: typing.Final             = 8         # empty position of a filterwheel

class PicturePlaneModality(enum.IntEnum):
    """
    Enum for modality of given plane.
    !!! warning
        In modern .nd2 files this modality enum should be converted to [PicturePlaneModalityFlags](metadata.md#limnd2.metadata.PicturePlaneModalityFlags) instance using [from_modality()](metadata.md#limnd2.metadata.PicturePlaneModalityFlags.from_modality) function.
    """
    eModWidefieldFluo: typing.Final       = 0
    eModBrightfield: typing.Final         = 1
    eModLaserScanConfocal: typing.Final   = 2
    eModSpinDiskConfocal: typing.Final    = 3
    eModSweptFieldConfocal: typing.Final  = 4
    eModMultiPhotonFluo: typing.Final     = 5
    eModPhaseContrast: typing.Final       = 6
    eModDIContrast: typing.Final          = 7
    eModSpectralConfocal: typing.Final    = 8
    eModVAASConfocal: typing.Final        = 9
    eModVAASConfocalIF: typing.Final      = 10
    eModVAASConfocalNF: typing.Final      = 11
    eModDSDConfocal: typing.Final         = 12

class PicturePlaneModalityFlags(enum.IntFlag):
    modFluorescence: typing.Final                     = 0x0000000000000001
    modBrightfield: typing.Final                      = 0x0000000000000002
    modDarkfield: typing.Final                        = 0x0000000000000004
    modMaskLight: typing.Final                        = (modFluorescence | modBrightfield | modDarkfield)
    modPhaseContrast: typing.Final                    = 0x0000000000000010
    modDIContrast: typing.Final                       = 0x0000000000000020
    modNAMC: typing.Final                             = 0x0000000000000008
    modMaskContrast: typing.Final                     = (modPhaseContrast | modDIContrast | modNAMC)
    modCamera: typing.Final                           = 0x0000000000000100
    modLaserScanConfocal: typing.Final                = 0x0000000000000200
    modSpinDiskConfocal: typing.Final                 = 0x0000000000000400
    modSweptFieldConfocalSlit: typing.Final           = 0x0000000000000800
    modSweptFieldConfocalPinholes: typing.Final       = 0x0000000000001000
    modDSDConfocal: typing.Final                      = 0x0000000000002000
    modSIM: typing.Final                              = 0x0000000000004000
    modISIM: typing.Final                             = 0x0000000000008000
    modRCM: typing.Final                              = 0x0000000000000040
    modSora: typing.Final                             = 0x0000000040000000
    modLiveSR: typing.Final                           = 0x0000000000040000
    modLightSheet: typing.Final                       = 0x0000000000080000
    modDeepSIM: typing.Final                          = 0x0000002000000000
    modMaskAcqHWType: typing.Final                    = (modCamera | modLaserScanConfocal | modSpinDiskConfocal | modSweptFieldConfocalSlit | modSweptFieldConfocalPinholes | modDSDConfocal | modRCM | modDeepSIM | modISIM | modSora | modLiveSR | modLightSheet)
    modMultiPhotonFluo: typing.Final                  = 0x0000000000010000
    modTIRF: typing.Final                             = 0x0000000000020000
    modPMT: typing.Final                              = 0x0000000000100000
    modSpectral: typing.Final                         = 0x0000000000200000
    modVAAS_IF: typing.Final                          = 0x0000000000400000
    modVAAS_NF: typing.Final                          = 0x0000000000800000
    modTransmitDetector: typing.Final                 = 0x0000000001000000
    modNonDescannedDetector: typing.Final             = 0x0000000002000000
    modVirtualFilter: typing.Final                    = 0x0000000004000000
    modGaAsP: typing.Final                            = 0x0000000008000000
    modRemainder: typing.Final                        = 0x0000000010000000
    modAUX: typing.Final                              = 0x0000000020000000
    modCustomDescChannel: typing.Final                = 0x0000000080000000
    modSTED: typing.Final                             = 0x0000000100000000
    modGalvano: typing.Final                          = 0x0000000200000000
    modResonant: typing.Final                         = 0x0000000400000000
    modAX: typing.Final                               = 0x0000000800000000
    modStorm: typing.Final                            = 0x0000001000000000
    modNSPARCDetector: typing.Final                   = 0x0000004000000000
    modPMT_IRGaAsP: typing.Final                      = 0x0000008000000000
    modPMT_GaAs: typing.Final                         = 0x0000010000000000
    modMaskDetector: typing.Final                     = (modSpectral | modVAAS_IF | modVAAS_NF | modTransmitDetector | modNonDescannedDetector | modVirtualFilter | modAUX | modNSPARCDetector)

    @staticmethod
    def from_modality(mod: PicturePlaneModality) -> PicturePlaneModalityFlags:
        """
        Converts modality enum to PicturePlaneModalityFlags.

        Parameters
        ----------
        mod : PicturePlaneModality
            modality enum instance

        Returns
        -------
        PicturePlaneModalityFlags
            Modalify flag for given modality
        """
        return {
            PicturePlaneModality.eModWidefieldFluo:      PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modCamera,
            PicturePlaneModality.eModBrightfield:        PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modCamera,
            PicturePlaneModality.eModLaserScanConfocal:  PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal,
            PicturePlaneModality.eModSpinDiskConfocal:   PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSpinDiskConfocal,
            PicturePlaneModality.eModSweptFieldConfocal: PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSweptFieldConfocalSlit,
            PicturePlaneModality.eModMultiPhotonFluo:    PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modMultiPhotonFluo|PicturePlaneModalityFlags.modLaserScanConfocal,
            PicturePlaneModality.eModPhaseContrast:      PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modPhaseContrast,
            PicturePlaneModality.eModDIContrast:         PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modDIContrast,
            PicturePlaneModality.eModSpectralConfocal:   PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal|PicturePlaneModalityFlags.modSpectral,
            PicturePlaneModality.eModVAASConfocal:       PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal,
            PicturePlaneModality.eModVAASConfocalIF:     PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal|PicturePlaneModalityFlags.modVAAS_IF,
            PicturePlaneModality.eModVAASConfocalNF:     PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal|PicturePlaneModalityFlags.modVAAS_NF,
            PicturePlaneModality.eModDSDConfocal:        PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modDSDConfocal
        }.get(mod, PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modCamera)

    @staticmethod
    def modality_string_map() -> dict[str, PicturePlaneModalityFlags]:
        """
        Returns mapping of known modality strings ("Wide-field", "Brightfield", ...) to PicturePlaneModalityFlags.
        """
        return {
            "Undefined": 0,
            "Wide-field":           PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modCamera,
            "Brightfield":          PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modCamera,
            "Phase":                PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modCamera | PicturePlaneModalityFlags.modPhaseContrast,
            "DIC":                  PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modCamera | PicturePlaneModalityFlags.modDIContrast,
            "DarkField":            PicturePlaneModalityFlags.modDarkfield    | PicturePlaneModalityFlags.modCamera,
            "TIRF":                 PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modCamera | PicturePlaneModalityFlags.modTIRF,
            "Confocal, Fluo":       PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal,
            "Confocal, Trans":      PicturePlaneModalityFlags.modBrightfield  | PicturePlaneModalityFlags.modLaserScanConfocal | PicturePlaneModalityFlags.modTransmitDetector,
            "Multi-photon":         PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal | PicturePlaneModalityFlags.modMultiPhotonFluo,
            "SFC pinhole":          PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSweptFieldConfocalPinholes,
            "SFC slit":             PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSweptFieldConfocalSlit,
            "Spinning Disc":        PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSpinDiskConfocal,
            "DSD":                  PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modDSDConfocal,
            "NSIM":                 PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSIM,
            "iSim":                 PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modISIM,
            "RCM":                  PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modRCM,
            "CSU W1-SoRa":          PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modSora,
            "NSPARC":               PicturePlaneModalityFlags.modFluorescence | PicturePlaneModalityFlags.modLaserScanConfocal | PicturePlaneModalityFlags.modNSPARCDetector,
        }

    @staticmethod
    def modality_list() -> list[str]:
        """
        Returns list of known modality strings ("Wide-field", "Brightfield", ...).
        """
        return list(PicturePlaneModalityFlags.modality_string_map().keys())


    @staticmethod
    def from_modality_string(modality: str) -> PicturePlaneModalityFlags:
        """
        Converts modality string to PicturePlaneModalityFlags.

        Parameters
        ----------
        modality : string
            modality string (for example "Wide-field", "Brightfield", "Phase", ...)

        Returns
        -------
        PicturePlaneModalityFlags
            Modalify flag for given modality, 0 for "undefined"
        """

        modality_map_parsed = {key.lower().replace("-", "").replace(" ", "").replace(",", "") : val for key, val in PicturePlaneModalityFlags.modality_string_map().items()}
        modality_parsed = modality.lower().replace("-", "").replace(" ", "").replace(",", "")

        if modality_parsed in ("undefined", "unknown"):
            return 0
        if modality_parsed in modality_map_parsed:
            return modality_map_parsed[modality_parsed]
        raise ValueError(f"Non-recognized modality string: {modality}")

    @staticmethod
    def to_str_list(flags : PicturePlaneModalityFlags) -> list[str]:
        """
        Converts modality flags to list of human readable strings.

        Parameters
        ----------
        flags : PicturePlaneModalityFlags
            odality flags

        Returns
        -------
        list[str]
            human readable string list, for example ["Brightfield", "Phase"]
        """
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
    eSptInvalid: typing.Final         = 0
    eSptPoint: typing.Final           = 1
    eSptRaisingEdge: typing.Final     = 2
    eSptFallingEdge: typing.Final     = 3
    eSptPeak: typing.Final            = 4
    eSptRange: typing.Final           = 5


@dataclass(frozen=True, kw_only=True, init=False)
class OpticalSpectrumPoint(LVSerializable):
    eType: OpticalSpectrumPointType             = LV_field(OpticalSpectrumPointType.eSptInvalid,  LVType.UINT32)
    dWavelength: float                          = LV_field(0.0,                                   LVType.DOUBLE)
    dTValue: float                              = LV_field(1.0,                                   LVType.DOUBLE)

    def __post_init__(self):
        object.__setattr__(self, "eType", OpticalSpectrumPointType(self.eType))
        if "uiWavelength" in self._unknown_fields:
            object.__setattr__(self, "dWavelength", self._unknown_fields.pop("uiWavelength"))

@dataclass(frozen=True, kw_only=True)
class OpticalSpectrum(LVSerializable):
    uiCount: int                                = LV_field(0,           LVType.UINT32)
    bPoints: bool                               = LV_field(False,       LVType.BOOL)
    pPoint: list[OpticalSpectrumPoint]          = LV_field(list,        LVType.LEVEL)

    def __post_init__(self):
        if isinstance(self.pPoint, dict):
            object.__setattr__(self, "pPoint", [OpticalSpectrumPoint(**self.pPoint[k]) for k in sorted(self.pPoint)])
        if isinstance(self.pPoint, list):
            object.__setattr__(self, "pPoint", [OpticalSpectrumPoint(**p) for p in self.pPoint])

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
            peak = (self.pPoint[ifirst].dWavelength + self.pPoint[ilast].dWavelength) / 2
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
                   return (pt.dWavelength + self.pPoint[index + 1].dWavelength) / 2, pt.dWavelength, self.pPoint[index + 1].dWavelength
                elif pt.eType == OpticalSpectrumPointType.eSptPeak:
                    return pt.dWavelength, pt.dWavelength, pt.dWavelength
            return 0, 0, 0      # more error handling?

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


@dataclass(frozen=True, kw_only=True)
class FluorescentProbe(LVSerializable):
    m_sName: str                            = LV_field("",                  LVType.STRING)
    m_uiColor: int                          = LV_field(0xFFFFFF,            LVType.UINT32)
    m_ExcitationSpectrum: OpticalSpectrum   = LV_field(OpticalSpectrum,     LVType.LEVEL)
    m_EmissionSpectrum: OpticalSpectrum     = LV_field(OpticalSpectrum,     LVType.LEVEL)

    def __post_init__(self):
        object.__setattr__(self, 'm_ExcitationSpectrum', OpticalSpectrum(**self.m_ExcitationSpectrum))
        object.__setattr__(self, 'm_EmissionSpectrum', OpticalSpectrum(**self.m_EmissionSpectrum))

    """
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
    """

@dataclass(frozen=True, kw_only=True, init=False)
class OpticalFilter(LVSerializable):
    m_sName: str                            = LV_field("",                                      LVType.STRING)
    m_sUserName: str                        = LV_field("",                                      LVType.STRING)
    m_ePlacement: OpticalFilterPlacement    = LV_field(OpticalFilterPlacement.eOfpNoFilter,     LVType.UINT32)
    m_eNature: OpticalFilterNature          = LV_field(OpticalFilterNature.eOfnGeneric,         LVType.UINT32)
    m_eSpctType: OpticalFilterSpectType     = LV_field(OpticalFilterSpectType.eOftBandpass,     LVType.UINT32)
    m_uiColor: int                          = LV_field(0xFFFFFF,                                LVType.UINT32)
    m_ExcitationSpectrum: OpticalSpectrum   = LV_field(OpticalSpectrum,                         LVType.LEVEL)
    m_EmissionSpectrum: OpticalSpectrum     = LV_field(OpticalSpectrum,                         LVType.LEVEL)
    m_MirrorSpectrum: OpticalSpectrum       = LV_field(OpticalSpectrum,                         LVType.LEVEL)

    def __post_init__(self):
        object.__setattr__(self, "m_ePlacement", OpticalFilterPlacement(self.m_ePlacement))
        object.__setattr__(self, "m_eNature", OpticalFilterNature(self.m_eNature))
        object.__setattr__(self, "m_eSpctType", OpticalFilterSpectType(self.m_eSpctType))

        object.__setattr__(self, "m_ExcitationSpectrum", OpticalSpectrum(**self.m_ExcitationSpectrum))
        object.__setattr__(self, "m_EmissionSpectrum", OpticalSpectrum(**self.m_EmissionSpectrum))
        object.__setattr__(self, "m_MirrorSpectrum", OpticalSpectrum(**self.m_MirrorSpectrum))

        if "m_wcTiName" in self._unknown_fields:
            self._unknown_fields.pop("m_wcTiName")


@dataclass(frozen=True, kw_only=True)
class OpticalFilterPath(LVSerializable):
    m_sDescr: str                       = LV_field("",          LVType.STRING)
    m_uiCount: int                      = LV_field(0,           LVType.UINT32)
    m_pFilter: list[OpticalFilter]      = LV_field(list,        LVType.LEVEL)

    def __post_init__(self):
        if isinstance(self.m_pFilter, dict):
            filters = [OpticalFilter(**filter) for filter in self.m_pFilter.values()]
            object.__setattr__(self, "m_pFilter", filters)
        elif isinstance(self.m_pFilter, list):
            filters = [OpticalFilter(**filter) for filter in self.m_pFilter]
            object.__setattr__(self, "m_pFilter", filters)


    """
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
    """

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
                    wl = (flt.m_ExcitationSpectrum.pPoint[0].dWavelength + flt.m_ExcitationSpectrum.pPoint[1].dWavelength) / 2
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

@dataclass(frozen=True, init=False, kw_only=True)
class PicturePlaneDesc(LVSerializable):
    uiCompCount: int                                = LV_field(1,                                           LVType.UINT32)
    uiSampleIndex: int                              = LV_field(0,                                           LVType.UINT32)
    # Specifies a sample relation of this instance. SLxPicturePlaneDesc instances are
    # grouped by this index. Index also determines the sample settings for this instance
    # (see SLxSampleSetting).

    dObjCalibration1to1: float                      = LV_field(0.0,                                         LVType.DOUBLE)
    # Calibration and camera chip used for acquisition

    uiModalityMask: PicturePlaneModalityFlags       = LV_field(PicturePlaneModalityFlags.modFluorescence,   LVType.UINT64)
    pFluorescentProbe: FluorescentProbe             = LV_field(FluorescentProbe,                            LVType.LEVEL)
    # Spectrum of the fluorescence fluorophore used. It can be specified by the user and
    # can be used for any calculations (spectral unmixing etc.)

    pFilterPath: OpticalFilterPath                  = LV_field(OpticalFilterPath,                           LVType.LEVEL)
    # Filter path description, comes from devices. It can contain information about all
    #   elements influencing the spectral properties in the optical path. Optionally there
    #   can be lamps (incl. spectra), filters (incl. CCD sensitivity) and shutters.
    #   It must enable a description of an experiment using e.g.: exc. filterwheel, ems. fw,
    #   mirrors, shutter, DIA lamp for DIC

    dLampVoltage: float                             = LV_field(0.0,                                         LVType.DOUBLE)
    dFadingCorr: float                              = LV_field(0.0,                                         LVType.DOUBLE)
    # The coefficient used for fluorescence fading correction.

    uiColor: int                                    = LV_field(0xFF6A02,                                    LVType.UINT32)
    # The colour used for representation of the plane and optionally for look-up table
    #   creation. By default same as excitation filter.

    sDescription: str                               = LV_field("",                                          LVType.STRING)
    # name

    dAcqTime: float                                 = LV_field(0.0,                                         LVType.DOUBLE)
    # acquistion time of one single image plane (value can be different for more planes inside one picture)

    dPinholeDiameter: float                         = LV_field(-1.0,                                        LVType.DOUBLE)
    # pinhole size in um

    iChannelSeriesIndex: int                        = LV_field(-1,                                          LVType.INT32)
    # channel series index

    iCapturedPlaneIndex: int                        = LV_field(-1,                                          LVType.INT32)
    # index of picture plane at the capture time,
    #   used to access correct camerasettings items corresponding to this picture plane

    #emissionWavelengthNm: float                     = LV_field(0.0,                                         LVType.DOUBLE)
    #excitationWavelengthNm: float                   = LV_field(0.0,                                         LVType.DOUBLE)


    # original names "sizeObjFullChip.cx" and "sizeObjFullChip.cy" - will need manual encoding
    sizeObjFullChip_cx: int                         = LV_field(0,                                           LVType.INT32)
    sizeObjFullChip_cy: int                         = LV_field(0,                                           LVType.INT32)

    pass
    """
    Atributes found in XML variant, but not in LV
    sOpticalConfigName: str                         = LV_field("",                                          LVType.STRING)                      #TODO
    sOpticalConfigFull: dict[str, str]              = LV_field(dict,                                        LVType.ENCODING_NOT_IMPLEMENTED)    #TODO
    sCameraSetting: dict[str, Any]                  = LV_field(dict,                                        LVType.ENCODING_NOT_IMPLEMENTED)    #TODO
    """

    def __post_init__(self):
        if "sizeObjFullChip.cx" in self._unknown_fields:
            object.__setattr__(self, "sizeObjFullChip_cx", self._unknown_fields.pop("sizeObjFullChip.cx"))
        if "sizeObjFullChip.cy" in self._unknown_fields:
            object.__setattr__(self, "sizeObjFullChip_cy", self._unknown_fields.pop("sizeObjFullChip.cy"))
        if "eModality" in self._unknown_fields:
            object.__setattr__(self, "uiModalityMask", PicturePlaneModalityFlags.from_modality(self._unknown_fields.pop("eModality")))

        if "sOpticalConfigName" in self._unknown_fields: self._unknown_fields.pop("sOpticalConfigName")
        if "sOpticalConfigFull" in self._unknown_fields: self._unknown_fields.pop("sOpticalConfigFull")
        if "sCameraSetting" in self._unknown_fields: self._unknown_fields.pop("sCameraSetting")

        object.__setattr__(self, "uiModalityMask", PicturePlaneModalityFlags(self.uiModalityMask))
        object.__setattr__(self, "pFluorescentProbe", FluorescentProbe(**self.pFluorescentProbe))
        object.__setattr__(self, "pFilterPath", OpticalFilterPath(**self.pFilterPath))



    def to_serializable_dict(self, parent_path=""):
        """
        Custom serialization for this object - "sizeObjFullChip_cy" has to be renamed to "sizeObjFullChip.cy"
        """
        result = super().to_serializable_dict(parent_path)
        if "sizeObjFullChip_cy" in result:
            result["sizeObjFullChip.cy"] = result.pop("sizeObjFullChip_cy")
        if "sizeObjFullChip_cx" in result:
            result["sizeObjFullChip.cx"] = result.pop("sizeObjFullChip_cx")
        return result



    """
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
    """
    @cached_property
    def emissionWavelengthNm(self) -> float:
        if self.pFluorescentProbe.m_EmissionSpectrum.isValid:
            return self.pFluorescentProbe.m_EmissionSpectrum.singleWavelength()
        elif self.pFilterPath.isValid:
            return self.pFilterPath.meanEmissionWavelength()
        return 0.0

    @cached_property
    def excitationWavelengthNm(self) -> float:
        if self.pFluorescentProbe.m_ExcitationSpectrum.isValid:
            return self.pFluorescentProbe.m_ExcitationSpectrum.singleWavelength()
        elif self.emissionWavelengthNm is not None and self.pFilterPath.isValid:
            return self.pFilterPath.closestExcitationWavelength(self.emissionWavelengthNm)
        return 0.0

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

@dataclass(frozen=True, kw_only=True, init=False)
class CameraSetting(LVSerializable):
    CameraUniqueName: str                   = LV_field("",                  LVType.STRING)
    CameraUserName: str                     = LV_field("",                  LVType.STRING)
    CameraFamilyName: str                   = LV_field("",                  LVType.STRING)
    OverloadedUniqueName: str               = LV_field("",                  LVType.STRING)

    # this class also has a LOT of other fields stored in self._unknown_fields

@dataclass(frozen=True, kw_only=True, init=False)
class DeviceSetting(LVSerializable):
    m_sMicroscopeFullName: str              = LV_field("",                  LVType.STRING)
    m_sMicroscopeShortName: str             = LV_field("",                  LVType.STRING)
    m_sMicroscopePhysFullName: str          = LV_field("",                  LVType.STRING)
    m_sMicroscopePhysShortName: str         = LV_field("",                  LVType.STRING)
    m_vectMicroscope_size: int              = LV_field(0,                   LVType.INT32)
    m_iMicroscopeUse: int                   = LV_field(0,                   LVType.INT32)
    m_ibMicroscopeExist: int                = LV_field(0,                   LVType.INT32)

    # this class also has a LOT of other fields stored in self._unknown_fields

@dataclass(frozen=True, kw_only=True)
class SampleSettingsOC(LVSerializable):
    uiOCTypeKey: int                        = LV_field(0,                   LVType.UINT32)
    sOpticalConfigName: str                 = LV_field("",                  LVType.STRING)

@dataclass(frozen=True, kw_only=True)
class ObjectiveSetting(LVSerializable):
    wsObjectiveName: str                    = LV_field("",                  LVType.STRING)
    wsObjectiveCode: str                    = LV_field("",                  LVType.STRING)
    dObjectiveMag: float                    = LV_field(0.0,                 LVType.DOUBLE)
    dObjectiveNA: float                     = LV_field(0.0,                 LVType.DOUBLE)
    dRefractIndex: float                    = LV_field(0.0,                 LVType.DOUBLE)
    bTiltingNosepiece: bool                 = LV_field(False,               LVType.BOOL)
    dHorizontalAngle: float                 = LV_field(0.0,                 LVType.DOUBLE)
    dVerticalAngle: float                   = LV_field(0.0,                 LVType.DOUBLE)
    dOpticalAxis: float                     = LV_field(0.0,                 LVType.DOUBLE)


@dataclass(frozen=True, kw_only=True, init=False)
class SampleSettings(LVSerializable):
    pCameraSetting: CameraSetting           = LV_field(CameraSetting,       LVType.LEVEL)
    pDeviceSetting: DeviceSetting           = LV_field(DeviceSetting,       LVType.LEVEL)
    pObjectiveSetting: ObjectiveSetting     = LV_field(ObjectiveSetting,    LVType.LEVEL)
    sOpticalConfigs: list[SampleSettingsOC] = LV_field(list,                LVType.LEVEL)
    sSpecSettings: str                      = LV_field("",                  LVType.STRING)
    uiModeFQ: int                           = LV_field(0,                   LVType.UINT32)
    baScanArea: bytes                       = LV_field(bytes,               LVType.BYTEARRAY)
    matCameraToStage: dict                  = LV_field(dict,                LVType.ENCODING_NOT_IMPLEMENTED)
    dExposureTime: float                    = LV_field(0.0,                 LVType.DOUBLE)
    dScalingToIntensity: float              = LV_field(0.0,                 LVType.DOUBLE)
    dRelayLensZoom: float                   = LV_field(1.0,                 LVType.DOUBLE)
    dObjectiveToPinholeZoom: float          = LV_field(1.0,                 LVType.DOUBLE)
    uiOpticalConfigs: type                  = LV_field(0,                   LVType.UINT32)
    dZOffset: type                          = LV_field(0.0,                 LVType.DOUBLE)
    dCalibration1To1: type                  = LV_field(0.0,                 LVType.DOUBLE)
    dCameraCalibrationZoom: type            = LV_field(1.0,                 LVType.DOUBLE)

    pass
    """
    Atributes found in XML variant, but not in LV
    eRepresentation: object                 = LV_field(None,                LVType.ENCODING_NOT_IMPLEMENTED)    # always 0
    sOpticalConfigName: object              = LV_field(None,                LVType.ENCODING_NOT_IMPLEMENTED)    # always empty string
    """

    def __post_init__(self):
        object.__setattr__(self, "pObjectiveSetting", ObjectiveSetting(**self.pObjectiveSetting))
        object.__setattr__(self, "pCameraSetting", CameraSetting(**self.pCameraSetting))
        object.__setattr__(self, "pDeviceSetting", DeviceSetting(**self.pDeviceSetting))

        if isinstance(self.sOpticalConfigs, dict):
            if all(isinstance(d, dict) for d in self.sOpticalConfigs.values()):
                configs = [SampleSettingsOC(**config) for config in self.sOpticalConfigs.values()]
                object.__setattr__(self, "sOpticalConfigs", configs)
            else:
                object.__setattr__(self, "sOpticalConfigs", None)

        if isinstance(self.sOpticalConfigs, list):
            object.__setattr__(self, "sOpticalConfigs", [SampleSettingsOC(**conf) for conf in self.sOpticalConfigs])

        if "eRepresentation" in self._unknown_fields:
            self._unknown_fields.pop("eRepresentation")

        if "sOpticalConfigName" in self._unknown_fields:
            self._unknown_fields.pop("sOpticalConfigName")


    @property
    def cameraName(self) -> str:
        return self.pCameraSetting.CameraUserName

    @property
    def microscopeName(self) -> str:
        return self.pDeviceSetting.m_sMicroscopeFullName

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
        if not self.sOpticalConfigs:
            return []
        return [item.sOpticalConfigName for item in self.sOpticalConfigs]


@dataclass(frozen=True, kw_only=True, init=False)
class PictureMetadataPicturePlanes(LVSerializable):
    uiCount: int                                                = LV_field(0,                                                       LVType.UINT32)                   # == len(sPlane)
    uiCompCount: int                                            = LV_field(0,                                                       LVType.UINT32)    # the sum of uiCompCount of all sPlane members
    sPlaneNew: list[PicturePlaneDesc]                           = LV_field(list,                                                    LVType.LEVEL)
    uiSampleCount: int                                          = LV_field(0,                                                       LVType.UINT32)
    sSampleSetting: list[SampleSettings]                        = LV_field(list,                                                    LVType.LEVEL)
    sDescription: str                                           = LV_field("",                                                      LVType.STRING)
    eRepresentation: PictureMetadataPicturePlanesRepresentation = LV_field(PictureMetadataPicturePlanesRepresentation.eRepDefault,  LVType.UINT32)
    iExperimentSettingsCount: int                               = LV_field(0,                                                       LVType.INT32)
    sExperimentSetting: dict                                    = LV_field(dict,                                                    LVType.ENCODING_NOT_IMPLEMENTED)
    iStimulationSettingsCount: int                              = LV_field(0,                                                       LVType.INT32)
    sStimulationSetting: dict                                   = LV_field(dict,                                                    LVType.ENCODING_NOT_IMPLEMENTED)

    def __post_init__(self):

        if self.sPlaneNew and isinstance(self.sPlaneNew, dict):
            planes = []
            for key in sorted(self.sPlaneNew):
                planes.append(PicturePlaneDesc(**self.sPlaneNew[key]))
            object.__setattr__(self, "sPlaneNew", planes)

        if "sPlane" in self._unknown_fields and isinstance(self._unknown_fields["sPlane"], dict):
            planes = []
            for key in sorted(self._unknown_fields["sPlane"]):
                planes.append(PicturePlaneDesc(**self._unknown_fields["sPlane"][key]))
            object.__setattr__(self, "sPlaneNew", planes)
            self._unknown_fields.pop("sPlane")

        if self.sPlaneNew and isinstance(self.sPlaneNew, list):
            object.__setattr__(self, "sPlaneNew", [PicturePlaneDesc(**p) for p in self.sPlaneNew])


        if isinstance(self.sSampleSetting, dict):
            ssettings = []
            for setting in self.sSampleSetting.values():
                ssettings.append(SampleSettings(**setting))
            object.__setattr__(self, "sSampleSetting", ssettings)

        elif isinstance(self.sSampleSetting, list):
            object.__setattr__(self, "sSampleSetting", [SampleSettings(**s) for s in self.sSampleSetting])

        object.__setattr__(self, "eRepresentation", PictureMetadataPicturePlanesRepresentation(self.eRepresentation))


    @property
    def valid(self) -> bool:
        """
        Checks if PictureMetadataPicturePlanes insance has valid number of channels.

        Returns
        -------
        bool
            True if number of channels is valid, False otherwise.
        """
        return 0 < self.uiCount and self.uiCount <= self.uiCompCount and self.uiCount == len(self.sPlaneNew)

    def makeValid(self, comps: int, **kwargs) -> None:
        """
        Attempts to fix info about channels using specified number of channels.

        This function creates channel info basec on component count like this:

        ```
        comps == 1: function creates one Mono channel
        comps == 2: function creates channels Channel_1, Channel_2
        comps == 3: function creates one RGB channel
        comps >= 4: function creates channels Channel_1, ..., Channel_N
        ```

        Parameters
        ----------
        comps : int
            The number of components in the image.
        **kwargs : dict
            Additional parameters to pass to each plane.
        """
        if comps in (1, 3):
            args = dict(uiCompCount=comps,
                        uiModalityMask=PicturePlaneModalityFlags.modBrightfield if comps == 3 else PicturePlaneModalityFlags.modFluorescence,
                        sDescription="RGB" if comps == 3 else "Mono")
            args.update(kwargs)
            object.__setattr__(self, 'uiCount', 1)
            object.__setattr__(self, 'uiCompCount', comps)
            object.__setattr__(self, 'uiSampleCount', 1)
            object.__setattr__(self, 'sSampleSetting', [ SampleSettings() ])
            object.__setattr__(self, 'sPlaneNew', [ PicturePlaneDesc(**args) ])
        else:
            planes = []
            for i in range(comps):
                args = dict(uiCompCount = 1,
                            uiModalityMask = PicturePlaneModalityFlags.modFluorescence,
                            sDescription = f"Channel_{i + 1}")
                args.update(kwargs)
                planes.append(PicturePlaneDesc(**args))
            object.__setattr__(self, 'uiCount', comps)
            object.__setattr__(self, 'uiCompCount', comps)
            object.__setattr__(self, 'uiSampleCount', 1)
            object.__setattr__(self, 'sSampleSetting', [ SampleSettings() ])
            object.__setattr__(self, 'sPlaneNew', planes)

    def to_serializable_dict(self, parent_path=""):
        """
        Converts dataclass to Python dictionary encodeable with LV encoder.
        """

        res = super().to_serializable_dict(parent_path)

        # generally items in lists in LVSerializable data structures can be encoded with empty keys
        # channels and sample settings however can not, we will manually change keys
        # from integer key (int keys are replaces with "") to "a{key}"

        for key in sorted(res["sPlaneNew"]):
            res["sPlaneNew"][f"a{key}"] = res["sPlaneNew"].pop(key)

        for key in sorted(res["sSampleSetting"]):
            res["sSampleSetting"][f"a{key}"] = res["sSampleSetting"].pop(key)
        return res


    def to_table(self) -> dict[str, any]:
        """
        Converts picture planes metadata to a treeview table.
        """

        rows=[]
        col_defs=[ dict(id="id", hidden=True), dict(id="camera", title="Camera"), dict(id="channel", title="Channel"), dict(id="feature", title="Feature"), dict(id="value", title="Value") ]
        settings = self.sSampleSetting
        for plane in self.sPlaneNew:
            setting = settings[plane.uiSampleIndex] if 0 <= plane.uiSampleIndex < len(settings) else None
            if setting:
                camera = setting.cameraName or "Unknown camera"
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="OC name:", value=','.join(oc for oc in setting.opticalConfigurations)))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Microscope name:", value=setting.microscopeName))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Objective name:", value=setting.objectiveName))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Objective magnification:", value=setting.objectiveMagnification))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Objective numerical aperture:", value=setting.objectiveNumericAperture))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Refractive index:", value=setting.refractiveIndex))
            else:
                camera = "Unknown camera"
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="OC name:", value='N/A'))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Microscope name:", value='N/A'))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Objective name:", value='N/A'))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Objective magnification:", value='N/A'))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Objective numerical aperture:", value='N/A'))
                rows.append(dict(id=str(len(rows)+1), camera=camera, channel=plane.sDescription, feature="Refractive index:", value='N/A'))
        rows.sort(key=lambda row: row["camera"])
        return dict(coldefs=col_defs, groups=create_treeview_grouping(rows, ['camera', 'channel']), rowdata=rows)


@dataclass(frozen=True, kw_only=True)
class PictureMetadataPhysicalQuantity(LVSerializable):
    wsName: str                                         = LV_field("",                                  LVType.STRING)
    uiIntepretation: int                                = LV_field(0,                                   LVType.ENCODING_NOT_IMPLEMENTED)
    dValue: float                                       = LV_field(0.0,                                 LVType.DOUBLE)

@dataclass(frozen=True, kw_only=True, init=False)
class PictureMetadata(LVSerializable):
    dTimeAbsolute: float                                = LV_field(jdn_now,                             LVType.DOUBLE)
    # time specification when the picture was captured [Julian Day Number]

    dTimeMSec: float                                    = LV_field(0.0,                                 LVType.DOUBLE)
    # time offset of captured frame [ms]

    eTimeSource: PictureMetadataTimeSourceType          = LV_field(PictureMetadataTimeSourceType.etsSW, LVType.INT32)
    dXPos: float                                        = LV_field(0.0,                                 LVType.DOUBLE)
    dYPos: float                                        = LV_field(0.0,                                 LVType.DOUBLE)
    uiRow: int                                          = LV_field(0,                                   LVType.UINT32)
    uiCol: int                                          = LV_field(0,                                   LVType.UINT32)
    dZPos: float                                        = LV_field(0.0,                                 LVType.DOUBLE)
    bZPosAbsolute: bool                                 = LV_field(False,                               LVType.BOOL)
    dAngle: float                                       = LV_field(0.0,                                 LVType.DOUBLE)
    sPicturePlanes: PictureMetadataPicturePlanes        = LV_field(PictureMetadataPicturePlanes,        LVType.LEVEL)
    dTemperK: float                                     = LV_field(293.0,                               LVType.DOUBLE)
    # temperature (in Kelvins)

    dCalibration: float                                 = LV_field(-1.0,                                LVType.DOUBLE)
    # microns to pixel

    dAspect: float                                      = LV_field(-1.0,                                LVType.DOUBLE)
    # pixel aspect ratio

    dCalibPrecision: float                              = LV_field(-1.0,                                LVType.DOUBLE)
    # calibration precision in microns

    bCalibrated: bool                                   = LV_field(False,                               LVType.BOOL)
    # is calibration valid

    wsObjectiveName: str                                = LV_field("",                                  LVType.STRING)
    dObjectiveMag: float                                = LV_field(-1.0,                                LVType.DOUBLE)
    dObjectiveNA: float                                 = LV_field(-1.0,                                LVType.DOUBLE)
    dRefractIndex1: float                               = LV_field(-1.0,                                LVType.DOUBLE)
    dRefractIndex2: float                               = LV_field(-1.0,                                LVType.DOUBLE)
    dZoom: float                                        = LV_field(-1.0,                                LVType.DOUBLE)
    pPhysicalVar: list[PictureMetadataPhysicalQuantity] = LV_field(list,                                LVType.LEVEL)
    uiPhysicalVarCount: int                             = LV_field(0,                                   LVType.UINT32)
    # == len(pPhysicalVar)

    wsCustomData: str                                   = LV_field("",                                  LVType.STRING)
    ePictureXAxis: PictureMetadataAxisDescription       = LV_field(PictureMetadataAxisDescription.eaxdX,LVType.INT32)
    ePictureYAxis: PictureMetadataAxisDescription       = LV_field(PictureMetadataAxisDescription.eaxdY,LVType.INT32)
    dTimeAxisCalibration: float                         = LV_field(-1.0,                                LVType.DOUBLE)
    # valid when there is eaxdT axis, in ms

    dZAxisCalibration: float                            = LV_field(-1.0,                                LVType.DOUBLE)
    # valid when there is eaxdZ axis

    dStgLgCT11: float                                   = LV_field(1.0,                                 LVType.DOUBLE)
    dStgLgCT21: float                                   = LV_field(0.0,                                 LVType.DOUBLE)
    dStgLgCT12: float                                   = LV_field(0.0,                                 LVType.DOUBLE)
    dStgLgCT22: float                                   = LV_field(1.0,                                 LVType.DOUBLE)
    # transformation matrix, more general than dAngle

    baOpticalPathsCorrections: bytes                    = LV_field(b'',                                 LVType.BYTEARRAY)

    pass
    """
    Atributes found in XML variant, but not in LV
    # sCameraSetting: dict[str, Any]                      = LV_field(dict,                                LVType.UNKNOWN)       # only seems to hold default values
    # dProjectiveMag: float                               = LV_field(-1.0,                                LVType.DOUBLE)        # looks unused even in NIS

    # Probably error in NIS? should be uiCol
    # uicon20_L: int                                      = LV_field(0,                                   LVType.UINT32)
    """


    def __post_init__(self):
        object.__setattr__(self, "eTimeSource", PictureMetadataTimeSourceType(self.eTimeSource))
        object.__setattr__(self, "sPicturePlanes", PictureMetadataPicturePlanes(**self.sPicturePlanes))

        if isinstance(self.pPhysicalVar, dict):
            physical = [PictureMetadataPhysicalQuantity(**p) for p in self.pPhysicalVar.values()]
            object.__setattr__(self, "pPhysicalVar", physical)

        object.__setattr__(self, "ePictureXAxis", PictureMetadataAxisDescription(self.ePictureXAxis))
        object.__setattr__(self, "ePictureYAxis", PictureMetadataAxisDescription(self.ePictureYAxis))

        if "dPinholeRadius" in self._unknown_fields:
            radius = self._unknown_fields.pop("dPinholeRadius")
            for plane in self.sPicturePlanes.sPlaneNew:
                object.__setattr__(plane, "dPinholeDiameter", radius)

        if "dProjectiveMag" in self._unknown_fields:
            self._unknown_fields.pop("dProjectiveMag")

        if "uiCon20(L" in self._unknown_fields:
            self._unknown_fields.pop("uiCon20(L")

        if "sCameraSetting" in self._unknown_fields:
            self._unknown_fields.pop("sCameraSetting")


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
        return self.sPicturePlanes.sPlaneNew

    @property
    def channelNames(self) -> list[str]:
        return [ "RGB" if 3 == plane.uiCompCount else plane.sDescription for plane in self.sPicturePlanes.sPlaneNew ]

    @property
    def componentNames(self) -> list[str]:
        ret = []
        for plane in self.sPicturePlanes.sPlaneNew:
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
        for plane in self.sPicturePlanes.sPlaneNew:
            if 3 == plane.uiCompCount:
                ret.append((0, 0, 1))
                ret.append((0, 1, 0))
                ret.append((1, 0, 0))
            else:
                ret.append(color_as_tuple(plane.uiColor))
        return ret

    def sampleSettings(self, plane: int | PicturePlaneDesc = 0) -> SampleSettings|None:
        try:
            if isinstance(plane, int):
                return self.sPicturePlanes.sSampleSetting[self.sPicturePlanes.sPlaneNew[plane].uiSampleIndex]
            if isinstance(plane, PicturePlaneDesc):
                return self.sPicturePlanes.sSampleSetting[plane.uiSampleIndex]
        except (AttributeError, IndexError) as _:
            return None

    def cameraName(self, plane: int | PicturePlaneDesc = 0) -> str:
        try:
            return self.sampleSettings(plane).cameraName
        except (AttributeError, IndexError) as _:
            return ""

    def microscopeName(self, plane: int | PicturePlaneDesc = 0) -> str:
        try:
            return self.sampleSettings(plane).microscopeName
        except (AttributeError, IndexError):
            return ""

    def refractiveIndex(self, plane: int | PicturePlaneDesc = 0) -> float:
        try:
            return self.sampleSettings(plane).refractiveIndex
        except (AttributeError, IndexError):
            return -1.0

    def objectiveName(self, plane: int | PicturePlaneDesc = 0) -> str:
        try:
            return self.sampleSettings(plane).objectiveName
        except (AttributeError, IndexError):
            return ""

    def objectiveMagnification(self, plane: int | PicturePlaneDesc = 0) -> float:
        try:
            return self.sampleSettings(plane).objectiveMagnification
        except (AttributeError, IndexError):
            return -1.0

    def objectiveNumericAperture(self, plane: int | PicturePlaneDesc = 0) -> float:
        try:
            return self.sampleSettings(plane).objectiveNumericAperture
        except (AttributeError, IndexError):
            return -1.0

    def opticalConfigurations(self, plane: int | PicturePlaneDesc = 0) -> list[str]:
        try:
            return self.sampleSettings(plane).opticalConfigurations
        except (AttributeError, IndexError):
            return []

    def to_lv(self) -> bytes:
        return encode_lv({"SLxPictureMetadata" : self.to_serializable_dict()})

    @staticmethod
    def from_lv(data: bytes|memoryview) -> PictureMetadata:
        decoded = decode_lv(data)
        return PictureMetadata(**decoded.get('SLxPictureMetadata', {}))

    @staticmethod
    def from_var(data: bytes|memoryview) -> PictureMetadata:
        decoded = decode_var(data)
        return PictureMetadata(**decoded[0])