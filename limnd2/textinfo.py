from __future__ import annotations

from dataclasses import dataclass
from .litevariant import decode_lv
from .variant import decode_var


@dataclass(init=False, frozen=True)
class ImageTextInfo:
    sImageID: str = ""
    sType: str = ""
    sGroup: str = ""
    sSampleID: str = ""
    sAuthor: str = ""
    sDescription: str = ""
    sCapturing: str = ""
    sSampling: str = ""
    sLocation: str = ""
    sDate: str = ""
    sConclusion: str = ""
    sInfo1: str = ""
    sInfo2: str = ""
    sOptics: str = ""

    def __init__(   self,
                    TextInfoItem_0: str = "",
                    TextInfoItem_1: str = "",
                    TextInfoItem_2: str = "",
                    TextInfoItem_3: str = "",
                    TextInfoItem_4: str = "",
                    TextInfoItem_5: str = "",
                    TextInfoItem_6: str = "",
                    TextInfoItem_7: str = "",
                    TextInfoItem_8: str = "",
                    TextInfoItem_9: str = "",
                    TextInfoItem_10: str = "",
                    TextInfoItem_11: str = "",
                    TextInfoItem_12: str = "", 
                    TextInfoItem_13: str = ""):
        object.__setattr__(self, 'sImageID', TextInfoItem_0)
        object.__setattr__(self, 'sType', TextInfoItem_1)
        object.__setattr__(self, 'sGroup', TextInfoItem_2)
        object.__setattr__(self, 'sSampleID', TextInfoItem_3)
        object.__setattr__(self, 'sAuthor', TextInfoItem_4)
        object.__setattr__(self, 'sDescription', TextInfoItem_5)
        object.__setattr__(self, 'sCapturing', TextInfoItem_6)
        object.__setattr__(self, 'sSampling', TextInfoItem_7)
        object.__setattr__(self, 'sLocation', TextInfoItem_8)
        object.__setattr__(self, 'sDate', TextInfoItem_9)
        object.__setattr__(self, 'sConclusion', TextInfoItem_10)
        object.__setattr__(self, 'sInfo1', TextInfoItem_11)
        object.__setattr__(self, 'sInfo2', TextInfoItem_12)
        object.__setattr__(self, 'sOptics', TextInfoItem_13)
    
    def to_dict(self) -> dict[str, str]:
        return dict(
            sImageID=self.sImageID,
            sType=self.sType,
            sGroup=self.sGroup,
            sSampleID=self.sSampleID,
            sAuthor=self.sAuthor,
            sDescription=self.sDescription,
            sCapturing=self.sCapturing,
            sSampling=self.sSampling,
            sLocation=self.sLocation,
            sDate=self.sDate,
            sConclusion=self.sConclusion,
            sInfo1=self.sInfo1,
            sInfo2=self.sInfo2, 
            sOptics=self.sOptics)

    @staticmethod
    def from_lv(data: bytes|memoryview) -> ImageTextInfo:
        return ImageTextInfo(**(decode_lv(data).get('SLxImageTextInfo', {})))
    
    @staticmethod
    def from_var(data: bytes|memoryview) -> ImageTextInfo:
        decoded = decode_var(data)
        return ImageTextInfo(**decoded.get('SLxImageTextInfo', {}))
    

@dataclass(frozen=True, kw_only=True)
class AppInfo:
    m_SWNameString: str = ""
    m_GrabberString: str = ""
    m_VersionString: str = ""
    m_CopyrightString: str = ""
    m_CompanyString: str = ""
    m_NFRString: str = ""

    @property
    def software(self) -> str:
        return f"{self.m_SWNameString} {self.m_VersionString}"

    def from_var(data: bytes|memoryview) -> AppInfo:
        decoded = decode_var(data)
        return AppInfo(**decoded[0])
