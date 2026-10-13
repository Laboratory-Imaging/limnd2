from __future__ import annotations

import json
from dataclasses import dataclass
from .lite_variant import decode_lv, encode_lv, ELxLiteVariantType as LVType
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
            imageId=self.sImageID,
            type=self.sType,
            group=self.sGroup,
            sampleId=self.sSampleID,
            author=self.sAuthor,
            description=self.sDescription,
            capturing=self.sCapturing,
            sampling=self.sSampling,
            location=self.sLocation,
            date=self.sDate,
            conclusion=self.sConclusion,
            info1=self.sInfo1,
            info2=self.sInfo2,
            optics=self.sOptics)

    def toDict(self) -> dict[str, str]:
        """Return the public image-text metadata representation."""
        return self.to_dict()

    @classmethod
    def fromDict(cls, data: dict[str, str]) -> "ImageTextInfo":
        """Create image text information from :meth:`toDict` output."""
        fields = (
            "imageId", "type", "group", "sampleId", "author", "description",
            "capturing", "sampling", "location", "date", "conclusion", "info1",
            "info2", "optics",
        )
        return cls(**{
            f"TextInfoItem_{index}": data.get(name, "")
            for index, name in enumerate(fields)
        })

    def toJson(self, *, indent: int | None = None) -> str:
        """Serialize the image text fields to JSON."""
        return json.dumps(self.toDict(), ensure_ascii=False, indent=indent)

    @classmethod
    def fromJson(cls, data: str) -> "ImageTextInfo":
        """Create image text information from JSON produced by :meth:`toJson`."""
        decoded = json.loads(data)
        if not isinstance(decoded, dict):
            raise ValueError("ImageTextInfo.fromJson() expects a JSON object")

        return cls.fromDict(decoded)

    def to_lv(self) -> bytes:
        """Encode this object as an ND2 ``ImageTextInfoLV`` chunk."""
        return encode_lv({
            "SLxImageTextInfo": {
                f"TextInfoItem_{index}": (value, LVType.STRING)
                for index, value in enumerate((
                    self.sImageID, self.sType, self.sGroup, self.sSampleID,
                    self.sAuthor, self.sDescription, self.sCapturing,
                    self.sSampling, self.sLocation, self.sDate,
                    self.sConclusion, self.sInfo1, self.sInfo2, self.sOptics,
                ))
            }
        })

    @staticmethod
    def from_lv(data: bytes|memoryview) -> ImageTextInfo:
        return ImageTextInfo(**(decode_lv(data).get('SLxImageTextInfo', {})))

    @staticmethod
    def from_var(data: bytes|memoryview) -> ImageTextInfo:
        decoded = decode_var(data)
        return ImageTextInfo(**decoded[0]) # type: ignore


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

    @staticmethod
    def from_var(data: bytes|memoryview) -> AppInfo:
        try:
            decoded = decode_var(data)
            return AppInfo(**decoded[0]) # type: ignore
        except:
            return AppInfo()
