from __future__ import annotations

import collections, enum, numpy as np, datetime
from xml.etree.ElementTree import Element, SubElement, tostring
from dataclasses import dataclass, field
from .lite_variant import decode_lv

class RecordedDataType(enum.IntEnum):
    eUnknown = 0
    eString  = 1
    eInt     = 2
    eDouble  = 3

@dataclass(frozen=True, kw_only=True)
class RecordedDataItem:
    ID: str = ""
    Desc: str = ""
    Unit: str = ""
    Type: RecordedDataType = RecordedDataType.eUnknown
    Group: int = 0
    Size: int = 0
    Data: np.ndarray | None = None

    @staticmethod
    def from_desc_and_data(desc: dict, data: bytes) -> RecordedDataItem:
        size = desc.get("Size", 0)
        type = RecordedDataType(desc.get("Type", 0))
        if type == RecordedDataType.eString:
            strings = []
            item_size = 2*256
            for i in range(0, item_size*size, item_size):
                strings.append(data[i:i+item_size].decode("utf-16").rstrip("\x00"))
            return RecordedDataItem(**desc, Data=np.array(strings))
        elif type == RecordedDataType.eInt:
            return RecordedDataItem(**desc, Data=np.ndarray(
                    buffer=data, dtype=np.int32,
                    shape=(size, ),
                    strides=(4, ),
                    ))
        elif type == RecordedDataType.eDouble:
            return RecordedDataItem(**desc, Data=np.ndarray(
                    buffer=data, dtype=np.float64,
                    shape=(size, ),
                    strides=(8, ),
                    ))
        return RecordedDataItem()

    @property
    def data(self) -> np.ndarray:
        if self.Data is None:
            return np.array([])
        return self.Data.astype(object) if self.Type == RecordedDataType.eInt else self.Data

class RecordedData(collections.UserList):
    def __init__(self, iterable = []):
        super().__init__(RecordedDataItem(**item) for item in iterable)

    def findById(self, id: str) -> int:
        for index, item in enumerate(self.data):
            if item.ID == id:
                return index
        return -1

    @property
    def rowCount(self) -> int:
        return max(col.Size for col in self.data)

    def sort(self) -> None:     # pyright: ignore[reportIncompatibleMethodOverride] custom sort
        order = ['INDEX', 'ACQTIME', 'X', 'Y', 'Z', 'Z1', 'Z2', 'PFS_OFFSET', 'PFS_STATUS']
        for id in reversed(order):
            if 0 < (index := self.findById(id)):
                self.data.insert(0, self.data.pop(index))


@dataclass(frozen=True)
class RecordedDataSidecar:
    """Recorded-data columns and acquisition caches decoded from a sidecar."""
    columns: RecordedData
    acq_times: np.ndarray | None = None
    acq_times2: np.ndarray | None = None
    acq_frames: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "columns": [
                {
                    "ID": item.ID,
                    "Desc": item.Desc,
                    "Unit": item.Unit,
                    "Type": int(item.Type),
                    "Group": item.Group,
                    "Size": item.Size,
                    "Data": item.data.tolist(),
                }
                for item in self.columns
                if item.ID not in {"INDEX", "ACQTIME"}
            ],
            "acqTimes": self.acq_times.tolist() if self.acq_times is not None else None,
            "acqTimes2": self.acq_times2.tolist() if self.acq_times2 is not None else None,
            "acqFrames": self.acq_frames.tolist() if self.acq_frames is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecordedDataSidecar":
        columns = RecordedData(
            {
                **item,
                "Type": RecordedDataType(item["Type"]),
                "Data": np.asarray(item.get("Data", []), dtype=object if item["Type"] == int(RecordedDataType.eString) else None),
            }
            for item in data.get("columns", [])
        )
        return cls(
            columns=columns,
            acq_times=np.asarray(data["acqTimes"], dtype=np.float64) if data.get("acqTimes") is not None else None,
            acq_times2=np.asarray(data["acqTimes2"], dtype=np.float64) if data.get("acqTimes2") is not None else None,
            acq_frames=np.asarray(data["acqFrames"], dtype=np.uint32) if data.get("acqFrames") is not None else None,
        )


def encode_recorded_data_descriptor(items: list[RecordedDataItem]) -> bytes:
    """Encode the standard ``CustomDataV2_0`` column-descriptor Variant.

    This intentionally covers only the scalar column types parsed by
    :class:`RecordedDataItem`: UTF-16 strings, int32, and float64.
    """
    root = Element("variant", {"version": "1.0"})
    container = SubElement(root, "CustomTagDescription_v1.0", {"runtype": "CLxListVariant"})
    for index, item in enumerate(items):
        tag = SubElement(container, f"Tag{index}", {"runtype": "CLxListVariant"})
        for name, value, runtype in (
            ("ID", item.ID, "CLxStringW"),
            ("Type", int(item.Type), "lx_int32"),
            ("Group", item.Group, "lx_int32"),
            ("Size", item.Size, "lx_int32"),
            ("Desc", item.Desc, "CLxStringW"),
            ("Unit", item.Unit, "CLxStringW"),
        ):
            SubElement(tag, name, {"runtype": runtype, "value": str(value)})
    return b'<?xml version="1.0" encoding="UTF-8"?>' + tostring(root, encoding="utf-8")


def encode_recorded_data_values(item: RecordedDataItem) -> bytes:
    """Encode one standard recorded-data column payload."""
    values = item.data[:item.Size]
    if len(values) != item.Size:
        raise ValueError(
            f"Recorded-data column {item.ID!r} has {len(values)} values; "
            f"descriptor requires {item.Size}."
        )
    if item.Type == RecordedDataType.eDouble:
        return np.asarray(values, dtype="<f8").tobytes()
    if item.Type == RecordedDataType.eInt:
        return np.asarray(values, dtype="<i4").tobytes()
    if item.Type == RecordedDataType.eString:
        return b"".join(
            (b"\xff\xfe" + str(value).encode("utf-16-le")[:510]).ljust(512, b"\0")
            for value in values
        )
    raise ValueError(f"Unsupported recorded-data type: {item.Type!r}")


class CustomDescriptionItemType(enum.IntEnum):
    Unknown    = 0
    Check      = 1  # CheckBox
    Number     = 2  # EditBox with number
    Text       = 3  # EditBox with string
    Selection  = 4  # DropDown ComboBox
    LongText   = 5  # Multi-line EditBox
    Date       = 6


@dataclass(init=False, frozen=True)
class CustomDescriptionItem:
    type: CustomDescriptionItemType = field(default=CustomDescriptionItemType(0))
    id: int = 0
    name: str = ''
    desc: str = ''
    isEmpty: bool = False
    isDefaultEmpty: bool = False
    isMandatory: bool = False
    isEnabled: bool = False
    checked: int|None = None
    value: float|None = None
    unit: str|None = None
    format: int|None = None
    digits: int|None = None
    text: str|None = None
    selected: int|None = None
    labels: list[str]|None = None
    date: datetime.datetime|None = None

    def __init__(self,
                 CLxItem: dict = {},
                 **kwargs):
        if "CLxText" in kwargs:
            kwargs = kwargs.get("CLxText", {})
            CLxItem = kwargs.get("CLxItem", {})
        if kwargs is None:
            kwargs = {}
        object.__setattr__(self, 'type', CustomDescriptionItemType(CLxItem.get('eType', 0)))
        object.__setattr__(self, 'id', CLxItem.get('iID', 0))
        object.__setattr__(self, 'name', CLxItem.get('sName', ''))
        object.__setattr__(self, 'desc', CLxItem.get('sDescription', ''))
        object.__setattr__(self, 'isEmpty', CLxItem.get('bEmpty', False))
        object.__setattr__(self, 'isDefaultEmpty', CLxItem.get('bEmptyDefault', False))
        object.__setattr__(self, 'isMandatory', CLxItem.get('bMandatory', False))
        object.__setattr__(self, 'isEnabled', CLxItem.get('bEnabled', False))
        if self.type == CustomDescriptionItemType.Check:
            object.__setattr__(self, 'checked', kwargs.get('iCheck', kwargs.get('iDefault', 0)))
        elif self.type == CustomDescriptionItemType.Number:
            object.__setattr__(self, 'value', kwargs.get('dValue', kwargs.get('dDefault', 0.0)))
            object.__setattr__(self, 'unit', kwargs.get('sUnit', ""))
            object.__setattr__(self, 'format', kwargs.get('eFormat', 0)) # 0 - float, 1 - scientific
            object.__setattr__(self, 'digits', kwargs.get('uiPlaces', 3))
        elif self.type == CustomDescriptionItemType.Text:
            object.__setattr__(self, 'text', kwargs.get('sText', kwargs.get('sDefault', "")))
        elif self.type == CustomDescriptionItemType.Selection:
            object.__setattr__(self, 'selected', kwargs.get('iSelection', kwargs.get('iDefault', 0)))
            object.__setattr__(self, 'labels', kwargs.get('vLabels', []))
        elif self.type == CustomDescriptionItemType.LongText:
            object.__setattr__(self, 'text', kwargs.get('sText', kwargs.get('sDefault', "")))
        elif self.type == CustomDescriptionItemType.Date:
            object.__setattr__(self, 'date', datetime.datetime.fromtimestamp(kwargs.get('aDate', kwargs.get('aDefault', 0))//1000))
            object.__setattr__(self, 'format', kwargs.get('eDateFormat', 0)) # 0 - date time sec, 1 - date time, 2 - date only

    @property
    def valueAsText(self) -> str | None:
        if self.type == CustomDescriptionItemType.Check:
            return "ON" if self.checked else "OFF"
        elif self.type == CustomDescriptionItemType.Number:
            u =  f' {self.unit}'.strip()
            return f'{self.value:.{self.digits}e}{u}' if self.format else f'{self.value:.{self.digits}f}{u}'
        elif self.type == CustomDescriptionItemType.Text:
            return self.text
        elif self.type == CustomDescriptionItemType.Selection:
            labels = list(self.labels or [])
            idx = self.selected if isinstance(self.selected, int) else -1
            return labels[idx] if 0 <= idx < len(labels) else ""
        elif self.type == CustomDescriptionItemType.LongText:
            return self.text
        elif self.type == CustomDescriptionItemType.Date:
            date = self.date
            if date is None:
                return None
            return date.strftime('%x %X') if self.format in (0, 1) else date.strftime('%x')

class CustomDescription(collections.UserList):
    def __init__(self, content: dict):
        super().__init__(CustomDescriptionItem(**item) for item in content.get('vData', {}).values())
        self.name: str = content.get('sName', "")

    @staticmethod
    def from_lv(data: bytes|memoryview) -> CustomDescription:
        decoded = decode_lv(data)
        return CustomDescription(decoded.get('CLxCustomDescription', {}))

