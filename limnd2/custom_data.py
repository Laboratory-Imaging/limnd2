from __future__ import annotations

import collections, enum, numpy as np, datetime
from dataclasses import dataclass, field
from .lite_variant import decode_lv
from .treeview_helper import get_format_fn

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
    Data: np.ndarray|None = None

    @staticmethod
    def from_desc_and_data(desc: dict, data: bytes) -> RecordedDataItem:
        size = desc.get("Size", 0)
        type = RecordedDataType(desc.get("Type", 0))
        if type == RecordedDataType.eString:
            strings = []
            item_size = 2*256
            for i in range(0, item_size*size, item_size):
                strings.append(data[i:i+item_size].decode("utf-16"))
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
        
    @property
    def data(self) -> np.ndarray:
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
    
    def sort(self) -> None:
        order = ['INDEX', 'ACQTIME', 'X', 'Y', 'Z', 'Z1', 'Z2', 'PFS_OFFSET', 'PFS_STATUS']
        for id in reversed(order):
            if 0 < (index := self.findById(id)):
                self.data.insert(0, self.data.pop(index))

    def to_table(self) -> dict[str, any]:
        coldefs = []
        coldefs.append(dict(id='id', hidden=True))
        rowdata = [ dict(id=i+1) for i in range(self.rowCount) ]
        for col in self.data:
            coldefs.append(dict(id=col.ID, title=f"{col.Desc} [{col.Unit}]" if col.Unit else col.Desc, fmtfncode=_get_recorded_data_fmt_function(col), style=_get_recorded_data_styles(col)))
            for index, datavalue in enumerate(col.data):
                rowdata[index][col.ID] = datavalue
        coldefs.append(dict(id='tail'))
        return dict(coldefs=coldefs, rowdata=rowdata)                


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
            kwargs = kwargs.get("CLxText")
            CLxItem = kwargs.get("CLxItem")
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
            object.__setattr__(self, 'format', kwargs.get('eFormat', 0)) # 0 - float, 1 - scientic
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
    def valueAsText(self) -> str:
        if self.type == CustomDescriptionItemType.Check:
            return "ON" if self.checked else "OFF"
        elif self.type == CustomDescriptionItemType.Number:
            u =  f' {self.unit}'.strip()
            return f'{self.value:.{self.digits}e}{u}' if self.format else f'{self.value:.{self.digits}f}{u}'
        elif self.type == CustomDescriptionItemType.Text:
            return self.text
        elif self.type == CustomDescriptionItemType.Selection:
            return self.labels[self.selected] if type(self.labels) == list and 0 <= self.selected and self.selected < len(self.labels) else ""
        elif self.type == CustomDescriptionItemType.LongText:
            return self.text
        elif self.type == CustomDescriptionItemType.Date:
            return self.date.strftime('%x %X') if self.format in (0, 1) else self.date.strftime('%x')

class CustomDescription(collections.UserList):
    def __init__(self, content: dict):
        super().__init__(CustomDescriptionItem(**item) for item in content.get('vData', {}).values())
        self.name: str = content.get('sName', "")

    @staticmethod
    def from_lv(data: bytes|memoryview) -> CustomDescription:
        decoded = decode_lv(data)
        return CustomDescription(decoded.get('CLxCustomDescription', {}))



def _get_recorded_data_fmt_function(col: RecordedDataItem) -> str:
    if col.Type == RecordedDataType.eDouble:
        digits = 2 if col.ID in ('X', 'Y') else 3
        return get_format_fn(digits)
    else:
        return "(coldef) => { coldef.fmtfn = String };"
    
def _get_recorded_data_styles(col: RecordedDataItem) -> dict[str, str]:
    if col.Type in (RecordedDataType.eDouble, RecordedDataType.eInt) or col.ID == "ACQTIME":
        return { "text-align": "right" }
    else:
        return { "text-align": "left" }    
