from __future__ import annotations

import collections, enum, numpy as np
from dataclasses import dataclass

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
        order = ['$INDEX', '$ACQTIME', 'X', 'Y', 'Z', 'Z1', 'Z2', 'PFS_OFFSET', 'PFS_STATUS']
        for id in reversed(order):
            if 0 < (index := self.findById(id)):
                self.data.insert(0, self.data.pop(index))