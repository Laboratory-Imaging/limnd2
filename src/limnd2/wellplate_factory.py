"""
Helpers for creating wellplate metadata structures (`WellplateDesc`,
`WellplateFrameInfo`) for writing to ND2 files.
"""

from __future__ import annotations

import re
from typing import Iterable

from .experiment import WellplateDesc, WellplateFrameInfo, WellplateFrameInfoItem

_WELL_RE = re.compile(r"^\s*([A-Za-z]+)\s*(\d+)\s*$")


def _row_label_to_index(label: str) -> int:
    value = 0
    for char in label.upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid row label {label!r}.")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _index_to_row_label(index: int) -> str:
    if index < 0:
        raise ValueError("Row index must be >= 0.")
    value = index + 1
    out = []
    while value > 0:
        value, rem = divmod(value - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


def _parse_well(well: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(well, tuple):
        if len(well) != 2:
            raise ValueError(f"Well tuple must have 2 items, got {well!r}.")
        return int(well[0]), int(well[1])

    match = _WELL_RE.match(well)
    if not match:
        raise ValueError(f"Well {well!r} must be in form like 'A1' or 'B03'.")
    row_label, col = match.groups()
    row_idx = _row_label_to_index(row_label)
    col_idx = int(col) - 1
    return row_idx, col_idx


class WellplateFactory:
    """
    Helper class for creating [`WellplateDesc`] and [`WellplateFrameInfo`].

    The intended use is:
    1. Set plate geometry (`rows`, `columns`, naming).
    2. Add frame-to-well mappings with `addItem`/`addWell`/`addGrid`.
    3. Create structures with `createWellplateDesc`/`createWellplateFrameInfo` (or call the instance).
    """

    def __init__(
        self,
        *,
        name: str = "",
        rows: int = 0,
        columns: int = 0,
        rowNaming: str = "",
        columnNaming: str = "",
        plateIndex: int = 0,
        plateUuid: str = "",
    ) -> None:
        self.name = str(name)
        self.rows = int(rows)
        self.columns = int(columns)
        self.rowNaming = str(rowNaming) if rowNaming else self._default_row_naming(self.rows)
        self.columnNaming = (
            str(columnNaming) if columnNaming else self._default_column_naming(self.columns)
        )
        self.plateIndex = int(plateIndex)
        self.plateUuid = str(plateUuid)

        self._items: list[WellplateFrameInfoItem] = []
        self._well_index_map: dict[tuple[int, int], int] = {}
        self._next_well_index = 0

    @staticmethod
    def _default_row_naming(rows: int) -> str:
        if rows <= 0:
            return ""
        return f"A-{_index_to_row_label(rows - 1)}"

    @staticmethod
    def _default_column_naming(columns: int) -> str:
        if columns <= 0:
            return ""
        return f"1-{columns}"

    @staticmethod
    def _well_name(row_idx: int, col_idx: int) -> str:
        return f"{_index_to_row_label(row_idx)}{col_idx + 1}"

    def setPlate(
        self,
        *,
        name: str | None = None,
        rows: int | None = None,
        columns: int | None = None,
        rowNaming: str | None = None,
        columnNaming: str | None = None,
        plateIndex: int | None = None,
        plateUuid: str | None = None,
    ) -> WellplateFactory:
        if name is not None:
            self.name = str(name)
        if rows is not None:
            self.rows = int(rows)
            if rowNaming is None and not self.rowNaming:
                self.rowNaming = self._default_row_naming(self.rows)
        if columns is not None:
            self.columns = int(columns)
            if columnNaming is None and not self.columnNaming:
                self.columnNaming = self._default_column_naming(self.columns)
        if rowNaming is not None:
            self.rowNaming = str(rowNaming)
        if columnNaming is not None:
            self.columnNaming = str(columnNaming)
        if plateIndex is not None:
            self.plateIndex = int(plateIndex)
        if plateUuid is not None:
            self.plateUuid = str(plateUuid)
        return self

    def _resolve_well_index(self, row_idx: int, col_idx: int, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)
        key = (row_idx, col_idx)
        if key not in self._well_index_map:
            self._well_index_map[key] = self._next_well_index
            self._next_well_index += 1
        return self._well_index_map[key]

    def addItem(
        self,
        *,
        seqIndex: int,
        well: str | tuple[int, int] | None = None,
        wellName: str | None = None,
        wellRowIndex: int | None = None,
        wellColIndex: int | None = None,
        wellIndex: int | None = None,
        plateIndex: int | None = None,
        plateUuid: str | None = None,
    ) -> WellplateFrameInfoItem:
        if well is not None:
            row_idx, col_idx = _parse_well(well)
        else:
            if wellRowIndex is None or wellColIndex is None:
                raise ValueError(
                    "Provide either `well` or both `wellRowIndex` and `wellColIndex`."
                )
            row_idx, col_idx = int(wellRowIndex), int(wellColIndex)

        if row_idx < 0 or col_idx < 0:
            raise ValueError("Well row/column indices must be non-negative.")

        if self.rows > 0 and row_idx >= self.rows:
            raise ValueError(
                f"Well row index {row_idx} is outside configured rows ({self.rows})."
            )
        if self.columns > 0 and col_idx >= self.columns:
            raise ValueError(
                f"Well column index {col_idx} is outside configured columns ({self.columns})."
            )

        item = WellplateFrameInfoItem(
            plateIndex=self.plateIndex if plateIndex is None else int(plateIndex),
            plateUuid=self.plateUuid if plateUuid is None else str(plateUuid),
            seqIndex=int(seqIndex),
            wellIndex=self._resolve_well_index(row_idx, col_idx, wellIndex),
            wellName=str(wellName) if wellName is not None else self._well_name(row_idx, col_idx),
            wellColIndex=col_idx,
            wellRowIndex=row_idx,
        )
        self._items.append(item)
        return item

    def addWell(
        self,
        well: str | tuple[int, int],
        *,
        seqStart: int,
        frameCount: int = 1,
        seqStep: int = 1,
        plateIndex: int | None = None,
        plateUuid: str | None = None,
    ) -> list[WellplateFrameInfoItem]:
        if frameCount <= 0:
            return []
        if seqStep == 0:
            raise ValueError("seqStep must not be 0.")

        out: list[WellplateFrameInfoItem] = []
        for offset in range(frameCount):
            out.append(
                self.addItem(
                    seqIndex=int(seqStart) + offset * int(seqStep),
                    well=well,
                    plateIndex=plateIndex,
                    plateUuid=plateUuid,
                )
            )
        return out

    def addGrid(
        self,
        wells: Iterable[str | tuple[int, int]],
        *,
        seqStart: int = 0,
        framesPerWell: int = 1,
        seqStep: int = 1,
        plateIndex: int | None = None,
        plateUuid: str | None = None,
    ) -> WellplateFactory:
        seq = int(seqStart)
        for well in wells:
            self.addWell(
                well,
                seqStart=seq,
                frameCount=int(framesPerWell),
                seqStep=int(seqStep),
                plateIndex=plateIndex,
                plateUuid=plateUuid,
            )
            seq += int(framesPerWell) * int(seqStep)
        return self

    def createWellplateDesc(self) -> WellplateDesc:
        return WellplateDesc(
            name=self.name,
            rows=self.rows,
            columns=self.columns,
            rowNaming=self.rowNaming,
            columnNaming=self.columnNaming,
        )

    def createWellplateFrameInfo(self, *, sort_by_seq: bool = True) -> WellplateFrameInfo:
        items = list(self._items)
        if sort_by_seq:
            items.sort(key=lambda item: item.seqIndex)
        return WellplateFrameInfo(items)

    def create(self) -> tuple[WellplateDesc, WellplateFrameInfo]:
        return self.createWellplateDesc(), self.createWellplateFrameInfo()

    def __call__(self) -> tuple[WellplateDesc, WellplateFrameInfo]:
        return self.create()

