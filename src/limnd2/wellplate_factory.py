"""
Helpers for creating wellplate metadata structures (`WellplateDesc`,
`WellplateFrameInfo`) for writing to ND2 files.
"""

from __future__ import annotations

import re
from .experiment import WellplateDesc, WellplateFrameInfo, WellplateFrameInfoItem
from .wellplate import Wellplate

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
    Helper class for creating a [`Wellplate`](wellplate.md).

    The intended use is:
    1. Set plate geometry (`rows`, `columns`, naming).
    2. Add frame-to-well mappings with `addItem`, `addWell`, or `addWellRect`.
    3. Call `create` to obtain a wellplate.
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
        self.rowNaming = str(rowNaming) if rowNaming else "letter"
        self.columnNaming = str(columnNaming) if columnNaming else "number"
        self.plateIndex = int(plateIndex)
        self.plateUuid = str(plateUuid)

        self._items: list[WellplateFrameInfoItem] = []
        self._next_seq_index = 0

    def _well_name(self, row_idx: int, col_idx: int) -> str:
        row = _index_to_row_label(row_idx) if self.rowNaming == "letter" else str(row_idx + 1)
        col = _index_to_row_label(col_idx) if self.columnNaming == "letter" else str(col_idx + 1)
        return f"{row}{col}"

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
        if self._items and ((rows is not None and int(rows) != self.rows) or
                            (columns is not None and int(columns) != self.columns)):
            raise ValueError("Cannot change plate geometry after adding frames.")
        if name is not None:
            self.name = str(name)
        if rows is not None:
            self.rows = int(rows)
        if columns is not None:
            self.columns = int(columns)
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
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Set positive plate rows and columns before adding wells.")
        expected = row_idx * self.columns + col_idx
        if explicit is not None and int(explicit) != expected:
            raise ValueError(f"wellIndex must be {expected} for the given coordinates.")
        return expected

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
        """Add a sequence frame using a well name or zero-based coordinates."""
        if isinstance(well, str) and (self.rowNaming, self.columnNaming) != ("letter", "number"):
            raise ValueError("String well names require letter rows and number columns; use coordinates.")
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

        if int(seqIndex) < 0:
            raise ValueError("seqIndex must be non-negative.")
        if any(existing.seqIndex == int(seqIndex) for existing in self._items):
            raise ValueError(f"Sequence index {seqIndex} is already mapped.")
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
        self._next_seq_index = max(self._next_seq_index, item.seqIndex + 1)
        return item

    def addWell(
        self,
        well: str | tuple[int, int] | list[str | tuple[int, int]],
        *,
        seqStart: int | None = None,
        frameCount: int = 1,
        seqStep: int = 1,
        wellIndex: int | None = None,
        plateIndex: int | None = None,
        plateUuid: str | None = None,
    ) -> list[WellplateFrameInfoItem]:
        """
        Map one or more wells to consecutive sequence frames.

        Parameters
        ----------
        well : str | list[str]
            Letter-row, number-column names, such as ``B2``.
        seqStart : int | None
            First sequence index. If omitted, continue after the highest index used.
        frameCount : int
            Number of consecutive frames per well.
        seqStep : int
            Positive interval between sequence indexes.
        wellIndex : int | None
            Optional physical index for a single well; must match its coordinates.

        Returns
        -------
        list[WellplateFrameInfoItem]
            The newly added frame mappings.
        """
        if frameCount <= 0:
            raise ValueError("frameCount must be positive.")
        if seqStep <= 0:
            raise ValueError("seqStep must be positive.")

        wells = well if isinstance(well, list) else [well]
        if not wells:
            raise ValueError("At least one well is required.")
        if (self.rowNaming, self.columnNaming) != ("letter", "number") or not all(isinstance(w, str) for w in wells):
            raise ValueError("addWell requires letter-number well names; use addItem for coordinates.")
        if wellIndex is not None and len(wells) != 1:
            raise ValueError("wellIndex can only be supplied for a single well.")
        seq = self._next_seq_index if seqStart is None else int(seqStart)
        # Roll back on failure so list and rectangle additions are atomic.
        original_length, original_next = len(self._items), self._next_seq_index

        out: list[WellplateFrameInfoItem] = []
        try:
            for current in wells:
                for _ in range(frameCount):
                    out.append(self.addItem(
                        seqIndex=seq, well=current, plateIndex=plateIndex,
                        plateUuid=plateUuid, wellIndex=wellIndex,
                    ))
                    seq += int(seqStep)
        except Exception:
            del self._items[original_length:]
            self._next_seq_index = original_next
            raise
        return out

    def addWellRect(
        self, start: str, end: str, *, seqStart: int | None = None,
        frameCount: int = 1, seqStep: int = 1,
    ) -> list[WellplateFrameInfoItem]:
        """
        Map a rectangular region in row-major order, including both endpoints.

        Parameters
        ----------
        start, end : str
            First and last wells, for example ``B2`` and ``G11``.
        seqStart : int | None
            First sequence index, or the next available index when omitted.
        frameCount : int
            Number of consecutive frames per well.
        seqStep : int
            Positive interval between sequence indexes.

        Returns
        -------
        list[WellplateFrameInfoItem]
            The newly added frame mappings.
        """
        if (self.rowNaming, self.columnNaming) != ("letter", "number"):
            raise ValueError("Rectangles require letter rows and number columns.")
        first_row, first_col = _parse_well(start)
        last_row, last_col = _parse_well(end)
        if first_row > last_row or first_col > last_col:
            raise ValueError("Rectangle end must not precede its start.")
        wells = [self._well_name(row, col)
                 for row in range(first_row, last_row + 1)
                 for col in range(first_col, last_col + 1)]
        return self.addWell(wells, seqStart=seqStart, frameCount=frameCount, seqStep=seqStep)

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

    def create(self) -> Wellplate:
        """Create a wellplate from the configured description and mappings."""
        return Wellplate(self.createWellplateDesc(), self.createWellplateFrameInfo())

    def __call__(self) -> Wellplate:
        return self.create()

