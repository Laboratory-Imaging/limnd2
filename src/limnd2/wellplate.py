from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

from .experiment import WellplateDesc, WellplateFrameInfo


@dataclass
class Wellplate:
    """
    Combined wellplate metadata for one plate.

    !!! example
        See the [reading example](wellplate.md#wellplate-metadata) above.

    This object currently represents plate descriptor 0 and does not include
    every possible wellplate-related chunk. A file may contain only some of
    these fields.

    Attributes
    ----------
    description : WellplateDesc | None
        Plate name, row and column counts, and axis naming conventions.
        ``None`` if the plate descriptor is absent.
    frames : WellplateFrameInfo | None
        Frame-to-well mappings. Each item identifies an ND2 sequence index,
        plate index, well index, well name, and zero-based row and column.
        Multiple frames may belong to the same well. ``None`` if the frame
        information chunk is absent.
    definition : bytes | None
        Optional detailed plate definition, stored as raw JSON bytes. It may
        contain geometry, naming details, and a UUID. ``None`` if the chunk is
        absent; the factory does not invent a definition for a new plate.

    """

    description: WellplateDesc | None = None
    frames: WellplateFrameInfo | None = None
    definition: bytes | None = None

    @classmethod
    def from_reader(cls, reader) -> Wellplate | None:
        """Read existing wellplate chunks from an ND2 reader.

        Return ``None`` when none of the supported chunks are present.
        Partial metadata is retained with missing fields set to ``None``.
        """
        from .base import ND2_CHUNK_NAME_WellPlateDescJson

        desc, frames = reader.wellplateDesc, reader.wellplateFrameInfo
        definition = reader.chunk(ND2_CHUNK_NAME_WellPlateDescJson)
        if desc is None and frames is None and definition is None:
            return None
        return cls(desc, frames, bytes(definition) if definition is not None else None)

    def __iter__(self) -> Iterator[WellplateDesc | WellplateFrameInfo | None]:
        """Allow existing ``desc, frames = factory.create()`` calls to work."""
        return iter((self.description, self.frames))
