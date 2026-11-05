# Wellplate metadata

Use [`Nd2Reader.wellplate`](nd2.md#limnd2.nd2.Nd2Reader.wellplate) to read the plate description and its
frame-to-well mappings together. It returns `None` if no wellplate metadata
is present. Individual fields can also be `None` if the file contains only
part of the metadata.

```python
import limnd2

with limnd2.Nd2Reader("input.nd2") as reader:
    plate = reader.wellplate
    if plate is not None:
        if plate.description is not None:
            print(f"Plate name: {plate.description.name}")
            print(f"Plate rows: {plate.description.rows}")
            print(f"Plate columns: {plate.description.columns}")

        if plate.frames is not None:
            for frame in plate.frames:
                print(
                    f"Frame {frame.seqIndex}: well {frame.wellName!r}, "
                    f"row index {frame.wellRowIndex}, "
                    f"column index {frame.wellColIndex}, "
                    f"well index {frame.wellIndex}"
                )
```

??? example "See example output"
    For a plate with four mapped frames, the output could look like this:

    ```text
    Plate name: My plate
    Plate rows: 8
    Plate columns: 12
    Frame 0: well 'B2', row index 1, column index 1, well index 13
    Frame 1: well 'B3', row index 1, column index 2, well index 14
    Frame 2: well 'C2', row index 2, column index 1, well index 25
    Frame 3: well 'C3', row index 2, column index 2, well index 26
    ```

Sequence, plate, well, row, and column indexes are zero-based. Some ND2 files
have an empty `wellName`; their row and column indexes can still identify the
well. `plate.definition` contains the optional detailed plate definition as
raw bytes. This interface currently reads plate descriptor 0; it does not
include every possible wellplate-related chunk.

For the individual fields, see [`WellplateDesc`](experiment.md#limnd2.experiment.WellplateDesc),
[`WellplateFrameInfo`](experiment.md#limnd2.experiment.WellplateFrameInfo), and
[`WellplateFrameInfoItem`](experiment.md#limnd2.experiment.WellplateFrameInfoItem).

!!! info
    To create wellplate metadata or write it to an ND2 file, see the
    [wellplate factory](wellplate_factory.md).

::: limnd2.wellplate
    options:
      heading_level: 3
