# ND2 Conversion Examples

This folder contains example scripts demonstrating how to convert TIFF image sequences into ND2 files using the `limnd2` library. Each script shows how to handle different ND2 dimensions and metadata.

## Example Scripts

| Script                                                     | TIFF Folder              | ND2 Dimensions         | Description                                                             |
|------------------------------------------------------------|--------------------------|------------------------|-------------------------------------------------------------------------|
| [`example_convert_simple.py`](./example_convert_simple.py) | [`tiffs_t`](./tiffs_t)   | Time (T)               | Converts TIFFs into an ND2 file with a single time loop dimension.      |
| [`example_convert_tc.py`](./example_convert_tc.py)         | [`tiffs_tc`](./tiffs_tc) | Time (T), Channels (C) | Converts TIFFs into an ND2 file with time and multiple simple channels. |
| [`example_convert_tc2.py`](./example_convert_tc2.py)       | [`tiffs_tc`](./tiffs_tc) | Time (T), Channels (C) | Similar to above, but adds more detailed channel metadata.              |
| [`example_convert_tz.py`](./example_convert_tz.py)         | [`tiffs_tz`](./tiffs_tz) | Time (T), Z-stack (Z)  | Converts TIFFs into an ND2 file with time and z-stack dimensions.       |

Each script serves as a practical example of using `limnd2` for different ND2 file structures. See the individual scripts for usage details and specific implementation.
