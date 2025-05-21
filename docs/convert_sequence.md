# Convert image sequence to ND2 format

Functions for converting image sequences are not part of the base `limnd2` namespace, but are instead accessible through the `limnd2.tools` namespace.

In order to use functions and classes described on this page, import them like this:

```python
from limnd2.tools import convert_sequence_to_nd2,\
    convert_sequence_to_nd2_args, convert_sequence_to_nd2_cli
```

## Summary

- [convert_sequence_to_nd2](convert_sequence.md#limnd2.tools.conversion.LimConvertSequence.convert_sequence_to_nd2):
 Convert a sequence of images to ND2 format.
- [convert_sequence_to_nd2_cli](convert_sequence.md#limnd2.tools.conversion.LimConvertSequence.convert_sequence_to_nd2_args):
 Convert a sequence of images to ND2 format with CLI arguments (for CLI usage).
- [convert_sequence_to_nd2_args](convert_sequence.md#limnd2.tools.conversion.LimConvertSequence.convert_sequence_to_nd2_cli):
 Convert a sequence of images to ND2 format with CLI arguments (for usage in Python scripts).

## Examples

Since converting image sequences is pretty advanced task and requires correct order of images, and correctly set attributes and metadata,
we prepared a few examples that showcase how to use the [`convert_sequence_to_nd2`](convert_sequence.md#limnd2.tools.conversion.LimConvertSequence.convert_sequence_to_nd2) function to create one dimensional, multi-dimensional and even multi-channel ND2 files.

To see the examples, please check out the [convert_to_nd2_examples](https://github.com/Laboratory-Imaging/limnd2/tree/main/examples/convert_to_nd2_examples) folder in the `limnd2` repository on GitHub.

## Function documentation

::: limnd2.tools.conversion.LimConvertSequence
    options:
      heading_level: 3
      members:
        - convert_sequence_to_nd2
        - convert_sequence_to_nd2_cli
        - convert_sequence_to_nd2_args
