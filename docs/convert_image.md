# Convert image to ND2 format

Functions for converting images are not part of the base `limnd2` namespace, but are instead accessible through the `limnd2.tools` namespace.

In order to use functions and classes described on this page, import them like this:

```python
from limnd2.tools import convert_file_to_nd2,\
    convert_file_to_nd2_args, convert_file_to_nd2_cli
```

## Summary

- [convert_file_to_nd2](convert_image.md#limnd2.tools.conversion.LimConvertFile.convert_file_to_nd2):
 Convert an image to ND2.
- [convert_file_to_nd2_cli](convert_image.md#limnd2.tools.conversion.LimConvertFile.convert_file_to_nd2_cli):
 Convert an image to ND2 format with CLI arguments (for CLI usage).
- [convert_file_to_nd2_args](convert_image.md#limnd2.tools.conversion.LimConvertFile.convert_file_to_nd2_args):
 Convert an image ND2 format with CLI arguments (for usage in Python scripts).

## Function documentation

::: limnd2.tools.conversion.LimConvertFile
    options:
      heading_level: 3
      members:
        - convert_file_to_nd2
        - convert_file_to_nd2_cli
        - convert_file_to_nd2_args
