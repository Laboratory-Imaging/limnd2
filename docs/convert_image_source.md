
# LimImageSource class

This class is mainly used for converting images of different formats to ND2 format. It serves as a locator for specific image frame (just filepath for most image formats, filepath + IDF for TIFF files).

This class is not a part of the base `limnd2` namespace, but is instead accessible through the `limnd2.tools` namespace like this:

```python
from limnd2.tools import LimImageSource
```

::: limnd2.tools.conversion.LimImageSource
    options:
      heading_level: 3
