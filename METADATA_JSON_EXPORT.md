# Metadata JSON Export Feature

## Overview

The `metadataAsJSON()` function exports all ND2 metadata as LLM-friendly JSON with embedded documentation. This allows you to:

1. **Preserve existing structure** - Exports data exactly as structured in ND2 files
2. **Add explanatory documentation** - Includes `_description` and `_doc` fields based on docstrings
3. **Enable LLM queries** - Makes it easy for AI to answer questions about the image

## Usage

### Basic Export

```python
import limnd2

with limnd2.Nd2Reader("file.nd2") as nd2:
    json_str = limnd2.metadataAsJSON(nd2)
    print(json_str)
```

### Export to File

```python
with limnd2.Nd2Reader("file.nd2") as nd2:
    limnd2.metadataAsJSON(nd2, output_path="metadata.json")
```

### Compact Export (No Documentation)

```python
with limnd2.Nd2Reader("file.nd2") as nd2:
    json_str = limnd2.metadataAsJSON(
        nd2,
        include_documentation=False,  # Remove _doc fields
        indent=None  # Single-line compact format
    )
```

## JSON Structure

The exported JSON includes:

### 1. **Summary Section** (Most Important for LLMs)
Quick-reference flags for common queries:
```json
{
  "summary": {
    "is3D": true,
    "_is3D_doc": "True if this is a 3D image (has Z-stack acquisition)",
    "hasTimeSeries": true,
    "_hasTimeSeries_doc": "True if this is a time-series acquisition",
    "hasMultipleXYSites": false,
    "_hasMultipleXYSites_doc": "True if acquisition includes multiple XY positions (multipoint)",
    "dimensionSummary": {
      "zPlanes": 5,
      "timePoints": 25,
      "xyPositions": 1,
      "channels": 2
    }
  }
}
```

### 2. **Attributes Section**
Core image dimensions and properties:
```json
{
  "attributes": {
    "_description": "Core image dimensions and pixel properties",
    "width": 1024,
    "_width_doc": "Image width in pixels",
    "height": 1024,
    "_height_doc": "Image height in pixels",
    "componentCount": 2,
    "_componentCount_doc": "Number of color channels/planes per frame",
    "frameCount": 125,
    "_frameCount_doc": "Total number of image frames in the sequence"
  }
}
```

### 3. **Experiment Section**
Acquisition loop definitions:
```json
{
  "experiment": {
    "_description": "Acquisition loop definitions organizing the image sequence",
    "levels": [
      {
        "type": "ExperimentTimeLoop",
        "_type_long_name": "Time",
        "count": 25,
        "step": 150,
        "stepUnit": "ms",
        "_step_doc": "Step size in ms"
      },
      {
        "type": "ExperimentZStackLoop",
        "_type_long_name": "Z-Stack",
        "count": 5,
        "step": 4.0,
        "stepUnit": "µm",
        "_step_doc": "Step size in µm"
      }
    ]
  }
}
```

### 4. **Metadata Section**
Channel information and microscope settings:
```json
{
  "metadata": {
    "_description": "Comprehensive channel information including wavelengths, microscope settings, and objectives",
    "channels": [
      {
        "name": "DETECTOR A",
        "emissionWavelengthNm": 520.0,
        "_emissionWavelengthNm_doc": "Emission wavelength in nanometers",
        "excitationWavelengthNm": 488.0,
        "modality": ["Camera", "AUX"]
      }
    ],
    "sampleSettings": {
      "cameraName": "Nikon A1 LFOV",
      "microscopeName": "Ti2 Microscope",
      "objectiveMagnification": 40.0
    },
    "calibration": {
      "pixelSizeUM": 0.432,
      "_pixelSizeUM_doc": "Pixel size in micrometers"
    }
  }
}
```

### 5. **Text Info Section**
Human-readable annotations:
```json
{
  "textInfo": {
    "_description": "Human-readable text metadata and annotations",
    "author": "Lab Technician",
    "_author_doc": "Person who acquired the image",
    "description": "Detailed acquisition settings...",
    "date": "9/22/2068 12:28:28 AM"
  }
}
```

## Common LLM Queries

After exporting, you can ask LLMs questions like:

| Question | JSON Path |
|----------|-----------|
| "Is this a 3D image?" | `summary.is3D` |
| "Does it have multiple XY sites?" | `summary.hasMultipleXYSites` |
| "Is this a time series?" | `summary.hasTimeSeries` |
| "What wavelengths were used?" | `metadata.channels[].emissionWavelengthNm` |
| "What's the pixel size?" | `metadata.calibration.pixelSizeUM` |
| "How many Z planes?" | `summary.dimensionSummary.zPlanes` |
| "What camera was used?" | `metadata.sampleSettings.cameraName` |
| "What objective magnification?" | `metadata.sampleSettings.objectiveMagnification` |

## Implementation Details

### Documentation Fields

All documentation fields are prefixed with `_`:
- `_description`: Section-level documentation
- `_*_doc`: Field-level documentation (e.g., `_width_doc`)
- `_note`: Additional contextual information

### Serialization

The implementation handles:
- NumPy types → Python native types
- Datetime → ISO format strings
- Enums → String representations
- Nested dataclasses → Dictionaries
- Arrays → Lists

### Source Files

- **Main function**: `src/limnd2/export.py::metadataAsJSON()`
- **Helper functions**: `_get_attributes_dict()`, `_get_experiment_dict()`, `_get_metadata_dict()`, etc.
- **Example**: `examples/example_metadata_json_export.py`

## Future Enhancements

The JSON can later be transformed to other formats using LLMs:
1. **OME-XML** - For interoperability with Bio-Formats
2. **Simplified views** - Custom schemas for specific workflows
3. **Natural language summaries** - LLM-generated descriptions
4. **Validation** - Check metadata completeness and consistency

## Notes

- Documentation strings are extracted from existing docstrings and comments in the codebase
- The structure mirrors the internal Python object structure
- Fields with `None` or empty values are still included (for schema consistency)
- Metadata and experiment data may be absent in simple images (RGB/mono, single frame)
