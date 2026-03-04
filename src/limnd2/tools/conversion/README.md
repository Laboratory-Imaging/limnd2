# Conversion Readers

This folder contains file readers used by the conversion pipeline (`limnd2-convert-file-to-nd2`, `limnd2-convert-sequence-to-nd2`, and `limnd2-get-image-dimensions`).

## Supported Input Formats

| Format | Extensions | Required extra | Main libraries used |
|---|---|---|---|
| ND2 | `.nd2` | none | built-in `limnd2` ND2 reader |
| TIFF family | `.tif`, `.tiff`, `.btf` | `limnd2[commonff]` | `tifffile`, `zarr`, `ome_types` |
| Zeiss LSM | `.lsm` | `limnd2[commonff]` | `tifffile` (LSM-specific source class) |
| Zeiss CZI | `.czi` | `limnd2[czi]` | `czifile`, `imagecodecs` |
| Olympus OIF/OIB | `.oif`, `.oib` | `limnd2[olympus]` | `oiffile` |
| PNG | `.png` | `limnd2[commonff]` | `Pillow` |
| JPEG | `.jpg`, `.jpeg` | `limnd2[commonff]` | `Pillow` |

## TIFF Dispatch Subtypes

`LimImageSourceTiff` is an entry point that dispatches to:

1. `LimImageSourceTiffOmeTiff` for OME-TIFF (OME XML metadata).
2. `LimImageSourceTiffMeta` for MetaSeries / MetaXpress style `<MetaData>...` payloads.
3. `LimImageSourceTiffBase` as fallback for plain TIFF.

## Optional Extras Behavior

If a required extra is not installed, only that specific format is unavailable.  
Other formats continue to work, and the converter raises a clear `ImportError` with the suggested `pip install "limnd2[...]"` command.

