# Conversion Readers

This folder contains file readers used by the conversion pipeline (`limnd2-convert-file-to-nd2`, `limnd2-convert-sequence-to-nd2`, and `limnd2-get-image-dimensions`).

## Supported Input Formats

| Format | Extensions | Required extra | Main libraries used |
|---|---|---|---|
| ND2 | `.nd2` | none | built-in `limnd2` ND2 reader |
| OME-Zarr / OME-NGFF | `zarr.json` inside an `.ome.zarr` store | `limnd2[commonff]` | `zarr` |
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

For OME-Zarr, pass either the store folder itself or its root `zarr.json` / legacy
`.zattrs` file. The reader opens the store, reads level 0, and expands OME T/Z/C
axes and compatible image groups into the normal conversion pipeline. Plate
well paths are retained as multipoint labels, allowing the existing wellplate
inference to reconstruct well information. If `nd2_chunks/manifest.json` is
present, the reader exposes its decoded limnd2 metadata to the conversion
writer. The `limnd2-get-image-dimensions` metadata probe also accepts a store
folder and returns the dimensions and channel/calibration settings used by the
import dialog before conversion starts.

