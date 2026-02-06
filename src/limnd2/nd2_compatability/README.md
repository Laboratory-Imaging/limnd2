# ND2 compatibility layer

This folder contains the compatibility layer that mirrors Talley Lambert’s
`nd2` API while using the `limnd2` backend (chunker, metadata, variants).

Entry point: `limnd2.nd2_compatability.ND2File` (also re-exported as
`limnd2.nd2file.ND2File` for backward compatibility).

## ND2File API summary

The compat class always uses `limnd2.Nd2Reader` (and its chunker) under the hood.
Below is the public API summary; private helpers are intentionally omitted.

| Method / property | Uses limnd2 | Status / notes |
| --- | --- | --- |
| `is_supported_file` | No (reads header bytes directly) | Checks ND2/JP2 magic. |
| `version` | Yes (`Nd2Reader.version`) | Works. |
| `path` | No | Returns path or `<memory>`. |
| `is_legacy` | Yes (`Nd2Reader.version`) | True for (1,0) legacy. |
| `open` / `close` / `closed` | Yes | Wraps `Nd2Reader` lifecycle; `.close()` calls `finalize()`. |
| `attributes` | Yes (`imageAttributes`, `pictureMetadata`) | Works; compat mapping of channels/bit depth. |
| `text_info` | Yes (`imageTextInfo`) | Works; returns dict. |
| `rois` | No | Not implemented; returns `{}`. |
| `experiment` | Yes (`experiment`) | Partial: Time/NETime/Z/XY loops mapped; Spect loop skipped. |
| `events` | Yes (`recordedData`) | Works for recorded data; returns records/list/dict. |
| `unstructured_metadata` | Yes (chunker + `decode_lv`/`decode_var`) | Works; compat decoding with prefix stripping. |
| `metadata` | Yes (`pictureMetadata`, attributes) | Builds compat `Metadata`/`Channel`; partial but functional. |
| `frame_metadata` | Yes | Per-frame metadata assembled from picture metadata. |
| `custom_data` | Yes (custom chunks + descriptions) | Works; returns CustomDescription/SmartExperiment when present. |
| `jobs` | No | Not implemented; returns `None`. |
| `ndim` / `shape` | Yes | Derived from `sizes` + frame shape. |
| `sizes` | Yes + compat heuristics | Uses experiment + textinfo inference; may differ from nd2 in edge cases. |
| `is_rgb` / `components_per_channel` | Yes | Derived from attributes/channel count. |
| `size` / `nbytes` / `dtype` | Yes | Derived from attributes/shape. |
| `voxel_size` | Yes (`pictureMetadata`, Z loop) | Works. |
| `asarray` | Yes (`read_frame`) | Stacks frames; supports `position=` slicing. |
| `__array__` | Yes | NumPy interop via `asarray()`. |
| `write_tiff` | Yes + `tifffile` | Works; OME optional, requires `ome-types`. |
| `to_dask` | Yes + `dask` | Works; optional `resource_backed_dask_array`. |
| `to_xarray` | Yes + `xarray` | Works; coords from loops + voxel sizes. |
| `read_frame` | Yes (`Nd2Reader.image`) | Works; compat reshape/axis heuristics. |
| `loop_indices` | Yes (`experiment`) | Works; axis mapping from loops. |
| `binary_data` | Yes (binary metadata + data) | Works when binary layers exist. |
| `ome_metadata` | Yes + `ome-types` | Works for modern ND2; not supported for legacy. |

## Missing features

- Chunkmap rescue helpers (`nd2._parse._chunk_decode.get_chunkmap`) are not wired up yet.
- ROIs parsing is not implemented (currently returns empty).
- `write_ome_zarr` is not implemented.
- Some metadata/shape parity mismatches remain.
