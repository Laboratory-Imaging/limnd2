# limnd2

A Python library for reading and writing Nikon NIS-Elements `.nd2` files.

The `limnd2` is inspired by the [tlambert03/nd2](https://github.com/tlambert03/nd2) implementation.
It provides read/write [interfaces](https://laboratory-imaging.github.io/limnd2/docs/nd2/)
that aim to support:
- reading **all** data and metadata stored in ND2 files, and
- writing ND2 files that enable **meaningful analysis** in NIS-Elements and NIS-Express.

The library also implements the same public [interface](https://laboratory-imaging.github.io/limnd2/docs/nd2file/)
as [tlambert03/nd2](https://github.com/tlambert03/nd2), so it can be used as a drop-in replacement in most cases.

> [!WARNING]
> This library is still in active development.
> Current version: `0.3.0`.
> Until `1.0`, behavior and APIs can change, and some changes may be released without a version bump.
> GitHub Issues and Pull Requests are currently disabled.
> If you have a problem or question, contact: `techsupp@lim.cz`.

## Install

Install from our package index with `pip`:

```sh
pip install --index-url https://pypi.laboratory-imaging.com/simple limnd2
```

Install from our package index with `uv`:

```sh
uv pip install --index-url https://pypi.laboratory-imaging.com/simple limnd2
```

Quick install check:

```sh
python -c "import limnd2; print(limnd2.__version__)"
```

## Choose extras

Install only what your workflow needs:

- `limnd2[results]`: enables reading ND2 results/analysis tables stored in `.h5` data (`h5py`, `pandas`).
- `limnd2[commonff]`: enables common file-format workflows, mainly conversions and exports
(TIFF/OME-TIFF/PNG/JPEG inputs and TIFF export via `Pillow`, `tifffile`, `zarr`).
- `limnd2[ome-zarr]`: enables OME-Zarr export and the OME-Zarr GUI/CLI tools
(`dask`, `ome-zarr`, `zarr`, `fsspec`, `s3fs`).
- `limnd2[legacy]`: enables reading legacy ND2 files that use JPEG2000 compression (`imagecodecs`).
- `limnd2[all]`: installs the main runtime extras above, but does not include `ome-zarr` yet.

Examples with `pip`:

```sh
pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[results]"
pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[commonff,legacy]"
pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[ome-zarr]"
pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[all]"
```

Examples with `uv`:

```sh
uv pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[results]"
uv pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[commonff,legacy]"
uv pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[ome-zarr]"
uv pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[all]"
```

## Documentation and examples

- [Documentation](https://laboratory-imaging.github.io/limnd2/docs/)
- [Quick start](https://laboratory-imaging.github.io/limnd2/docs)
- [Command-line tools](https://laboratory-imaging.github.io/limnd2/docs/cli_index/)
- [Releases](https://github.com/Laboratory-Imaging/limnd2/releases)
- [Usage examples](examples/)

## OME-Zarr export

Install the OME-Zarr extra first:

```sh
pip install --index-url https://pypi.laboratory-imaging.com/simple "limnd2[ome-zarr]"
```

Python API:

```python
import limnd2

with limnd2.Nd2Reader("file.nd2") as reader:
    reader.to_ome_zarr(
        "file.ome.zarr",
        include_binaries=True,
        use_dask=True,
        overwrite=True,
    )
```

CLI:

```sh
limnd2-ome-zarr-export file.nd2
limnd2-ome-zarr-export file.nd2 --output-folder .\exports
limnd2-ome-zarr-export file.nd2 --s3-prefix s3://my-bucket/ome-zarr
```

GUI:

```sh
limnd2-ome-zarr-exporter
```
