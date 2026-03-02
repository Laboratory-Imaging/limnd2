# limnd2

A Python library for reading and writing Nikon NIS-Elements `.nd2` files.

`limnd2` is built on top of [tlambert03/nd2](https://github.com/tlambert03/nd2) and keeps a compatible interface while adding write support and extended metadata handling.

> [!WARNING]
> This library is still in active development.
> Current version: `0.3.0`.
> Until `1.0`, behavior and API can change, and some changes may be released without a version number bump.
> GitHub Issues and Pull Requests are currently disabled.
> If you have a problem or question, contact: `techsupp@lim.cz`.

## Install

Install from our package index with `pip`:

```sh
pip install --index-url https://pypi.lim-dev.xyz/simple limnd2
```

Install from our package index with `uv`:

```sh
uv pip install --index-url https://pypi.lim-dev.xyz/simple limnd2
```

Quick install check:

```sh
python -c "import limnd2; print(limnd2.__version__)"
```

## Choose extras

Install only what your workflow needs:

- `limnd2[results]`: enables reading ND2 results/analysis tables stored in `.h5` data (`h5py`, `pandas`).
- `limnd2[commonff]`: enables common file-format workflows, mainly conversions and exports (TIFF/OME-TIFF/PNG/JPEG inputs and TIFF export via `Pillow`, `tifffile`, `zarr`).
- `limnd2[legacy]`: enables reading legacy ND2 files that use JPEG2000 compression (`imagecodecs`).
- `limnd2[all]`: installs all runtime extras above; use this if you want full runtime functionality without picking extras one by one.

Examples with `pip`:

```sh
pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[results]"
pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[commonff,legacy]"
pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[all]"
```

Examples with `uv`:

```sh
uv pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[results]"
uv pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[commonff,legacy]"
uv pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[all]"
```

## Documentation and examples

- [Documentation](https://laboratory-imaging.github.io/limnd2/docs/)
- [Quick start](https://laboratory-imaging.github.io/limnd2/docs/index/)
- [Command-line tools](https://laboratory-imaging.github.io/limnd2/docs/cli_index/)
- [Releases](https://github.com/Laboratory-Imaging/limnd2/releases)
- [Usage examples](examples/)
