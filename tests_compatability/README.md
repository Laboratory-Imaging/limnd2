# ND2 compatibility tests

These tests exercise the `limnd2.nd2file` compatibility layer (API parity with
Talley Lambert's `nd2` package) and related helpers.

## What lives here

- `test_*.py`: pytest suites that mirror nd2 behavior and verify metadata,
  read paths, dask/xarray adapters, and I/O helpers.
- `nd2/`: a lightweight shim package used by the tests to prefer the local
  `limnd2.nd2file` implementation.
- `data/`: sample ND2 files used by the test suite (downloaded on demand).

## Running the tests (manual)

These tests are not part of the default suite. Run them explicitly from the repo
root:

```bash
pytest tests_compatability
```

Optional: run a single file

```bash
pytest tests_compatability/test_reader.py
```

On Windows, you can also use the helper script that generates HTML + logs:

```bat
tests_compatability\run_compatability_tests.bat
```

It writes:
- `tests_compatability/test_report/report.html`
- `tests_compatability/test_report/pytest_output.txt`
- `tests_compatability/test_report/coverage_output.txt`

## Test data

On first run, the suite will attempt to download ND2 sample data into
`tests_compatability/data` (from the official nd2 test set, with a fallback S3 mirror).
If downloads are blocked, tests will still be discovered but will show as skipped.

## Notes

- These tests are intentionally kept out of the main `tests/` folder and are
  excluded from default test discovery (see `.vscode/settings.json` and
  `pyproject.toml`). Run them manually as shown above.
- Some tests require optional deps like `dask`, `xarray`, `ome-types`, or
  `resource_backed_dask_array`.

## Skips you may still see

- Missing optional deps (`dask`, `xarray`, `ome-types`, `pandas`,
  `resource_backed_dask_array`, `tifffile`, `imagecodecs`, `yaozarrs`).
- Large-file guards in `test_reader.py` (to keep runtime reasonable).
- `test_codspeed.py` unless you run with `--codspeed`.
- `test_index.py` (nd2.index not supported in limnd2 compat).
- `test_jobs.py` (jobs metadata not supported in limnd2 compat).

## Known gaps (will fail if un-skipped)

- `write_ome_zarr` is not implemented on the compat `ND2File`.
- Chunkmap rescue helpers (`nd2._parse._chunk_decode.get_chunkmap`) are not
  wired up to limnd2 yet (rescue tests fail).
- ROIs parsing is incomplete (currently returns empty in `test_rois.py`).
- Bioformats parity has known mismatches (e.g. `jonas_control002.nd2` sizes).

## ND2File API summary

See `src/limnd2/nd2_compatability/README.md` for the compat layer API summary.
