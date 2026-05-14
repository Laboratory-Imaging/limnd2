from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ome_zarr_export_cli_module = importlib.import_module("limnd2.tools.ome_zarr_export_cli")


class _FakeReader:
    last_instance: _FakeReader | None = None

    def __init__(self, path: Path) -> None:
        self.path = path
        self.imageDataShape = (1, 1, 1, 8, 8, 1)
        self.imageAttributes = type("Attrs", (), {"dtype": "uint16"})()
        self.binaryRasterMetadata = ()
        self.calls: list[tuple[str | Path, dict[str, object]]] = []
        _FakeReader.last_instance = self

    def __enter__(self) -> _FakeReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def to_ome_zarr(self, target: str | Path, **kwargs: object) -> str | Path:
        self.calls.append((target, kwargs))
        return target


def test_ome_zarr_export_cli_local_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sample.nd2"
    source.touch()

    dep_calls: list[tuple[bool, bool]] = []
    s3_checks: list[str] = []
    uploads: list[tuple[Path, str]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        ["limnd2-ome-zarr-export", str(source)],
    )
    monkeypatch.setattr(
        ome_zarr_export_cli_module,
        "ensure_ome_zarr_dependencies",
        lambda **kwargs: dep_calls.append(
            (bool(kwargs["require_s3"]), bool(kwargs["require_dask"]))
        ),
    )
    monkeypatch.setattr(ome_zarr_export_cli_module, "_s3_write_check", lambda prefix: s3_checks.append(prefix))
    monkeypatch.setattr(
        ome_zarr_export_cli_module,
        "_upload_local_ome_zarr",
        lambda local_path, dest_uri: uploads.append((local_path, dest_uri)),
    )
    monkeypatch.setattr(ome_zarr_export_cli_module.limnd2, "Nd2Reader", _FakeReader)

    ome_zarr_export_cli_module.main()

    assert dep_calls == [(False, True)]
    assert s3_checks == []
    assert uploads == []

    reader = _FakeReader.last_instance
    assert reader is not None
    assert len(reader.calls) == 1
    target, kwargs = reader.calls[0]
    assert target == source.parent / "sample.ome.zarr"
    assert kwargs["overwrite"] is False
    assert kwargs["use_dask"] is None
    assert kwargs["chunks"] == (1, 1, 1, 512, 512)
    assert kwargs["shard_shape"] is None
    assert kwargs["include_binaries"] is False
    assert callable(kwargs["progress_callback"])


def test_ome_zarr_export_cli_direct_s3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sample.nd2"
    source.touch()

    dep_calls: list[tuple[bool, bool]] = []
    s3_checks: list[str] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limnd2-ome-zarr-export",
            str(source),
            "--s3-prefix",
            "s3://bucket/prefix",
            "--output-name",
            "custom_export",
            "--include-binaries",
            "--overwrite",
        ],
    )
    monkeypatch.setattr(
        ome_zarr_export_cli_module,
        "ensure_ome_zarr_dependencies",
        lambda **kwargs: dep_calls.append(
            (bool(kwargs["require_s3"]), bool(kwargs["require_dask"]))
        ),
    )
    monkeypatch.setattr(ome_zarr_export_cli_module, "_s3_write_check", lambda prefix: s3_checks.append(prefix))
    monkeypatch.setattr(ome_zarr_export_cli_module.limnd2, "Nd2Reader", _FakeReader)

    ome_zarr_export_cli_module.main()

    assert dep_calls == [(True, True)]
    assert s3_checks == ["s3://bucket/prefix"]

    reader = _FakeReader.last_instance
    assert reader is not None
    target, kwargs = reader.calls[0]
    assert target == "s3://bucket/prefix/custom_export.ome.zarr"
    assert kwargs["overwrite"] is True
    assert kwargs["include_binaries"] is True


def test_ome_zarr_export_cli_local_then_upload_to_s3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sample.nd2"
    source.touch()
    cwd = tmp_path / "work"
    cwd.mkdir()

    dep_calls: list[tuple[bool, bool]] = []
    s3_checks: list[str] = []
    uploads: list[tuple[Path, str]] = []

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "limnd2-ome-zarr-export",
            str(source),
            "--output-folder",
            ".\\exports",
            "--s3-prefix",
            "s3://bucket/prefix",
        ],
    )
    monkeypatch.setattr(
        ome_zarr_export_cli_module,
        "ensure_ome_zarr_dependencies",
        lambda **kwargs: dep_calls.append(
            (bool(kwargs["require_s3"]), bool(kwargs["require_dask"]))
        ),
    )
    monkeypatch.setattr(ome_zarr_export_cli_module, "_s3_write_check", lambda prefix: s3_checks.append(prefix))
    monkeypatch.setattr(
        ome_zarr_export_cli_module,
        "_upload_local_ome_zarr",
        lambda local_path, dest_uri: uploads.append((local_path, dest_uri)),
    )
    monkeypatch.setattr(ome_zarr_export_cli_module.limnd2, "Nd2Reader", _FakeReader)

    ome_zarr_export_cli_module.main()

    assert dep_calls == [(False, True)]
    assert s3_checks == ["s3://bucket/prefix"]

    expected_local = (cwd / "exports" / "sample.ome.zarr").resolve()
    reader = _FakeReader.last_instance
    assert reader is not None
    target, _kwargs = reader.calls[0]
    assert target == expected_local
    assert uploads == [(expected_local, "s3://bucket/prefix/sample.ome.zarr")]
