from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from limnd2.tools.export import frame_export_cli, sequence_export_cli


@pytest.fixture()
def sample_nd2_path(nd2_files: list[Path]) -> Path:
    return nd2_files[0]


def test_frame_export_cli(tmp_path: Path, sample_nd2_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output_path = tmp_path / "frame_cli.tiff"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frame_export",
            str(sample_nd2_path),
            "--frame-index",
            "0",
            "--output-path",
            str(output_path),
            "--progress-to-json",
        ],
    )

    frame_export_cli()

    assert output_path.exists()
    captured = capsys.readouterr().out.strip()
    if captured:
        payload = json.loads(captured)
        assert payload.get("file")
        assert Path(payload["file"]).exists()



def test_sequence_export_cli(tmp_path: Path, sample_nd2_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    export_dir = tmp_path / "series_cli"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sequence_export",
            str(sample_nd2_path),
            "--folder",
            str(export_dir),
            "--prefix",
            "cli",
            "--bits",
            "8",
            "--progress-to-json",
        ],
    )

    sequence_export_cli()

    assert export_dir.exists()
    generated = sorted(export_dir.glob("*.tiff"))
    assert generated

    captured = capsys.readouterr().out.strip()
    if captured:
        lines = [json.loads(line) for line in captured.splitlines() if line.strip()]
        assert len(lines) == len(generated)
        for entry in lines:
            assert Path(entry["file"]).exists()
