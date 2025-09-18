import json
import sys
from pathlib import Path

import pytest

from limnd2.tools.export import frame_export_cli, sequence_export_cli

ND2_BASE = Path(__file__).parent / "test_files" / "nd2_files"
ND2_FILES = sorted(ND2_BASE.rglob("*.nd2")) if ND2_BASE.exists() else []


@pytest.fixture()
def sample_nd2_path() -> Path:
    if not ND2_FILES:
        pytest.skip(f"No .nd2 files found under {ND2_BASE}")
    return ND2_FILES[0]


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
