import json
from pathlib import Path

import pytest

from limnd2.tools import index

ND2_BASE = Path(__file__).parent / "test_files" / "nd2_files"
ND2_FILES = sorted(ND2_BASE.rglob("*.nd2")) if ND2_BASE.exists() else []


@pytest.fixture(autouse=True)
def restore_plain_print(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure we capture standard JSON output even if rich is installed.
    monkeypatch.setattr(index, "print", index.original_print)


@pytest.fixture()
def nd2_directory() -> Path:
    if not ND2_FILES:
        pytest.skip(f"No .nd2 files found under {ND2_BASE}")
    return ND2_BASE


def test_index_main_json_output(nd2_directory: Path, capsys: pytest.CaptureFixture[str]):
    capsys.readouterr()  # clear any pending output

    index.main([str(nd2_directory), "--recurse", "--format", "json"])

    out = capsys.readouterr().out.strip()
    assert out, "Expected JSON output from index tool"

    payload = json.loads(out)
    assert isinstance(payload, list) and payload, "Index should return at least one record"
    for entry in payload:
        assert "Name" in entry and entry["Name"].lower().endswith(".nd2")
        assert "Frames" in entry and entry["Frames"] >= 1


def test_index_main_include_columns(nd2_directory: Path, capsys: pytest.CaptureFixture[str]):
    capsys.readouterr()

    index.main(
        [
            str(nd2_directory),
            "--recurse",
            "--format",
            "json",
            "--include",
            "Name,Frames",
        ]
    )

    out = capsys.readouterr().out.strip()
    assert out
    payload = json.loads(out)
    assert payload
    for entry in payload:
        assert set(entry.keys()) == {"Name", "Frames"}
