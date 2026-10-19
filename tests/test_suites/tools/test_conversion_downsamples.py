from pathlib import Path
from unittest.mock import Mock

import limnd2

from limnd2.tools.conversion.LimConvertSequenceArgparser import create_parser
from limnd2.tools.conversion.LimImageSourceConvert import LIMND2Utils


def test_sequence_parser_accepts_generate_downsamples_flag():
    parsed = create_parser().parse_args(
        [".", ".*", "-n", "output.nd2", "--generate-downsamples"]
    )

    assert parsed.generate_downsamples is True


def test_writer_generates_downsamples_after_nd2_writer_closes(tmp_path, monkeypatch):
    output_path = tmp_path / "output.nd2"
    state = {"closed": False}

    class FakeWriter:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            state["closed"] = True

    def generate(path):
        assert state["closed"] is True
        assert path == output_path

    monkeypatch.setattr(limnd2, "Nd2Writer", FakeWriter)
    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceConvert.ProgressPrinter",
        lambda *_args, **_kwargs: Mock(),
    )
    generate_mock = Mock(side_effect=generate)
    monkeypatch.setattr(limnd2, "generate_downsamples", generate_mock)

    result = LIMND2Utils.write_files_to_nd2(
        output_path,
        [],
        object(),
        object(),
        object(),
        multiprocessing=False,
        generate_downsamples=True,
    )

    assert result is True
    generate_mock.assert_called_once_with(output_path)


def test_writer_skips_downsamples_by_default(tmp_path, monkeypatch):
    class FakeWriter:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

    monkeypatch.setattr(limnd2, "Nd2Writer", FakeWriter)
    monkeypatch.setattr(
        "limnd2.tools.conversion.LimImageSourceConvert.ProgressPrinter",
        lambda *_args, **_kwargs: Mock(),
    )
    generate_mock = Mock()
    monkeypatch.setattr(limnd2, "generate_downsamples", generate_mock)

    result = LIMND2Utils.write_files_to_nd2(
        Path(tmp_path / "output.nd2"),
        [],
        object(),
        object(),
        object(),
        multiprocessing=False,
    )

    assert result is True
    generate_mock.assert_not_called()
