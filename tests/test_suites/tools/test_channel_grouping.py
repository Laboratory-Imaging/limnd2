from __future__ import annotations

from limnd2.tools.conversion.LimConvertUtils import ConvertSequenceArgs
from limnd2.tools.conversion.LimImageSourceConvert import (
    _ensure_metadata_plane_count,
    group_by_channel_with_padding,
)
from limnd2.tools.conversion.LimPlanSequence import _build_qml_settings
from limnd2.metadata_factory import MetadataFactory, Plane


class _DummySource:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"_DummySource({self.name})"


def test_group_by_channel_with_padding_maps_one_based_channel_values() -> None:
    # Simulate filename-derived channel groups from patterns like "..._w1.TIF", "..._w2.TIF", "..._w3.TIF".
    src_w1 = _DummySource("w1")
    src_w2 = _DummySource("w2")
    src_w3 = _DummySource("w3")

    files = {
        src_w1: [1],
        src_w2: [2],
        src_w3: [3],
    }
    args = ConvertSequenceArgs(channels=None)
    exp_count = {"channel": 3}

    frames = group_by_channel_with_padding(
        files=files,
        arguments=args,
        exp_count=exp_count,
        channel_count=3,
        zero_source_factory=lambda: _DummySource("zero"),
        source_wrapper_factory=lambda source, _slot: source,
        allow_missing_files=False,
    )

    assert len(frames) == 1
    assert [src.name for src in frames[0]] == ["w1", "w2", "w3"]


def test_group_by_channel_with_padding_uses_auto_mode_when_channels_not_user_provided() -> None:
    # Simulate metadata-added channels (e.g. from sample file) while filename values are one-based.
    src_w1 = _DummySource("w_channel")
    src_w2 = _DummySource("w_another")
    src_w3 = _DummySource("w_third")

    files = {
        src_w1: [1],
        src_w2: [2],
        src_w3: [3],
    }
    args = ConvertSequenceArgs(
        channels={0: object()},  # auto metadata already present
        channels_user_provided=False,
    )
    exp_count = {"channel": 3}

    frames = group_by_channel_with_padding(
        files=files,
        arguments=args,
        exp_count=exp_count,
        channel_count=3,
        zero_source_factory=lambda: _DummySource("zero"),
        source_wrapper_factory=lambda source, _slot: source,
        allow_missing_files=False,
    )

    assert len(frames) == 1
    assert [src.name for src in frames[0]] == ["w_channel", "w_another", "w_third"]


def test_group_by_channel_with_padding_numeric_fallback_with_user_channel_keys() -> None:
    # Simulate explicit channel settings (string keys) while filename channels are numeric 1..N.
    src_w1 = _DummySource("w1")
    src_w2 = _DummySource("w2")
    src_w3 = _DummySource("w3")

    files = {
        src_w1: [1],
        src_w2: [2],
        src_w3: [3],
    }
    args = ConvertSequenceArgs(
        channels={"w_channel": object(), "w_another": object(), "w_third": object()},
        channels_user_provided=True,
    )
    exp_count = {"channel": 3}

    frames = group_by_channel_with_padding(
        files=files,
        arguments=args,
        exp_count=exp_count,
        channel_count=3,
        zero_source_factory=lambda: _DummySource("zero"),
        source_wrapper_factory=lambda source, _slot: source,
        allow_missing_files=False,
    )

    assert len(frames) == 1
    assert [src.name for src in frames[0]] == ["w1", "w2", "w3"]


def test_group_by_channel_with_padding_returns_slot_original_values() -> None:
    src_w1 = _DummySource("w1")
    src_w2 = _DummySource("w2")
    src_w3 = _DummySource("w3")

    files = {
        src_w1: [1],
        src_w2: [2],
        src_w3: [3],
    }
    args = ConvertSequenceArgs(channels=None)
    exp_count = {"channel": 3}

    grouped = group_by_channel_with_padding(
        files=files,
        arguments=args,
        exp_count=exp_count,
        channel_count=3,
        zero_source_factory=lambda: _DummySource("zero"),
        source_wrapper_factory=lambda source, _slot: source,
        allow_missing_files=False,
        return_channel_slot_values=True,
    )
    frames, slot_values = grouped

    assert len(frames) == 1
    assert [src.name for src in frames[0]] == ["w1", "w2", "w3"]
    assert slot_values == [1, 2, 3]


def test_build_qml_settings_prefers_original_channel_labels_column() -> None:
    args = ConvertSequenceArgs(metadata=MetadataFactory(), time_step=None, z_step=None)
    args.metadata.addPlane({"name": "Blue", "modality": "Undefined", "color": "#4080FF"})
    args.metadata.addPlane({"name": "Green", "modality": "Undefined", "color": "#00B050"})

    qml = _build_qml_settings(
        args,
        channel_labels=["Blue", "Green"],
        channel_original_labels=["1", "2"],
    )

    assert qml["channels"][0][0] == "1"
    assert qml["channels"][0][1] == "Blue"
    assert qml["channels"][1][0] == "2"
    assert qml["channels"][1][1] == "Green"


def test_ensure_metadata_plane_count_limits_excess_user_channel_settings() -> None:
    args = ConvertSequenceArgs(
        metadata=MetadataFactory(),
        channels={
            "10": Plane(name="channel_10"),
            "11": Plane(name="channel_11"),
            "2": Plane(name="channel_2"),
            "3": Plane(name="channel_3"),
        },
        channels_user_provided=True,
    )

    _ensure_metadata_plane_count(
        args,
        component_count=2,
        is_rgb=False,
        channel_labels=["Channel1", "Channel2"],
    )

    assert len(args.metadata.planes) == 2
    assert [plane.name for plane in args.metadata.planes] == ["channel_2", "channel_3"]
