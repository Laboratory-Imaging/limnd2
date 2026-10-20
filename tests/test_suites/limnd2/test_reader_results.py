import pytest

pytest.importorskip("h5py")
pytest.importorskip("pandas")

from limnd2.nd2 import Nd2Reader
from limnd2.results import (
    ResultPanesConfiguration,
    create_table_data_from_h5,
    read_result_binary_data,
    read_results_from_h5,
)


def test_nd2_reader_exposes_pivot_table_results(nd2_with_result_path):
    h5_path = nd2_with_result_path.with_suffix(".h5")
    direct_results = read_results_from_h5(h5_path)

    with Nd2Reader(nd2_with_result_path) as nd2:
        reader_results = nd2.results

    assert direct_results, "Failed to load results directly from the .h5 file"
    assert reader_results, "Nd2Reader.results should not be empty"
    assert direct_results.keys() == reader_results.keys()

    pivot = reader_results["Pivot table"]
    assert pivot.load_error is None
    assert pivot.result_panes_configuration is ResultPanesConfiguration.simple
    assert "main-left" in pivot.result_panes

    pane = pivot.result_panes["main-left"]
    assert pane.state.get("panes"), "Expected runtime state with panes"
    assert set(pane.private_table_locations) == {"a", "b"}

    for alias, location in pane.private_table_locations.items():
        assert location.startswith("/Pivot table/"), f"Unexpected location for {alias}: {location}"

def test_create_table_data_from_h5_produces_dataframe(nd2_with_result_path, monkeypatch):
    import limnd2.results as results_module

    monkeypatch.setattr(results_module, "_TableData__lazy_pandas", results_module.__lazy_pandas, raising=False)

    h5_path = nd2_with_result_path.with_suffix(".h5")
    pivot = read_results_from_h5(h5_path)["Pivot table"]
    pane = pivot.result_panes["main-left"]
    table_location = pane.private_table_locations["b"]

    table_data = create_table_data_from_h5(h5_path, table_location)

    assert table_data is not None
    assert table_data.name == "a"
    assert list(table_data.df.columns[:3]) == ["TimeLapseIndex", "Time", "Entity"]
    assert len(table_data.df) == 6

    roi_meta = table_data.column_metadata["_ObjId"]
    assert roi_meta["globalRange"]["min"] == int(table_data.df["RoiId"].min())
    assert roi_meta["globalRange"]["max"] == int(table_data.df["RoiId"].max())

    assert table_data.column_lookup["_ObjId"] == "RoiId"
    assert set(table_data.df["Entity"].unique()) == {"Bin"}
    assert str(table_data.df["BinMeanOf340"].dtype) == "Float64"


def test_read_results_lists_and_reads_binary_layers(tmp_path):
    import h5py
    import numpy as np

    h5_path = tmp_path / "result.h5"
    with h5py.File(h5_path, "w") as h5:
        result = h5.create_group("Segmentation")
        variant = result.create_group("latest")
        variant.attrs["Creation_time"] = 1
        binary = variant.create_group("Binary_Cell")
        binary.attrs.update({
            "Object": "Binary",
            "Id": "Cell",
            "binName": "Cell labels",
            "width": 4,
            "height": 3,
            "bpc": 32,
        })
        binary.create_group("0").create_group("2").create_dataset(
            "image", data=np.arange(12, dtype=np.uint32).reshape(3, 4)
        )
        binary.create_group("1").create_group("2").create_dataset(
            "image", data=np.arange(4, dtype=np.uint32).reshape(2, 2)
        )

    binary_item = read_results_from_h5(h5_path)["Segmentation"].binaries[0]
    assert binary_item.result_name == "Segmentation"
    assert binary_item.name == "Cell labels"
    assert binary_item.location.endswith("/Binary_Cell")
    assert binary_item.dtype == "uint32"
    assert binary_item.shape == (3, 4)
    assert binary_item.levels == {0: (2,), 1: (2,)}
    assert binary_item.metadata["width"] == 4
    np.testing.assert_array_equal(
        read_result_binary_data(h5_path, binary_item, frame_index=2, rect=(1, 1, 2, 2)),
        np.array([[5, 6], [9, 10]], dtype=np.uint32),
    )
