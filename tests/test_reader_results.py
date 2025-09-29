import pytest

pytest.importorskip("h5py")
pytest.importorskip("pandas")

from limnd2.nd2 import Nd2Reader
from limnd2.results import ResultPanesConfiguration, create_table_data_from_h5, read_results_from_h5


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
