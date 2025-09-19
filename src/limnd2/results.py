from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import errno, enum, json

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import h5py
    import pandas

_RESULTS_EXTRA_INSTALL_HINT = '[results] extra not installed. Please install it with `pip install "limnd2[results]"`.'

def _missing_results_dependency(package: str) -> ImportError:
    msg = f'Missing optional dependency "{package}" required for limnd2 results support. {_RESULTS_EXTRA_INSTALL_HINT}'
    return ImportError(msg)

def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise _missing_results_dependency('h5py') from exc
    return h5py

def _require_pandas():
    try:
        import pandas
    except ImportError as exc:
        raise _missing_results_dependency('pandas') from exc
    return pandas

class PaneDataVersionTooLow(Exception):
    def __init__(self, version: int = 0):
        self.version = version
        self.message = f"Pane version too low: {self.version}"
        super().__init__(self.message)

def __lazy_pandas():
    return _require_pandas()

@dataclass(kw_only=True)
class TableData:
    name: str = ""
    metadata: dict[str, any] = field(default_factory=dict)
    private_tables: dict[str, TableData] = field(default_factory=dict)
    column_metadata: dict[str, dict[str, any]] = field(default_factory=dict)
    df: pandas.DataFrame = field(default_factory=lambda: __lazy_pandas().DataFrame())

    @property
    def column_lookup(self) -> dict[str, str]:
        if self.df is None:
            return {}
        col_ids = list(self.column_metadata.keys())
        col_names = list(self.df.columns)
        return { k: v for k, v in zip(col_ids, col_names) }


@dataclass(frozen=True, kw_only=True)
class BinaryResultItem:
    pass

@dataclass(kw_only=True)
class ResultPane:
    state: dict[str, any]
    private_tables: dict[str, TableData] = field(default_factory=dict)
    private_table_locations: dict[str, str]

class ResultPanesConfiguration(enum.StrEnum):
    none = "none"
    simple = "simple"
    complex = "complex"

@dataclass(frozen=True, kw_only=True)
class ResultItem:
    name: str
    load_error: str|None = None
    attributes: dict[str, any] = field(default_factory=dict)
    binaries: list[BinaryResultItem] = field(default_factory=list)
    result_panes: dict[str, ResultPane] = field(default_factory=dict)
    result_panes_configuration: ResultPanesConfiguration = ResultPanesConfiguration.none

    def __post_init__(self) -> None:
        if "side" in self.result_panes:
            object.__setattr__(self, 'result_panes_configuration', ResultPanesConfiguration.complex)
        elif "main-left" in self.result_panes:
            object.__setattr__(self, 'result_panes_configuration', ResultPanesConfiguration.simple)


def read_results_from_h5(h5_filename: str|Path) -> dict[str, ResultItem]:
    h5py = _require_h5py()
    try:
        results: dict[str, any] = {}
        with h5py.File(h5_filename, 'r') as h5:
            for result_name in h5.keys():
                result = h5[result_name]
                latest_time, latest_result_variant = 0, None
                for result_variant_key in result.keys():
                    result_variant = result[result_variant_key]
                    if (ctime := result_variant.attrs.get("Creation_time", 0)) < latest_time:
                        continue
                    latest_result_variant = result_variant
                    latest_time = ctime
                results[result_name] = read_result_item(result_name, latest_result_variant)
        return results
    except OSError as e:
        if e.errno == 33:
            raise PermissionError(f"Permission denied: {h5_filename}")
        elif e.errno == errno.ENOENT:
            return {}
        else:
            raise e
    except Exception as e:
        print(e)

def create_table_data_from_h5(h5_filename: str|Path, tbl_location : str) -> TableData:
    h5py = _require_h5py()
    pandas = _require_pandas()
    try:
        with h5py.File(h5_filename, 'r') as h5:
            tbl = h5[tbl_location]
            id_meta = {}
            table_data = TableData(name=tbl.attrs["Id"], metadata=json.loads(tbl.attrs["Metadata"]))
            for key in tbl.keys():
                item = tbl[key]
                if (obj := item.attrs["Object"]) == "Column":
                    column = item
                    id = column.attrs["Id"]
                    meta = json.loads(column.attrs["Metadata"])
                    if "title" not in meta:
                        meta["title"] = key[11:]
                    title = _make_unique_name(meta["title"], list(table_data.df.columns))
                    id_meta[id] = meta
                    decltype = meta.get('decltype', '')
                    if decltype == "int":
                        NA = column.attrs.get("NullValue", -1)
                        a = pandas.array(column[:], dtype="Int64")
                        a[a == NA] = pandas.NA
                        table_data.df[title] = a
                        meta["globalRange"] = { "min": int(a.min()), "max": int(a.max()) }
                    elif decltype == "double":
                        a = pandas.array(column[:], dtype="float64")
                        table_data.df[title] = a
                        meta["globalRange"] = { "min": float(a.min()), "max": float(a.max()) }
                    else:
                        a = pandas.array(column[:])
                        a[a == b''] = pandas.NA
                        table_data.df[title] = a
                        table_data.df[title] = table_data.df[title].str.decode("utf-8")

                elif obj == "Table":
                    private_table = create_table_data_from_h5(item)
                    table_data.private_tables[private_table.name] = private_table

            table_data.df = table_data.df.convert_dtypes()
            table_data.column_metadata = id_meta

            return table_data

    except Exception as e:
        print(e)

def read_result_item(name: str, result: h5py.Group) -> ResultItem:
    bins = []
    panes = {}
    all_panes = []
    result_version_too_low: int|None = None
    for table_name in result.keys():
        try:
            table = result[table_name]
            if table.attrs["Object"] == "Table":
                meta = json.loads(table.attrs["Metadata"])
                sys_flags = meta.get("_systemFlags", [])
                if "main-left" in sys_flags:
                    panes["main-left"] = _generate_result_pane(table)
                    all_panes.append((table, meta))
                elif "main-right" in sys_flags:
                    panes["main-right"] = _generate_result_pane(table)
                    all_panes.append((table, meta))
                elif "side-by-image" in sys_flags:
                    panes["side"] = _generate_result_pane(table)
                    all_panes.append((table, meta))
                else:
                    all_panes.append((table, meta))

            elif table.attrs["Object"] == "BinaryLayer":
                pass

            elif table_name == "_FileContent":
                pass

        except PaneDataVersionTooLow as e:
            result_version_too_low = e.version
        except KeyError:
            continue
        except json.decoder.JSONDecodeError:
            continue

    attribute_keys = [ "Application_build_number", "Application_name", "Application_version", "Creation_time", "GA3_recipe_hash", "Username" ]
    attributes = { k: result.attrs.get(k) for k in attribute_keys if k in result.attrs }
    if "Creation_time" in attributes and type(attributes["Creation_time"]) == float:
        attributes["Creation_time"] = int(jdn_to_timestamp(attributes["Creation_time"]))

    if result_version_too_low is not None:
        if "Application_name" in attributes and "Application_version" in attributes:
            error_text = f"Error: Analysis results version is too low ({attributes['Application_name']} {attributes['Application_version']})."
        else:
            error_text = f"Error: Analysis results version is too low ({result_version_too_low})."
        return ResultItem(name=name, load_error=error_text,  binaries=bins)

    if 0 == len(all_panes):
        panes = {}

    elif 1 == len(all_panes) and "layout" in all_panes[0][1].get("_systemFlags", []):
        panes["main-left"] = _generate_result_pane(all_panes[0][0])

    elif not ("main-left" in panes and "main-right" in panes and "side" in panes) and 0 < len(all_panes):
        try:
            panes["main-left"] = _generate_fake_result_pane(all_panes)
        except PaneDataVersionTooLow:
            raise

    return ResultItem(name=name, attributes=attributes, binaries=bins, result_panes=panes)

def _read_runtime_state_from_group(tbl : h5py.Group):
    initialSateColumn = runtimeSateColumn = None
    for colName in tbl.keys():
        column = tbl[colName]
        if "Object" in column.attrs and column.attrs["Object"] == "Column" and "Metadata" in column.attrs:
            meta = json.loads(column.attrs["Metadata"])
            ver = meta.get("ver", 0)
            decltype = meta.get("decltype", "")
            featureName = meta.get("feature", "")
            if featureName == "jsonInitialState" and decltype == "QString":
                if ver < 2:
                    raise PaneDataVersionTooLow(ver)
                initialSateColumn = column
            elif featureName == "jsonRuntimeState" and decltype == "QString":
                if ver < 2:
                    raise PaneDataVersionTooLow(ver)
                runtimeSateColumn = column
    column = runtimeSateColumn or initialSateColumn
    return json.loads(column[0].decode("utf-8") if type(column[0]) == bytes else column[0]) if column else None

def _generate_result_pane(table_group: h5py.Group) -> ResultPane:
    private_tables = {}
    for table_name in table_group.keys():
        table = table_group[table_name]
        if table.attrs["Object"] != "Table":
            continue
        private_tables[table_name] = table.name
    return ResultPane(
        private_table_locations = private_tables,
        state = _read_runtime_state_from_group(table_group)
        )

def _generate_fake_result_pane(table_meta: list[tuple[h5py.Group, dict[str, any]]]) -> ResultPane:
    tabs = []
    private_tables = {}
    curr_name_ord = ord("a")
    for tbl, meta in table_meta:
        sys_flags = meta.get("_systemFlags", [])
        parameterUuid = meta.get("parameterUuid", "")
        if "result" not in sys_flags:
            continue
        if "html" in sys_flags or "webpage" in sys_flags:
            rt_state = _read_runtime_state_from_group(tbl)
            table_name = chr(curr_name_ord)
            tab = rt_state["panes"][0]["state"]["tabs"][0]
            orig_table_name = tab["state"]["_tableName"]
            tab["state"]["_tableName"] = table_name
            tabs.append(tab)
            private_tables[table_name] = tbl[orig_table_name].name
            curr_name_ord += 1
        else:
            table_name = chr(curr_name_ord)
            state = dict(_tableName=table_name, _tableParamUuid=parameterUuid, tableRowVisibility="all")
            tabs.append(dict(className="LimDataGrid", state=state))
            private_tables[table_name] = tbl.name
            curr_name_ord += 1

    return ResultPane(
        private_table_locations = private_tables,
        state = dict(panes=[ dict(className="LimPane", state=dict(tabs=tabs)) ])
        )

def _make_unique_name(name: str, keys: list[str]) -> str:
    index = 1
    original = name
    while name in keys:
        name += f"{original}_{index}"
        index += 1
    return name

def jdn_to_timestamp(jdn):
   return (jdn - 2_440_587.5) * 86_400.0