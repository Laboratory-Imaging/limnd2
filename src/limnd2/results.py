from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import errno, enum, json

from typing import TYPE_CHECKING, Any
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
    metadata: dict[str, Any] = field(default_factory=dict)
    private_tables: dict[str, TableData] = field(default_factory=dict)
    column_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    """A binary label layer stored in an analysis-result HDF5 file.

    The pixel data remains in the HDF5 file and is loaded on demand by
    :func:`read_result_binary_data`.  ``location`` is the absolute HDF5 group
    path of the layer; levels map pyramid levels to the available frame
    indices.
    """

    result_name: str
    name: str
    location: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    levels: dict[int, tuple[int, ...]] = field(default_factory=dict)

@dataclass(kw_only=True)
class ResultPane:
    state: dict[str, Any]
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
    attributes: dict[str, Any] = field(default_factory=dict)
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
        results: dict[str, Any] = {}
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
    return {}

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


def _h5_value_to_python(value: Any) -> Any:
    """Convert HDF5 attribute scalars into ordinary, displayable values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and getattr(value, "shape", ()) == ():
        return _h5_value_to_python(value.item())
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _read_binary_result_item(
    result_name: str, group_name: str, binary: h5py.Group
) -> BinaryResultItem:
    """Read a result binary's catalog metadata without loading its pixels."""
    h5py = _require_h5py()
    metadata = {key: _h5_value_to_python(value) for key, value in binary.attrs.items()}
    levels: dict[int, tuple[int, ...]] = {}
    dtype: str | None = None
    shape: tuple[int, ...] = ()

    for level_name, level_group in binary.items():
        try:
            level = int(level_name)
        except ValueError:
            continue
        if not isinstance(level_group, h5py.Group):
            continue

        frames: list[int] = []
        for frame_name, frame_group in level_group.items():
            try:
                frame = int(frame_name)
            except ValueError:
                continue
            if not isinstance(frame_group, h5py.Group) or "image" not in frame_group:
                continue
            image = frame_group["image"]
            if not isinstance(image, h5py.Dataset):
                continue
            frames.append(frame)
            if not shape:
                shape = tuple(image.shape)
                dtype = str(image.dtype)
        if frames:
            levels[level] = tuple(sorted(frames))

    return BinaryResultItem(
        result_name=result_name,
        name=str(metadata.get("binName") or metadata.get("Id") or group_name),
        location=binary.name,
        metadata=metadata,
        dtype=dtype,
        shape=shape,
        levels=levels,
    )


def read_result_binary_data(
    h5_filename: str | Path,
    binary: BinaryResultItem,
    *,
    frame_index: int,
    downsample_level: int = 0,
    rect: tuple[int, int, int, int] | None = None,
):
    """Load one result-binary frame, optionally cropped to ``(x, y, w, h)``."""
    h5py = _require_h5py()
    try:
        with h5py.File(h5_filename, "r") as h5:
            path = f"{binary.location}/{downsample_level}/{frame_index}/image"
            image = h5[path]
            if not isinstance(image, h5py.Dataset):
                raise ValueError(f"Result binary data at {path} is not a dataset.")
            if rect is None:
                return image[:]
            x, y, width, height = rect
            if x < 0 or y < 0 or width < 0 or height < 0:
                raise ValueError("Result binary crop values must be non-negative.")
            return image[y:y + height, x:x + width]
    except KeyError as exc:
        raise KeyError(
            f"Result binary {binary.name!r} has no level {downsample_level}, "
            f"frame {frame_index}."
        ) from exc

def read_result_item(name: str, result: h5py.Group) -> ResultItem:
    bins = []
    panes = {}
    all_panes = []
    result_version_too_low: int|None = None
    for table_name in result.keys():
        try:
            table = result[table_name]
            object_type = _h5_value_to_python(table.attrs["Object"])
            if object_type == "Table":
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

            elif object_type in ("Binary", "BinaryLayer"):
                bins.append(_read_binary_result_item(name, table_name, table))

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

def _generate_fake_result_pane(table_meta: list[tuple[h5py.Group, dict[str, Any]]]) -> ResultPane:
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
