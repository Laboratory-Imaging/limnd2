"""
Script to list all nd2 file and some of its metadata in a table.

Example usage:
    > index.py "C:/Users/<username>/Desktop" -r                 recursively lists information about all nd2 files in Desktop

Use -h flag to list all options for this sctipt:
    > index.py -h
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import RawTextHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Sequence,
    TypedDict,
    cast,
    no_type_check,
)

import limnd2

original_print = print

try:
    import rich
    print = rich.print
except ImportError:
    rich = None


class Record(TypedDict):
    """Dict returned by `index_file`."""

    Path: str
    Name: str
    Version: str
    Size: str
    Modified: str
    Experiment: str
    Frames: int
    Dtype: str
    Bits: int
    Resolution: str
    Channels: int
    Binary: str
    Software: str
    Grabber: str


HEADERS = list(Record.__annotations__)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # YYYY-MM-DD HH:MM:SS


def index_file(path: Path) -> Record:
    """Return a dict with the index file data."""

    file_version = None
    file_obj = None
    bin_info = ""
    try:
        file_obj = limnd2.Nd2Reader(str(path.resolve()))
        raster_bin_count = len(file_obj.chunker.binaryRasterMetadata)
        if 0 < raster_bin_count:
            bin_info = f"{raster_bin_count}x BIN"
        else:
            rle_bin_count = len(file_obj.chunker.binaryRleMetadata)
            if 0 < rle_bin_count:
                bin_info = f"{rle_bin_count}x RLEv{file_obj.chunker.rleBinaryVersion()}"
    except limnd2.UnsupportedChunkmapError as e:
        file_version = e.file_version

    def size_fmt(size):
        kB = 1024
        MB = kB*1024
        GB = MB*1024
        TB = GB*1024
        if TB < size:
            return f"{size/TB:.1f} TB"
        if GB < size:
            return f"{size/GB:.1f} GB"
        if MB < size:
            return f"{size/MB:.1f} MB"
        if kB < size:
            return f"{size/kB:.1f} kB"
        return f"{size} B"

    #if(file_obj.imageAttributes.uiWidth != file_obj.imageAttributes.uiTileWidth):
    #    print(path.name, "\n", file_obj.imageAttributes)
    """
    print(path.name)

    for e in file_obj.experiment:
        if e.wsMeasProbes != "":
            copy = e.__dict__
            del copy["ppNextLevelEx"]
            #print(copy["uLoopPars"])
            print(copy)
    """
    #print(path.name, "\n", "\n\n".join([f"{e}" for e in file_obj.experiment]), "\n\n\n")

    if file_obj is not None:
        return Record({
            "Path": str(path.parent),
            "Name": path.name,
            "Version": f"{file_obj.version[0]}.{file_obj.version[1]}",
            "Size": size_fmt(path.stat().st_size),
            "Modified": datetime.fromtimestamp(path.stat().st_mtime).strftime('%x %X'),
            "Experiment": ", ".join([f"{e.shortName} ({e.count})" for e in file_obj.experiment]) if file_obj.experiment else "",
            "Frames": max(1, file_obj.imageAttributes.frameCount),
            "Dtype": file_obj.imageAttributes.dtype.__name__,
            "Bits": file_obj.imageAttributes.uiBpcSignificant,
            "Resolution": f"{file_obj.imageAttributes.uiWidth} x {file_obj.imageAttributes.uiHeight}",
            "Channels": file_obj.imageAttributes.componentCount,
            "Binary": bin_info,
            "Software": f'{file_obj.appInfo.m_SWNameString} {file_obj.appInfo.m_VersionString}',
            "Grabber": file_obj.appInfo.m_GrabberString
        })
    else:
        return Record({
            "Path": str(path.parent),
            "Name": path.name,
            "Version": f"{file_version[0]}.{file_version[1]}" if file_version is not None else "",
            "Size": size_fmt(path.stat().st_size),
            "Modified": datetime.fromtimestamp(path.stat().st_mtime).strftime('%x %X'),
            "Frames": 1,
            "Experiment": "",
            "Dtype": "",
            "Bits": 0,
            "Resolution": "",
            "Channels": 1,
            "Binary": "",
            "Software": "",
            "Grabber": "",
        })


def _gather_files(
    paths: Iterable[Path], recurse: bool = False, glob: str = "*.nd2"
) -> Iterator[Path]:
    """Return a generator of all files in the given path."""
    for p in paths:
        if p.is_dir():
            yield from p.rglob(glob) if recurse else p.glob(glob)
        else:
            yield p


def _index_files(
    paths: Iterable[Path], recurse: bool = False, glob: str = "*.nd2", format="table"
) -> list[Record]:

    files = _gather_files(paths, recurse, glob)
    results = []
    error = False

    for file_path in files:
        try:
            results.append(index_file(file_path))
        except Exception as e:
            if not error:
                import sys
                error = True
                if format == "table":
                    print("[red]Following files could not be processed:[/red]")
                else:
                    original_print("Following files could not be processed:", file=sys.stderr)
            if format == "table":
                print(f"\t[white]{file_path}:[/white] [red italic]{e}[/red italic]")
            else:
                original_print(f"\t{file_path}:{e}", file=sys.stderr)

    return results

def _pretty_print_table(data: list[Record], sort_column: str | None = None) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

    except ImportError:
        raise sys.exit(
            "rich is required to print a pretty table. "
            "Install it with `pip install rich`."
        ) from None

    table = Table(show_header=True, header_style="bold")
    headers = list(data[0])

    # add headers, and highlight any sorted columns
    sort_col = ""
    if sort_column:
        sort_col = (sort_column or "").rstrip("-")
        direction = " ↓" if sort_column.endswith("-") else " ↑"
    for header in headers:
        if header == sort_col:
            table.add_column(header + direction, style="green")
        else:
            table.add_column(header)

    for row in data:
        table.add_row(*[_stringify(value) for value in row.values()])

    Console().print(table)


def _stringify(val: Any) -> str:
    if isinstance(val, bool):
        return "✅" if val else ""
    return str(val)


def _print_csv(records: list[Record], skip_header: bool = False) -> None:
    import csv

    writer = csv.DictWriter(sys.stdout, fieldnames=records[0].keys())
    if not skip_header:
        writer.writeheader()
    writer.writerows(records)


def _print_json(records: list[Record]) -> None:
    print(json.dumps(records, indent=2))


def _parse_args(argv: Sequence[str] = ()) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Create an index of important metadata in ND2 files."
        f"\n\nValid column names are:\n{HEADERS!r}",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Path to ND2 file or directory containing ND2 files.",
    )
    parser.add_argument(
        "--recurse",
        "-r",
        action="store_true",
        default=False,
        help="Recursively search directories",
    )
    parser.add_argument(
        "--glob-pattern",
        "-g",
        type=str,
        default="*.nd2",
        help="Glob pattern to search for",
    )
    parser.add_argument(
        "--sort-by",
        "-s",
        default="",
        type=str,
        choices=[*HEADERS, "", *(f"{x}-" for x in HEADERS)],
        metavar="COLUMN_NAME",
        help="Column to sort by. If not specified, the order is not guaranteed. "
        "\nTo sort in reverse, append a hyphen.",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="table" if rich is not None else "json",
        type=str,
        choices=["table", "csv", "json"],
    )
    parser.add_argument(
        "--include",
        "-i",
        type=str,
        help="Comma-separated columns to include in the output",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        help="Comma-separated columns to exclude in the output",
    )
    parser.add_argument(
        "--no-header",
        default=False,
        action="store_true",
        help="Don't write the CSV header",
    )
    parser.add_argument(
        "--filter",
        "-F",
        type=str,
        action="append",
        help="Filter the output. Each filter "
        "should be a python expression (string)\nthat evaluates to True or False. "
        "It will be evaluated in the context\nof each row. You can use any of the "
        "column names as variables.\ne.g.: \"Frames > 50 and 'T' in Experiment\". (May "
        "be used multiple times).",
    )

    return parser.parse_args(argv or sys.argv[1:])


@no_type_check
def _filter_data(
    data: list[Record],
    sort_by: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    filters: Sequence[str] = (),
) -> list[Record]:
    """Filter and sort the data.

    Parameters
    ----------
    data : list[Record]
        the data to filter
    sort_by : str | None, optional
        Name of column to sort by, by default None
    include : str | None, optional
        Comma-separated list of columns to include, by default None
    exclude : str | None, optional
        Comma-separated list of columns to exclude, by default None
    filters : Sequence[str], optional
        Sequence of python expression strings to filter the data, by default ()

    Returns
    -------
    list[Record]
        _description_
    """
    includes = include.split(",") if include else []
    unrecognized = set(includes) - set(HEADERS)
    if unrecognized:  # pragma: no cover
        print(f"Unrecognized columns: {', '.join(unrecognized)}", file=sys.stderr)
        includes = [x for x in includes if x not in unrecognized]

    if sort_by in ["Size", "Size-"]:

        def to_bytes(size):
            number, unit = size.split()
            number = float(number)

            size_units = {
                'B': 1,
                'KB': 1024,
                'MB': 1024 ** 2,
                'GB': 1024 ** 3,
                'TB': 1024 ** 4
            }

            return number * size_units[unit.upper()]

        if sort_by.endswith("-"):
            data.sort(key=lambda x: to_bytes(x[sort_by[:-1]]), reverse=True)
        else:
            data.sort(key=lambda x: to_bytes(x[sort_by]))

    if sort_by in ["Modified", "Modified-"]:

        if sort_by.endswith("-"):
            data.sort(key=lambda x: datetime.strptime(x[sort_by[:-1]], "%m/%d/%y %H:%M:%S"), reverse=True)
        else:
            data.sort(key=lambda x: datetime.strptime(x[sort_by], "%m/%d/%y %H:%M:%S"))

    elif sort_by:
        if sort_by.endswith("-"):
            data.sort(key=lambda x: x[sort_by[:-1]], reverse=True)
        else:
            data.sort(key=lambda x: x[sort_by])

    if includes:
        # preserve order of to_include
        data = [{h: row[h] for h in includes} for row in data]

    to_exclude = cast("list[str]", exclude.split(",") if exclude else [])

    if to_exclude:
        data = [{h: row[h] for h in HEADERS if h not in to_exclude} for row in data]

    if filters:
        # filters are in the form of a string expression, to be evaluated
        # against each row. For example, "'TimeLoop' in experiment"
        for f in filters:
            try:
                data = [row for row in data if bool(eval(f, None, row))]  # noqa: S307
            except Exception as e:  # pragma: no cover
                print(f"Error evaluating filter {f!r}: {e}", file=sys.stderr)
                sys.exit(1)

    return data


def main(argv: Sequence[str] = ()) -> None:
    """Index ND2 files and print the results as a table."""
    args = _parse_args(argv)

    data = _index_files(paths=args.paths, recurse=args.recurse, glob=args.glob_pattern, format=args.format)

    if not data:
        print("[red]No ND2 files found.[/red]")
        return

    data = _filter_data(
        data,
        sort_by=args.sort_by,
        include=args.include,
        exclude=args.exclude,
        filters=args.filter,
    )

    if args.format == "table":
        _pretty_print_table(data, args.sort_by)
    elif args.format == "csv":
        _print_csv(data, args.no_header)
    elif args.format == "json":
        _print_json(data)


if __name__ == "__main__":
    main()